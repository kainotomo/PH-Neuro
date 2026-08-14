"""Unit tests for the E033 predictive-coding plastic mode (Phase 1.3).

Covers:
* Injection-point construction (A/B/U/V shapes per rank/inv_rank and
  architecture).
* Identity invariant (I1): with B = 0 (injection zero) and V = 0 (inverse
  x̂ = 0) the wrapped model is bit-identical to the raw frozen model.
* No-backprop invariant (I3): learn() under ``predictive_coding`` never calls
  ``backward()``.
* The derived error-driven update for W = B·A with reconstruction error
  ``ε = pre − U@(V@post)``:
      ΔA = η·M·mean((Bᵀ·post) ⊗ ε)/(rms(Bᵀpost)·rms(ε))
      ΔB = η·M·mean(post ⊗ (A·ε))/(rms(post)·rms(A·ε))
  and the local recirculation inverse update:
      ΔV = η_inv·mean((Uᵀε) ⊗ post)/(rms(Uᵀε)·rms(post))
      ΔU = η_inv·mean(ε ⊗ (V·post))/(rms(ε)·rms(V·post))
  verified numerically against a manual recomputation from captured pre/post
  activations (constant-M = 1.0, so M is known).
* Parameter budget: A+B in predictive-coding mode equals A+B in low-rank mode
  at the same rank (the matched-budget comparison to the E032 LoRA baseline);
  the inverse U+V is auxiliary and reported separately.
* Serialization round-trips (A/B/U/V) and shape validation.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from ph_neuro.brain import BrainWrapper
from ph_neuro.brain.block_wrappers import (
    GPT2BlockWrapper,
    SmolLM2BlockWrapper,
    get_block_container,
    get_block_wrapper,
)
from ph_neuro.brain.brain_wrapper import cyclic_batch_iter
from tests.brain._models import random_token_ids, tiny_gpt2, tiny_llama
from tests.brain.test_brain_wrapper import FakeTok, flat_tokens

log = logging.getLogger("test")

INV_RANK = 8


def make_pc(model=None, rank=1, inv_rank=INV_RANK, **kwargs):
    if model is None:
        torch.manual_seed(42)
        model = tiny_llama()
    return BrainWrapper(
        model,
        plasticity="predictive_coding",
        rank=rank,
        inv_rank=inv_rank,
        tokenizer=FakeTok(),
        log=log,
        **kwargs,
    )


class TestInjectionConstruction:
    def test_rank_required(self):
        with pytest.raises(ValueError):
            BrainWrapper(
                tiny_llama(), plasticity="predictive_coding", rank=0,
                tokenizer=FakeTok(), log=log,
            )

    def test_abuv_shapes_llama(self):
        model = tiny_llama(layers=2, hidden=32)  # intermediate_size = 64
        brain = make_pc(model, rank=1, inv_rank=4)
        by_name = {ip.name: ip for ip in brain._injection_points}  # noqa: SLF001
        op = by_name["L00.o_proj"]
        assert tuple(op.A.shape) == (1, 32)
        assert tuple(op.B.shape) == (32, 1)
        # U: (d_in, r_inv), V: (r_inv, d_out)
        assert tuple(op.U.shape) == (32, 4)
        assert tuple(op.V.shape) == (4, 32)
        dp = by_name["L00.down_proj"]
        assert tuple(dp.A.shape) == (1, 64)
        assert tuple(dp.B.shape) == (32, 1)
        assert tuple(dp.U.shape) == (64, 4)
        assert tuple(dp.V.shape) == (4, 32)
        # B = 0 (injection zero) and V = 0 (x̂ = 0) at construction; A/U random.
        assert torch.equal(op.B, torch.zeros_like(op.B))
        assert torch.equal(op.V, torch.zeros_like(op.V))
        assert op.A.abs().sum() > 0
        assert op.U.abs().sum() > 0

    def test_abuv_shapes_gpt2(self):
        model = tiny_gpt2(layers=2, hidden=32)
        brain = BrainWrapper(
            model, plasticity="predictive_coding", rank=1, inv_rank=4,
            tokenizer=FakeTok(), log=log,
        )
        by_name = {ip.name: ip for ip in brain._injection_points}  # noqa: SLF001
        ac = by_name["L00.attn_c_proj"]
        assert tuple(ac.A.shape) == (1, 32)
        assert tuple(ac.U.shape) == (32, 4)
        assert tuple(ac.V.shape) == (4, 32)
        mc = by_name["L00.mlp_c_proj"]
        assert tuple(mc.A.shape) == (1, 128)
        assert tuple(mc.U.shape) == (128, 4)

    def test_matched_budget_vs_low_rank(self):
        """A+B param count is identical to low_rank at the same rank → the
        matched-budget comparison to the E032 LoRA baseline is exact."""
        model = tiny_llama(layers=2, hidden=32)
        lr_brain = BrainWrapper(
            model, plasticity="low_rank", rank=1, tokenizer=FakeTok(), log=log,
        )
        pc_brain = BrainWrapper(
            model, plasticity="predictive_coding", rank=1, inv_rank=8,
            tokenizer=FakeTok(), log=log,
        )
        assert pc_brain.plastic_parameter_count() == lr_brain.plastic_parameter_count()
        # inverse is auxiliary and reported separately
        assert pc_brain.inverse_parameter_count() > 0
        assert lr_brain.inverse_parameter_count() == 0

    def test_vector_mode_has_no_inverse(self):
        brain = BrainWrapper(
            tiny_llama(layers=2, hidden=32), tokenizer=FakeTok(), log=log,
        )
        assert brain.inverse_parameter_count() == 0


class TestIdentityInvariant:
    def test_identity_with_b_zero(self):
        raw = tiny_llama(layers=2)
        model = tiny_llama(layers=2)
        model.load_state_dict(raw.state_dict())
        brain = make_pc(model, rank=1)
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            ref = raw(input_ids=ids).logits
            wrapped = brain.model(input_ids=ids).logits  # B=0, V=0 -> no-op
        assert torch.equal(ref, wrapped)

    def test_without_plasticity_identical(self):
        raw = tiny_llama(layers=2)
        model = tiny_llama(layers=2)
        model.load_state_dict(raw.state_dict())
        brain = make_pc(model, rank=1)
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            ref = raw(input_ids=ids).logits
            with brain.without_plasticity():
                wrapped = brain.model(input_ids=ids).logits
        assert torch.equal(ref, wrapped)


class TestNoBackprop:
    def test_learn_never_backward(self, monkeypatch):
        calls = {"n": 0}

        def spy(*a, **k):  # noqa: ARG002
            calls["n"] += 1

        monkeypatch.setattr(torch.Tensor, "backward", spy)
        brain = make_pc(rank=1)
        metrics = brain.learn(
            flat_tokens(4096), steps=4, batch_size=2, seq_len=16, seed=42
        )
        assert len(metrics) == 4
        assert calls["n"] == 0


class TestUpdateRule:
    def test_b_and_v_move_off_zero(self):
        brain = make_pc(rank=1)
        brain.learn(flat_tokens(4096), steps=3, batch_size=2, seq_len=16, seed=42)
        for ip in brain._injection_points:  # noqa: SLF001
            assert ip.B.abs().sum() > 0, ip.name  # plastic bootstrapped
            assert ip.V.abs().sum() > 0, ip.name  # inverse bootstrapped

    def test_update_matches_derived_rule(self):
        """With constant M = 1.0 and the known initial state, one learn step's
        ΔA/ΔB/ΔU/ΔV must match the derived error-driven rule recomputed from
        the captured pre/post of the very same batch."""
        torch.manual_seed(7)
        model = tiny_llama(layers=2, hidden=32)
        brain = BrainWrapper(
            model,
            plasticity="predictive_coding",
            rank=1,
            inv_rank=4,
            modulator_cfg={"mode": "constant", "M": 1.0},
            lr=1e-3,
            inv_lr=1e-3,
            inv_decay=0.0,
            decay_rate=0.0,
            tokenizer=FakeTok(),
            log=log,
        )
        tokens = flat_tokens(1024)
        batch = next(iter(cyclic_batch_iter(tokens, 2, 16, 99)))["input_ids"]

        # Capture pre/post on this batch with A/B/U/V at their initial state.
        brain._capture = True  # noqa: SLF001
        with torch.no_grad():
            brain.model(input_ids=batch)
        brain._capture = False  # noqa: SLF001
        pre = brain._last_pre  # noqa: SLF001
        post = brain._last_post_frozen  # noqa: SLF001

        eta, M, eta_inv = brain.lr, 1.0, brain.inv_lr
        expected = {}
        for ip in brain._injection_points:  # noqa: SLF001
            p, q = pre[ip.name], post[ip.name]
            n = p.size(0) * p.size(1)
            vpost = torch.einsum("ro,bso->bsr", ip.V, q)
            xhat = torch.einsum("ir,bsr->bsi", ip.U, vpost)
            eps = p - xhat

            rms_q = math.sqrt(float(q.pow(2).mean())) + 1e-8
            rms_e = math.sqrt(float(eps.pow(2).mean())) + 1e-8
            ut = torch.einsum("ir,bsi->bsr", ip.U, eps)
            rms_ut = math.sqrt(float(ut.pow(2).mean())) + 1e-8
            dV = (eta_inv / (n * rms_ut * rms_q)) * torch.einsum("bsr,bso->ro", ut, q)
            rms_vp = math.sqrt(float(vpost.pow(2).mean())) + 1e-8
            dU = (eta_inv / (n * rms_e * rms_vp)) * torch.einsum("bsi,bsr->ir", eps, vpost)

            pB = q @ ip.B
            rms_pB = math.sqrt(float(pB.pow(2).mean())) + 1e-8
            dA = (eta * M / (n * rms_pB * rms_e)) * torch.einsum("bsr,bsi->ri", pB, eps)
            pA = torch.einsum("ri,bsi->bsr", ip.A, eps)
            rms_pA = math.sqrt(float(pA.pow(2).mean())) + 1e-8
            dB = (eta * M / (n * rms_q * rms_pA)) * torch.einsum("bso,bsr->or", q, pA)

            expected[ip.name] = (
                ip.A.detach().clone() + dA,
                ip.B.detach().clone() + dB,
                ip.U.detach().clone() + dU,
                ip.V.detach().clone() + dV,
            )

        brain.learn(tokens, steps=1, batch_size=2, seq_len=16, seed=99)
        for ip in brain._injection_points:  # noqa: SLF001
            ea, eb, eu, ev = expected[ip.name]
            assert torch.allclose(ip.A, ea, atol=1e-6), f"{ip.name} A mismatch"
            assert torch.allclose(ip.B, eb, atol=1e-6), f"{ip.name} B mismatch"
            assert torch.allclose(ip.U, eu, atol=1e-6), f"{ip.name} U mismatch"
            assert torch.allclose(ip.V, ev, atol=1e-6), f"{ip.name} V mismatch"


class TestInverseDecay:
    def test_inverse_decay_applied(self):
        brain = make_pc(rank=1, inv_decay=1e-2)
        brain.learn(flat_tokens(4096), steps=1, batch_size=2, seq_len=16, seed=42)
        for ip in brain._injection_points:  # noqa: SLF001
            v_before = ip.V.abs().mean().item()
            brain.learn(flat_tokens(4096), steps=1, batch_size=2, seq_len=16, seed=42)
            assert ip.V.abs().mean().item() <= v_before + 1e-3


class TestSerialization:
    def test_state_dict_round_trip(self):
        brain = make_pc(rank=1)
        brain.learn(flat_tokens(4096), steps=2, batch_size=2, seq_len=16, seed=42)
        sd = brain.state_dict()
        assert any(k.endswith(".A") for k in sd)
        assert any(k.endswith(".B") for k in sd)
        assert any(k.endswith(".U") for k in sd)
        assert any(k.endswith(".V") for k in sd)
        brain2 = make_pc(rank=1)
        brain2.load_state_dict(sd)
        for ip1, ip2 in zip(
            brain._injection_points, brain2._injection_points, strict=True
        ):  # noqa: SLF001
            assert torch.equal(ip1.A, ip2.A)
            assert torch.equal(ip1.B, ip2.B)
            assert torch.equal(ip1.U, ip2.U)
            assert torch.equal(ip1.V, ip2.V)

    def test_save_load(self, tmp_path):
        brain = make_pc(rank=1)
        brain.learn(flat_tokens(4096), steps=3, batch_size=2, seq_len=16, seed=42)
        path = str(tmp_path / "pc_state.pt")
        brain.save(path)
        brain2 = make_pc(rank=1)
        brain2.load(path)
        for ip1, ip2 in zip(
            brain._injection_points, brain2._injection_points, strict=True
        ):  # noqa: SLF001
            assert torch.equal(ip1.A, ip2.A)
            assert torch.equal(ip1.B, ip2.B)
            assert torch.equal(ip1.U, ip2.U)
            assert torch.equal(ip1.V, ip2.V)

    def test_load_shape_mismatch(self):
        brain = make_pc(rank=1, inv_rank=8)
        sd = brain.state_dict()
        brain3 = make_pc(rank=1, inv_rank=4)
        with pytest.raises(ValueError):
            brain3.load_state_dict(sd)


class TestFactory:
    def test_wrapper_factory_accepts_inv_rank(self):
        model = tiny_llama(layers=2, hidden=32)
        wrapper = get_block_wrapper(model)
        pts = wrapper.get_injection_points(
            get_block_container(model)[0], 0, rank=1, inv_rank=4
        )
        assert pts[0].inv_rank == 4
        assert tuple(pts[0].U.shape) == (32, 4)
        assert isinstance(get_block_wrapper(tiny_llama()), SmolLM2BlockWrapper)
        assert isinstance(get_block_wrapper(tiny_gpt2()), GPT2BlockWrapper)


class TestSummary:
    def test_summary_includes_inverse(self):
        brain = make_pc(rank=1)
        s = brain.summary()
        assert s["plasticity"] == "predictive_coding"
        assert s["inv_rank"] == INV_RANK
        assert s["inverse_params"] == brain.inverse_parameter_count()
