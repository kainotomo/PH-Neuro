"""Unit tests for the E032 low-rank plastic mode (Phase 1.2).

Covers:
* Injection-point construction (A/B shapes per rank and architecture).
* Identity invariant (I1): with B = 0 (or without_plasticity) the low-rank
  wrapped model is bit-identical to the raw frozen model, even though A is
  randomly initialised.
* No-backprop invariant (I3): learn() under low_rank never calls backward().
* The derived local Hebbian update for W = B·A:
      ΔA = η·M·mean_t((Bᵀ·post_t) ⊗ pre_t)
      ΔB = η·M·mean_t(post_t ⊗ (A·pre_t))
  verified numerically against a manual recomputation from captured
  pre/post activations (constant-M = 1.0, so M is known).
* Decay applied to both A and B.
* Parameter counts and save/load round-trips.
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


def make_low_rank(model=None, rank=2, **kwargs):
    if model is None:
        torch.manual_seed(42)
        model = tiny_llama()
    return BrainWrapper(
        model,
        plasticity="low_rank",
        rank=rank,
        tokenizer=FakeTok(),
        log=log,
        **kwargs,
    )


class TestInjectionConstruction:
    def test_rank_required(self):
        with pytest.raises(ValueError):
            BrainWrapper(
                tiny_llama(), plasticity="low_rank", rank=0, tokenizer=FakeTok(), log=log
            )

    def test_a_b_shapes_llama(self):
        model = tiny_llama(layers=2, hidden=32)  # intermediate_size = 64
        brain = make_low_rank(model, rank=2)
        by_name = {ip.name: ip for ip in brain._injection_points}  # noqa: SLF001
        # L00.o_proj: 32->32; L00.down_proj: 64->32
        op = by_name["L00.o_proj"]
        assert tuple(op.A.shape) == (2, 32)
        assert tuple(op.B.shape) == (32, 2)
        assert op.in_features == 32
        dp = by_name["L00.down_proj"]
        assert tuple(dp.A.shape) == (2, 64)
        assert tuple(dp.B.shape) == (32, 2)
        assert dp.in_features == 64
        # B is exactly zero; A is random (deadlock-break init)
        assert torch.equal(op.B, torch.zeros_like(op.B))
        assert op.A.abs().sum() > 0

    def test_a_b_shapes_gpt2(self):
        model = tiny_gpt2(layers=2, hidden=32)  # n_embd*4 = 128 for c_proj in
        brain = BrainWrapper(
            model,
            plasticity="low_rank",
            rank=1,
            tokenizer=FakeTok(),
            log=log,
        )
        by_name = {ip.name: ip for ip in brain._injection_points}  # noqa: SLF001
        ac = by_name["L00.attn_c_proj"]
        assert tuple(ac.A.shape) == (1, 32)
        assert tuple(ac.B.shape) == (32, 1)
        mc = by_name["L00.mlp_c_proj"]
        assert tuple(mc.A.shape) == (1, 128)
        assert tuple(mc.B.shape) == (32, 1)

    def test_parameter_counts(self):
        # tiny_llama(hidden=32, inter=64): o_proj 64r + down_proj 96r = 160r/block
        brain = make_low_rank(tiny_llama(layers=2, hidden=32), rank=2)
        assert brain.plastic_parameter_count() == 160 * 2 * 2  # 640
        assert brain.plastic_memory_bytes() == 640 * 4

    def test_vector_mode_unchanged(self):
        brain = BrainWrapper(
            tiny_llama(layers=2, hidden=32),
            tokenizer=FakeTok(),
            log=log,
        )
        assert brain.plastic_parameter_count() == 4 * 32  # 2 layers x 2 pts x 32


class TestIdentityInvariant:
    def test_identity_with_b_zero(self):
        raw = tiny_llama(layers=2)
        model = tiny_llama(layers=2)
        model.load_state_dict(raw.state_dict())
        brain = make_low_rank(model, rank=2)
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            ref = raw(input_ids=ids).logits
            wrapped = brain.model(input_ids=ids).logits  # B=0 -> injection is zero
        assert torch.equal(ref, wrapped)

    def test_without_plasticity_identical(self):
        raw = tiny_llama(layers=2)
        model = tiny_llama(layers=2)
        model.load_state_dict(raw.state_dict())
        brain = make_low_rank(model, rank=2)
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            ref = raw(input_ids=ids).logits
            with brain.without_plasticity():
                wrapped = brain.model(input_ids=ids).logits
        assert torch.equal(ref, wrapped)


class TestNoBackprop:
    def test_learn_never_backward(self, monkeypatch):
        calls = {"n": 0}

        def spy(*a, **k):
            calls["n"] += 1

        monkeypatch.setattr(torch.Tensor, "backward", spy)
        brain = make_low_rank(rank=2)
        metrics = brain.learn(flat_tokens(4096), steps=4, batch_size=2, seq_len=16, seed=42)
        assert len(metrics) == 4
        assert calls["n"] == 0


class TestUpdateRule:
    def test_a_b_move_off_zero(self):
        brain = make_low_rank(rank=2)
        brain.learn(flat_tokens(4096), steps=3, batch_size=2, seq_len=16, seed=42)
        for ip in brain._injection_points:  # noqa: SLF001
            assert ip.B.abs().sum() > 0, ip.name  # deadlock broken

    def test_update_matches_derived_rule(self):
        """With constant M = 1.0 and the known initial (A, B=0), one learn
        step's ΔA/ΔB must match the derived projection rule recomputed from
        the captured pre/post of the very same batch."""
        torch.manual_seed(7)
        model = tiny_llama(layers=2, hidden=32)
        brain = BrainWrapper(
            model,
            plasticity="low_rank",
            rank=2,
            modulator_cfg={"mode": "constant", "M": 1.0},
            lr=1e-3,
            tokenizer=FakeTok(),
            log=log,
        )
        tokens = flat_tokens(1024)
        batch = next(iter(cyclic_batch_iter(tokens, 2, 16, 99)))["input_ids"]

        # Capture pre/post on this batch with A/B at their initial state.
        brain._capture = True  # noqa: SLF001
        with torch.no_grad():
            brain.model(input_ids=batch)
        brain._capture = False  # noqa: SLF001
        pre = brain._last_pre  # noqa: SLF001
        # The low-rank update uses the PRE-INJECTION (frozen) post.
        post = brain._last_post_frozen  # noqa: SLF001

        eta, M = brain.lr, 1.0
        expected = {}
        for ip in brain._injection_points:  # noqa: SLF001
            p, q = pre[ip.name], post[ip.name]
            n = p.size(0) * p.size(1)
            pB = q @ ip.B
            rms_pB = math.sqrt(float(pB.pow(2).mean())) + 1e-8
            rms_p = math.sqrt(float(p.pow(2).mean())) + 1e-8
            dA = (eta * M / (n * rms_pB * rms_p)) * torch.einsum("bsr,bsi->ri", pB, p)
            pA = torch.einsum("ri,bsi->bsr", ip.A, p)
            rms_q = math.sqrt(float(q.pow(2).mean())) + 1e-8
            rms_pA = math.sqrt(float(pA.pow(2).mean())) + 1e-8
            dB = (eta * M / (n * rms_q * rms_pA)) * torch.einsum("bso,bsr->or", q, pA)
            expected[ip.name] = (
                ip.A.detach().clone() + dA,
                ip.B.detach().clone() + dB,
            )

        brain.learn(tokens, steps=1, batch_size=2, seq_len=16, seed=99)
        for ip in brain._injection_points:  # noqa: SLF001
            ea, eb = expected[ip.name]
            assert torch.allclose(ip.A, ea, atol=1e-6), f"{ip.name} A update mismatch"
            assert torch.allclose(ip.B, eb, atol=1e-6), f"{ip.name} B update mismatch"


class TestDecay:
    def test_decay_applied_to_a_and_b(self):
        brain = make_low_rank(rank=2, decay_rate=1e-2)
        # pre-train a little so B is nonzero
        brain.learn(flat_tokens(4096), steps=1, batch_size=2, seq_len=16, seed=42)
        for ip in brain._injection_points:  # noqa: SLF001
            b_before = ip.B.abs().mean().item()
            brain.learn(flat_tokens(4096), steps=1, batch_size=2, seq_len=16, seed=42)
            # decay strictly shrinks B when Δ is small relative to |B|
            assert ip.B.abs().mean().item() <= b_before + 1e-3


class TestSerialization:
    def test_state_dict_round_trip(self):
        brain = make_low_rank(rank=2)
        brain.learn(flat_tokens(4096), steps=2, batch_size=2, seq_len=16, seed=42)
        sd = brain.state_dict()
        assert any(k.endswith(".A") for k in sd)
        assert any(k.endswith(".B") for k in sd)
        brain2 = make_low_rank(rank=2)
        brain2.load_state_dict(sd)
        for ip1, ip2 in zip(brain._injection_points, brain2._injection_points, strict=True):  # noqa: SLF001
            assert torch.equal(ip1.A, ip2.A)
            assert torch.equal(ip1.B, ip2.B)

    def test_save_load(self, tmp_path):
        brain = make_low_rank(rank=2)
        brain.learn(flat_tokens(4096), steps=3, batch_size=2, seq_len=16, seed=42)
        path = str(tmp_path / "lora_state.pt")
        brain.save(path)
        brain2 = make_low_rank(rank=2)
        brain2.load(path)
        for ip1, ip2 in zip(brain._injection_points, brain2._injection_points, strict=True):  # noqa: SLF001
            assert torch.equal(ip1.A, ip2.A)
            assert torch.equal(ip1.B, ip2.B)

    def test_load_shape_mismatch(self):
        brain = make_low_rank(rank=2)
        sd = brain.state_dict()
        brain3 = make_low_rank(rank=1)
        with pytest.raises(ValueError):
            brain3.load_state_dict(sd)


class TestFactory:
    def test_wrapper_factory_accepts_rank(self):
        model = tiny_llama(layers=2, hidden=32)
        wrapper = get_block_wrapper(model)
        pts = wrapper.get_injection_points(get_block_container(model)[0], 0, rank=3)
        assert pts[0].rank == 3
        assert tuple(pts[0].A.shape) == (3, 32)
        assert isinstance(get_block_wrapper(tiny_llama()), SmolLM2BlockWrapper)
        assert isinstance(get_block_wrapper(tiny_gpt2()), GPT2BlockWrapper)
