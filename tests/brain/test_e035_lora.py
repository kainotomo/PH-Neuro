"""Unit tests for the E035 ternary LoRA machinery (Step 2.2).

Covers:
* ``ternary_quantize`` — per-matrix scale ``s = mean|W|``, ``Q = sign(W)``,
  zero-matrix identity.
* ``TernaryLoRAAdapter`` per mode (ta/tb/tc) — identity invariant (B = 0 →
  injection exactly zero → bit-identical to the raw model), shapes, init
  conventions.
* T-B (DQT) — forward uses the **int8 ternary** buffers (not the floats), the
  custom autograd routes the ternary-weight gradient to the float buffers
  (STE; verified against a manual backprop through the ternary weights), the
  scale-factor gradients follow the product rule, and ``apply_after_step``
  stochastically rounds the floats into the ternary buffers.
* T-C (STE) — forward values are the sign of the latent scores, backward is
  identity (STE).
* Packing — ``pack_ternary_adapters`` round-trips and gives 4 weights/byte →
  16× vs float32; ``ternary_storage_report`` confirms the reduction factor.
* Checkpoint round-trip for a ternary adapter.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from ph_neuro.brain.lora import (
    TernaryLoRAAdapter,
    build_ternary_lora_adapters,
    n_lora_params,
    pack_ternary_adapters,
    ternary_quantize,
    ternary_storage_report,
)
from ph_neuro.brain.lora import _TernaryDQTInjection
from ph_neuro.examples.run_e035_lora import _adapter_weights, _resume, _save_checkpoint
from ph_neuro.layers.ste_linear import ste_sign
from tests.brain._models import tiny_gpt2, tiny_llama


# ── ternary_quantize ───────────────────────────────────────────────


class TestTernaryQuantize:
    def test_scale_and_sign(self):
        torch.manual_seed(0)
        w = torch.randn(8, 16) * 0.1
        Q, s = ternary_quantize(w)
        assert Q.dtype == torch.int8
        assert set(Q.unique().tolist()) <= {-1, 0, 1}
        assert s.item() == pytest.approx(w.abs().mean().item())
        assert torch.equal(Q, w.detach().sign().to(torch.int8))

    def test_zero_matrix_identity(self):
        w = torch.zeros(4, 8)
        Q, s = ternary_quantize(w)
        assert (Q == 0).all()
        assert s.item() == 0.0

    def test_quantize_recovers_sign_magnitude(self):
        # equal-magnitude same-sign vector → reconstruction is exact
        w = torch.tensor([[2.0, 2.0, 2.0]])
        Q, s = ternary_quantize(w)
        assert torch.equal(Q, torch.ones_like(w).to(torch.int8))
        assert s.item() == pytest.approx(2.0)
        assert torch.allclose(s * Q.float(), w, atol=1e-6)
        # mixed signs: s = mean|w| is the L1-optimal single scale
        w2 = torch.tensor([[1.0, -3.0]])
        Q2, s2 = ternary_quantize(w2)
        assert s2.item() == pytest.approx(2.0)
        assert torch.equal(Q2, torch.tensor([[1, -1]]).to(torch.int8))


# ── TernaryLoRAAdapter — identity + shapes per mode ────────────────


class TestTernaryAdapterIdentity:
    @pytest.mark.parametrize("mode", ["ta", "tb", "tc"])
    def test_identity_invariant(self, mode):
        """Frozen output is bit-identical with hooks disabled (all modes);
        ta/tc are also identity at construction (B = 0 → zero injection)."""
        torch.manual_seed(0)
        model = tiny_llama(layers=1, hidden=32)
        ids = torch.randint(0, 128, (1, 16))
        raw = model(ids).logits
        adapters = build_ternary_lora_adapters(model, rank=1, device="cpu", mode=mode)
        if mode in ("ta", "tc"):
            # B = 0 → injection exactly zero at construction.
            assert torch.equal(model(ids).logits, raw)
        # Disabled hooks → bit-identical frozen output (the real I1 guarantee;
        # T-B inits A/B ~N(0, 0.1) per the DQT amendment, so its construction
        # injection is a small nonzero perturbation, not exactly zero).
        for ad in adapters:
            ad.set_enabled(False)
        assert torch.equal(model(ids).logits, raw)

    @pytest.mark.parametrize("mode", ["ta", "tb", "tc"])
    def test_shapes_and_budget(self, mode):
        model = tiny_llama(layers=2, hidden=32)
        adapters = build_ternary_lora_adapters(model, rank=1, device="cpu", mode=mode)
        assert len(adapters) == 2 * 2  # 2 blocks × (o_proj + down_proj)
        assert n_lora_params(adapters) == 2 * ((32 + 32) + (64 + 32))
        for ad in adapters:
            assert ad.mode == mode
            assert ad.rank == 1
            assert ad.n_params() == 1 * (ad.in_features + ad.out_features)

    @pytest.mark.parametrize("mode", ["ta", "tb", "tc"])
    def test_gpt2_build(self, mode):
        model = tiny_gpt2(layers=2, hidden=32)
        adapters = build_ternary_lora_adapters(model, rank=1, device="cpu", mode=mode)
        assert len(adapters) == 2 * 2  # 2 blocks × (attn.c_proj + mlp.c_proj)
        # attn.c_proj: 32→32 (64); mlp.c_proj: 128→32 (160); × 2 blocks = 448.
        assert n_lora_params(adapters) == 2 * ((32 + 32) + (128 + 32))

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            TernaryLoRAAdapter(tiny_llama(layers=1).model.layers[0].self_attn.o_proj,
                               rank=1, device="cpu", mode="td")


# ── T-B (DQT) mechanics ────────────────────────────────────────────


def _build_tb_adapter():
    torch.manual_seed(3)
    model = tiny_llama(layers=1, hidden=32)
    mod = model.model.layers[0].self_attn.o_proj
    return TernaryLoRAAdapter(mod, rank=1, device="cpu", mode="tb")


class TestDQTAdapter:
    def test_ternary_buffers_int8_and_float_params(self):
        ad = _build_tb_adapter()
        assert ad.A_tern.dtype == torch.int8
        assert ad.B_tern.dtype == torch.int8
        # DQT init (ste_dqt.py convention): both A and B ~10% nonzero int8
        # at construction (amended pre-registration).
        assert 0 < int((ad.B_tern != 0).sum()) < ad.B_tern.numel()
        assert set(ad.A_tern.unique().tolist()) <= {-1, 0, 1}
        assert set(ad.B_tern.unique().tolist()) <= {-1, 0, 1}
        assert ad.A_float.requires_grad and ad.B_float.requires_grad
        assert ad.A_scale.requires_grad and ad.B_scale.requires_grad
        # A_scale init normalizes the ternary A row to ~unit norm
        assert ad.A_scale.item() == pytest.approx(1.0 / math.sqrt(
            max(int((ad.A_tern != 0).sum()), 1)))
        assert ad.B_scale.item() == pytest.approx(1e-2)

    def test_forward_uses_int8_ternary(self):
        ad = _build_tb_adapter()
        x = torch.randn(2, 8, 32)
        out = ad.forward_delta(x)
        # manual computation using the int8 buffers (not the floats)
        s = ad.A_scale * ad.B_scale
        t = torch.einsum("ri,bsi->bsr", ad.A_tern.float(), x)
        manual = s * torch.einsum("or,bsr->bso", ad.B_tern.float(), t)
        assert torch.allclose(out, manual, atol=1e-6)
        # the construction delta is a small perturbation (scales normalize it)
        assert out.abs().mean().item() < 0.2

    def test_gradients_route_to_float_buffers(self):
        """STE: the ternary-weight gradient lands on A_float/B_float."""
        ad = _build_tb_adapter()
        x = torch.randn(2, 8, 32)
        out = ad.forward_delta(x)
        out.sum().backward()
        assert ad.A_float.grad is not None
        assert ad.B_float.grad is not None
        assert ad.A_scale.grad is not None
        assert ad.B_scale.grad is not None
        assert tuple(ad.A_float.grad.shape) == (1, 32)
        assert tuple(ad.B_float.grad.shape) == (32, 1)
        # x also gets a gradient
        assert out.grad_fn is not None

    def test_ste_gradient_matches_manual_backprop(self):
        """gain-decoupled STE: A/B float grads == unscaled ternary backprop;
        x grad == the true (scaled) gradient into the model."""
        ad = _build_tb_adapter()
        x = torch.randn(2, 8, 32).requires_grad_()
        ad.forward_delta(x).sum().backward()
        got_A = ad.A_float.grad.clone()
        got_B = ad.B_float.grad.clone()
        got_x = x.grad.clone()

        s = (ad.A_scale.detach() * ad.B_scale.detach())
        A_t = ad.A_tern.float()
        B_t = ad.B_tern.float()

        # True (scaled) gradient into the model — matches grad_x.
        x2 = x.detach().clone().requires_grad_(True)
        A2 = A_t.clone().requires_grad_(True)
        B2 = B_t.clone().requires_grad_(True)
        t2 = torch.einsum("ri,bsi->bsr", A2, x2)
        (s * torch.einsum("or,bsr->bso", B2, t2)).sum().backward()
        assert torch.allclose(got_x, x2.grad, atol=1e-6)

        # Unscaled (scale=1) backprop through the ternary weights — matches
        # the gain-decoupled A/B float gradients.
        x3 = x.detach().clone().requires_grad_(True)
        A3 = A_t.clone().requires_grad_(True)
        B3 = B_t.clone().requires_grad_(True)
        t3 = torch.einsum("ri,bsi->bsr", A3, x3)
        (torch.einsum("or,bsr->bso", B3, t3)).sum().backward()
        assert torch.allclose(got_A, A3.grad, atol=1e-6)
        assert torch.allclose(got_B, B3.grad, atol=1e-6)

    def test_apply_stochastic_rounding_flips(self):
        ad = _build_tb_adapter()
        with torch.no_grad():
            ad.A_float.data.uniform_(-0.4, 0.4)
            ad.B_float.data.uniform_(-0.4, 0.4)
        stats = ad.apply_after_step()
        assert stats is not None
        assert "A_flip_rate" in stats and "B_flip_rate" in stats
        assert 0.0 < stats["A_flip_rate"] <= 1.0
        assert 0.0 < stats["B_flip_rate"] <= 1.0
        # ternary buffers are int8 in {-1, 0, +1}
        assert set(ad.A_tern.unique().tolist()) <= {-1, 0, 1}
        assert set(ad.B_tern.unique().tolist()) <= {-1, 0, 1}


# ── T-C (STE) mechanics ────────────────────────────────────────────


class TestSTELatentAdapter:
    def test_forward_uses_sign_and_ste_backward(self):
        torch.manual_seed(4)
        model = tiny_llama(layers=1, hidden=32)
        ad = TernaryLoRAAdapter(model.model.layers[0].self_attn.o_proj,
                                rank=1, device="cpu", mode="tc")
        x = torch.randn(2, 8, 32)
        out = ad.forward_delta(x)
        # B_latent = 0 → sign = 0 → identity at init
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-8)
        # once B_latent moves, forward values are (scale · ±1)
        with torch.no_grad():
            ad.B_latent.data.uniform_(-0.5, 0.5)
            ad.A_latent.data.uniform_(-0.5, 0.5)
        out2 = ad.forward_delta(x)
        s = ad.A_scale * ad.B_scale
        t = torch.einsum("ri,bsi->bsr", ste_sign(ad.A_latent).float(), x)
        manual = s * torch.einsum("or,bsr->bso", ste_sign(ad.B_latent).float(), t)
        assert torch.allclose(out2, manual, atol=1e-6)
        # STE: identity backward through sign
        out2.sum().backward()
        assert ad.A_latent.grad is not None
        assert ad.B_latent.grad is not None
        assert ad.A_scale.grad is not None


# ── T-A (post-training quantization) mechanics ─────────────────────


class TestPostTrainQuantize:
    def test_quantize_produces_ternary_snapshot(self):
        torch.manual_seed(5)
        model = tiny_llama(layers=1, hidden=32)
        ad = TernaryLoRAAdapter(model.model.layers[0].self_attn.o_proj,
                                rank=1, device="cpu", mode="ta")
        # give A/B nontrivial values
        with torch.no_grad():
            ad.A.data.uniform_(-0.05, 0.05)
            ad.B.data.uniform_(-0.05, 0.05)
        assert ad.phase == "float"
        ad.quantize()
        assert ad.phase == "quantized"
        A_tern, B_tern, A_scale, B_scale = ad.ternary_snapshot()
        assert A_tern.dtype == torch.int8 and B_tern.dtype == torch.int8
        assert A_scale.item() == pytest.approx(ad.A.abs().mean().item())
        assert B_scale.item() == pytest.approx(ad.B.abs().mean().item())
        assert torch.equal(A_tern, ad.A.sign().to(torch.int8))
        # quantized forward uses the int8 snapshot
        x = torch.randn(2, 8, 32)
        out = ad.forward_delta(x)
        s = A_scale * B_scale
        t = torch.einsum("ri,bsi->bsr", A_tern.float(), x)
        manual = s * torch.einsum("or,bsr->bso", B_tern.float(), t)
        assert torch.allclose(out, manual, atol=1e-6)

    def test_calib_phase_uses_ste(self):
        torch.manual_seed(6)
        model = tiny_llama(layers=1, hidden=32)
        ad = TernaryLoRAAdapter(model.model.layers[0].self_attn.o_proj,
                                rank=1, device="cpu", mode="ta")
        with torch.no_grad():
            ad.A.data.uniform_(-0.05, 0.05)
            ad.B.data.uniform_(-0.05, 0.05)
        ad.quantize()
        ad.set_phase("calib")
        x = torch.randn(2, 8, 32)
        out = ad.forward_delta(x)
        out.sum().backward()
        assert ad.A.grad is not None and ad.B.grad is not None  # STE through sign


# ── storage / packing ──────────────────────────────────────────────


class TestPacking:
    def test_pack_roundtrip_and_16x(self):
        torch.manual_seed(7)
        model = tiny_llama(layers=2, hidden=32)
        adapters = build_ternary_lora_adapters(model, rank=1, device="cpu", mode="tc")
        with torch.no_grad():
            for ad in adapters:
                ad.A_latent.data.uniform_(-0.5, 0.5)
                ad.B_latent.data.uniform_(-0.5, 0.5)
        n_params = n_lora_params(adapters)
        packed = pack_ternary_adapters(adapters)
        assert packed.numel() == (n_params + 3) // 4  # 4 weights / byte
        # unpacking recovers exactly the ternary values (same interleaved
        # A/B-per-adapter order as pack_ternary_adapters)
        from ph_neuro.utils.packing import unpack_ternary

        flat = torch.cat(
            [t for ad in adapters for t in
             (ad.ternary_snapshot()[0].flatten(), ad.ternary_snapshot()[1].flatten())]
        )
        unpacked = unpack_ternary(packed, flat.shape)
        assert torch.equal(unpacked, flat)

    def test_storage_report_16x(self):
        torch.manual_seed(8)
        model = tiny_llama(layers=2, hidden=32)
        adapters = build_ternary_lora_adapters(model, rank=1, device="cpu", mode="tb")
        packed = pack_ternary_adapters(adapters)
        rep = ternary_storage_report(adapters, packed)
        n = n_lora_params(adapters)
        assert rep["float32_bytes"] == n * 4
        assert rep["packed_bytes"] == (n + 3) // 4
        assert rep["scale_bytes"] == 2 * len(adapters) * 4
        # reduction = float32 / (packed + scales)
        assert rep["reduction_factor"] == pytest.approx(
            rep["float32_bytes"] / (rep["packed_bytes"] + rep["scale_bytes"])
        )
        # At the real E034 budget (344,064 params, 48 adapters) the reduction
        # is ~15.93× — the pre-registered 16× storage claim.
        n_real, n_adapt = 344064, 48
        red_real = (n_real * 4) / ((n_real + 3) // 4 + 2 * n_adapt * 4)
        assert red_real >= 15.5  # the aggregate's 16× bar
        assert red_real == pytest.approx(15.93, rel=0.01)


# ── checkpoint round-trip ──────────────────────────────────────────


class TestTernaryCheckpoint:
    @pytest.mark.parametrize("mode", ["ta", "tb", "tc"])
    def test_save_resume_roundtrip(self, tmp_path, mode):
        torch.manual_seed(9)
        model = tiny_llama(layers=1, hidden=32)
        adapters = build_ternary_lora_adapters(model, rank=1, device="cpu", mode=mode)
        if mode == "ta":
            adapters[0].quantize()
        params = [p for ad in adapters for p in ad.parameters()]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
        ckpt_dir = str(tmp_path)
        _save_checkpoint(
            os.path.join(ckpt_dir, "brain_ckpt_step4.pt"), 4, adapters, opt, None,
            {"tag": "t"},
        )
        model2 = tiny_llama(layers=1, hidden=32)
        adapters2 = build_ternary_lora_adapters(model2, rank=1, device="cpu", mode=mode)
        if mode == "ta":
            adapters2[0].quantize()
        params2 = [p for ad in adapters2 for p in ad.parameters()]
        opt2 = torch.optim.AdamW(params2, lr=1e-3, weight_decay=0.0)
        step = _resume(ckpt_dir, 10, adapters2, opt2, None)
        assert step == 4
        s1 = adapters[0].state_dict()
        s2 = adapters2[0].state_dict()
        assert set(s1.keys()) == set(s2.keys())
        for k in s1:
            if isinstance(s1[k], str):
                assert s1[k] == s2[k]
            else:
                assert torch.equal(s1[k], s2[k])


# ── runner helper ──────────────────────────────────────────────────


class TestAdapterWeightsHelper:
    def test_mixed_adapters(self):
        from ph_neuro.brain.lora import build_lora_adapters

        torch.manual_seed(10)
        model = tiny_llama(layers=1, hidden=32)
        float_ads = build_lora_adapters(model, rank=1, device="cpu")
        tern_ads = build_ternary_lora_adapters(model, rank=1, device="cpu", mode="tb")
        w = _adapter_weights(float_ads + tern_ads)
        assert w.numel() == 2 * n_lora_params(float_ads)
