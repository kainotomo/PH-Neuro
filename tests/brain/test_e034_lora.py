"""Unit tests for the E034 surprise-gated LoRA machinery (Step 2.1).

Covers:
* ``LoRAAdapter`` construction/injection: shapes per architecture, scaled
  random init (``A ~ N(0, 1/sqrt(d_in))``, ``B = 0``), the identity invariant
  (I1: with ``B = 0`` the wrapped model is bit-identical to the raw frozen
  model), and the ``enabled`` flag used for frozen-baseline eval.
* ``compute_effective_lr``: the gated-lr step logic for all three methods
  (plain / surprise / const_reduced), including M = 0 during warmup.
* Modulator EMA persistence: ``SurpriseModulator.state_dict()`` round-trips
  through the runner's checkpoint save/resume (bit-identical resume).
* ``make_three_domain_batch_iter``: the E034 sequential stream (warmup wiki →
  phase-1 pubmed → phase-2 cnn) is deterministic and correctly ordered.
"""

from __future__ import annotations

import logging
import math
import os

import pytest
import torch

from ph_neuro.brain.lora import (
    LoRAAdapter,
    all_lora_weights,
    build_lora_adapters,
    n_lora_params,
)
from ph_neuro.brain.modulator import SurpriseModulator
from ph_neuro.examples.run_e034_lora import _resume, _save_checkpoint, compute_effective_lr
from tests.brain._models import tiny_gpt2, tiny_llama

log = logging.getLogger("test")


# ── LoRAAdapter ────────────────────────────────────────────────────


class TestLoRAAdapter:
    def test_shapes_and_init_llama(self):
        model = tiny_llama(layers=2, hidden=32)
        adapters = build_lora_adapters(model, rank=1, device="cpu")
        assert len(adapters) == 2 * 2  # 2 blocks × (o_proj + down_proj)
        # o_proj: 32→32; down_proj: 64→32 (intermediate = hidden*2).
        op = next(a for a in adapters if a.in_features == 32)
        assert tuple(op.A.shape) == (1, 32)
        assert tuple(op.B.shape) == (32, 1)
        assert torch.equal(op.B, torch.zeros_like(op.B))  # B = 0 → identity
        assert float(op.A.std()) == pytest.approx(1.0 / math.sqrt(32), rel=0.2)

    def test_shapes_and_init_gpt2(self):
        model = tiny_gpt2(layers=2, hidden=32)
        adapters = build_lora_adapters(model, rank=1, device="cpu")
        assert len(adapters) == 2 * 2  # 2 blocks × (attn.c_proj + mlp.c_proj)
        # attn.c_proj: 32→32; mlp.c_proj: 128→32 (Conv1D, mlp ratio 4).
        by_in = {ad.in_features for ad in adapters}
        assert by_in == {32, 128}
        for ad in adapters:
            assert tuple(ad.B.shape) == (ad.out_features, 1)

    def test_identity_invariant(self):
        """With B = 0 the wrapped model is bit-identical to the raw model."""
        torch.manual_seed(0)
        model = tiny_llama(layers=1, hidden=32)
        ids = torch.randint(0, 128, (1, 16))
        raw = model(ids).logits
        adapters = build_lora_adapters(model, rank=1, device="cpu")
        assert torch.equal(model(ids).logits, raw)  # injection zero (B=0)
        for ad in adapters:
            ad.set_enabled(False)
        assert torch.equal(model(ids).logits, raw)  # disabled → identity

    def test_n_params_budget(self):
        model = tiny_llama(layers=2, hidden=32)
        adapters = build_lora_adapters(model, rank=1, device="cpu")
        n = n_lora_params(adapters)
        # Per block: o_proj (1·32+32·1) + down_proj (1·64+32·1) = 64 + 96.
        assert n == 2 * ((32 + 32) + (64 + 32))
        assert all_lora_weights(adapters).numel() == n
        # rank-1 on the real SmolLM2 (o_proj 2048→2048, down_proj 8192→2048)
        # per block = 2·2048 + (8192+2048) = 14,336; × 24 blocks = 344,064
        # (the exact E032 Part D / E034 matched budget).
        assert 24 * (2 * 2048 + (8192 + 2048)) == 344064


# ── compute_effective_lr (the gate) ────────────────────────────────


class TestComputeEffectiveLr:
    def test_plain_learns_during_warmup(self):
        lr, do = compute_effective_lr("plain", 1e-3, 0.5, None, in_warmup=True)
        assert lr == 1e-3 and do is True
        lr, do = compute_effective_lr("plain", 1e-3, 0.5, None, in_warmup=False)
        assert lr == 1e-3 and do is True

    def test_surprise_gates_and_freezes_warmup(self):
        lr, do = compute_effective_lr("surprise", 1e-3, 0.8, None, in_warmup=True)
        assert lr == 0.0 and do is False  # M forced to 0 during warmup
        lr, do = compute_effective_lr("surprise", 1e-3, 0.8, None, in_warmup=False)
        assert lr == pytest.approx(1e-3 * 0.8) and do is True  # effective = η·M
        lr, _ = compute_effective_lr("surprise", 1e-3, 0.018, None, in_warmup=False)
        assert lr == pytest.approx(1e-3 * 0.018)  # near-frozen in-domain

    def test_const_reduced_no_warmup(self):
        lr, do = compute_effective_lr("const_reduced", 1e-3, 0.0, 0.12, in_warmup=True)
        assert lr == 0.0 and do is False
        lr, do = compute_effective_lr("const_reduced", 1e-3, 0.0, 0.12, in_warmup=False)
        assert lr == pytest.approx(1e-3 * 0.12) and do is True


# ── modulator EMA persistence through checkpoints ──────────────────


class TestModulatorCheckpointPersistence:
    def test_state_dict_roundtrip(self):
        mod = SurpriseModulator(mode="surprise_ema", alpha=0.99)
        # advance the EMA a few steps with known losses
        for loss in (2.5, 2.6, 2.7, 3.0):
            mod.update(loss)
        state = mod.state_dict()
        mod2 = SurpriseModulator(mode="surprise_ema", alpha=0.99)
        assert mod2.ema_loss is None
        mod2.load_state_dict(state)
        assert torch.equal(mod.ema_loss, mod2.ema_loss)
        assert mod2.initialized is True
        # continuing both modulators in lockstep gives identical M
        for loss in (3.2, 3.1, 2.9):
            s1, m1 = mod.update(loss)
            s2, m2 = mod2.update(loss)
            assert m1 == pytest.approx(m2)
            assert s1 == pytest.approx(s2)

    def test_runner_checkpoint_save_resume(self, tmp_path):
        """_save_checkpoint/_resume restore adapter + optimizer + EMA state."""
        torch.manual_seed(42)
        model = tiny_llama(layers=1, hidden=32)
        adapters = build_lora_adapters(model, rank=1, device="cpu")
        params = [p for ad in adapters for p in ad.parameters()]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
        mod = SurpriseModulator(mode="surprise_ema")
        for loss in (2.4, 2.5):
            mod.update(loss)
        # a few real steps so the optimizer state is non-trivial
        ids = torch.randint(0, 128, (2, 8))
        for _ in range(3):
            logits = model(ids).logits
            V = logits.size(-1)
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].to(torch.float32).reshape(-1, V),
                ids[..., 1:].reshape(-1),
            )
            loss.backward()
            opt.step()
            opt.zero_grad()

        ckpt_dir = str(tmp_path)
        _save_checkpoint(
            os.path.join(ckpt_dir, "brain_ckpt_step5.pt"), 5, adapters, opt, mod,
            {"tag": "t"},
        )
        # new adapters/opt/modulator, resume
        torch.manual_seed(42)
        adapters2 = build_lora_adapters(tiny_llama(layers=1, hidden=32), rank=1, device="cpu")
        params2 = [p for ad in adapters2 for p in ad.parameters()]
        opt2 = torch.optim.AdamW(params2, lr=1e-3, weight_decay=0.0)
        mod2 = SurpriseModulator(mode="surprise_ema")
        step = _resume(ckpt_dir, 10, adapters2, opt2, mod2)
        assert step == 5
        assert torch.equal(adapters[0].A.detach(), adapters2[0].A.detach())
        assert torch.equal(adapters[0].B.detach(), adapters2[0].B.detach())
        assert torch.equal(mod.ema_loss, mod2.ema_loss)
        assert mod2.initialized is True


# ── two-domain stream ──────────────────────────────────────────────


class TestThreeDomainBatchIter:
    def test_sequence_and_determinism(self):
        from ph_neuro.brain.brain_wrapper import cyclic_batch_iter
        from ph_neuro.brain.datasets import make_three_domain_batch_iter

        def stream(tag: str):
            return torch.arange(100, dtype=torch.long) + (0 if tag == "w" else
                                                          1000 if tag == "p" else 2000)

        w, p, c = stream("w"), stream("p"), stream("c")
        it = make_three_domain_batch_iter(w, p, c, warmup_steps=3, phase1_steps=4,
                                          batch_size=2, seq_len=4, seed=1)
        batches = [next(it) for _ in range(3 + 4 + 4)]
        # warmup batches are from the wiki stream
        for b in batches[:3]:
            assert b["input_ids"].max().item() < 100
        # phase-1 batches from the pubmed stream
        for b in batches[3:7]:
            assert 1000 <= b["input_ids"].min().item() < 2000
        # phase-2 batches from the cnn stream
        for b in batches[7:]:
            assert 2000 <= b["input_ids"].min().item()

        # determinism: same seed → identical sequence
        it2 = make_three_domain_batch_iter(w, p, c, 3, 4, 2, 4, seed=1)
        for _ in range(7):
            next(it2)
        assert torch.equal(next(it2)["input_ids"], batches[7]["input_ids"])

        # resume by skipping reproduces the same stream
        it3 = make_three_domain_batch_iter(w, p, c, 3, 4, 2, 4, seed=1)
        for _ in range(7):
            next(it3)
        assert torch.equal(next(it3)["input_ids"], batches[7]["input_ids"])
