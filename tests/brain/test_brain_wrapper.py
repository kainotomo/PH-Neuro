"""Unit tests for BrainWrapper — invariants I1–I5, save/load, checkpoints.

Invariants (Step 0.4 spec):
* I1 — active=False (or all-zero biases) ⇒ bit-identical to frozen model.
* I2 — zero-init ⇒ generate() == frozen generate().
* I3 — no autograd: learn() never calls .backward().
* I4 — eval mode: model stays in eval() throughout learn/generate.
* I5 — every plastic bias has shape (d_out,) = hidden_size, float32.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest
import torch

from ph_neuro.brain import BrainWrapper
from ph_neuro.brain.brain_wrapper import check_gpu_free, cyclic_batch_iter
from tests.brain._models import random_token_ids, tiny_gpt2, tiny_llama

log = logging.getLogger("test")


class FakeTok:
    """Minimal tokenizer stand-in exposing the BrainWrapper call surface."""

    name_or_path = "fake"
    pad_token_id = 2
    eos_token_id = 2
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [(ord(c) % 127) + 1 for c in text] or [1]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(int(i)) for i in ids)


def make_brain(model=None, **kwargs):
    if model is None:
        # Reseed so every fresh model instance has identical frozen weights
        # (each tiny_llama() construction consumes global RNG).
        torch.manual_seed(42)
        model = tiny_llama()
    return BrainWrapper(model, tokenizer=FakeTok(), log=log, **kwargs)


def flat_tokens(n: int = 8192) -> torch.Tensor:
    """Flat token stream over 1..127 (avoids token 0 = zeroed pad embedding)."""
    base = torch.arange(1, 128).long()
    return base.repeat(n // base.numel() + 1)[:n]


# ── I1 / I2 identity ───────────────────────────────────────────────


class TestIdentity:
    def test_without_plasticity_bit_identical(self):
        raw = tiny_llama()
        model = tiny_llama()
        model.load_state_dict(raw.state_dict())
        brain = make_brain(model)
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            a = raw(input_ids=ids).logits
            with brain.without_plasticity():
                b = brain.model(input_ids=ids).logits
        assert torch.equal(a, b)

    def test_active_zero_bias_bit_identical(self):
        raw = tiny_llama()
        model = tiny_llama()
        model.load_state_dict(raw.state_dict())
        brain = make_brain(model)
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            a = raw(input_ids=ids).logits
            b = brain.model(input_ids=ids).logits  # active=True, all-zero bias
        assert torch.equal(a, b)

    def test_generate_zero_init_equals_frozen(self):
        raw = tiny_llama()
        model = tiny_llama()
        model.load_state_dict(raw.state_dict())
        brain = make_brain(model)
        ids = torch.tensor([[5, 7, 9, 11, 13]])
        with torch.no_grad():
            frozen = raw.generate(
                ids, max_new_tokens=8, do_sample=False,
                pad_token_id=2, eos_token_id=2,
            )
            with brain.without_plasticity():
                p = brain.model.generate(
                    ids, max_new_tokens=8, do_sample=False,
                    pad_token_id=2, eos_token_id=2,
                )
        assert torch.equal(p, frozen)
        # I2: active (zero bias) generate runs without error
        brain.generate("ab", max_new_tokens=4, do_sample=False)


class TestNoBackprop:
    def test_learn_never_calls_backward(self, monkeypatch):
        calls = {"n": 0}

        def spy(*a, **k):
            calls["n"] += 1

        monkeypatch.setattr(torch.Tensor, "backward", spy)
        brain = make_brain()
        brain.learn(flat_tokens(), steps=3, batch_size=2, seq_len=16, seed=42)
        assert calls["n"] == 0


class TestEvalMode:
    def test_model_stays_in_eval(self):
        brain = make_brain()
        brain.learn(flat_tokens(), steps=2, batch_size=2, seq_len=16, seed=42)
        assert brain.model.training is False
        brain.generate("hello", max_new_tokens=4, do_sample=False)
        assert brain.model.training is False


class TestShapes:
    def test_biases_are_hidden_sized_float32(self):
        for model in (tiny_llama(hidden=32), tiny_gpt2(hidden=32)):
            brain = make_brain(model)
            for ip in brain._injection_points:  # noqa: SLF001
                assert tuple(ip.bias.shape) == (32,)
                assert ip.bias.dtype == torch.float32
            assert brain.plastic_parameter_count() > 0
            assert brain.plastic_memory_bytes() == brain.plastic_parameter_count() * 4


# ── without_plasticity ─────────────────────────────────────────────


class TestWithoutPlasticity:
    def test_nested_use(self):
        brain = make_brain()
        assert brain._active is True  # noqa: SLF001
        with brain.without_plasticity():
            assert brain._active is False  # noqa: SLF001
            with brain.without_plasticity():
                assert brain._active is False  # noqa: SLF001
            assert brain._active is False  # noqa: SLF001
        assert brain._active is True  # noqa: SLF001


# ── state_dict / save / load ───────────────────────────────────────


class TestStateDict:
    def test_roundtrip(self):
        brain = make_brain()
        brain.learn(flat_tokens(), steps=4, batch_size=2, seq_len=16, seed=42)
        sd = brain.state_dict()
        assert set(sd) == {f"plastic.{ip.name}" for ip in brain._injection_points}
        for v in sd.values():
            assert v.dtype == torch.float32
        brain2 = make_brain()
        brain2.load_state_dict(sd)
        for ip1, ip2 in zip(brain._injection_points, brain2._injection_points, strict=True):  # noqa: SLF001
            assert torch.equal(ip1.bias, ip2.bias)

    def test_strict_missing_key(self):
        brain = make_brain()
        with pytest.raises(KeyError):
            brain.load_state_dict({})

    def test_strict_extra_key(self):
        brain = make_brain()
        sd = brain.state_dict()
        sd["plastic.bogus"] = torch.zeros(32)
        with pytest.raises(KeyError):
            brain.load_state_dict(sd)
        brain.load_state_dict(sd, strict=False)  # extras ignored

    def test_shape_mismatch_raises(self):
        brain = make_brain()
        sd = brain.state_dict()
        key = next(iter(sd))
        bad = dict(sd)
        bad[key] = torch.zeros(64)
        with pytest.raises(ValueError):
            brain.load_state_dict(bad)


class TestSaveLoad:
    def test_save_atomic_and_load(self, tmp_path):
        brain = make_brain()
        brain.learn(flat_tokens(), steps=4, batch_size=2, seq_len=16, seed=42)
        path = str(tmp_path / "brain.pt")
        brain.save(path)
        assert os.path.exists(path)
        # no leftover temp files
        assert not [f for f in os.listdir(tmp_path) if ".tmp." in f]
        brain2 = make_brain()
        brain2.load(path)
        for ip1, ip2 in zip(brain._injection_points, brain2._injection_points, strict=True):  # noqa: SLF001
            assert torch.equal(ip1.bias, ip2.bias)

    def test_load_wrong_architecture(self, tmp_path):
        brain = make_brain(tiny_llama(hidden=64))
        path = str(tmp_path / "big.pt")
        brain.save(path)
        small = make_brain(tiny_gpt2(hidden=32))
        with pytest.raises(ValueError):
            small.load(path)


# ── checkpoints / resume / skip-if-exists ──────────────────────────


class TestCheckpoints:
    def test_skip_if_exists(self, tmp_path):
        brain = make_brain(checkpoint_dir=str(tmp_path))
        m = brain.learn(flat_tokens(), steps=5, batch_size=2, seq_len=16, seed=42)
        assert len(m) == 5
        brain2 = make_brain(checkpoint_dir=str(tmp_path))
        m2 = brain2.learn(flat_tokens(), steps=5, batch_size=2, seq_len=16, seed=42)
        assert m2 == []  # already complete → never restarted

    def test_resume_matches_fresh_run(self, tmp_path):
        ckpt = str(tmp_path / "sub")
        brain1 = make_brain(checkpoint_dir=ckpt)
        brain1.learn(flat_tokens(), steps=5, batch_size=2, seq_len=16, seed=42)

        # a fresh brain resumes from step 5 and continues to step 8
        resumed = make_brain(checkpoint_dir=ckpt)
        m = resumed.learn(flat_tokens(), steps=8, batch_size=2, seq_len=16, seed=42)
        assert len(m) == 3  # steps 5,6,7
        assert m[0]["step"] == 5

        # determinism: a fresh run of 8 steps must match the resumed run
        fresh = make_brain(checkpoint_dir=None)
        fresh.learn(flat_tokens(), steps=8, batch_size=2, seq_len=16, seed=42)
        for ip_r, ip_f in zip(resumed._injection_points, fresh._injection_points, strict=True):  # noqa: SLF001
            assert torch.allclose(ip_r.bias, ip_f.bias, atol=1e-6)

    def test_checkpoint_contains_ema(self, tmp_path):
        brain = make_brain(checkpoint_dir=str(tmp_path))
        brain.learn(flat_tokens(), steps=3, batch_size=2, seq_len=16, seed=42)
        ckpt = torch.load(os.path.join(str(tmp_path), "brain_ckpt_step3.pt"),
                          weights_only=False)
        assert ckpt["format"] == "ph_neuro_brain_checkpoint"
        assert ckpt["ema_initialized"] is True
        assert "plastic" in ckpt

    def test_no_checkpoint_dir_disables(self, tmp_path):
        brain = make_brain()
        brain.learn(flat_tokens(), steps=3, batch_size=2, seq_len=16, seed=42)
        assert not list(tmp_path.iterdir())  # nothing written


# ── GPU gate (mocked nvidia-smi) ───────────────────────────────────


class TestGpuGate:
    def test_exit_when_contended(self, monkeypatch):
        monkeypatch.setattr(
            "ph_neuro.brain.brain_wrapper.gpu_free_mb", lambda: 1024
        )  # 1 GiB free
        with pytest.raises(SystemExit):
            check_gpu_free(6.0, "exit", log)

    def test_warn_when_contended(self, monkeypatch):
        monkeypatch.setattr(
            "ph_neuro.brain.brain_wrapper.gpu_free_mb", lambda: 1024
        )
        check_gpu_free(6.0, "warn", log)  # proceeds, no raise

    def test_pass_when_enough(self, monkeypatch):
        monkeypatch.setattr(
            "ph_neuro.brain.brain_wrapper.gpu_free_mb", lambda: 8192
        )
        check_gpu_free(6.0, "exit", log)

    def test_cpu_skips_check(self, monkeypatch):
        called = {"n": 0}

        def boom():
            called["n"] += 1
            return 1

        monkeypatch.setattr("ph_neuro.brain.brain_wrapper.gpu_free_mb", boom)
        brain = make_brain()  # CPU device
        brain._check_gpu("exit", 6.0)  # noqa: SLF001
        assert called["n"] == 0


# ── evaluate ───────────────────────────────────────────────────────


class TestEvaluate:
    def test_frozen_equals_plastic_when_zero_bias(self):
        brain = make_brain()
        res = brain.evaluate(ids=flat_tokens(4096), window=32, stride=16)
        assert res["frozen"]["ppl"] == pytest.approx(res["plastic"]["ppl"], rel=1e-5)
        assert res["frozen"]["n_tokens"] == res["plastic"]["n_tokens"]
        assert res["frozen"]["per_block"]["nll"] == pytest.approx(
            res["plastic"]["per_block"]["nll"]
        )

    def test_modes(self):
        brain = make_brain()
        f = brain.evaluate(ids=flat_tokens(2048), window=32, stride=16, mode="frozen")
        assert f["plastic"] is None and f["frozen"] is not None
        p = brain.evaluate(ids=flat_tokens(2048), window=32, stride=16, mode="plastic")
        assert p["frozen"] is None and p["plastic"] is not None

    def test_plastic_bias_changes_ppl(self):
        brain = make_brain()
        before = brain.evaluate(ids=flat_tokens(4096), window=32, stride=16, mode="plastic")
        for ip in brain._injection_points:  # noqa: SLF001
            ip.bias.fill_(5.0)
        after = brain.evaluate(ids=flat_tokens(4096), window=32, stride=16, mode="plastic")
        # a large injected bias must move the output distribution
        assert (
            after["plastic"]["per_block"]["nll"]
            != before["plastic"]["per_block"]["nll"]
        )
        assert after["plastic"]["mean_nll"] != pytest.approx(
            before["plastic"]["mean_nll"], abs=1e-4
        )


# ── misc API ───────────────────────────────────────────────────────


class TestMisc:
    def test_consolidate_placeholder(self):
        brain = make_brain()
        assert brain.consolidate() == {"status": "not_implemented", "phase": "2.3"}

    def test_summary_and_helpers(self):
        brain = make_brain()
        s = brain.summary()
        assert s["model_type"] == "llama"
        assert s["plastic_params"] == brain.plastic_parameter_count()
        assert len(brain.injection_point_names()) == len(brain._injection_points)  # noqa: SLF001
        brain.set_lr(2e-3)
        assert brain.lr == 2e-3
        brain.set_decay_rate(0.1)
        assert brain.decay_rate == 0.1

    def test_unsupported_plasticity(self):
        with pytest.raises(NotImplementedError):
            make_brain(plasticity="low_rank")

    def test_invalid_modulator_cfg(self):
        with pytest.raises(ValueError):
            make_brain(modulator_cfg={"mode": "surprise_ema", "bogus": 1})

    def test_steps_zero_returns_empty(self):
        brain = make_brain()
        assert brain.learn(flat_tokens(), steps=0) == []


class TestCyclicBatchIter:
    def test_deterministic(self):
        tokens = flat_tokens(2048)
        a = [b["input_ids"] for b in
             [next(iter(cyclic_batch_iter(tokens, 2, 16, 42))) for _ in range(3)]]
        b = [b["input_ids"] for b in
             [next(iter(cyclic_batch_iter(tokens, 2, 16, 42))) for _ in range(3)]]
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)
        assert a[0].shape == (2, 16)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            next(iter(cyclic_batch_iter(torch.empty(0), 2, 16, None)))
