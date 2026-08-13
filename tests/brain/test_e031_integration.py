"""Integration tests — full BrainWrapper stack on a tiny real model.

Runs the complete learn → evaluate pipeline on tiny LLaMA/GPT-2 models
(CPU) and asserts the E031 invariants: no backprop, warmup gates M, plastic
updates actually move biases, and ``without_plasticity()`` reproduces the
raw frozen model exactly.
"""

from __future__ import annotations

import logging

import pytest
import torch

from ph_neuro.brain import BrainWrapper
from tests.brain._models import random_token_ids, tiny_gpt2, tiny_llama
from tests.brain.test_brain_wrapper import FakeTok, flat_tokens

log = logging.getLogger("test")


def test_full_learn_loop_no_backward(monkeypatch):
    calls = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    brain = BrainWrapper(tiny_llama(layers=2), tokenizer=FakeTok(), log=log)
    metrics = brain.learn(flat_tokens(4096), steps=4, batch_size=2, seq_len=16, seed=42)
    assert len(metrics) == 4
    assert calls["n"] == 0
    # plastic biases moved off zero
    total = sum(float(ip.bias.abs().sum()) for ip in brain._injection_points)  # noqa: SLF001
    assert total > 0
    # EMA initialized and each metric has the expected keys
    assert brain.modulator.initialized
    for m in metrics:
        assert set(m) == {"step", "loss", "ema_loss", "surprise_s",
                          "modulator_M", "mean_abs_delta_b", "mean_abs_b", "tokens_seen"}


def test_warmup_zeroes_modulator():
    brain = BrainWrapper(tiny_llama(layers=2), tokenizer=FakeTok(), log=log)
    metrics = brain.learn(flat_tokens(4096), steps=4, batch_size=2, seq_len=16,
                          warmup_steps=2, seed=42)
    assert metrics[0]["modulator_M"] == 0.0
    assert metrics[1]["modulator_M"] == 0.0
    # EMA still advanced during warmup (warmup gates M, not the EMA):
    # L̂₁ = α·L̂₀ + (1−α)·L₁
    l0, l1 = metrics[0]["loss"], metrics[1]["loss"]
    assert metrics[1]["ema_loss"] == pytest.approx(0.99 * l0 + 0.01 * l1, abs=1e-5)
    assert l1 > 0


def test_constant_m_applies_every_step():
    brain = BrainWrapper(
        tiny_llama(layers=2),
        tokenizer=FakeTok(),
        modulator_cfg={"mode": "constant", "M": 1.0},
        log=log,
    )
    metrics = brain.learn(flat_tokens(4096), steps=3, batch_size=2, seq_len=16, seed=42)
    assert all(m["modulator_M"] == 1.0 for m in metrics)
    assert all(m["surprise_s"] == 0.0 for m in metrics)


def test_without_plasticity_identical_to_raw():
    raw = tiny_llama(layers=2)
    model = tiny_llama(layers=2)
    model.load_state_dict(raw.state_dict())
    brain = BrainWrapper(model, tokenizer=FakeTok(), log=log)
    ids = random_token_ids(batch=2, seq=16)
    with torch.no_grad():
        ref = raw(input_ids=ids).logits
        with brain.without_plasticity():
            wrapped = brain.model(input_ids=ids).logits
    assert torch.equal(ref, wrapped)


def test_hook_capture_shapes_after_learn():
    brain = BrainWrapper(tiny_llama(layers=2, hidden=32), tokenizer=FakeTok(), log=log)
    brain.learn(flat_tokens(4096), steps=1, batch_size=2, seq_len=16, seed=42)
    post = brain._last_post  # noqa: SLF001
    assert set(post) == {ip.name for ip in brain._injection_points}  # noqa: SLF001
    for t in post.values():
        assert tuple(t.shape) == (2, 16, 32)
        assert t.dtype == torch.float32


def test_gpt2_wrapper_learns_and_identity():
    raw = tiny_gpt2(layers=2)
    model = tiny_gpt2(layers=2)
    model.load_state_dict(raw.state_dict())
    brain = BrainWrapper(model, tokenizer=FakeTok(), log=log)
    ids = random_token_ids(batch=2, seq=16)
    with torch.no_grad():
        ref = raw(input_ids=ids).logits
        with brain.without_plasticity():
            wrapped = brain.model(input_ids=ids).logits
    assert torch.equal(ref, wrapped)
    metrics = brain.learn(flat_tokens(4096), steps=3, batch_size=2, seq_len=16, seed=42)
    assert len(metrics) == 3
    assert sum(float(ip.bias.abs().sum()) for ip in brain._injection_points) > 0  # noqa: SLF001


def test_learn_then_evaluate(tmp_path):
    brain = BrainWrapper(tiny_llama(layers=2), tokenizer=FakeTok(), log=log,
                         checkpoint_dir=str(tmp_path / "ckpt"))
    brain.learn(flat_tokens(4096), steps=6, batch_size=2, seq_len=16, seed=42)
    res = brain.evaluate(ids=flat_tokens(2048), window=32, stride=16)
    assert res["frozen"]["ppl"] > 1.0
    assert res["plastic"]["ppl"] > 1.0
    # checkpoint written at the final step
    assert (tmp_path / "ckpt" / "brain_ckpt_step6.pt").exists()


def test_save_load_learned(tmp_path):
    brain = BrainWrapper(tiny_llama(layers=2), tokenizer=FakeTok(), log=log)
    brain.learn(flat_tokens(4096), steps=5, batch_size=2, seq_len=16, seed=42)
    path = str(tmp_path / "learned.pt")
    brain.save(path)
    brain2 = BrainWrapper(tiny_llama(layers=2), tokenizer=FakeTok(), log=log)
    brain2.load(path)
    for ip1, ip2 in zip(brain._injection_points, brain2._injection_points, strict=True):  # noqa: SLF001
        assert torch.equal(ip1.bias, ip2.bias)


def test_modulator_moves_with_domain_shift():
    """A sustained loss jump after warmup should raise M (the surprise window)."""
    brain = BrainWrapper(tiny_llama(layers=2), tokenizer=FakeTok(), log=log)
    metrics = brain.learn(flat_tokens(4096), steps=4, batch_size=2, seq_len=16,
                          warmup_steps=2, seed=42)
    # warmup steps M=0
    assert metrics[0]["modulator_M"] == 0.0 and metrics[1]["modulator_M"] == 0.0
