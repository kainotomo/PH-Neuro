"""Integration tests for Milestone M2.1 — DQT Transformer on TinyStories.

Verifies the end-to-end DQT transformer language-modeling pipeline: data
loading/sequence packing, the model factory, single-batch overfitting
(sanity that DQT attention + FFN backward + stochastic rounding can learn),
a short training loop where loss decreases, the annealing switch, and
perplexity calculation.

All tests run on CPU with synthetic (learnable) data so they stay fast and
never require downloading TinyStories.
"""

from __future__ import annotations

import math
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.examples.run_m2_1_dqt_transformer import (
    ANNEAL_FRACTION,
    apply_dqt_rounding,
    compute_perplexity,
    evaluate_perplexity,
    find_latest_checkpoint,
    load_checkpoint,
    should_use_stochastic,
    train_dqt_transformer,
)
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.models.dqt_transformer import count_parameters, count_ternary_weights, dqt_gpt2
from ph_neuro.training.tinystories import (
    make_gpt2_tokenizer,
    make_synthetic_lm_loader,
    make_synthetic_token_sequences,
    pack_sequences,
    tokenize_texts,
)

DEVICE = torch.device("cpu")


def _build_small_model(vocab: int = 64, seq_len: int = 32) -> nn.Module:
    """A tiny DQT transformer for fast integration tests."""
    return dqt_gpt2(
        vocab_size=vocab,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=seq_len + 1,
        device=DEVICE,
    )


def _check_ternary_invariants(model: nn.Module) -> bool:
    """All DQT weights are int8 ternary in {-1, 0, +1}."""
    for m in model.modules():
        if isinstance(m, TernaryDQTLinear):
            w = m.weight_ternary
            if w.dtype != torch.int8:
                return False
            if not bool(torch.all((w >= -1) & (w <= 1)).item()):
                return False
    return True


# ── 1. Data loading ────────────────────────────────────────────────


class TestTinyStoriesData:
    """Dataset loading + sequence packing tests."""

    def test_pack_sequences_shift(self):
        """pack_sequences() produces shift-by-1 (input, target) windows."""
        tokens = torch.arange(0, 100, dtype=torch.long)
        seq_len = 8
        input_ids, targets = pack_sequences(tokens, seq_len)
        assert input_ids.shape == targets.shape
        assert input_ids.shape[1] == seq_len
        n_windows = input_ids.shape[0]
        # target[n, t] == input[n, t+1] for all but the last column
        for n in range(n_windows):
            assert torch.equal(targets[n, :-1], input_ids[n, 1:])
            if n + 1 < n_windows:
                assert targets[n, -1] == input_ids[n + 1, 0]

    def test_tokenize_texts(self):
        """tokenize_texts() returns a flat int tensor."""
        texts = ["hello world", "another story here", "third"]
        tok = make_gpt2_tokenizer()
        ids = tokenize_texts(texts, tok)
        assert ids.dtype == torch.int32
        assert ids.ndim == 1
        assert ids.numel() > 0
        # Token count should equal the sum of the per-story encodings
        expected = sum(len(tok.encode(t)) for t in texts)
        assert ids.numel() == expected

    def test_synthetic_loader_shapes(self):
        """Synthetic loader yields (input_ids, targets) of the right shape."""
        vocab, seq_len, batch = 64, 32, 4
        loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=batch, n_batches=4
        )
        input_ids, targets = next(iter(loader))
        assert input_ids.shape == (batch, seq_len)
        assert targets.shape == (batch, seq_len)
        assert targets.dtype == torch.long

    def test_synthetic_is_learnable(self):
        """The synthetic corpus has a deterministic next-token function."""
        seqs = make_synthetic_token_sequences(1, 16, 64, seed=0)
        # t_{k+1} == (3 * t_k + 5) % 64 must hold (the LCG structure)
        assert torch.equal(seqs[0, 1:], (3 * seqs[0, :-1] + 5) % 64)


# ── 2. Overfit a single batch ──────────────────────────────────────


class TestTransformerOverfitBatch:
    """DQT transformer can overfit a single synthetic batch (sanity)."""

    def test_transformer_overfit_batch(self):
        """Loss on one learnable batch should drop well below its start."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=8, n_batches=1, seed=7
        )
        # Slightly higher LR than the milestone default speeds up this sanity
        # check; the milestone itself uses lr=0.01 (validated in M1.1).
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)

        input_ids, targets = next(iter(loader))
        input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)

        def step_loss() -> float:
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab), targets.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            apply_dqt_rounding(model, use_stochastic=True)
            return float(loss.item())

        initial = step_loss()
        final = initial
        for _ in range(200):
            final = step_loss()

        assert _check_ternary_invariants(model), "Ternary invariant broken"
        assert final < initial * 0.5, (
            f"Single-batch loss did not drop: {initial:.3f} -> {final:.3f}"
        )
        # Real learning: clearly below the uniform-predictor baseline ln(vocab)
        assert final < math.log(vocab), (
            f"Loss not below uniform baseline ln({vocab})={math.log(vocab):.3f}: {final:.3f}"
        )


# ── 3. Short training loop ─────────────────────────────────────────


class TestTransformerTrainingLoop:
    """10-step training loop: loss should decrease."""

    def test_transformer_training_loop(self):
        """Train 60 steps and assert the loss trend decreases.

        Uses ``anneal_fraction=1.0`` so the deterministic-sign switch is
        NOT triggered: with only 60 steps the float buffers are still
        near-zero, so an early ``sign()`` snap would chaotically flip
        ~90% of weights (the known DQT annealing failure mode). The
        annealing switch itself is covered by
        :class:`TestTransformerAnnealing`.

        DQT converges slowly in its first steps (M1.1 showed the same),
        so per-step loss is noisy; we compare a 10-step windowed mean
        instead of raw first-vs-last step.
        """
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=10, seed=3
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=4
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

        results = train_dqt_transformer(
            model, train_loader, val_loader, optimizer, scheduler, DEVICE,
            epochs=6, max_steps=60, anneal_fraction=1.0,
            record_steps=True, verbose=False,
        )
        step_losses = results["step_loss_history"]
        assert len(step_losses) == 60, f"Expected 60 recorded steps, got {len(step_losses)}"
        window = 10
        first = sum(step_losses[:window]) / window
        last = sum(step_losses[-window:]) / window
        # Loss trend must decrease (windowed mean, last vs first 10 steps)
        assert last < first, (
            f"Windowed step loss did not decrease: first10={first:.4f} last10={last:.4f}"
        )
        assert results["steps_trained"] == 60
        assert results["final_val_ppl"] > 1.0
        assert math.isfinite(results["final_val_ppl"])


# ── 4. Annealing switch ────────────────────────────────────────────


class TestTransformerAnnealing:
    """Annealing from stochastic to deterministic rounding works."""

    def test_annealing_switch(self):
        """Deterministic rounding snaps weights to sign(float)."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        # Perturb the float buffers so sign() differs from current ternary
        for m in model.modules():
            if isinstance(m, TernaryDQTLinear):
                with torch.no_grad():
                    m.weight_float.data.add_(0.5)
        apply_dqt_rounding(model, use_stochastic=False)
        # After deterministic rounding, ternary == sign(float) everywhere
        for m in model.modules():
            if isinstance(m, TernaryDQTLinear):
                expected = m.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
                assert torch.equal(m.weight_ternary, expected)

    def test_annealing_schedule(self):
        """should_use_stochastic() switches at the anneal fraction."""
        total = 100
        anneal_start = int(total * ANNEAL_FRACTION)
        assert should_use_stochastic(anneal_start - 1, total, ANNEAL_FRACTION) is True
        assert should_use_stochastic(anneal_start, total, ANNEAL_FRACTION) is False
        assert should_use_stochastic(total - 1, total, ANNEAL_FRACTION) is False


# ── 5. Perplexity calculation ──────────────────────────────────────


class TestPerplexityCalculation:
    """Perplexity is computed correctly (exp of mean cross-entropy loss)."""

    def test_perplexity_calculation(self):
        """compute_perplexity(loss) == exp(loss)."""
        assert compute_perplexity(0.0) == 1.0
        assert math.isclose(compute_perplexity(2.0), math.exp(2.0), rel_tol=1e-6)
        assert compute_perplexity(4.0) > compute_perplexity(2.0)

    def test_evaluate_perplexity(self):
        """evaluate_perplexity() returns exp(mean loss) and is finite."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=5
        )
        ppl = evaluate_perplexity(model, loader, DEVICE)
        assert math.isfinite(ppl)
        assert ppl > 1.0
        # Initial (untrained) perplexity should be near vocab size for a
        # uniform predictor, so must be comfortably below exp(vocab).
        assert ppl < math.exp(vocab)

    def test_model_param_counts(self):
        """Model factory reports sensible float/ternary parameter counts."""
        model = _build_small_model()
        n_total = count_parameters(model)
        n_ternary = count_ternary_weights(model)
        assert n_ternary > 0
        assert n_total > n_ternary  # float embedding + RMSNorm add to it
        assert _check_ternary_invariants(model)


# ── 6. Pause/resume ────────────────────────────────────────────────


class TestTransformerResume:
    """Checkpointed training can be paused and resumed."""

    def _make_runner(self, vocab=64, seq_len=32):
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=10, seed=3
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=4
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        return model, train_loader, val_loader, optimizer, scheduler

    def test_checkpoint_and_find_latest(self):
        """Training writes checkpoints and find_latest picks the newest."""
        model, train_loader, val_loader, optimizer, scheduler = self._make_runner()
        with tempfile.TemporaryDirectory() as ckpt_dir:
            train_dqt_transformer(
                model, train_loader, val_loader, optimizer, scheduler, DEVICE,
                epochs=2, max_steps=4, anneal_fraction=1.0,
                checkpoint_every=2, checkpoint_dir=ckpt_dir, verbose=False,
            )
            latest = find_latest_checkpoint(ckpt_dir)
            assert latest is not None
            assert "ckpt_step4.pt" in latest, f"Expected step-4 checkpoint, got {latest}"

    def test_resume_continues_training(self):
        """A fresh run resumed from a checkpoint continues the step count."""
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Phase 1: train 4 steps (checkpoints at step 2 and step 4)
            m1, tl, vl, o1, s1 = self._make_runner()
            r1 = train_dqt_transformer(
                m1, tl, vl, o1, s1, DEVICE,
                epochs=2, max_steps=4, anneal_fraction=1.0,
                checkpoint_every=2, checkpoint_dir=ckpt_dir, verbose=False,
            )
            assert r1["steps_trained"] == 4

            # Phase 2: build a FRESH model, resume from the latest checkpoint
            m2, tl, vl, o2, s2 = self._make_runner()
            loaded = load_checkpoint(
                "auto", ckpt_dir, m2, o2, s2, DEVICE
            )
            assert loaded["step"] == 4, f"Expected resume at step 4, got {loaded['step']}"
            r2 = train_dqt_transformer(
                m2, tl, vl, o2, s2, DEVICE,
                epochs=3, max_steps=8, anneal_fraction=1.0,
                checkpoint_every=2, checkpoint_dir=ckpt_dir,
                start_step=loaded["step"],
                start_epoch=loaded["epoch"] + 1,
                best_val_ppl=loaded["best_val_ppl"],
                best_step=loaded["best_step"],
                verbose=False,
            )
            # Continued from step 4 to step 8 (4 more steps)
            assert r2["steps_trained"] == 8
            assert _check_ternary_invariants(m2)
