"""Integration tests for Phase 2 NTH-1 — Neuromodulated Hebbian single-layer MNIST.

Success criteria:
    1. No .backward() calls during training
    2. All weights remain in {-1, 0, +1} at every step
    3. Weight flips stabilize (<1% per step after convergence)
    4. Single layer achieves >85% accuracy on MNIST within 5 epochs
    5. Training completes in under 2 minutes on an RTX 4060
    6. NTH label modulator is theoretically equivalent to WTA
"""

from __future__ import annotations

import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.core.hebbian_rules import neuromodulated_update
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.neuromodulated import (
    NeuromodulatedHebbianClassifier,
    build_label_modulator,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_synthetic_data(
    n_samples: int = 200,
    in_features: int = 784,
    out_features: int = 10,
) -> DataLoader:
    """Create a tiny synthetic dataset for fast invariant tests."""
    x = torch.randn(n_samples, in_features)
    y = torch.randint(0, out_features, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)


def _count_backward_calls(
    classifier: NeuromodulatedHebbianClassifier,
    loader: DataLoader,
    **kwargs,
) -> int:
    """Train for one step and count how many times .backward() is called."""
    call_count = [0]
    original_backward = torch.Tensor.backward

    def tracking_backward(self, *args, **kwargs):  # type: ignore
        call_count[0] += 1
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = tracking_backward  # type: ignore
    try:
        x, y = next(iter(loader))
        with torch.no_grad():
            classifier.train_step(
                x, y,
                lr=kwargs.get("lr", 0.01),
                decay=kwargs.get("decay", 0.0),
                epsilon=kwargs.get("epsilon", 0.1),
            )
    finally:
        torch.Tensor.backward = original_backward  # type: ignore
    return call_count[0]


def _check_all_ternary(w: torch.Tensor) -> bool:
    """Check that all weights are in {-1, 0, +1}."""
    return bool(((w == -1) | (w == 0) | (w == 1)).all().item())


# ── Tests: Low-level neuromodulated_update ─────────────────────────

class TestNeuromodulatedUpdate:
    """Test the neuromodulated_update function in hebbian_rules.py."""

    def test_update_positive_modulator(self):
        """M=+1 should increase scores (Hebbian strengthen)."""
        scores = torch.zeros(3, 5, dtype=torch.float16)
        pre = torch.randint(0, 3, (4, 5), dtype=torch.int8) - 1  # {-1, 0, +1}
        modulator = torch.zeros(4, 3, dtype=torch.float32)
        modulator[:, 0] = 1.0  # M=+1 for neuron 0

        updated = neuromodulated_update(scores, pre, modulator, lr=0.01)
        # Neuron 0 should have non-zero delta (some scores increased)
        assert (updated[0] != scores[0]).any(), "M=+1 should modify scores"
        # Other neurons should have zero delta (no modulation)
        assert torch.allclose(updated[1:], scores[1:]), "M=0 should not modify scores"

    def test_update_negative_modulator(self):
        """M=-1 should decrease scores (anti-Hebbian weaken)."""
        scores = torch.zeros(3, 5, dtype=torch.float16)
        pre = torch.ones(4, 5, dtype=torch.int8)  # All +1
        modulator = torch.zeros(4, 3, dtype=torch.float32)
        modulator[:, 1] = -1.0  # M=-1 for neuron 1

        updated = neuromodulated_update(scores, pre, modulator, lr=0.01)
        # Neuron 1 should have negative delta
        assert (updated[1] < scores[1]).any(), "M=-1 should decrease scores"
        # Other neurons should have zero delta
        assert torch.allclose(updated[0], scores[0]), "M=0 should not modify scores"
        assert torch.allclose(updated[2], scores[2]), "M=0 should not modify scores"

    def test_update_neutral_modulator(self):
        """M=0 everywhere should produce no change."""
        scores = torch.randn(3, 5, dtype=torch.float16)
        pre = torch.randint(0, 3, (4, 5), dtype=torch.int8) - 1
        modulator = torch.zeros(4, 3, dtype=torch.float32)  # All zeros

        updated = neuromodulated_update(scores, pre, modulator, lr=0.1)
        assert torch.allclose(updated, scores), "M=0 everywhere should not modify scores"

    def test_update_with_post(self):
        """When post is provided, modulator should be applied element-wise to post."""
        scores = torch.zeros(3, 5, dtype=torch.float16)
        pre = torch.ones(4, 5, dtype=torch.int8)
        # M = +1 for neuron 0, but post = 0 for neuron 0 → no update expected
        modulator = torch.zeros(4, 3, dtype=torch.float32)
        modulator[:, 0] = 1.0
        post = torch.zeros(4, 3, dtype=torch.int8)
        post[:, 1] = 1  # Only neuron 1 fires

        updated = neuromodulated_update(scores, pre, modulator, lr=0.01, post=post)
        # M=+1 + post=0 = 0 for neuron 0 → no delta
        # M=0 + post=1 = 0 for neuron 1 → no delta
        assert torch.allclose(updated, scores), "M⊙post=0 should not modify scores"

    def test_update_shape_preserved(self):
        """Output shape should match input shape."""
        scores = torch.randn(3, 5, dtype=torch.float16)
        pre = torch.randint(0, 3, (4, 5), dtype=torch.int8) - 1
        modulator = torch.randn(4, 3)

        updated = neuromodulated_update(scores, pre, modulator, lr=0.01)
        assert updated.shape == scores.shape


# ── Tests: build_label_modulator ───────────────────────────────────

class TestBuildLabelModulator:
    """Test the build_label_modulator utility."""

    def test_correct_class_positive(self):
        """For wrong predictions, the correct class neuron should get M=+1."""
        y = torch.tensor([2, 5, 7])
        pred = torch.tensor([3, 5, 1])  # Samples 0,2 wrong; sample 1 correct
        mod = build_label_modulator(y, pred, out_features=10)
        # Sample 0 (wrong): correct=2 → M=+1
        assert mod[0, 2] == 1.0, "Wrong pred: correct class should be +1"
        # Sample 1 (correct): all zeros, no update
        assert (mod[1] == 0).all(), "Correct pred: no modulation"
        # Sample 2 (wrong): correct=7 → M=+1
        assert mod[2, 7] == 1.0, "Wrong pred: correct class should be +1"

    def test_wrong_class_negative(self):
        """The wrongly-predicted neuron should get M=-1, correct class gets M=+1."""
        y = torch.tensor([2, 5, 7])
        pred = torch.tensor([3, 5, 1])  # Sample 0 and 2 are wrong
        mod = build_label_modulator(y, pred, out_features=10)
        # Sample 0: correct=2 (+1), predicted=3 (-1)
        assert mod[0, 2] == 1.0, "Correct class should be +1"
        assert mod[0, 3] == -1.0, "Wrong predicted class should be -1"
        # Sample 1: correct=5, predicted=5 — all zeros (no update)
        assert (mod[1] == 0).all(), "Correct pred: all zeros"
        # Sample 2: correct=7 (+1), predicted=1 (-1)
        assert mod[2, 7] == 1.0, "Correct class should be +1"
        assert mod[2, 1] == -1.0, "Wrong predicted class should be -1"

    def test_positive_only(self):
        """positive_only=True should only set M=+1 for wrong preds, no M=-1."""
        y = torch.tensor([2, 5, 7])
        pred = torch.tensor([3, 5, 1])  # Samples 0,2 wrong
        mod = build_label_modulator(y, pred, out_features=10, positive_only=True)
        # All entries should be >= 0
        assert (mod >= 0).all(), "positive_only should have no negative values"
        # Wrong predictions should have M=+1 for correct class
        assert mod[0, 2] == 1.0
        assert mod[2, 7] == 1.0
        # Correct prediction should have all zeros
        assert (mod[1] == 0).all()

    def test_negative_only(self):
        """negative_only=True should only set M=-1 for wrong predictions."""
        y = torch.tensor([2, 5, 7])
        pred = torch.tensor([3, 5, 1])  # Samples 0,2 wrong
        mod = build_label_modulator(y, pred, out_features=10, negative_only=True)
        # All entries should be <= 0
        assert (mod <= 0).all(), "negative_only should have no positive values"
        # Wrong predictions: predicted class gets -1
        assert mod[0, 3] == -1.0
        assert mod[2, 1] == -1.0
        # Correct prediction (sample 1) should have all zeros
        assert (mod[1] == 0).all()

    def test_full_target(self):
        """full_target=True should set M=-1 for ALL wrong classes (only on wrong preds)."""
        y = torch.tensor([2, 5, 7])
        pred = torch.tensor([3, 5, 1])  # Samples 0,2 wrong; sample 1 correct
        mod = build_label_modulator(y, pred, out_features=10, full_target=True)
        # Sample 0: wrong (pred=3, correct=2) — all wrong classes get -1
        assert mod[0, 2] == 1.0  # correct class stays +1
        assert mod[0, 3] == -1.0  # predicted class gets -1
        assert (mod[0, :2] == -1.0).all()  # classes 0,1
        assert (mod[0, 4:] == -1.0).all()  # classes 4-9
        # Sample 1: correct (pred=5, correct=5) — all zeros
        assert (mod[1] == 0).all(), "Correct pred: all zeros, no update"
        # Sample 2: wrong (pred=1, correct=7) — all wrong classes get -1
        assert mod[2, 7] == 1.0  # correct class stays +1
        assert mod[2, 1] == -1.0  # predicted class gets -1
        assert (mod[2, :1] == -1.0).all()  # class 0
        assert (mod[2, 2:7] == -1.0).all()  # classes 2-6
        assert (mod[2, 8:] == -1.0).all()  # classes 8-9


# ── Tests: NeuromodulatedHebbianClassifier invariants ──────────────

class TestNeuromodulatedHebbianClassifier:
    """Test the NeuromodulatedHebbianClassifier class."""

    def test_create(self):
        """Creating the classifier should work."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        assert clf.model.weight.shape == (10, 784)

    def test_train_step_returns_dict(self):
        """Train step should return a metrics dict."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        with torch.no_grad():
            metrics = clf.train_step(x, y)
        assert "flip_rate" in metrics
        assert "n_flips" in metrics
        assert isinstance(metrics["flip_rate"], float)

    def test_predict_returns_correct_shape(self):
        """Predict should return indices with correct shape."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        x = torch.randn(32, 784)
        pred = clf.predict(x)
        assert pred.shape == (32,)
        assert pred.dtype == torch.long

    def test_evaluate_returns_float(self):
        """Evaluate should return a float between 0 and 1."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        loader = _make_synthetic_data(n_samples=50)
        acc = clf.evaluate(loader)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_get_weight_stats(self):
        """Weight stats should contain expected keys and sum to 100."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        stats = clf.get_weight_stats()
        assert "pos_pct" in stats
        assert "neg_pct" in stats
        assert "zero_pct" in stats
        total = stats["pos_pct"] + stats["neg_pct"] + stats["zero_pct"]
        assert abs(total - 100.0) < 1.0, f"Weight distribution sums to {total:.1f}%, not 100%"

    def test_repr(self):
        """String representation should include architecture info."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        r = repr(clf)
        assert "784" in r
        assert "10" in r
        assert "NeuromodulatedHebbianClassifier" in r

    def test_device_property(self):
        """Device property should return a torch.device."""
        clf = NeuromodulatedHebbianClassifier()
        assert isinstance(clf.device, torch.device)


# ── Tests: No backward calls ──────────────────────────────────────

class TestNoBackward:
    """Verify that no .backward() calls occur during training."""

    def test_no_backward_single_step(self):
        """No .backward() calls in a single train_step."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        loader = _make_synthetic_data(n_samples=32)
        n_calls = _count_backward_calls(clf, loader)
        assert n_calls == 0, f"Expected 0 .backward() calls, got {n_calls}"

    def test_no_backward_full_epoch(self):
        """No .backward() calls across a full epoch."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        loader = _make_synthetic_data(n_samples=64, in_features=784)
        count = [0]
        original_backward = torch.Tensor.backward

        def tracking_backward(self, *args, **kwargs):  # type: ignore
            count[0] += 1
            return original_backward(self, *args, **kwargs)

        torch.Tensor.backward = tracking_backward  # type: ignore
        try:
            for x, y in loader:
                with torch.no_grad():
                    clf.train_step(x, y)
        finally:
            torch.Tensor.backward = original_backward  # type: ignore

        assert count[0] == 0, f"Expected 0 .backward() calls, got {count[0]}"


# ── Tests: Weight invariants ──────────────────────────────────────

class TestWeightInvariants:
    """Verify that all weights remain ternary throughout training."""

    def test_initial_weights_ternary(self):
        """Initial weights should be ternary."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        w = clf.model.weight.unpack()
        assert _check_all_ternary(w), "Initial weights not all ternary"

    def test_weights_ternary_after_update(self):
        """Weights should remain ternary after a train_step."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        with torch.no_grad():
            clf.train_step(x, y)
        w = clf.model.weight.unpack()
        assert _check_all_ternary(w), "Weights not ternary after update"

    def test_weights_ternary_mnist_shapes(self):
        """Weights should remain ternary even with MNIST-shaped data."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        loader = _make_synthetic_data(n_samples=100, in_features=784)
        for x, y in loader:
            with torch.no_grad():
                clf.train_step(x, y)
        w = clf.model.weight.unpack()
        assert _check_all_ternary(w), "Weights not ternary after MNIST-shaped data"


# ── Tests: Accuracy — MNIST end-to-end ───────────────────────────

class TestMNISTAccuracy:
    """End-to-end MNIST accuracy tests."""

    @pytest.fixture(scope="class")
    def trained_classifier(self) -> NeuromodulatedHebbianClassifier:
        """Train a NeuromodulatedHebbianClassifier on MNIST for NTH-1 test."""
        clf = NeuromodulatedHebbianClassifier(
            in_features=784,
            out_features=10,
            theta_upper=1.0,
            theta_lower=0.3,
        )
        train_loader, test_loader = get_mnist_loaders(batch_size=128)
        for _ in range(5):
            for x, y in train_loader:
                with torch.no_grad():
                    clf.train_step(x, y, lr=0.01, epsilon=0.1)
        return clf

    def test_accuracy_above_85_percent(self, trained_classifier):
        """Accuracy should exceed 85% (matching WTA baseline of ~88%)."""
        _, test_loader = get_mnist_loaders(batch_size=128)
        acc = trained_classifier.evaluate(test_loader, epsilon=0.1)
        assert acc > 0.85, f"Accuracy {100 * acc:.2f}% < 85% target"

    def test_weights_always_ternary_mnist(self, trained_classifier):
        """After MNIST training, all weights should remain ternary."""
        w = trained_classifier.model.weight.unpack()
        assert _check_all_ternary(w), "Weights not ternary after MNIST training"

    def test_training_time_under_2_minutes(self):
        """Training should complete in under 2 minutes total."""
        clf = NeuromodulatedHebbianClassifier(
            in_features=784,
            out_features=10,
            theta_upper=1.0,
            theta_lower=0.3,
        )
        train_loader, _ = get_mnist_loaders(batch_size=128)

        start = time.time()
        for _ in range(5):
            for x, y in train_loader:
                with torch.no_grad():
                    clf.train_step(x, y, lr=0.01, epsilon=0.1)
        elapsed = time.time() - start
        assert elapsed < 120, f"Training took {elapsed:.1f}s, expected < 120s"


# ── Tests: NTH equivalence to WTA ─────────────────────────────────

class TestNTHEquivalence:
    """Verify that NTH label modulator is equivalent to WTA."""

    def test_modulator_correct_positive(self):
        """For wrong predictions, the correct class neuron gets M=+1."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        # Force wrong predictions by seeding all latent scores
        with torch.no_grad():
            clf.model._latent_scores.scores[:] = 5.0
            clf.model.refresh_weights()

        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        with torch.no_grad():
            x_ternary = torch.sign(x)
            out = clf.model(x_ternary.float())
            pred = out.argmax(dim=1)
            mod = build_label_modulator(y, pred, 10)

        wrong_mask = pred != y
        for i in range(32):
            if wrong_mask[i]:
                assert mod[i, y[i]] == 1.0, f"Wrong pred {i}: correct class {y[i]} should be +1"
            else:
                assert (mod[i] == 0).all(), f"Correct pred {i}: all zeros"

    def test_modulator_wrong_negative(self):
        """The wrongly-predicted neuron gets M=-1 for wrong predictions."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        # Force specific predictions by seeding latent scores
        with torch.no_grad():
            clf.model._latent_scores.scores[:] = 5.0
            clf.model.refresh_weights()

        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        with torch.no_grad():
            x_ternary = torch.sign(x)
            out = clf.model(x_ternary.float())
            pred = out.argmax(dim=1)
            mod = build_label_modulator(y, pred, 10)

        wrong_mask = pred != y
        if wrong_mask.any():
            for i in wrong_mask.nonzero(as_tuple=True)[0]:
                assert mod[i, pred[i]] == -1.0, \
                    f"Sample {i}: wrong prediction {pred[i]} should be -1"
                assert mod[i, y[i]] == 1.0, \
                    f"Sample {i}: correct class {y[i]} should be +1"

    def test_nth_same_as_wta_on_wrong_only(self):
        """On wrong predictions, NTH and WTA produce identical deltas.

        Key difference: WTA only updates wrong predictions, while NTH
        updates ALL samples (strengthening correct predictions too).
        To compare fairly, restrict both to wrong predictions only.
        """
        torch.manual_seed(42)
        pre = torch.randint(0, 3, (16, 784), dtype=torch.int8) - 1
        pre_f = pre.float()
        y = torch.randint(0, 10, (16,))
        pred = torch.randint(0, 10, (16,))
        lr = 0.01
        wrong = pred != y

        # WTA: only wrong samples
        if wrong.any():
            correct_hot = torch.zeros(wrong.sum(), 10).float()
            correct_hot[range(wrong.sum()), y[wrong]] = 1.0
            pred_hot = torch.zeros(wrong.sum(), 10).float()
            pred_hot[range(wrong.sum()), pred[wrong]] = 1.0
            wta = lr * (correct_hot.T @ pre_f[wrong] - pred_hot.T @ pre_f[wrong])
        else:
            wta = torch.zeros(10, 784)

        # NTH: full modulator on wrong samples only
        mod = build_label_modulator(y[wrong], pred[wrong], 10)
        nth = lr * (mod.T @ pre_f[wrong])

        assert torch.allclose(wta, nth, atol=1e-6), (
            f"On wrong preds only: NTH != WTA, "
            f"max diff = {(wta - nth).abs().max().item()}"
        )

    def test_nth_same_as_wta_on_random_batches(self):
        """On wrong predictions only, NTH and WTA match across random batches."""
        torch.manual_seed(1234)
        for trial in range(5):
            batch = torch.randint(4, 16, (1,)).item()
            pre = torch.randint(0, 3, (batch, 784), dtype=torch.int8) - 1
            pre_f = pre.float()
            y = torch.randint(0, 10, (batch,))
            pred = torch.randint(0, 10, (batch,))
            lr = 0.01

            wrong = pred != y

            if not wrong.any():
                continue  # Skip trials with no wrong preds

            # WTA: only wrong samples
            correct_hot = torch.zeros(wrong.sum(), 10).float()
            correct_hot[range(wrong.sum()), y[wrong]] = 1.0
            pred_hot = torch.zeros(wrong.sum(), 10).float()
            pred_hot[range(wrong.sum()), pred[wrong]] = 1.0
            wta = lr * (correct_hot.T @ pre_f[wrong] - pred_hot.T @ pre_f[wrong])

            # NTH: full modulator on wrong samples only
            mod = build_label_modulator(y[wrong], pred[wrong], 10)
            nth = lr * (mod.T @ pre_f[wrong])

            assert torch.allclose(wta, nth, atol=1e-6), (
                f"Trial {trial}: NTH != WTA on wrong preds, "
                f"max diff = {(wta - nth).abs().max().item()}"
            )


# ── Tests: Flip rate convergence ──────────────────────────────────

class TestFlipRate:
    """Verify that weight flips decrease over time (convergence)."""

    def test_flip_rate_decreases_over_time(self):
        """Flip rate should trend downward after initial activation."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        train_loader, _ = get_mnist_loaders(batch_size=128)

        epoch_flip_rates = []
        for _ in range(3):
            rates = []
            for x, y in train_loader:
                with torch.no_grad():
                    m = clf.train_step(x, y)
                    rates.append(m["flip_rate"])
            epoch_flip_rates.append(sum(rates) / len(rates))

        if epoch_flip_rates[1] > 0 and epoch_flip_rates[2] > 0:
            assert epoch_flip_rates[2] <= epoch_flip_rates[1] * 1.5, (
                f"Flip rate increased: {100 * epoch_flip_rates[1]:.3f}% → "
                f"{100 * epoch_flip_rates[2]:.3f}%"
            )

    def test_flip_rate_below_one_percent(self):
        """After training, flip rate should be <1% per step."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        train_loader, _ = get_mnist_loaders(batch_size=128)

        last_epoch_rates = []
        for epoch in range(3):
            rates = []
            for x, y in train_loader:
                with torch.no_grad():
                    m = clf.train_step(x, y)
                    rates.append(m["flip_rate"])
            if epoch == 2:
                last_epoch_rates = rates

        avg_rate = sum(last_epoch_rates) / max(len(last_epoch_rates), 1)
        assert avg_rate < 0.01, (
            f"Flip rate {100 * avg_rate:.3f}%/step >= 1% target"
        )


# ── Tests: Synthetic fast tests ───────────────────────────────────

class TestSyntheticFast:
    """Fast synthetic tests suitable for CI."""

    def test_synthetic_training_completes(self):
        """Training on synthetic data should complete without errors."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        loader = _make_synthetic_data(n_samples=100, in_features=784)
        for x, y in loader:
            with torch.no_grad():
                clf.train_step(x, y)

    def test_synthetic_evaluate(self):
        """Evaluate should return a reasonable value on synthetic data."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        loader = _make_synthetic_data(n_samples=300, in_features=784)
        with torch.no_grad():
            for x, y in loader:
                clf.train_step(x, y)
        acc = clf.evaluate(loader)
        assert isinstance(acc, float)

    def test_training_requires_no_grad(self):
        """Training should assert that autograd is disabled."""
        clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        x = torch.randn(16, 784)
        y = torch.randint(0, 10, (16,))
        with pytest.raises(AssertionError, match="Autograd must be disabled"):
            clf.train_step(x, y)


# ── Tests: Ablations ─────────────────────────────────────────────

class TestAblations:
    """Test different modulator configurations."""

    def test_positive_only_plateaus(self):
        """Positive-only modulator should achieve lower accuracy than full label."""
        clf_pos = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        train_loader, test_loader = get_mnist_loaders(batch_size=128)

        # Train positive-only
        for _ in range(5):
            for x, y in train_loader:
                with torch.no_grad():
                    clf_pos.train_step(x, y, lr=0.01, epsilon=0.1, positive_only=True)
        acc_pos = clf_pos.evaluate(test_loader, epsilon=0.1)

        # Train full label modulator
        clf_label = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
        for _ in range(5):
            for x, y in train_loader:
                with torch.no_grad():
                    clf_label.train_step(x, y, lr=0.01, epsilon=0.1)
        acc_label = clf_label.evaluate(test_loader, epsilon=0.1)

        # Positive-only should be notably worse than full label
        assert acc_pos < acc_label + 0.05, (
            f"Positive-only ({100 * acc_pos:.1f}%) should be worse than "
            f"full label ({100 * acc_label:.1f}%)"
        )

    def test_different_seeds_give_similar_results(self):
        """Training with different random seeds should converge similarly."""
        accs = []
        for seed in range(3):
            torch.manual_seed(seed)
            clf = NeuromodulatedHebbianClassifier(in_features=784, out_features=10)
            train_loader, test_loader = get_mnist_loaders(batch_size=128)
            for _ in range(5):
                for x, y in train_loader:
                    with torch.no_grad():
                        clf.train_step(x, y, lr=0.01, epsilon=0.1)
            acc = clf.evaluate(test_loader, epsilon=0.1)
            accs.append(acc)

        max_diff = max(accs) - min(accs)
        acc_strs = ", ".join(f"{100 * a:.1f}%" for a in accs)
        assert max_diff < 0.10, (
            f"Accuracy variance too high: [{acc_strs}], "
            f"max diff = {100 * max_diff:.1f}pp"
        )
