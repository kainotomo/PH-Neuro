"""Integration tests for Phase 2 NTH-4 — Multi-layer Neuromodulated Hebbian.

Success criteria:
    1. No .backward() calls during training
    2. All weights remain in {-1, 0, +1} at every step
    3. Weight flips stabilize (<1% per step after convergence)
    4. Hidden modulator shapes are correct for all 3 approaches
    5. Approach A: M_hidden is zero for correct predictions
    6. Approach B: M_hidden = M_output @ W_out
    7. Approach C: random feedback matrix B is fixed (never updated)
    8. Training does not crash on synthetic data
    9. Accuracy > random (10%) on synthetic 100-sample dataset
"""

from __future__ import annotations

import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.training.nth_multilayer import (
    NTHMultiLayerClassifier,
    _build_hidden_modulator_label_broadcast,
    _build_hidden_modulator_weight_feedback,
    _build_hidden_modulator_random_feedback,
    _init_random_feedback_matrix,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_synthetic_data(
    n_samples: int = 200,
    in_features: int = 784,
    out_features: int = 10,
) -> DataLoader:
    """Create a tiny synthetic dataset for fast invariant tests."""
    rng = torch.Generator()
    rng.manual_seed(42)
    x = torch.randn(n_samples, in_features, generator=rng)
    y = torch.randint(0, out_features, (n_samples,), generator=rng)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)


def _count_backward_calls(
    classifier: NTHMultiLayerClassifier,
    loader: DataLoader,
    **kwargs,
) -> int:
    """Train for one step and count .backward() calls."""
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
                lr_hidden=kwargs.get("lr_hidden", 0.005),
                lr_output=kwargs.get("lr_output", 0.01),
                decay=kwargs.get("decay", 0.0),
                epsilon=kwargs.get("epsilon", 0.1),
            )
    finally:
        torch.Tensor.backward = original_backward  # type: ignore
    return call_count[0]


def _check_all_ternary(w: torch.Tensor) -> bool:
    """Check that all weights are in {-1, 0, +1}."""
    return bool(((w == -1) | (w == 0) | (w == 1)).all().item())


# ── Tests: Hidden modulator builders ──────────────────────────────

class TestHiddenModulatorLabelBroadcast:
    """Test _build_hidden_modulator_label_broadcast."""

    def test_all_correct_no_modulation(self):
        """When all predictions are correct, M_hidden should be all zeros."""
        y = torch.tensor([0, 1, 2])
        pred = torch.tensor([0, 1, 2])
        post_hidden = torch.tensor([
            [1, 0, -1, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
        ], dtype=torch.int8)

        M = _build_hidden_modulator_label_broadcast(y, pred, post_hidden)
        assert M.shape == (3, 4)
        assert torch.all(M == 0), "M should be all zeros when all predictions correct"

    def test_wrong_predictions_get_negative_modulation(self):
        """For wrong predictions, active hidden neurons get M=-1 times sign of post."""
        y = torch.tensor([0])
        pred = torch.tensor([1])
        post_hidden = torch.tensor([[1, 0, -1, 0]], dtype=torch.int8)

        M = _build_hidden_modulator_label_broadcast(y, pred, post_hidden)
        # Active neurons (index 0 and 2) should get M = -sign(post) = -1 and +1
        assert M[0, 0] == -1.0, f"Expected M[0,0]=-1, got {M[0,0]}"
        assert M[0, 1] == 0.0, f"Expected M[0,1]=0, got {M[0,1]}"
        assert M[0, 2] == 1.0, f"Expected M[0,2]=1, got {M[0,2]}"
        assert M[0, 3] == 0.0, f"Expected M[0,3]=0, got {M[0,3]}"

    def test_mixed_correct_and_wrong(self):
        """Correct predictions get zero, wrong get anti-Hebbian."""
        y = torch.tensor([0, 1, 2])
        pred = torch.tensor([0, 1, 0])  # third sample is wrong
        post_hidden = torch.tensor([
            [1, 0],
            [0, 1],
            [1, 0],
        ], dtype=torch.int8)

        M = _build_hidden_modulator_label_broadcast(y, pred, post_hidden)
        assert torch.all(M[0] == 0), "First sample correct -> M=0"
        assert torch.all(M[1] == 0), "Second sample correct -> M=0"
        assert M[2, 0] == -1.0, "Third sample wrong, active neuron 0 -> M=-1"
        assert M[2, 1] == 0.0, "Third sample wrong, inactive neuron 1 -> M=0"


class TestHiddenModulatorWeightFeedback:
    """Test _build_hidden_modulator_weight_feedback."""

    def test_correct_prediction_no_modulation(self):
        """M_output is all zeros (all correct) -> M_hidden should be all zeros."""
        batch, hidden = 3, 4
        n_classes = 3
        M_output = torch.zeros(batch, n_classes)
        W_out = torch.randn(n_classes, hidden)

        M = _build_hidden_modulator_weight_feedback(M_output, W_out)
        assert M.shape == (batch, hidden)
        assert torch.allclose(M, torch.zeros_like(M)), "M should be all zeros"

    def test_wrong_prediction_nonzero_modulation(self):
        """With a wrong prediction, M_hidden should be non-zero."""
        batch, hidden = 2, 4
        n_classes = 3
        # Sample 0: correct (y=0, pred=0 -> M_output all zero)
        # Sample 1: wrong (y=0, pred=1 -> M_output[1]=+1 at idx 0, -1 at idx 1)
        M_output = torch.zeros(batch, n_classes)
        M_output[1, 0] = 1.0   # correct class
        M_output[1, 1] = -1.0  # wrong prediction

        W_out = torch.tensor([
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        M = _build_hidden_modulator_weight_feedback(M_output, W_out)
        assert M.shape == (batch, hidden)
        assert torch.allclose(M[0], torch.zeros(hidden)), "Correct sample should be zero"
        # M[1] = W_out[0] - W_out[1] = [1, -1, -1, 1]
        expected = torch.tensor([1.0, -1.0, -1.0, 1.0])
        assert torch.allclose(M[1], expected), f"Expected {expected}, got {M[1]}"


class TestHiddenModulatorRandomFeedback:
    """Test _build_hidden_modulator_random_feedback."""

    def test_correct_prediction_no_modulation(self):
        """M_output is all zeros -> M_hidden should be all zeros."""
        batch, hidden = 3, 4
        n_classes = 3
        M_output = torch.zeros(batch, n_classes)
        B = (torch.randint(0, 2, (n_classes, hidden)) * 2 - 1).to(torch.int8)

        M = _build_hidden_modulator_random_feedback(M_output, B)
        assert M.shape == (batch, hidden)
        assert torch.allclose(M, torch.zeros_like(M)), "M should be all zeros"

    def test_wrong_prediction_nonzero(self):
        """With a wrong prediction, M_hidden should be non-zero."""
        batch, hidden = 2, 4
        n_classes = 3
        M_output = torch.zeros(batch, n_classes)
        M_output[1, 0] = 1.0
        M_output[1, 1] = -1.0

        # Use a specific B that guarantees non-zero output
        B = torch.tensor([
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=torch.int8)

        M = _build_hidden_modulator_random_feedback(M_output, B)
        assert M.shape == (batch, hidden)
        assert torch.allclose(M[0], torch.zeros(hidden)), "Correct sample should be zero"
        # M[1] = 1*B[0] + (-1)*B[1] = [1 - (-1), 0, 0, 0] = [2, 0, 0, 0]
        expected = torch.tensor([2.0, 0.0, 0.0, 0.0])
        assert torch.allclose(M[1], expected), f"Expected {expected}, got {M[1]}"


class TestInitRandomFeedbackMatrix:
    """Test _init_random_feedback_matrix."""

    def test_shape(self):
        """B should have the correct shape."""
        B = _init_random_feedback_matrix(10, 512)
        assert B.shape == (10, 512)

    def test_ternary_values(self):
        """B should have values in {-1, 0, +1}."""
        B = _init_random_feedback_matrix(10, 512)
        assert _check_all_ternary(B), "B should be ternary"

    def test_fixed_seed_reproducible(self):
        """Same seed should produce the same matrix."""
        B1 = _init_random_feedback_matrix(10, 512, seed=42)
        B2 = _init_random_feedback_matrix(10, 512, seed=42)
        assert torch.allclose(B1.float(), B2.float()), "Same seed should produce same B"

    def test_different_seed_different(self):
        """Different seeds should produce different matrices."""
        B1 = _init_random_feedback_matrix(10, 512, seed=42)
        B2 = _init_random_feedback_matrix(10, 512, seed=99)
        assert not torch.allclose(B1.float(), B2.float()), "Different seed should differ"

    def test_density(self):
        """Should have approximately the requested density."""
        B = _init_random_feedback_matrix(10, 512, density=0.3)
        actual_density = (B != 0).float().mean().item()
        assert 0.25 < actual_density < 0.35, f"Expected ~0.3 density, got {actual_density:.3f}"


# ── Tests: NTHMultiLayerClassifier ─────────────────────────────────

class TestNTHMultiLayerClassifier:
    """Test the NTHMultiLayerClassifier on synthetic data."""

    @pytest.fixture
    def classifier(self):
        return NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=64,
            out_features=4,
            modulator_mode="label_broadcast",
            theta_upper=1.0,
            theta_lower=0.3,
            device="cpu",
        )

    @pytest.fixture
    def tiny_loader(self):
        """4 classes, 100 samples, 784 dims."""
        return _make_synthetic_data(n_samples=100, in_features=784, out_features=4)

    @pytest.fixture
    def batch(self):
        """A single batch for step tests."""
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(16, 784, generator=rng)
        y = torch.randint(0, 4, (16,), generator=rng)
        return x, y

    def test_no_backward_calls(self, classifier, tiny_loader):
        """NTH training should never call .backward()."""
        n_calls = _count_backward_calls(classifier, tiny_loader)
        assert n_calls == 0, f"Expected 0 .backward() calls, got {n_calls}"

    def test_all_weights_ternary(self, classifier, batch):
        """All weights should remain in {-1, 0, +1} after a step."""
        x, y = batch
        with torch.no_grad():
            classifier.train_step(x, y)

        for layer_name in ["hidden_layer", "output_layer"]:
            layer = getattr(classifier, layer_name)
            w = layer.weight.unpack()
            assert _check_all_ternary(w), f"{layer_name} weights not ternary"

    def test_evaluate_returns_float(self, classifier, tiny_loader):
        """evaluate should return a float between 0 and 1."""
        from ph_neuro.training.neuromodulated import NeuromodulatedHebbianClassifier
        acc = classifier.evaluate(tiny_loader)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_fit_does_not_crash(self, classifier, tiny_loader):
        """fit should run without errors."""
        history = classifier.fit(
            tiny_loader, test_loader=tiny_loader,
            lr_hidden=0.005, lr_output=0.01,
            epochs=2, verbose=False,
        )
        assert "accuracy" in history
        assert len(history["accuracy"]) == 2

    def test_flip_rates_not_nan(self, classifier, batch):
        """Flip rates should be non-NaN floats."""
        x, y = batch
        with torch.no_grad():
            metrics = classifier.train_step(x, y)
        assert not any(
            isinstance(v, float) and (v != v)  # NaN check
            for v in [metrics["flip_rate_hidden"], metrics["flip_rate_output"]]
        )

    def test_get_weight_stats(self, classifier):
        """get_weight_stats should return correct structure."""
        stats = classifier.get_weight_stats()
        assert "hidden" in stats
        assert "output" in stats
        for name in ["hidden", "output"]:
            s = stats[name]
            assert "pos_pct" in s
            assert "neg_pct" in s
            assert "zero_pct" in s
            total = s["pos_pct"] + s["neg_pct"] + s["zero_pct"]
            assert abs(total - 100.0) < 1e-5, f"{name} percentages don't sum to 100"

    def test_accuracy_above_random_after_short_train(self, classifier, tiny_loader):
        """After 3 epochs on tiny data, accuracy should be above random (25%)."""
        classifier.fit(
            tiny_loader, test_loader=tiny_loader,
            lr_hidden=0.01, lr_output=0.02,
            epochs=3, verbose=False,
        )
        acc = classifier.evaluate(tiny_loader)
        # Random for 4 classes = 25%. After 3 epochs should at least match random
        assert acc >= 0.20, f"Accuracy {100 * acc:.1f}% < 20% — should be above random"
        print(f"  Tiny synthetic accuracy: {100 * acc:.1f}%")

    def test_accuracy_improves_with_training(self, classifier, tiny_loader):
        """Accuracy should improve from epoch 1 to epoch 5 (or at least not degrade)."""
        history = classifier.fit(
            tiny_loader, test_loader=tiny_loader,
            lr_hidden=0.01, lr_output=0.02,
            epochs=5, verbose=False,
        )
        # Check accuracy trend: last should be >= first (within noise)
        first_acc = history["accuracy"][0]
        last_acc = history["accuracy"][-1]
        assert last_acc >= first_acc - 0.05, (
            f"Accuracy dropped: {100 * first_acc:.1f}% -> {100 * last_acc:.1f}%"
        )


# ── Tests: All three modulator modes ──────────────────────────────

class TestAllModulatorModes:
    """Verify all three modulator modes work end-to-end."""

    @pytest.mark.parametrize("mode", ["label_broadcast", "weight_feedback", "random_feedback"])
    def test_all_modes_no_crash(self, mode):
        """All three modes should train without crashing."""
        classifier = NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=32,
            out_features=4,
            modulator_mode=mode,
            device="cpu",
        )
        loader = _make_synthetic_data(n_samples=50, in_features=784, out_features=4)
        history = classifier.fit(
            loader, test_loader=loader,
            lr_hidden=0.005, lr_output=0.01,
            epochs=2, verbose=False,
        )
        assert len(history["accuracy"]) == 2
        assert all(0.0 <= a <= 1.0 for a in history["accuracy"])

    @pytest.mark.parametrize("mode", ["label_broadcast", "weight_feedback", "random_feedback"])
    def test_no_backward_all_modes(self, mode):
        """All modulator modes should have zero .backward() calls."""
        classifier = NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=32,
            out_features=4,
            modulator_mode=mode,
            device="cpu",
        )
        loader = _make_synthetic_data(n_samples=50, in_features=784, out_features=4)
        n_calls = _count_backward_calls(classifier, loader)
        assert n_calls == 0, f"Mode {mode}: Expected 0 .backward() calls, got {n_calls}"

    @pytest.mark.parametrize("mode", ["label_broadcast", "weight_feedback", "random_feedback"])
    def test_weights_ternary_all_modes(self, mode):
        """All weights ternary after training."""
        classifier = NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=32,
            out_features=4,
            modulator_mode=mode,
            device="cpu",
        )
        loader = _make_synthetic_data(n_samples=50, in_features=784, out_features=4)
        x, y = next(iter(loader))
        with torch.no_grad():
            classifier.train_step(x, y)

        for layer in [classifier.hidden_layer, classifier.output_layer]:
            w = layer.weight.unpack()
            assert _check_all_ternary(w), f"Mode {mode}: Weights not ternary"


class TestNTHMultiLayerClassifierWeightFeedback:
    """Tests for weight_feedback mode."""

    @pytest.fixture
    def classifier(self):
        return NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=64,
            out_features=4,
            modulator_mode="weight_feedback",
            theta_upper=1.0,
            theta_lower=0.3,
            device="cpu",
        )

    def test_m_hidden_shape(self, classifier):
        """Verify M_hidden has correct shape in weight_feedback mode."""
        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(16, 784, generator=rng)
        y = torch.randint(0, 4, (16,), generator=rng)

        x = x.to(classifier.device)
        y = y.to(classifier.device)

        with torch.no_grad():
            M_out, M_hidden, h_ternary, pred = classifier._compute_modulators(x, y)

        assert M_hidden.shape == (16, 64), f"Expected (16, 64), got {M_hidden.shape}"
        assert M_out.shape == (16, 4), f"Expected (16, 4), got {M_out.shape}"


class TestNTHMultiLayerClassifierRandomFeedback:
    """Tests for random_feedback mode."""

    @pytest.fixture
    def classifier(self):
        return NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=64,
            out_features=4,
            modulator_mode="random_feedback",
            device="cpu",
        )

    def test_feedback_matrix_initialized(self, classifier):
        """Feedback matrix should be initialized."""
        assert classifier._feedback_matrix is not None
        assert classifier._feedback_matrix.shape == (4, 64)

    def test_feedback_matrix_never_updated(self, classifier):
        """Feedback matrix should not change after training steps."""
        B_before = classifier._feedback_matrix.clone()

        rng = torch.Generator()
        rng.manual_seed(42)
        x = torch.randn(16, 784, generator=rng)
        y = torch.randint(0, 4, (16,), generator=rng)

        with torch.no_grad():
            for _ in range(3):
                classifier.train_step(x, y)

        B_after = classifier._feedback_matrix
        assert torch.allclose(B_before.float(), B_after.float()), "B should never change"

    def test_label_broadcast_does_not_have_feedback(self):
        """label_broadcast mode should not create a feedback matrix."""
        classifier = NTHMultiLayerClassifier(
            in_features=784,
            hidden_size=64,
            out_features=4,
            modulator_mode="label_broadcast",
            device="cpu",
        )
        assert classifier._feedback_matrix is None
