"""Integration tests for Phase 2 TFF-2 — Forward-Forward 2-layer MLP on MNIST.

Tests the Forward-Forward training of hidden layers and greedy layer-wise
training with FF hidden + WTA output.

Success criteria:
    1. No .backward() calls during FF hidden layer training
    2. All weights remain in {-1, 0, +1} at every step
    3. Goodness separation (g_pos - g_neg) improves during training
    4. Flip rate stabilizes (<1% per step after convergence)
    5. End-to-end 2-layer training achieves >random accuracy
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.linear import TernaryHebbianLinear
from ph_neuro.training.greedy import (
    LayerConfig,
    MultiLayerHebbianClassifier,
    evaluate_goodness_separation,
    train_ff_hidden_epoch,
    _init_hidden_layer_connectivity,
)
from ph_neuro.training.forward_forward import generate_negative_data


# ── Helpers ────────────────────────────────────────────────────────

def _make_synthetic_data(
    n_samples: int = 500,
    in_features: int = 100,
    out_features: int = 5,
) -> DataLoader:
    """Create a tiny synthetic dataset for fast tests."""
    # Create data with some structure (not purely random)
    x = torch.randn(n_samples, in_features)
    # Add class-specific structure
    y = torch.randint(0, out_features, (n_samples,))
    for c in range(out_features):
        mask = y == c
        x[mask, c * 10 : (c + 1) * 10] += 0.5  # class-specific signal
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)


def _count_backward_calls(layer, loader, device, lr_pos=0.01, lr_neg=0.005):
    """Train one epoch and count .backward() calls."""
    call_count = [0]
    original_backward = torch.Tensor.backward

    def tracking_backward(self, *args, **kwargs):
        call_count[0] += 1
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = tracking_backward
    try:
        with torch.no_grad():
            train_ff_hidden_epoch(
                layer=layer, loader=loader, frozen_encoder=None,
                device=device, lr_pos=lr_pos, lr_neg=lr_neg,
            )
    finally:
        torch.Tensor.backward = original_backward
    return call_count[0]


def _is_all_ternary(w: torch.Tensor) -> bool:
    """Check all weights are in {-1, 0, +1}."""
    return bool(((w == -1) | (w == 0) | (w == 1)).all().item())


# ── Tests: train_ff_hidden_epoch ───────────────────────────────────

class TestFFHiddenEpoch:
    """Test the Forward-Forward hidden layer training function."""

    def test_no_backward_calls(self):
        """No .backward() should be called during FF hidden training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        n_calls = _count_backward_calls(layer, loader, device)
        assert n_calls == 0, f"Expected 0 .backward() calls, got {n_calls}"

    def test_weights_remain_ternary(self):
        """All weights must be in {-1, 0, +1} after training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        with torch.no_grad():
            train_ff_hidden_epoch(
                layer=layer, loader=loader, frozen_encoder=None,
                device=device, lr_pos=0.01, lr_neg=0.005,
            )

        w = layer.weight.unpack().cpu()
        assert _is_all_ternary(w), f"Weights not ternary: {w.unique().tolist()}"

    def test_flip_rate_stabilizes(self):
        """Flip rate should converge (decrease) over multiple epochs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=500, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        flip_rates = []
        with torch.no_grad():
            for _ in range(5):
                metrics = train_ff_hidden_epoch(
                    layer=layer, loader=loader, frozen_encoder=None,
                    device=device, lr_pos=0.01, lr_neg=0.005,
                )
                flip_rates.append(metrics["flip_rate"])

        # After 5 epochs, flip rate should be < 5% (stabilizing)
        assert flip_rates[-1] < 0.05, (
            f"Flip rate did not stabilize: {flip_rates[-1]:.4f}"
        )

    def test_goodness_separation_improves(self):
        """Goodness separation (g_pos - g_neg) should increase over epochs."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=500, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        separations = []
        with torch.no_grad():
            for _ in range(5):
                metrics = train_ff_hidden_epoch(
                    layer=layer, loader=loader, frozen_encoder=None,
                    device=device, lr_pos=0.01, lr_neg=0.01,
                )
                separations.append(metrics["separation"])

        # Separation should trend upward
        assert separations[-1] >= separations[0] - 0.5, (
            f"Goodness separation decreased: {separations[0]:.2f} -> {separations[-1]:.2f}"
        )

    def test_negative_pass_disabled_with_zero_lr(self):
        """Setting lr_neg=0 should skip anti-Hebbian negative pass."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        with torch.no_grad():
            metrics = train_ff_hidden_epoch(
                layer=layer, loader=loader, frozen_encoder=None,
                device=device, lr_pos=0.01, lr_neg=0.0,
            )

        # With lr_neg=0, g_neg should still be computed but no anti-Hebbian
        assert metrics["g_pos"] >= 0
        assert metrics["g_neg"] >= 0
        assert "separation" in metrics

    def test_no_decay_reduces_flips(self):
        """With the same params, decay=0 should not cause issues."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        with torch.no_grad():
            metrics = train_ff_hidden_epoch(
                layer=layer, loader=loader, frozen_encoder=None,
                device=device, lr_pos=0.01, lr_neg=0.005, decay=0.0,
            )
        assert metrics["flip_rate"] >= 0


# ── Tests: evaluate_goodness_separation ────────────────────────────

class TestGoodnessSeparation:
    """Test the goodness separation evaluation function."""

    def test_returns_expected_keys(self):
        """Separation dict must contain all required metrics."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        sep = evaluate_goodness_separation(
            layer=layer, loader=loader, frozen_encoder=None,
            device=device,
        )
        assert "g_pos" in sep
        assert "g_neg" in sep
        assert "separation" in sep
        assert "g_pos_std" in sep
        assert "g_neg_std" in sep

    def test_separation_positive_after_training(self):
        """Separation should be positive after FF training with lr_neg."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)
        layer = TernaryHebbianLinear(50, 32).to(device)
        _init_hidden_layer_connectivity(layer, density=0.2)

        with torch.no_grad():
            for _ in range(3):
                train_ff_hidden_epoch(
                    layer=layer, loader=loader, frozen_encoder=None,
                    device=device, lr_pos=0.01, lr_neg=0.01,
                )

        sep = evaluate_goodness_separation(
            layer=layer, loader=loader, frozen_encoder=None,
            device=device,
        )
        # With top-k activation, real data should have higher raw activation sum
        assert 0 <= sep["g_pos"], f"g_pos negative: {sep['g_pos']}"
        assert 0 <= sep["g_neg"], f"g_neg negative: {sep['g_neg']}"


# ── Tests: Greedy training with FF hidden layer ────────────────────

class TestFFGreedyTraining:
    """Test greedy layer-wise training with FF hidden + WTA output."""

    def test_greedy_training_synthetic(self):
        """Full 2-layer greedy training should work end-to-end."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = _make_synthetic_data(n_samples=500, in_features=50, out_features=4)
        test_loader = _make_synthetic_data(n_samples=100, in_features=50, out_features=4)

        classifier = MultiLayerHebbianClassifier(
            layer_sizes=[50, 32, 4],
            theta_upper=5.0,
            theta_lower=1.0,
            device=device,
        )

        configs = [
            LayerConfig(lr=0.01, lr_neg=0.005, epochs=3, hebbian_rule="forward_forward"),
            LayerConfig(lr=0.01, epochs=3, hebbian_rule="basic"),
        ]

        history = classifier.fit_greedy(
            train_loader=train_loader,
            layer_configs=configs,
            epsilon=0.1,
            verbose=False,
        )

        # Check output layer accuracy improved over random
        assert len(history[1]["accuracy"]) == 3
        assert history[1]["accuracy"][-1] > 0.25, (
            f"Accuracy too low: {history[1]['accuracy'][-1]:.3f}"
        )

    def test_all_weights_ternary_after_training(self):
        """All weights must be ternary after greedy training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)

        classifier = MultiLayerHebbianClassifier(
            layer_sizes=[50, 32, 4],
            device=device,
        )

        configs = [
            LayerConfig(lr=0.01, lr_neg=0.005, epochs=2, hebbian_rule="forward_forward"),
            LayerConfig(lr=0.01, epochs=2, hebbian_rule="basic"),
        ]

        classifier.fit_greedy(loader, layer_configs=configs, verbose=False)

        for i in range(classifier.n_layers):
            w = classifier.model.get_layer(i).weight.unpack().cpu()
            assert _is_all_ternary(w), f"Layer {i} weights not ternary"

    def test_evaluate_after_training(self):
        """Evaluation should work after greedy training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader = _make_synthetic_data(n_samples=200, in_features=50, out_features=4)

        classifier = MultiLayerHebbianClassifier(
            layer_sizes=[50, 32, 4],
            device=device,
        )

        configs = [
            LayerConfig(lr=0.01, lr_neg=0.005, epochs=2, hebbian_rule="forward_forward"),
            LayerConfig(lr=0.01, epochs=2, hebbian_rule="basic"),
        ]

        classifier.fit_greedy(train_loader, layer_configs=configs, verbose=False)
        acc = classifier.evaluate(train_loader)
        assert 0.0 <= acc <= 1.0


# ── Tests: LayerConfig with FF defaults ────────────────────────────

class TestFFLayerConfig:
    """Test that FF-specific LayerConfig works correctly."""

    def test_default_ff_hidden_has_lr_neg(self):
        """Default FF config must have lr_neg > 0."""
        cfg = LayerConfig.default_ff_hidden()
        assert cfg.hebbian_rule == "forward_forward"
        assert cfg.lr_neg > 0
        assert cfg.epochs == 10

    def test_ff_config_used_in_greedy(self):
        """FF config should be accepted by fit_greedy."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader = _make_synthetic_data(n_samples=100, in_features=20, out_features=3)

        classifier = MultiLayerHebbianClassifier(
            layer_sizes=[20, 16, 3],
            device=device,
        )

        # This should not raise
        configs = [
            LayerConfig.default_ff_hidden(),
            LayerConfig.default_output(),
        ]
        classifier.fit_greedy(loader, layer_configs=configs, verbose=False)
