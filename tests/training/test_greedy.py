"""Unit tests for greedy layer-wise Hebbian training.

Tests :class:`~ph_neuro.training.greedy.MultiLayerHebbianClassifier`
on synthetic data.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.training.greedy import LayerConfig, MultiLayerHebbianClassifier


# ── Helpers ──────────────────────────────────────────────────────

def _make_synthetic_loader(
    n_samples: int = 200,
    in_features: int = 784,
    out_features: int = 10,
    batch_size: int = 32,
) -> DataLoader:
    """Create a tiny synthetic dataset for fast tests."""
    x = torch.randn(n_samples, in_features)
    y = torch.randint(0, out_features, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def _count_backward_calls_during_fit(
    classifier: MultiLayerHebbianClassifier,
    loader: DataLoader,
    configs: list[LayerConfig],
) -> int:
    """Train greedily and count how many times .backward() is called."""
    call_count = [0]
    original_backward = torch.Tensor.backward

    def tracking_backward(self, *args, **kwargs):  # type: ignore
        call_count[0] += 1
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = tracking_backward  # type: ignore
    try:
        with torch.no_grad():
            classifier.fit_greedy(loader, layer_configs=configs, verbose=False)
    finally:
        torch.Tensor.backward = original_backward  # type: ignore

    return call_count[0]


def _check_weights_ternary(w: torch.Tensor) -> bool:
    """Check all weight values are in {-1, 0, +1}."""
    return bool(torch.all((w == -1) | (w == 0) | (w == 1)).item())


# ── Test 1: Construction ─────────────────────────────────────────

class TestConstruction:
    """Verify MultiLayerHebbianClassifier construction."""

    def test_3_layer_default(self):
        """Construct a 3-layer classifier with default params."""
        clf = MultiLayerHebbianClassifier([784, 256, 128, 10])
        assert clf.n_layers == 3
        assert clf.model.get_layer(0)._in_features == 784
        assert clf.model.get_layer(0)._out_features == 256
        assert clf.model.get_layer(1)._in_features == 256
        assert clf.model.get_layer(1)._out_features == 128
        assert clf.model.get_layer(2)._in_features == 128
        assert clf.model.get_layer(2)._out_features == 10

    def test_2_layer(self):
        """Construct a 2-layer classifier."""
        clf = MultiLayerHebbianClassifier([784, 256, 10])
        assert clf.n_layers == 2
        assert clf.model.get_layer(0)._out_features == 256
        assert clf.model.get_layer(1)._out_features == 10

    def test_single_layer(self):
        """Construct a single-layer classifier (mimics Phase 0)."""
        clf = MultiLayerHebbianClassifier([784, 10])
        assert clf.n_layers == 1
        assert clf.model.get_layer(0)._out_features == 10


# ── Test 2: Forward pass ─────────────────────────────────────────

class TestForwardPass:
    """Verify forward pass works correctly."""

    def test_predict_returns_correct_shape(self):
        """predict() should return a 1D tensor of class indices."""
        clf = MultiLayerHebbianClassifier([784, 256, 128, 10])
        x = torch.randn(16, 784)
        pred = clf.predict(x)
        assert pred.shape == (16,)
        assert pred.dtype == torch.int64
        assert torch.all((pred >= 0) & (pred < 10))

    def test_evaluate_returns_float(self):
        """evaluate() should return a float between 0 and 1."""
        loader = _make_synthetic_loader(n_samples=64, in_features=784, out_features=10)
        clf = MultiLayerHebbianClassifier([784, 256, 128, 10])
        acc = clf.evaluate(loader)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


# ── Test 3: Greedy training runs without error ───────────────────

class TestGreedyTraining:
    """Verify greedy layer-wise training runs without errors."""

    def test_fit_greedy_3_layer(self):
        """3-layer greedy training should complete."""
        loader = _make_synthetic_loader(
            n_samples=100, in_features=50, out_features=5, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([50, 32, 16, 5])
        configs = [
            LayerConfig(lr=0.01, epochs=2, hebbian_rule="basic"),
            LayerConfig(lr=0.01, epochs=2, hebbian_rule="basic"),
            LayerConfig(lr=0.01, epochs=2),
        ]
        history = clf.fit_greedy(loader, layer_configs=configs, verbose=False)
        assert len(history) == 3
        for i in range(3):
            assert "flip_rate" in history[i]
            assert len(history[i]["flip_rate"]) == 2  # 2 epochs

    def test_fit_greedy_default_configs(self):
        """Default configs should work."""
        loader = _make_synthetic_loader(
            n_samples=64, in_features=20, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([20, 10, 3])
        history = clf.fit_greedy(loader, verbose=False)
        assert len(history) == 2

    def test_fit_greedy_single_layer(self):
        """Single-layer greedy training should work."""
        loader = _make_synthetic_loader(
            n_samples=64, in_features=50, out_features=5, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([50, 5])
        configs = [LayerConfig(lr=0.01, epochs=2)]
        history = clf.fit_greedy(loader, layer_configs=configs, verbose=False)
        assert len(history) == 1


# ── Test 4: No backward calls ────────────────────────────────────

class TestNoBackward:
    """Verify that .backward() is never called during greedy training."""

    def test_no_backward_2_layer(self):
        """2-layer greedy training must not call .backward()."""
        loader = _make_synthetic_loader(
            n_samples=50, in_features=20, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([20, 10, 3])
        configs = [
            LayerConfig(lr=0.01, epochs=1, hebbian_rule="basic"),
            LayerConfig(lr=0.01, epochs=1),
        ]
        n_calls = _count_backward_calls_during_fit(clf, loader, configs)
        assert n_calls == 0, f".backward() was called {n_calls} time(s)"

    def test_no_backward_3_layer(self):
        """3-layer greedy training must not call .backward()."""
        loader = _make_synthetic_loader(
            n_samples=60, in_features=30, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([30, 20, 10, 3])
        configs = [
            LayerConfig(lr=0.01, epochs=1, hebbian_rule="basic"),
            LayerConfig(lr=0.01, epochs=1, hebbian_rule="basic"),
            LayerConfig(lr=0.01, epochs=1),
        ]
        n_calls = _count_backward_calls_during_fit(clf, loader, configs)
        assert n_calls == 0, f".backward() was called {n_calls} time(s)"


# ── Test 5: Weights always ternary ───────────────────────────────

class TestWeightsAlwaysTernary:
    """Verify all weights remain in {-1, 0, +1} throughout training."""

    def test_initial_weights_ternary(self):
        """Initial weights should all be 0 (ternary)."""
        clf = MultiLayerHebbianClassifier([50, 32, 16, 5])
        for i in range(clf.n_layers):
            w = clf.model.get_layer(i).weight.unpack()
            assert _check_weights_ternary(w), f"Layer {i} initial weights not ternary"

    def test_weights_ternary_after_training(self):
        """Weights must remain ternary after greedy training."""
        loader = _make_synthetic_loader(
            n_samples=100, in_features=20, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([20, 10, 3])
        configs = [
            LayerConfig(lr=0.01, epochs=2, hebbian_rule="basic"),
            LayerConfig(lr=0.01, epochs=2),
        ]
        clf.fit_greedy(loader, layer_configs=configs, verbose=False)
        for i in range(clf.n_layers):
            w = clf.model.get_layer(i).weight.unpack()
            assert _check_weights_ternary(w), f"Layer {i} weights not ternary after training"


# ── Test 6: Layer freezing ───────────────────────────────────────

class TestLayerFreezing:
    """Verify layers are properly frozen during greedy training."""

    def test_earlier_layers_frozen_while_later_layer_trains(self):
        """When training layer 2, layer 1's weights should not change."""
        loader = _make_synthetic_loader(
            n_samples=100, in_features=20, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([20, 10, 3])

        # Train layer 0: enable it, run one epoch, freeze it
        layer0 = clf.model.get_layer(0)
        layer0.requires_hebbian_(True)
        from ph_neuro.training.greedy import train_unsupervised_epoch, _init_hidden_layer_connectivity
        _init_hidden_layer_connectivity(layer0, density=0.1)
        train_unsupervised_epoch(layer0, loader, frozen_encoder=None, device=clf.device, lr=0.02, decay=0.0, epsilon=0.0)
        layer0.requires_hebbian_(False)

        w0_before = layer0.weight.unpack().clone()
        assert not torch.all(w0_before == 0), "Layer 0 didn't learn anything"

        # Now train layer 1 with full greedy pipeline — layer 0 must not change.
        # Use epochs=0 for layer 0 so it's not re-trained.
        cfg0 = LayerConfig(lr=0.02, epochs=0, hebbian_rule="basic")
        cfg1 = LayerConfig(lr=0.02, epochs=2)
        clf.fit_greedy(loader, layer_configs=[cfg0, cfg1], verbose=False)

        w0_after = layer0.weight.unpack()
        assert torch.equal(w0_before, w0_after), (
            "Layer 0 weights changed while training layer 1"
        )


# ── Test 7: Config application ───────────────────────────────────

class TestConfigApplication:
    """Verify layer configs are applied correctly."""

    def test_config_updates_theta(self):
        """Config should update theta_upper and theta_lower on the layer."""
        loader = _make_synthetic_loader(
            n_samples=50, in_features=20, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([20, 10, 3])
        configs = [
            LayerConfig(lr=0.01, epochs=1, hebbian_rule="basic",
                        theta_upper=3.0, theta_lower=0.5),
            LayerConfig(lr=0.01, epochs=1),
        ]
        clf.fit_greedy(loader, layer_configs=configs, verbose=False)
        assert clf.model.get_layer(0).theta_upper == 3.0
        assert clf.model.get_layer(0).theta_lower == 0.5

    def test_config_updates_hebbian_rule(self):
        """Config should update the Hebbian rule on hidden layers."""
        loader = _make_synthetic_loader(
            n_samples=50, in_features=20, out_features=3, batch_size=16
        )
        clf = MultiLayerHebbianClassifier([20, 10, 3])
        configs = [
            LayerConfig(lr=0.01, epochs=1, hebbian_rule="oja"),
            LayerConfig(lr=0.01, epochs=1),
        ]
        clf.fit_greedy(loader, layer_configs=configs, verbose=False)
        assert clf.model.get_layer(0)._hebbian_rule == "oja"


# ── Test 8: Weight stats ─────────────────────────────────────────

class TestWeightStats:
    """Verify weight statistics reporting."""

    def test_get_layer_weight_stats_keys(self):
        """get_layer_weight_stats should return expected keys."""
        clf = MultiLayerHebbianClassifier([50, 32, 16, 5])
        stats = clf.get_layer_weight_stats(0)
        for key in ("pos_pct", "neg_pct", "zero_pct"):
            assert key in stats

    def test_get_all_weight_stats_length(self):
        """get_all_weight_stats should return one entry per layer."""
        clf = MultiLayerHebbianClassifier([50, 32, 16, 5])
        all_stats = clf.get_all_weight_stats()
        assert len(all_stats) == 3

    def test_weight_stats_sum_to_100(self):
        """Percentages for each layer should sum to ~100."""
        clf = MultiLayerHebbianClassifier([50, 32, 16, 5])
        all_stats = clf.get_all_weight_stats()
        for stats in all_stats:
            total = stats["pos_pct"] + stats["neg_pct"] + stats["zero_pct"]
            assert abs(total - 100.0) < 1e-6
