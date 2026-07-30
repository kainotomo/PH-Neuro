"""Tests for Hysteresis-STE layers (ste_hysteresis module).

Tests cover:
- ``ste_sign_hysteresis`` forward and backward
- ``HysteresisSTELinear`` layer construction and forward
- ``HysteresisSTEConv2d`` layer construction and forward
- Ternary weight invariant (always {-1, 0, +1})
- Hysteresis state persistence across forward passes
- Gradient flow through hysteresis layers
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_hysteresis import (
    HysteresisSTEConv2d,
    HysteresisSTELinear,
    ste_sign_hysteresis,
)


# ═══════════════════════════════════════════════════════════════════
#  Tests for ste_sign_hysteresis
# ═══════════════════════════════════════════════════════════════════


class TestSTESignHysteresis:
    """Tests for the hysteresis STE sign function."""

    def test_forward_activation_threshold(self):
        """|x| > theta_upper should activate to sign(x)."""
        # Scores exceeding theta_upper with prev=0 -> should activate
        x = torch.tensor([[3.0, -0.1, 0.0, 0.1, -3.0]], dtype=torch.float32)
        prev = torch.zeros_like(x, dtype=torch.int8)
        result = ste_sign_hysteresis(x, prev, theta_upper=1.0, theta_lower=0.3)
        expected = torch.tensor([[1, 0, 0, 0, -1]], dtype=torch.float32)
        assert torch.equal(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_forward_deactivation_threshold(self):
        """|x| < theta_lower should deactivate to 0."""
        # Scores below theta_lower with prev=+/-1 -> should deactivate
        x = torch.tensor([[0.05, -0.05, 0.0, 0.2, -0.1]], dtype=torch.float32)
        prev = torch.tensor([[1, -1, 1, 0, -1]], dtype=torch.int8)
        result = ste_sign_hysteresis(x, prev, theta_upper=1.0, theta_lower=0.3)
        expected = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)
        assert torch.equal(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_forward_hysteresis_gap(self):
        """Values in hysteresis gap should preserve prev_ternary."""
        x = torch.tensor([[0.5, -0.6, 0.8]], dtype=torch.float32)
        prev = torch.tensor([[1, -1, 0]], dtype=torch.int8)
        result = ste_sign_hysteresis(x, prev, theta_upper=1.0, theta_lower=0.3)
        expected = torch.tensor([[1, -1, 0]], dtype=torch.float32)
        assert torch.equal(result, expected), (
            f"Hysteresis gap should preserve prev values. Expected {expected}, got {result}"
        )

    def test_forward_full_logic(self):
        """Combined test: activation + deactivation + preservation."""
        x = torch.tensor([[2.0, 0.5, 0.05, -0.5, -2.0]], dtype=torch.float32)
        prev = torch.tensor([[0, 1, 1, -1, 0]], dtype=torch.int8)
        result = ste_sign_hysteresis(x, prev, theta_upper=1.0, theta_lower=0.3)
        # 2.0 > 1.0 and prev=0 -> activate to 1
        # 0.5 in [0.3, 1.0] gap and prev=1 -> preserve 1
        # 0.05 < 0.3 and prev=1 -> deactivate to 0
        # -0.5 in [0.3, 1.0] gap and prev=-1 -> preserve -1
        # -2.0 < -1.0 and prev=0 -> activate to -1
        expected = torch.tensor([[1, 1, 0, -1, -1]], dtype=torch.float32)
        assert torch.equal(result, expected), (
            f"Expected {expected}, got {result}"
        )

    def test_backward_preserves_gradient(self):
        """STE backward should pass gradient through unchanged."""
        x = torch.tensor([[-1.5, 0.3, 2.0]], dtype=torch.float32, requires_grad=True)
        prev = torch.zeros_like(x, dtype=torch.int8)
        y = ste_sign_hysteresis(x, prev, theta_upper=1.0, theta_lower=0.3)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through hysteresis STE"
        assert torch.allclose(x.grad, torch.ones_like(x.grad)), (
            f"Hysteresis STE backward should be identity, got {x.grad}"
        )

    def test_gradient_flows_to_latent_scores(self):
        """Gradient should flow from output through STE to latent scores."""
        x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        prev = torch.zeros(3, 5, dtype=torch.int8)
        y = ste_sign_hysteresis(x, prev, theta_upper=1.0, theta_lower=0.3)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow to input"
        assert x.grad.shape == x.shape, "Gradient shape should match input"
        assert torch.allclose(x.grad, torch.ones_like(x.grad)), (
            "Hysteresis-STE backward should be identity"
        )

    def test_gradient_flows_with_different_theta(self):
        """Gradient should flow correctly with different thresholds."""
        x = torch.tensor([[0.2, 1.5, -2.0]], dtype=torch.float64, requires_grad=True)
        prev = torch.tensor([[1, 0, 0]], dtype=torch.int8)
        y = ste_sign_hysteresis(x, prev, theta_upper=0.5, theta_lower=0.1)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow with different theta"
        # 0.2 -> deactivated (was 1, now 0) -> gradient flows
        # 1.5 -> activated (was 0, now 1) -> gradient flows (sign has 0 derivative)
        # -2.0 -> activated (was 0, now -1) -> gradient flows
        assert x.grad.shape == x.shape
        # Values that changed (deactivated) get identity gradient
        # Values that stayed the same or activated via sign still get identity
        assert torch.allclose(x.grad, torch.ones_like(x.grad)), (
            "Backward should be identity regardless of thresholds"
        )


# ═══════════════════════════════════════════════════════════════════
#  Tests for HysteresisSTELinear
# ═══════════════════════════════════════════════════════════════════


class TestHysteresisSTELinear:
    """Suite of tests for the Hysteresis-STE linear layer."""

    def test_create(self):
        """Creating the layer should work with correct dimensions."""
        layer = HysteresisSTELinear(784, 10)
        assert layer.latent_scores.shape == (10, 784)
        assert layer.in_features == 784
        assert layer.out_features == 10
        assert layer.theta_upper == 1.0
        assert layer.theta_lower == 0.3

    def test_create_custom_thetas(self):
        """Custom thresholds should be stored correctly."""
        layer = HysteresisSTELinear(784, 10, theta_upper=0.5, theta_lower=0.1)
        assert layer.theta_upper == 0.5
        assert layer.theta_lower == 0.1

    def test_prev_ternary_buffer_exists(self):
        """prev_ternary buffer should be initialized to zeros."""
        layer = HysteresisSTELinear(20, 10)
        assert hasattr(layer, "prev_ternary")
        assert layer.prev_ternary.shape == (10, 20)
        assert layer.prev_ternary.dtype == torch.int8
        assert torch.all(layer.prev_ternary == 0)

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = HysteresisSTELinear(784, 10)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_forward_requires_grad(self):
        """Forward pass should create autograd graph."""
        layer = HysteresisSTELinear(784, 10)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.requires_grad, "Output should require grad for backprop"

    def test_ternary_weight_invariant(self):
        """Ternary weights should always be in {-1, 0, +1}."""
        layer = HysteresisSTELinear(20, 10)
        w = layer.ternary_weight()
        assert w.dtype == torch.int8
        assert torch.all((w >= -1) & (w <= 1)), (
            f"Weights have values outside {{-1, 0, +1}}: "
            f"min={w.min()}, max={w.max()}"
        )

    def test_prev_ternary_updates_after_forward(self):
        """prev_ternary buffer should be updated after each forward pass."""
        layer = HysteresisSTELinear(10, 5, theta_upper=0.5, theta_lower=0.1)
        assert torch.all(layer.prev_ternary == 0), "Initial prev_ternary should be zeros"

        # Run forward with strong scores -> should activate weights
        with torch.no_grad():
            layer.latent_scores.fill_(10.0)
        x = torch.randn(8, 10)
        _ = layer(x)

        # After forward with strong scores, prev_ternary should be all +1 (or -1 depending on sign)
        assert not torch.all(layer.prev_ternary == 0), (
            "prev_ternary should be updated after forward"
        )
        assert torch.all(layer.prev_ternary >= -1) and torch.all(layer.prev_ternary <= 1)

    def test_prev_ternary_persistence(self):
        """prev_ternary should persist between forward passes (hysteresis)."""
        layer = HysteresisSTELinear(
            10, 5, theta_upper=1.0, theta_lower=0.3,
        )

        # First forward: activate some weights with strong scores
        with torch.no_grad():
            layer.latent_scores.fill_(2.0)
            layer.latent_scores[:, :3] = -2.0
        x = torch.randn(8, 10)
        _ = layer(x)
        snapshot1 = layer.prev_ternary.clone()

        # Second forward: keep scores strong (same region) -> should preserve
        _ = layer(x)
        snapshot2 = layer.prev_ternary.clone()

        assert torch.equal(snapshot1, snapshot2), (
            "prev_ternary should be stable when scores stay in same region"
        )

    def test_backprop_updates_latent_scores(self):
        """Backprop through the layer should update latent scores."""
        layer = HysteresisSTELinear(10, 2)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)

        x = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))

        old_latent = layer.latent_scores.detach().clone()
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        assert layer.latent_scores.grad is not None, (
            "Gradients should be computed for latent_scores"
        )
        assert not torch.allclose(layer.latent_scores, old_latent), (
            "Latent scores should change after optimizer step"
        )

    def test_ternary_invariant_after_training_steps(self):
        """Ternary weights should stay in {-1, 0, +1} after training."""
        layer = HysteresisSTELinear(10, 5, theta_upper=1.0, theta_lower=0.3)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)

        for _step in range(50):
            x = torch.randn(16, 10)
            out = layer(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Check invariant
            w = layer.ternary_weight()
            assert torch.all((w >= -1) & (w <= 1)), (
                f"Weights outside {{-1, 0, +1}} at step {_step}: "
                f"min={w.min()}, max={w.max()}"
            )

    def test_reset_hysteresis_state(self):
        """reset_hysteresis_state() should zero out prev_ternary."""
        layer = HysteresisSTELinear(10, 5, theta_upper=0.5, theta_lower=0.1)
        with torch.no_grad():
            layer.latent_scores.fill_(10.0)
        x = torch.randn(8, 10)
        _ = layer(x)
        assert not torch.all(layer.prev_ternary == 0), (
            "prev_ternary should be non-zero after activation"
        )
        layer.reset_hysteresis_state()
        assert torch.all(layer.prev_ternary == 0), (
            "prev_ternary should be zero after reset"
        )

    def test_extra_repr(self):
        """extra_repr should include thresholds."""
        layer = HysteresisSTELinear(10, 5, theta_upper=2.0, theta_lower=0.5)
        rep = layer.extra_repr()
        assert "theta_upper=2.0" in rep
        assert "theta_lower=0.5" in rep

    def test_with_bias(self):
        """Layer with bias should work correctly."""
        layer = HysteresisSTELinear(10, 5, bias=True)
        x = torch.randn(8, 10)
        out = layer(x)
        assert out.shape == (8, 5)
        assert layer.bias is not None

    def test_without_bias(self):
        """Layer without bias should work, bias attr should be None."""
        layer = HysteresisSTELinear(10, 5, bias=False)
        x = torch.randn(8, 10)
        out = layer(x)
        assert out.shape == (8, 5)
        assert layer.bias is None


# ═══════════════════════════════════════════════════════════════════
#  Tests for HysteresisSTEConv2d
# ═══════════════════════════════════════════════════════════════════


class TestHysteresisSTEConv2d:
    """Suite of tests for the Hysteresis-STE conv layer."""

    def test_create(self):
        """Creating the layer should work with correct dimensions."""
        layer = HysteresisSTEConv2d(3, 64, kernel_size=3)
        assert layer.latent_scores.shape == (64, 3, 3, 3)
        assert layer.in_channels == 3
        assert layer.out_channels == 64

    def test_create_custom_thetas(self):
        """Custom thresholds should be stored."""
        layer = HysteresisSTEConv2d(3, 16, kernel_size=3, theta_upper=0.5, theta_lower=0.15)
        assert layer.theta_upper == 0.5
        assert layer.theta_lower == 0.15

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = HysteresisSTEConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.shape == (8, 16, 32, 32), f"Got shape {out.shape}"

    def test_forward_strided_shape(self):
        """Forward pass with stride=2 should halve spatial dims."""
        layer = HysteresisSTEConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.shape == (8, 16, 16, 16), f"Got shape {out.shape}"

    def test_forward_requires_grad(self):
        """Forward should create autograd graph for backprop."""
        layer = HysteresisSTEConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.requires_grad, "Output should require grad for backprop"

    def test_ternary_weight_invariant(self):
        """Ternary weights should always be in {-1, 0, +1}."""
        layer = HysteresisSTEConv2d(3, 16, kernel_size=3, padding=1)
        w = layer.ternary_weight()
        assert w.dtype == torch.int8
        assert torch.all((w >= -1) & (w <= 1)), (
            f"Weights have values outside {{-1, 0, +1}}: "
            f"min={w.min()}, max={w.max()}"
        )

    def test_prev_ternary_updates_after_forward(self):
        """prev_ternary buffer should update after forward."""
        layer = HysteresisSTEConv2d(3, 8, kernel_size=3, padding=1, theta_upper=0.5, theta_lower=0.1)
        assert torch.all(layer.prev_ternary == 0)

        with torch.no_grad():
            layer.latent_scores.fill_(10.0)
        x = torch.randn(4, 3, 16, 16)
        _ = layer(x)

        assert not torch.all(layer.prev_ternary == 0), (
            "prev_ternary should be updated after forward with strong scores"
        )

    def test_backprop_updates_latent_scores(self):
        """Backprop through conv layer should update latent scores."""
        layer = HysteresisSTEConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        x = torch.randn(4, 3, 8, 8)
        old_latent = layer.latent_scores.detach().clone()

        out = layer(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()

        assert layer.latent_scores.grad is not None, (
            "Gradients should be computed for latent_scores"
        )
        assert not torch.allclose(layer.latent_scores, old_latent), (
            "Latent scores should change after optimizer step"
        )

    def test_ternary_invariant_after_training_steps(self):
        """Ternary weights should stay in {-1, 0, +1} after training steps."""
        layer = HysteresisSTEConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)

        for _step in range(30):
            x = torch.randn(8, 3, 16, 16)
            out = layer(x)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            w = layer.ternary_weight()
            assert torch.all((w >= -1) & (w <= 1)), (
                f"Weights outside {{-1, 0, +1}} at step {_step}: "
                f"min={w.min()}, max={w.max()}"
            )

    def test_reset_hysteresis_state(self):
        """reset_hysteresis_state() should zero out prev_ternary."""
        layer = HysteresisSTEConv2d(3, 8, kernel_size=3, padding=1, theta_upper=0.5, theta_lower=0.1)
        with torch.no_grad():
            layer.latent_scores.fill_(10.0)
        x = torch.randn(4, 3, 8, 8)
        _ = layer(x)
        assert not torch.all(layer.prev_ternary == 0)
        layer.reset_hysteresis_state()
        assert torch.all(layer.prev_ternary == 0)

    def test_with_bias(self):
        """Layer with bias should produce correct shapes."""
        layer = HysteresisSTEConv2d(3, 8, kernel_size=3, padding=1, bias=True)
        x = torch.randn(4, 3, 16, 16)
        out = layer(x)
        assert out.shape == (4, 8, 16, 16)
        assert layer.bias is not None

    def test_without_bias(self):
        """Layer without bias should work, bias attr should be None."""
        layer = HysteresisSTEConv2d(3, 8, kernel_size=3, padding=1, bias=False)
        x = torch.randn(4, 3, 16, 16)
        out = layer(x)
        assert out.shape == (4, 8, 16, 16)
        assert layer.bias is None


# ═══════════════════════════════════════════════════════════════════
#  Integration: training with Hysteresis-STE
# ═══════════════════════════════════════════════════════════════════


class TestHysteresisSTETraining:
    """Integration tests for training with Hysteresis-STE."""

    def test_mlp_trains_mnist_subset(self):
        """A small Hysteresis-STE MLP should improve accuracy when trained on MNIST subset."""
        from ph_neuro.models.ste_models import hyst_ste_mlp

        # Generate synthetic MNIST-like data (small subset for speed)
        torch.manual_seed(42)
        n_samples = 500
        n_features = 784
        n_classes = 10

        x_train = torch.randn(n_samples, n_features)
        y_train = torch.randint(0, n_classes, (n_samples,))
        x_test = torch.randn(200, n_features)
        y_test = torch.randint(0, n_classes, (200,))

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

        model = hyst_ste_mlp(
            [784, 128, 10],
            theta_upper=1.0,
            theta_lower=0.3,
            batch_norm=False,
            flatten=False,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        epochs = 5

        initial_acc = evaluate_model(model, test_loader)

        model.train()
        for _epoch in range(epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

        final_acc = evaluate_model(model, test_loader)

        # Accuracy should improve
        assert final_acc > initial_acc, (
            f"Accuracy should improve: {initial_acc:.4f} -> {final_acc:.4f}"
        )

    def test_higher_theta_upper_gives_more_sparsity(self):
        """Higher theta_upper should produce higher sparsity."""
        from ph_neuro.models.ste_models import hyst_ste_mlp

        torch.manual_seed(42)
        n_samples = 300
        x = torch.randn(n_samples, 100)
        y = torch.randint(0, 5, (n_samples,))
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        # Train with low theta_upper
        model_low = hyst_ste_mlp([100, 50, 5], theta_upper=0.3, theta_lower=0.1, batch_norm=False, flatten=False)
        opt_low = torch.optim.AdamW(model_low.parameters(), lr=0.001)
        model_low.train()
        for _ in range(10):
            for bx, by in loader:
                opt_low.zero_grad()
                out = model_low(bx)
                F.cross_entropy(out, by).backward()
                opt_low.step()
        sparsity_low = _compute_sparsity_from_model(model_low)

        # Train with high theta_upper
        model_high = hyst_ste_mlp([100, 50, 5], theta_upper=3.0, theta_lower=1.0, batch_norm=False, flatten=False)
        opt_high = torch.optim.AdamW(model_high.parameters(), lr=0.001)
        model_high.train()
        for _ in range(10):
            for bx, by in loader:
                opt_high.zero_grad()
                out = model_high(bx)
                F.cross_entropy(out, by).backward()
                opt_high.step()
        sparsity_high = _compute_sparsity_from_model(model_high)

        assert sparsity_high > sparsity_low, (
            f"Higher theta_upper should give higher sparsity: "
            f"low={sparsity_low:.1f}%, high={sparsity_high:.1f}%"
        )


# ── Helpers ─────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        out = model(x)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def _compute_sparsity_from_model(model: torch.nn.Module) -> float:
    """Compute sparsity % across all HysteresisSTELinear layers."""
    from ph_neuro.layers.ste_hysteresis import HysteresisSTELinear

    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, HysteresisSTELinear):
            w = module.ternary_weight().flatten()
            total += w.numel()
            zeros += (w == 0).sum().item()
    if total == 0:
        return 0.0
    return 100.0 * zeros / total
