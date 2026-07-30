"""Tests for TernarySTELinear layer (STE backprop)."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_linear import TernarySTELinear, ste_sign


class TestSTESign:
    """Tests for the STE sign function."""

    def test_forward_values(self):
        """Forward pass should return values in {-1, 0, +1}."""
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = ste_sign(x)
        expected = torch.tensor([-1, -1, 0, 1, 1], dtype=torch.float32)
        assert torch.equal(y, expected), f"Got {y}, expected {expected}"

    def test_backward_preserves_gradient(self):
        """STE backward should pass gradient through unchanged."""
        x = torch.tensor([-1.5, 0.3, 2.0], requires_grad=True)
        y = ste_sign(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None, "Gradient should flow through STE"
        assert torch.allclose(x.grad, torch.ones_like(x.grad)), (
            "STE backward should be identity"
        )

    def test_ste_enables_training(self):
        """A simple model with STE should be trainable."""
        layer = TernarySTELinear(10, 2)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        initial_weight = layer.latent_scores.detach().clone()
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        # Weight should have changed after one step
        assert not torch.allclose(layer.latent_scores, initial_weight), (
            "Latent scores should change after optimization"
        )


class TestTernarySTELinear:
    """Suite of tests for the STE linear layer."""

    def test_create(self):
        """Creating the layer should work with correct dimensions."""
        layer = TernarySTELinear(784, 10)
        assert layer.latent_scores.shape == (10, 784)
        assert layer.in_features == 784
        assert layer.out_features == 10

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = TernarySTELinear(784, 10)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_forward_requires_grad(self):
        """Forward pass should create autograd graph (unlike Hebbian)."""
        layer = TernarySTELinear(784, 10)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.requires_grad, "Output should require grad for backprop"

    def test_ternary_weight_invariant(self):
        """Ternary weights should always be in {-1, 0, +1}."""
        layer = TernarySTELinear(20, 10)
        w = layer.ternary_weight()
        assert w.dtype == torch.int8
        assert torch.all((w >= -1) & (w <= 1)), (
            f"Weights have values outside {-1, 0, +1}: "
            f"min={w.min()}, max={w.max()}"
        )

    def test_ternary_weight_derived_from_latent(self):
        """Ternary weight should equal sign of latent scores."""
        layer = TernarySTELinear(5, 3)
        expected = layer.latent_scores.sign().to(torch.int8)
        actual = layer.ternary_weight()
        assert torch.equal(actual, expected), (
            "ternary_weight() should equal sign(latent_scores)"
        )

    def test_backprop_updates_latent_scores(self):
        """Backprop through the layer should update latent scores."""
        layer = TernarySTELinear(10, 2)
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

    def test_multilayer_training_converges(self):
        """A 2-layer STE MLP should converge on a simple binary task."""
        model = torch.nn.Sequential(
            TernarySTELinear(2, 16),
            torch.nn.ReLU(),
            TernarySTELinear(16, 2),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        # Simple 2D binary classification (XOR-like)
        rng = torch.Generator().manual_seed(42)
        x = torch.rand(500, 2, generator=rng)
        y = ((x[:, 0] > 0.5) ^ (x[:, 1] > 0.5)).long()

        for _step in range(200):
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out = model(x)
            acc = out.argmax(dim=1).eq(y).float().mean().item()

        assert acc > 0.8, f"Model should converge, got accuracy={acc:.3f}"

    def test_ternary_weight_invariant_after_training(self):
        """Ternary weights should stay in {-1, 0, +1} even after training."""
        model = torch.nn.Sequential(
            TernarySTELinear(4, 8),
            torch.nn.ReLU(),
            TernarySTELinear(8, 2),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.randn(50, 4)
        y = torch.randint(0, 2, (50,))

        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        # Check all STE layers maintain ternary invariant
        for module in model.modules():
            if isinstance(module, TernarySTELinear):
                w = module.ternary_weight()
                assert torch.all((w >= -1) & (w <= 1)), (
                    "Ternary weights must stay in {-1, 0, +1} after training"
                )

    def test_extra_repr(self):
        """String representation should include key parameters."""
        layer = TernarySTELinear(784, 10, bias=True)
        rep = repr(layer)
        assert "in_features=784" in rep
        assert "out_features=10" in rep
        assert "bias=True" in rep
