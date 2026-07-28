"""Tests for TernaryHebbianLinear layer."""

from __future__ import annotations

import torch

from ph_neuro.layers.linear import TernaryHebbianLinear, ternary_sign


class TestTernaryHebbianLinear:
    """Suite of tests for the Hebbian linear layer."""

    def test_create(self):
        """Creating the layer should work with correct dimensions."""
        layer = TernaryHebbianLinear(784, 10)
        assert layer.weight.shape == (10, 784)
        assert layer.latent_scores.shape == (10, 784)

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = TernaryHebbianLinear(784, 10)
        x = torch.randn(32, 784)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_forward_no_backward(self):
        """Forward pass should not create autograd graph."""
        layer = TernaryHebbianLinear(784, 10)
        x = torch.randn(32, 784)
        out = layer(x)
        assert not out.requires_grad, "Output should not require grad"

    def test_hebbian_update_shape(self):
        """Hebbian update should work with correct shapes."""
        layer = TernaryHebbianLinear(784, 10)
        pre = torch.randint(0, 2, (32, 784), dtype=torch.int8) * 2 - 1  # {-1, +1}
        post = torch.randint(0, 2, (32, 10), dtype=torch.int8) * 2 - 1
        layer.hebbian_update(pre, post, lr=0.01)

    def test_refresh_weights(self):
        """Refresh weights should not crash."""
        layer = TernaryHebbianLinear(5, 3)
        # Push a latent score above threshold
        layer._latent_scores.scores[0, 0] = 10.0
        layer.refresh_weights()
        # After refresh, the corresponding ternary weight should be 1
        w = layer.weight.unpack()
        assert w[0, 0] == 1, "Weight should activate after refresh"

    def test_apply_decay(self):
        """Decay should reduce latent scores."""
        layer = TernaryHebbianLinear(5, 3)
        layer._latent_scores.scores = torch.full((3, 5), 10.0, dtype=torch.float16)
        old = layer.latent_scores[0, 0].clone()
        layer.apply_decay(decay_rate=0.1)
        assert layer.latent_scores[0, 0] < old, "Decay should reduce scores"

    def test_ternary_sign(self):
        """Ternary sign should map values to {-1, 0, +1}."""
        x = torch.tensor([2.0, 0.0, -3.0, 0.5, -0.5, 0.0])
        result = ternary_sign(x)
        expected = torch.tensor([1, 0, -1, 1, -1, 0], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_ternary_sign_with_epsilon(self):
        """With epsilon, small values should become 0."""
        x = torch.tensor([2.0, 0.1, -0.05, -3.0])
        result = ternary_sign(x, epsilon=0.5)
        expected = torch.tensor([1, 0, 0, -1], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_extra_repr(self):
        """String representation should include key parameters."""
        layer = TernaryHebbianLinear(784, 10, theta_upper=5.0, theta_lower=1.0)
        r = repr(layer)
        assert "784" in r
        assert "10" in r
        assert "5.0" in r or "5" in r
