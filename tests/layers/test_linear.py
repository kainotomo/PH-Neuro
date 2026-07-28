"""Tests for TernaryHebbianLinear layer."""

from __future__ import annotations

import torch

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.linear import TernaryHebbianLinear


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

    def test_extra_repr(self):
        """String representation should include key parameters."""
        layer = TernaryHebbianLinear(784, 10, theta_upper=5.0, theta_lower=1.0)
        r = repr(layer)
        assert "784" in r
        assert "10" in r
        assert "5.0" in r or "5" in r

    # --- Freeze / unfreeze tests ---

    def test_requires_hebbian_false_prevents_update(self):
        """Freezing should make hebbian_update a no-op."""
        layer = TernaryHebbianLinear(5, 3)
        layer.requires_hebbian_(False)
        old_scores = layer._latent_scores.scores.clone()

        pre = torch.tensor([[1, 0, -1, 1, 0]], dtype=torch.int8)
        post = torch.tensor([[1, -1, 0]], dtype=torch.int8)
        layer.hebbian_update(pre, post, lr=10.0)

        assert torch.equal(layer._latent_scores.scores, old_scores), (
            "Scores should not change when Hebbian learning is disabled"
        )

    def test_requires_hebbian_false_prevents_decay(self):
        """Freezing should make apply_decay a no-op."""
        layer = TernaryHebbianLinear(5, 3)
        layer.requires_hebbian_(False)
        old_scores = layer._latent_scores.scores.clone()

        layer.apply_decay(decay_rate=1.0)  # would zero out scores

        assert torch.equal(layer._latent_scores.scores, old_scores), (
            "Scores should not change when decay is disabled"
        )

    # --- No-autograd verification ---

    def test_no_autograd_graph(self):
        """Forward pass and hebbian_update should produce no autograd nodes."""
        layer = TernaryHebbianLinear(784, 10)
        x = torch.randn(32, 784)

        # Forward pass
        out = layer(x)
        assert out.grad_fn is None, "Forward output should not have grad_fn"

        # Hebbian update
        pre = ternary_sign(x)
        post = ternary_sign(out.detach())
        layer.hebbian_update(pre, post, lr=0.01)

        # Verify no gradient info on parameters
        for name, param in layer.named_parameters():
            assert param.grad is None, f"Parameter {name} should not have grad"
            assert param.requires_grad is False, (
                f"Parameter {name} should not require grad"
            )
