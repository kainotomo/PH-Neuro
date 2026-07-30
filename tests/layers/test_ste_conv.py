"""Tests for TernarySTEConv2d layer (STE backprop for convolutions)."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_conv import TernarySTEConv2d


class TestTernarySTEConv2d:
    """Suite of tests for the STE conv layer."""

    def test_create(self):
        """Creating the layer should work with correct dimensions."""
        layer = TernarySTEConv2d(3, 64, kernel_size=3)
        assert layer.latent_scores.shape == (64, 3, 3, 3)
        assert layer.in_channels == 3
        assert layer.out_channels == 64

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = TernarySTEConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.shape == (8, 16, 32, 32), f"Got shape {out.shape}"

    def test_forward_strided_shape(self):
        """Forward pass with stride=2 should halve spatial dims."""
        layer = TernarySTEConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.shape == (8, 16, 16, 16), f"Got shape {out.shape}"

    def test_forward_requires_grad(self):
        """Forward should create autograd graph for backprop."""
        layer = TernarySTEConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.requires_grad, "Output should require grad for backprop"

    def test_ternary_weight_invariant(self):
        """Ternary weights should always be in {-1, 0, +1}."""
        layer = TernarySTEConv2d(3, 16, kernel_size=3, padding=1)
        w = layer.ternary_weight()
        assert w.dtype == torch.int8
        assert torch.all((w >= -1) & (w <= 1)), (
            f"Weights have values outside {-1, 0, +1}: "
            f"min={w.min()}, max={w.max()}"
        )

    def test_ternary_weight_derived_from_latent(self):
        """Ternary weight should equal sign of latent scores."""
        layer = TernarySTEConv2d(3, 8, kernel_size=3)
        expected = layer.latent_scores.sign().to(torch.int8)
        actual = layer.ternary_weight()
        assert torch.equal(actual, expected)

    def test_backprop_updates_latent_scores(self):
        """Backprop through the layer should update latent scores."""
        layer = TernarySTEConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        x = torch.randn(4, 3, 8, 8)
        old_latent = layer.latent_scores.detach().clone()

        out = layer(x)
        # Simple loss: reduce mean
        loss = out.mean()
        loss.backward()
        optimizer.step()

        assert layer.latent_scores.grad is not None, (
            "Gradients should be computed for latent_scores"
        )
        assert not torch.allclose(layer.latent_scores, old_latent), (
            "Latent scores should change after optimizer step"
        )

    def test_with_bias(self):
        """Layer with bias should produce correct shapes."""
        layer = TernarySTEConv2d(3, 8, kernel_size=3, padding=1, bias=True)
        x = torch.randn(4, 3, 16, 16)
        out = layer(x)
        assert out.shape == (4, 8, 16, 16)
        assert layer.bias is not None

    def test_without_bias(self):
        """Layer without bias should work, bias attr should be None."""
        layer = TernarySTEConv2d(3, 8, kernel_size=3, padding=1, bias=False)
        x = torch.randn(4, 3, 16, 16)
        out = layer(x)
        assert out.shape == (4, 8, 16, 16)
        assert layer.bias is None

    def test_ternary_invariant_after_training(self):
        """Ternary weights should stay in {-1, 0, +1} after training."""
        layer = TernarySTEConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.randn(8, 3, 16, 16)

        for _ in range(5):
            optimizer.zero_grad()
            out = layer(x)
            loss = out.mean()
            loss.backward()
            optimizer.step()

            w = layer.ternary_weight()
            assert torch.all((w >= -1) & (w <= 1)), (
                "Ternary weights must stay in {-1, 0, +1} after training"
            )
