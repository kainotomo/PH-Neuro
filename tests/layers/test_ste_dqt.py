"""Tests for TernaryDQTLinear layer (DQT with stochastic rounding).

Covers construction, forward pass (incl. flattened inputs), ternary weight
invariants (int8, values in {-1, 0, +1}), stochastic rounding, flip-rate
tracking, weight statistics, and the deterministic (sign-based) rounding
used for annealing during the fine-tuning phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear, stochastic_round


class TestTernaryDQTLinear:
    """Suite of tests for the DQT linear layer."""

    def test_create(self):
        """Creating the layer should produce correct dimensions."""
        layer = TernaryDQTLinear(784, 256)
        assert layer.weight_float.shape == (256, 784)
        assert layer.weight_ternary.shape == (256, 784)
        assert layer.in_features == 784
        assert layer.out_features == 256

    def test_ternary_weight_invariant(self):
        """weight_ternary should always be in {-1, 0, +1} and int8."""
        layer = TernaryDQTLinear(784, 256)
        w = layer.weight_ternary
        assert w.dtype == torch.int8
        assert torch.all((w >= -1) & (w <= 1)), (
            f"Ternary weights outside {{-1, 0, +1}}: min={w.min()}, max={w.max()}"
        )

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = TernaryDQTLinear(784, 10)
        x = torch.randn(8, 784)
        out = layer(x)
        assert out.shape == (8, 10), f"Got shape {out.shape}"

    def test_forward_flattened(self):
        """Forward pass should flatten (batch, *, in_features) inputs."""
        layer = TernaryDQTLinear(784, 10)
        x = torch.randn(8, 1, 784)
        out = layer(x)
        assert out.shape == (8, 10), f"Got shape {out.shape}"

    def test_forward_requires_grad(self):
        """Forward should create an autograd graph for backprop."""
        layer = TernaryDQTLinear(64, 16)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.requires_grad, "Output should require grad for backprop"

    def test_stochastic_rounding_returns_ternary(self):
        """apply_stochastic_rounding() should keep weights ternary int8."""
        layer = TernaryDQTLinear(64, 16)
        for _ in range(3):
            stats = layer.apply_stochastic_rounding()
            w = layer.weight_ternary
            assert w.dtype == torch.int8
            assert torch.all((w >= -1) & (w <= 1))
            assert 0.0 <= stats["flip_rate"] <= 1.0

    def test_deterministic_rounding(self):
        """After deterministic rounding, weight_ternary == sign(weight_float)."""
        layer = TernaryDQTLinear(64, 16)
        # Perturb the float buffer so sign() differs from the initial ternary
        with torch.no_grad():
            layer.weight_float.data.add_(0.5)
        old = layer.weight_ternary.clone()
        stats = layer.apply_deterministic_rounding()

        expected = layer.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
        assert torch.equal(layer.weight_ternary, expected), (
            "weight_ternary should equal sign(weight_float) after deterministic rounding"
        )
        assert layer.weight_ternary.dtype == torch.int8
        assert 0.0 <= stats["flip_rate"] <= 1.0
        # n_flips should count the actual old->new differences
        n_actual = (old != expected).sum().item()
        assert stats["n_flips"] == n_actual
        assert stats["flip_rate"] == n_actual / max(expected.numel(), 1)

    def test_flip_rate_tracking(self):
        """get_flip_rate() should return a value in [0, 1]."""
        layer = TernaryDQTLinear(64, 16)
        layer.apply_stochastic_rounding()
        rate = layer.get_flip_rate()
        assert 0.0 <= rate <= 1.0, f"Flip rate out of range: {rate}"

    def test_weight_stats_sum(self):
        """get_weight_stats() should have pos + neg + zero ≈ 100%."""
        layer = TernaryDQTLinear(64, 16)
        stats = layer.get_weight_stats()
        total = stats["pos_pct"] + stats["neg_pct"] + stats["zero_pct"]
        assert abs(total - 100.0) < 1e-4, f"Stats don't sum to 100%: {stats}"

    def test_backprop_updates_weight_float(self):
        """Backward + optimizer.step() should change weight_float."""
        layer = TernaryDQTLinear(64, 16)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(8, 64)
        old = layer.weight_float.detach().clone()

        out = layer(x)
        out.mean().backward()
        optimizer.step()

        assert layer.weight_float.grad is not None
        assert not torch.allclose(layer.weight_float, old), (
            "weight_float should change after optimizer step"
        )

    def test_ternary_invariant_after_training(self):
        """Ternary weights should stay in {-1, 0, +1} after training."""
        layer = TernaryDQTLinear(64, 16)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.randn(8, 64)

        for _ in range(5):
            optimizer.zero_grad()
            out = layer(x)
            out.mean().backward()
            optimizer.step()
            layer.apply_stochastic_rounding()

            w = layer.weight_ternary
            assert w.dtype == torch.int8
            assert torch.all((w >= -1) & (w <= 1))
