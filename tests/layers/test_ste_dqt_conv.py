"""Tests for TernaryDQTConv2d layer (DQT backprop for convolutions).

Covers construction, forward shapes (incl. stride), ternary weight
invariants (int8, values in {-1, 0, +1}), backprop through the custom
autograd Function, stochastic rounding, flip-rate tracking, and weight
statistics. Also validates the conv backward against a reference autograd
graph through the same float ternary weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d


class _RefConv(nn.Module):
    """Reference conv using a float parameter (for gradient comparison)."""

    def __init__(self, w: torch.Tensor, stride=1, padding=0, dilation=1):
        super().__init__()
        self.w = nn.Parameter(w.clone())
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x, self.w, stride=self.stride, padding=self.padding, dilation=self.dilation
        )


class TestTernaryDQTConv2d:
    """Suite of tests for the DQT conv layer."""

    def test_create(self):
        """Creating the layer should produce correct dimensions."""
        layer = TernaryDQTConv2d(3, 64, kernel_size=3)
        assert layer.weight_float.shape == (64, 3, 3, 3)
        assert layer.weight_ternary.shape == (64, 3, 3, 3)
        assert layer.in_channels == 3
        assert layer.out_channels == 64
        assert layer.kernel_size == (3, 3)

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        layer = TernaryDQTConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.shape == (8, 16, 32, 32), f"Got shape {out.shape}"

    def test_forward_strided(self):
        """Forward pass with stride=2 should halve spatial dims."""
        layer = TernaryDQTConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.shape == (8, 16, 16, 16), f"Got shape {out.shape}"

    def test_forward_requires_grad(self):
        """Forward should create an autograd graph for backprop."""
        layer = TernaryDQTConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 32, 32)
        out = layer(x)
        assert out.requires_grad, "Output should require grad for backprop"

    def test_ternary_weight_invariant(self):
        """weight_ternary should always be in {-1, 0, +1}."""
        layer = TernaryDQTConv2d(3, 16, kernel_size=3, padding=1)
        w = layer.ternary_weight()
        assert torch.all((w >= -1) & (w <= 1)), (
            f"Ternary weights have values outside {{-1, 0, +1}}: "
            f"min={w.min()}, max={w.max()}"
        )

    def test_ternary_weight_int8(self):
        """weight_ternary should be stored as int8."""
        layer = TernaryDQTConv2d(3, 16, kernel_size=3, padding=1)
        assert layer.weight_ternary.dtype == torch.int8

    def test_backprop_updates_weight_float(self):
        """Backward + optimizer.step() should change weight_float."""
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1, bias=False)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        x = torch.randn(4, 3, 16, 16)
        old = layer.weight_float.detach().clone()

        out = layer(x)
        out.mean().backward()
        optimizer.step()

        assert layer.weight_float.grad is not None, (
            "Gradients should be computed for weight_float"
        )
        assert not torch.allclose(layer.weight_float, old), (
            "weight_float should change after optimizer step"
        )

    def test_stochastic_rounding(self):
        """apply_stochastic_rounding() should produce valid ternary weights."""
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.randn(4, 3, 16, 16)

        for _ in range(5):
            optimizer.zero_grad()
            out = layer(x)
            out.mean().backward()
            optimizer.step()
            stats = layer.apply_stochastic_rounding()

            assert layer.weight_ternary.dtype == torch.int8
            assert torch.all((layer.weight_ternary >= -1) & (layer.weight_ternary <= 1))
            assert 0.0 <= stats["flip_rate"] <= 1.0

    def test_flip_rate_tracking(self):
        """get_flip_rate() should return a value in [0, 1]."""
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1)
        layer.apply_stochastic_rounding()
        rate = layer.get_flip_rate()
        assert 0.0 <= rate <= 1.0, f"Flip rate out of range: {rate}"

    def test_weight_stats(self):
        """get_weight_stats() should have pos + neg + zero ≈ 100%."""
        layer = TernaryDQTConv2d(3, 16, kernel_size=3, padding=1)
        stats = layer.get_weight_stats()
        total = stats["pos_pct"] + stats["neg_pct"] + stats["zero_pct"]
        assert abs(total - 100.0) < 1e-4, f"Stats don't sum to 100%: {stats}"

    def test_with_bias(self):
        """Layer with bias should produce correct shapes and bias gradient."""
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1, bias=True)
        x = torch.randn(4, 3, 16, 16)
        out = layer(x)
        assert out.shape == (4, 8, 16, 16)
        assert layer.bias is not None
        out.mean().backward()
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == (8,)

    def test_without_bias(self):
        """Layer without bias should work, bias attr should be None."""
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1, bias=False)
        x = torch.randn(4, 3, 16, 16)
        out = layer(x)
        assert out.shape == (4, 8, 16, 16)
        assert layer.bias is None

    def test_grad_input_matches_reference(self):
        """grad_input should match reference autograd through same ternary W."""
        torch.manual_seed(1)
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1, bias=False)
        x = torch.randn(4, 3, 16, 16, requires_grad=True)
        out = layer(x)
        grad_input, = torch.autograd.grad(out.mean(), x)

        ref = _RefConv(layer.weight_ternary.float(), padding=1)
        out_ref = ref(x)
        grad_input_ref, _ = torch.autograd.grad(out_ref.mean(), (x, ref.w))
        assert torch.allclose(grad_input, grad_input_ref, atol=1e-5)

    def test_grad_weight_matches_reference(self):
        """grad_weight should match reference autograd through same ternary W."""
        torch.manual_seed(1)
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1, bias=False)
        x = torch.randn(4, 3, 16, 16, requires_grad=True)
        out = layer(x)
        out.mean().backward()
        grad_weight = layer.weight_float.grad

        ref = _RefConv(layer.weight_ternary.float(), padding=1)
        out_ref = ref(x)
        _, grad_weight_ref = torch.autograd.grad(out_ref.mean(), (x, ref.w))
        assert torch.allclose(grad_weight, grad_weight_ref, atol=1e-5)

    def test_grad_matches_reference_strided(self):
        """Both grads should match reference for stride=2."""
        torch.manual_seed(2)
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        out = layer(x)
        grad_input, = torch.autograd.grad(out.mean(), x, retain_graph=True)
        out.mean().backward()
        grad_weight = layer.weight_float.grad

        ref = _RefConv(layer.weight_ternary.float(), stride=2, padding=1)
        out_ref = ref(x)
        grad_input_ref, grad_weight_ref = torch.autograd.grad(out_ref.mean(), (x, ref.w))
        assert torch.allclose(grad_input, grad_input_ref, atol=1e-5)
        assert torch.allclose(grad_weight, grad_weight_ref, atol=1e-5)

    def test_ternary_invariant_after_training(self):
        """Ternary weights should stay in {-1, 0, +1} after training."""
        layer = TernaryDQTConv2d(3, 8, kernel_size=3, padding=1)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.randn(8, 3, 16, 16)

        for _ in range(5):
            optimizer.zero_grad()
            out = layer(x)
            out.mean().backward()
            optimizer.step()
            layer.apply_stochastic_rounding()

            w = layer.weight_ternary
            assert w.dtype == torch.int8
            assert torch.all((w >= -1) & (w <= 1)), (
                "Ternary weights must stay in {-1, 0, +1} after training"
            )
