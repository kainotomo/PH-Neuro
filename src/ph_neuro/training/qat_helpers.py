"""Quantization-Aware Training (QAT) helpers for INT8 and INT4 baselines.

Provides thin wrappers using a custom Straight-Through Estimator for
fake quantization, avoiding the device-management pitfalls of
``torch.ao.quantization``.

These are standard QAT baselines — not novel contributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Custom STE Fake Quantize (device-safe) ─────────────────────────


class _FakeQuantizeSTE(torch.autograd.Function):
    """STE-based fake quantization.

    Forward: quantize input to INT N bits, then dequantize.
    Backward: identity pass-through (STE).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        num_bits: int = 8,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        # Symmetric quantization: scale = 2 * max(|x|) / (2^(num_bits) - 2)
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1

        # Per-tensor symmetric scale
        abs_max = x.abs().max().clamp(min=eps)
        scale = abs_max / float(qmax)

        # Quantize: round(x / scale), clamp, dequantize: * scale
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, qmin, qmax)
        x_dq = x_q * scale

        return x_dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


def fake_quantize_ste(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Apply fake quantization with STE backward.

    Args:
        x: Input tensor (typically weights).
        num_bits: Number of bits (e.g. 8 for INT8, 4 for INT4).

    Returns:
        Fake-quantized tensor (same shape).
    """
    return _FakeQuantizeSTE.apply(x, num_bits)


# ── Quantized Linear module ─────────────────────────────────────────


class _QuantizedLinear(nn.Module):
    """Linear layer with fake-quantized weights (STE)."""

    def __init__(self, in_features: int, out_features: int, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fake_quantize_ste(self.weight, self.num_bits)
        return nn.functional.linear(x, w_q, self.bias)


class _QuantizedConv2d(nn.Module):
    """Conv2d layer with fake-quantized weights (STE)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        num_bits: int = 8,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.stride = stride
        self.padding = padding
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, ks[0], ks[1]))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fake_quantize_ste(self.weight, self.num_bits)
        return nn.functional.conv2d(x, w_q, self.bias, stride=self.stride, padding=self.padding)


# ── Model builders ──────────────────────────────────────────────────


def _build_quant_mlp(
    layer_sizes: list[int],
    num_bits: int = 8,
    flatten: bool = True,
    device: torch.device | str | None = None,
) -> nn.Sequential:
    """Build an MLP with quantized linear layers."""
    layers: list[nn.Module] = []
    if flatten:
        layers.append(nn.Flatten())
    for i in range(len(layer_sizes) - 1):
        layers.append(_QuantizedLinear(layer_sizes[i], layer_sizes[i + 1], num_bits=num_bits))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
    model = nn.Sequential(*layers)
    if device is not None:
        model = model.to(device)
    return model


def _build_quant_cnn(
    num_bits: int = 8,
    in_channels: int = 3,
    img_size: int = 32,
    hidden_channels: int = 64,
    n_classes: int = 10,
    device: torch.device | str | None = None,
) -> nn.Sequential:
    """Build a CNN with quantized conv and linear layers."""
    flat_features = (2 * hidden_channels) * (img_size // 4) * (img_size // 4)
    model = nn.Sequential(
        _QuantizedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, num_bits=num_bits),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(hidden_channels),
        nn.MaxPool2d(2),
        _QuantizedConv2d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1, num_bits=num_bits),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(2 * hidden_channels),
        nn.MaxPool2d(2),
        nn.Flatten(),
        _QuantizedLinear(flat_features, 512, num_bits=num_bits),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        _QuantizedLinear(512, n_classes, num_bits=num_bits),
    )
    if device is not None:
        model = model.to(device)
    return model


# ── Public API ──────────────────────────────────────────────────────


def create_int8_qat_mlp(
    layer_sizes: list[int],
    device: torch.device | str | None = None,
) -> nn.Module:
    """Create an MLP with INT8 fake-quantized weights (QAT)."""
    return _build_quant_mlp(layer_sizes, num_bits=8, device=device)


def create_int4_qat_mlp(
    layer_sizes: list[int],
    device: torch.device | str | None = None,
) -> nn.Module:
    """Create an MLP with INT4 fake-quantized weights (QAT)."""
    return _build_quant_mlp(layer_sizes, num_bits=4, device=device)


def create_int8_qat_cnn(
    in_channels: int = 3,
    img_size: int = 32,
    hidden_channels: int = 64,
    n_classes: int = 10,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Create a CNN with INT8 fake-quantized weights (QAT)."""
    return _build_quant_cnn(num_bits=8, in_channels=in_channels, img_size=img_size,
                            hidden_channels=hidden_channels, n_classes=n_classes, device=device)


def create_int4_qat_cnn(
    in_channels: int = 3,
    img_size: int = 32,
    hidden_channels: int = 64,
    n_classes: int = 10,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Create a CNN with INT4 fake-quantized weights (QAT)."""
    return _build_quant_cnn(num_bits=4, in_channels=in_channels, img_size=img_size,
                            hidden_channels=hidden_channels, n_classes=n_classes, device=device)

