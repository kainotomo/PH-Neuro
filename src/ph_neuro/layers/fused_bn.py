"""Fused ternary layers and element-wise affine BN replacements.

After training, BatchNorm layers (at inference) perform a simple affine
transform::

    z = γ * (x - μ) / √(σ² + ε) + β

This is mathematically equivalent to a per-channel element-wise affine
operation::

    z = scale * x + bias
    where scale = γ / √(σ² + ε),  bias = β - γ * μ / √(σ² + ε)

The ``ElementWiseAffine1d`` / ``ElementWiseAffine2d`` layers replace
``BatchNorm1d`` / ``BatchNorm2d`` at inference time, providing the same
output but with less compute (no mean/variance normalization).

Additionally, ``FusedTernaryLinear`` and ``FusedTernaryConv2d`` combine
a ternary STE layer with its following BN into a single layer, for use
when the model structure is ``TernarySTE* → BN`` with no intervening
activation.

Usage::

    from ph_neuro.models.fuse_bn import fuse_bn_layers

    model = ste_mlp([784, 512, 256, 10], batch_norm=True)
    model.eval()
    fused = fuse_bn_layers(model)   # BN → ElementWiseAffine
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_linear import ste_sign


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an int to a 2-tuple if needed."""
    if isinstance(x, tuple):
        return x
    return (x, x)


# ── Element-Wise Affine Layers (BatchNorm replacement) ──────────────


class ElementWiseAffine1d(nn.Module):
    """Element-wise affine transform, replaces ``BatchNorm1d`` at inference.

    Forward: ``y = scale * x + bias``

    This is mathematically identical to ``BatchNorm1d`` in ``eval()``
    mode but faster: no running mean/variance tracking, no normalization.

    Args:
        num_features: Number of input/output features.
        scale: Per-channel scale, shape ``(num_features,)``.
            Computed as ``γ / √(σ² + ε)``.
        bias: Per-channel bias, shape ``(num_features,)``.
            Computed as ``β - γ * μ / √(σ² + ε)``.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        num_features: int,
        scale: torch.Tensor,
        bias: torch.Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._num_features = num_features
        self._scale = nn.Parameter(
            scale.to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self._bias = nn.Parameter(
            bias.to(device=device, dtype=dtype),
            requires_grad=False,
        )

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def bias(self) -> torch.Tensor:
        return self._bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise affine transform.

        Args:
            x: Input tensor, shape ``(batch, num_features)`` or
                ``(batch, num_features, *)``.

        Returns:
            Same shape as input.
        """
        return self._scale * x + self._bias

    def extra_repr(self) -> str:
        return f"num_features={self._num_features}"


class ElementWiseAffine2d(nn.Module):
    """Element-wise affine transform, replaces ``BatchNorm2d`` at inference.

    Forward: ``y = scale[:, None, None] * x + bias[:, None, None]``

    Args:
        num_channels: Number of channels.
        scale: Per-channel scale, shape ``(num_channels,)``.
            Computed as ``γ / √(σ² + ε)``.
        bias: Per-channel bias, shape ``(num_channels,)``.
            Computed as ``β - γ * μ / √(σ² + ε)``.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        num_channels: int,
        scale: torch.Tensor,
        bias: torch.Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._num_channels = num_channels
        self._scale = nn.Parameter(
            scale.to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self._bias = nn.Parameter(
            bias.to(device=device, dtype=dtype),
            requires_grad=False,
        )

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def bias(self) -> torch.Tensor:
        return self._bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise affine transform.

        Args:
            x: Input tensor, shape ``(batch, num_channels, H, W)``.

        Returns:
            Same shape as input.
        """
        return self._scale.view(1, -1, 1, 1) * x + self._bias.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"num_channels={self._num_channels}"


# ── Fused Ternary + BN Layers (for Linear/BN or Conv/BN w/o activation) ──


class FusedTernaryLinear(nn.Module):
    """Ternary linear layer with BatchNorm1d parameters fused in.

    Stores frozen ``latent_scores`` (ternary via ``sign()``) plus the
    per-channel ``scale`` and ``bias`` derived from BN parameters.

    Forward pass::

        z = scale * (sign(latent_scores) @ x) + bias

    The ternary MatMul ``sign(latent_scores) @ x`` is identical to the
    original ``TernarySTELinear``; the element-wise ``scale * y + bias``
    replaces the BatchNorm1d normalization step.

    .. note::
        Use this when the model has ``TernarySTELinear → BN`` with no
        intervening activation. For the common ``TernarySTELinear →
        ReLU → BN`` pattern, use ``ElementWiseAffine1d`` instead.

    Args:
        latent_scores: Frozen ``nn.Parameter`` copied from the original
            ``TernarySTELinear.latent_scores``.
        scale: Per-channel scale, shape ``(out_features,)``.
            Computed as ``γ / √(σ² + ε)``.
        bias: Per-channel bias, shape ``(out_features,)``.
            Computed as ``β - γ * μ / √(σ² + ε)``.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        latent_scores: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._in_features = latent_scores.shape[1]
        self._out_features = latent_scores.shape[0]

        self.latent_scores = nn.Parameter(
            latent_scores.to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self._scale = nn.Parameter(
            scale.to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self._bias = nn.Parameter(
            bias.to(device=device, dtype=dtype),
            requires_grad=False,
        )

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def bias(self) -> torch.Tensor:
        return self._bias

    def ternary_weight(self) -> torch.Tensor:
        return self.latent_scores.sign().to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_tern = ste_sign(self.latent_scores)
        y = F.linear(x, w_tern)
        return self._scale * y + self._bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"fused_bn=True"
        )


class FusedTernaryConv2d(nn.Module):
    """Ternary 2D convolution with BatchNorm2d parameters fused in.

    Stores frozen ``latent_scores`` plus per-channel ``scale`` and
    ``bias`` derived from BN parameters.

    Forward pass::

        z = scale * (sign(latent_scores) ⋆ x) + bias

    Args:
        latent_scores: Frozen ``nn.Parameter`` copied from the original
            ``TernarySTEConv2d.latent_scores``.
        scale: Per-channel scale, shape ``(out_channels,)``.
        bias: Per-channel bias, shape ``(out_channels,)``.
        stride: Stride of the convolution.
        padding: Padding of the convolution.
        dilation: Dilation of the convolution.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        latent_scores: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._in_channels = latent_scores.shape[1]
        self._out_channels = latent_scores.shape[0]
        self._kernel_size = _pair(latent_scores.shape[2])
        self._stride = _pair(stride)
        self._padding = _pair(padding)
        self._dilation = _pair(dilation)

        self.latent_scores = nn.Parameter(
            latent_scores.to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self._scale = nn.Parameter(
            scale.to(device=device, dtype=dtype),
            requires_grad=False,
        )
        self._bias = nn.Parameter(
            bias.to(device=device, dtype=dtype),
            requires_grad=False,
        )

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def bias(self) -> torch.Tensor:
        return self._bias

    def ternary_weight(self) -> torch.Tensor:
        return self.latent_scores.sign().to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_tern = ste_sign(self.latent_scores)
        y = F.conv2d(
            x, w_tern,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )
        return self._scale.view(1, -1, 1, 1) * y + self._bias.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return (
            f"{self._in_channels} -> {self._out_channels}, "
            f"kernel={self._kernel_size}, stride={self._stride}, "
            f"padding={self._padding}, dilation={self._dilation}, "
            f"fused_bn=True"
        )
