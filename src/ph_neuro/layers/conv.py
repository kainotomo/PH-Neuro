"""Ternary Hebbian convolutional layer placeholder.

Convolutional variant of :class:`~ph_neuro.layers.linear.TernaryHebbianLinear`.
Will be implemented in Phase 1 (Vision POC).
"""

from __future__ import annotations

import torch.nn as nn


class TernaryHebbianConv2d(nn.Module):
    """Ternary Hebbian 2D convolution layer.

    Note: This is a placeholder for Phase 1. The convolutional Hebbian
    update operates on local patches: each filter weight connects a local
    patch of the input to one output neuron.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        theta_upper: Hysteresis upper threshold.
        theta_lower: Hysteresis lower threshold.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        theta_upper: float = 5.0,
        theta_lower: float = 1.0,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._theta_upper = theta_upper
        self._theta_lower = theta_lower

    def forward(self, x):
        """Forward pass (placeholder)."""
        raise NotImplementedError(
            "TernaryHebbianConv2d is not yet implemented. It will be available in Phase 1."
        )

    def hebbian_update(self, pre_patches, post_activation, lr):
        """Apply Hebbian update (placeholder)."""
        raise NotImplementedError("TernaryHebbianConv2d.hebbian_update is not yet implemented.")

    def refresh_weights(self):
        """Refresh ternary weights from latent scores (placeholder)."""
        raise NotImplementedError("TernaryHebbianConv2d.refresh_weights is not yet implemented.")
