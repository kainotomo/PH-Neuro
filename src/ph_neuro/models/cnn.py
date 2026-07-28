"""Pre-built Hebbian CNN model.

A convolutional neural network built from
:class:`~ph_neuro.layers.conv.TernaryHebbianConv2d` and
:class:`~ph_neuro.layers.linear.TernaryHebbianLinear` layers, with
greedy layer-wise training support.

Architecture (default):
    ``Conv(3→64, 3×3) → sign → MaxPool2d(2) → Conv(64→128, 3×3) → sign → MaxPool2d(2)
    → Flatten → Linear(128×8×8→10)``

This is ~300K ternary weights — fits trivially on RTX 4060.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.conv import TernaryHebbianConv2d
from ph_neuro.layers.linear import TernaryHebbianLinear


class HebbianCNN(nn.Module):
    """Ternary Hebbian CNN for vision tasks.

    Args:
        in_channels: Input image channels (default 3 for CIFAR-10).
        img_size: Input image size (assumed square, default 32).
        hidden_channels: Number of channels for the first conv layer
            (default 64). The second conv gets ``2× hidden_channels``.
        n_classes: Number of output classes (default 10 for CIFAR-10).
        theta_upper: Hysteresis upper threshold for conv layers.
        theta_lower: Hysteresis lower threshold for conv layers.
        output_theta_upper: Hysteresis upper threshold for output layer
            (lower, since WTA needs faster flipping).
        output_theta_lower: Hysteresis lower threshold for output layer.
        device: Torch device.

    Attributes:
        conv1: First ``TernaryHebbianConv2d`` (e.g. ``3→64``).
        conv2: Second ``TernaryHebbianConv2d`` (e.g. ``64→128``).
        pool: ``MaxPool2d(2)``.
        output: Output ``TernaryHebbianLinear`` (e.g. ``8192→10``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 32,
        hidden_channels: int = 64,
        n_classes: int = 10,
        theta_upper: float = 7.0,
        theta_lower: float = 1.5,
        output_theta_upper: float = 1.0,
        output_theta_lower: float = 0.3,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._img_size = img_size
        self._hidden_channels = hidden_channels
        self._n_classes = n_classes

        # After two MaxPool2d(2) layers, spatial size is img_size // 4
        self._flat_features = (2 * hidden_channels) * (img_size // 4) * (img_size // 4)

        self.conv1 = TernaryHebbianConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            theta_upper=theta_upper,
            theta_lower=theta_lower,
            device=device,
        )

        self.conv2 = TernaryHebbianConv2d(
            in_channels=hidden_channels,
            out_channels=2 * hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            theta_upper=theta_upper,
            theta_lower=theta_lower,
            device=device,
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.output = TernaryHebbianLinear(
            in_features=self._flat_features,
            out_features=n_classes,
            theta_upper=output_theta_upper,
            theta_lower=output_theta_lower,
            device=device,
        )

        # Keep layers in a ModuleList for easy iteration (used by greedy training)
        self._layer_list = nn.ModuleList([self.conv1, self.conv2, self.output])

    @property
    def layer_list(self) -> nn.ModuleList:
        """All trainable layers (conv1, conv2, output) as a ModuleList."""
        return self._layer_list

    def forward(self, x: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """Forward pass: conv1 → sign → pool → conv2 → sign → pool → flatten → linear.

        Args:
            x: Input tensor, shape ``(N, C, H, W)``.
            epsilon: Dead-zone for ``ternary_sign`` between layers.

        Returns:
            Raw logits, shape ``(N, n_classes)``.
        """
        # Conv block 1
        h = self.conv1(x)
        h = ternary_sign(h, epsilon=epsilon).float()
        h = self.pool(h)

        # Conv block 2
        h = self.conv2(h)
        h = ternary_sign(h, epsilon=epsilon).float()
        h = self.pool(h)

        # Flatten → Linear output
        h = h.reshape(h.shape[0], -1)
        out = self.output(h)
        return out

    def forward_through(
        self,
        x: torch.Tensor,
        stop_at: int,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """Forward pass through layers up to (but not including) ``stop_at``.

        Useful for greedy training: pass input through frozen encoder layers
        up to the layer currently being trained.

        Args:
            x: Input tensor, shape ``(N, C, H, W)``.
            stop_at: Stop index (0 = no layers, 1 = after conv1, etc.).
            epsilon: Dead-zone for ``ternary_sign``.

        Returns:
            Activations at layer ``stop_at - 1`` (raw float for the next
            layer's input).
        """
        h = x.clone()
        if stop_at >= 1:
            h = self.conv1(h)
            h = ternary_sign(h, epsilon=epsilon).float()
            h = self.pool(h)
        if stop_at >= 2:
            h = self.conv2(h)
            h = ternary_sign(h, epsilon=epsilon).float()
            h = self.pool(h)
        if stop_at >= 3:
            h = h.reshape(h.shape[0], -1)
            h = self.output(h)
        return h
