"""DQT-based model definitions for ternary vision experiments.

Provides factory functions for CNNs that use :class:`TernaryDQTConv2d`
(convolutions) and :class:`TernaryDQTLinear` (linear layers) with standard
ReLU activations and optional BatchNorm. These are the DQT counterparts of
the STE factories in :mod:`ph_neuro.models.ste_models`.

The architecture mirrors ``ste_cnn()`` so results are directly comparable;
the only difference is the weight-bearing layers are DQT (int8 ternary
weights + stochastic rounding, no latent float scores).
"""

from __future__ import annotations

import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d


def dqt_cnn(
    in_channels: int = 3,
    img_size: int = 32,
    hidden_channels: int = 64,
    n_classes: int = 10,
    device: torch.device | str | None = None,
) -> nn.Sequential:
    """Build a small CNN with ternary DQT layers for CIFAR-style images.

    Architecture (mirrors :func:`ste_cnn` for direct comparison, except the
    FC head is 8192→256→10 instead of 8192→512→10 — M1.1-RETRY reduces
    classifier flip noise while the conv feature path stays identical):
        ``TernaryDQTConv2d → ReLU → BN → MaxPool →
         TernaryDQTConv2d → ReLU → BN → MaxPool →
         Flatten → TernaryDQTLinear → ReLU → BN → TernaryDQTLinear``

    Conv layers use no bias (BatchNorm handles the per-channel shift);
    MaxPool and activations remain float (not quantized in this milestone).

    Args:
        in_channels: Input image channels (default 3 for CIFAR).
        img_size: Input image size (assumed square, default 32).
        hidden_channels: Number of channels for the first conv layer.
            The second conv gets ``2 * hidden_channels``.
        n_classes: Number of output classes.
        device: Torch device.

    Returns:
        ``nn.Sequential`` with ternary DQT conv and linear layers.
    """
    # After two MaxPool2d(2), spatial size is img_size // 4.
    # flat_features is the fan-IN of the first linear layer (8192 for 32x32):
    # (2 * hidden_channels) * (img_size // 4) * (img_size // 4)
    flat_features = (2 * hidden_channels) * (img_size // 4) * (img_size // 4)

    # Smaller FC head than ste_cnn() (8192->512) — M1.1-RETRY: halves the
    # classifier flip noise while leaving the conv feature path untouched.
    head_hidden = 256

    layers: list[nn.Module] = [
        # Conv block 1
        TernaryDQTConv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(hidden_channels),
        nn.MaxPool2d(2),
        # Conv block 2
        TernaryDQTConv2d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(2 * hidden_channels),
        nn.MaxPool2d(2),
        # Flatten
        nn.Flatten(),
        # Linear block
        TernaryDQTLinear(flat_features, head_hidden),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(head_hidden),
        # Output layer (no activation — raw logits)
        TernaryDQTLinear(head_hidden, n_classes),
    ]

    model = nn.Sequential(*layers)
    if device is not None:
        model = model.to(device)
    return model
