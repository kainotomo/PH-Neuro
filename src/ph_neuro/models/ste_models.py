"""STE-based model definitions for ternary vision experiments.

Provides factory functions for MLP and CNN models that use
TernarySTELinear and TernarySTEConv2d layers with standard
ReLU activations and optional BatchNorm.

These models are used by the L1 baseline suite experiment.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from ph_neuro.layers.ste_conv import TernarySTEConv2d
from ph_neuro.layers.ste_linear import TernarySTELinear


def ste_mlp(
    layer_sizes: Sequence[int],
    batch_norm: bool = True,
    flatten: bool = True,
    device: torch.device | str | None = None,
) -> nn.Sequential:
    """Build an MLP with ternary STE layers and ReLU activations.

    Args:
        layer_sizes: Sequence of layer sizes, e.g. ``[784, 512, 256, 10]``.
        batch_norm: Whether to insert ``BatchNorm1d`` after each hidden layer.
        flatten: If ``True``, prepend ``nn.Flatten()`` for image inputs.
        device: Torch device.

    Returns:
        ``nn.Sequential`` with ``(Flatten?) → TernarySTELinear → ReLU → ...``.
    """
    layers: list[nn.Module] = []
    sizes = list(layer_sizes)

    if flatten:
        layers.append(nn.Flatten())

    for i in range(len(sizes) - 1):
        layers.append(TernarySTELinear(sizes[i], sizes[i + 1], bias=not batch_norm))
        if i < len(sizes) - 2:
            # Hidden layers: ReLU + optional BatchNorm
            layers.append(nn.ReLU(inplace=True))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
        # Last layer: no activation (raw logits for CrossEntropyLoss)

    model = nn.Sequential(*layers)
    if device is not None:
        model = model.to(device)
    return model


def ste_cnn(
    in_channels: int = 3,
    img_size: int = 32,
    hidden_channels: int = 64,
    n_classes: int = 10,
    device: torch.device | str | None = None,
) -> nn.Sequential:
    """Build a small CNN with ternary STE layers for CIFAR-style images.

    Architecture:
        ``Conv → ReLU → BN → MaxPool → Conv → ReLU → BN → MaxPool →
         Flatten → Linear → ReLU → BN → Linear``

    Args:
        in_channels: Input image channels (default 3 for CIFAR).
        img_size: Input image size (assumed square, default 32).
        hidden_channels: Number of channels for the first conv layer.
            The second conv gets ``2 * hidden_channels``.
        n_classes: Number of output classes.
        device: Torch device.

    Returns:
        ``nn.Sequential`` with ternary STE conv and linear layers.
    """
    # After two MaxPool2d(2), spatial size is img_size // 4
    flat_features = (2 * hidden_channels) * (img_size // 4) * (img_size // 4)

    layers: list[nn.Module] = [
        # Conv block 1
        TernarySTEConv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(hidden_channels),
        nn.MaxPool2d(2),
        # Conv block 2
        TernarySTEConv2d(hidden_channels, 2 * hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(2 * hidden_channels),
        nn.MaxPool2d(2),
        # Flatten
        nn.Flatten(),
        # Linear block
        TernarySTELinear(flat_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        # Output layer (no activation — raw logits)
        TernarySTELinear(512, n_classes),
    ]

    model = nn.Sequential(*layers)
    if device is not None:
        model = model.to(device)
    return model
