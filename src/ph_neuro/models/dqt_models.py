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

    Architecture (mirrors :func:`ste_cnn` for direct comparison):
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

    # FC head matches ste_cnn(): 8192->512->10 (M1.1-RETRY-3 restores the
    # 512-head from E020, keeping anneal@80% / patience=25 from E021.2).
    head_hidden = 512

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


def dqt_cnn_cifar100(
    in_channels: int = 3,
    img_size: int = 32,
    n_classes: int = 100,
    device: torch.device | str | None = None,
) -> nn.Sequential:
    """Build a larger 3-conv ternary DQT CNN for CIFAR-100 (M1.2).

    Architecture (larger than the 2-conv ``dqt_cnn()`` from M1.1 — CIFAR-100
    needs a bigger capacity model to exceed the 38.2% STE baseline):
        ``TernaryDQTConv2d(3→64) → ReLU → BN → MaxPool(2)   # 32→16
         TernaryDQTConv2d(64→128) → ReLU → BN → MaxPool(2)  # 16→8
         TernaryDQTConv2d(128→256) → ReLU → BN → MaxPool(2) # 8→4
         Flatten                                            # 256*4*4 = 4096
         TernaryDQTLinear(4096→512) → ReLU → BN
         TernaryDQTLinear(512→n_classes)``

    Conv layers use no bias (BatchNorm handles the per-channel shift);
    MaxPool and activations remain float (not quantized in this milestone).

    Args:
        in_channels: Input image channels (default 3 for CIFAR).
        img_size: Input image size (assumed square, default 32). ``flat_features``
            is derived dynamically from ``img_size`` after three ``MaxPool2d(2)``.
        n_classes: Number of output classes (default 100 for CIFAR-100).
        device: Torch device.

    Returns:
        ``nn.Sequential`` with ternary DQT conv and linear layers.
    """
    # After three MaxPool2d(2), spatial size is img_size // 8.
    # flat_features is the fan-IN of the first linear layer (4096 for 32x32):
    # (4 * 64) * (img_size // 8) * (img_size // 8)
    flat_features = (4 * 64) * (img_size // 8) * (img_size // 8)

    # FC head matches the brief: 4096 -> 512 -> n_classes
    head_hidden = 512

    layers: list[nn.Module] = [
        # Conv block 1
        TernaryDQTConv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        # Conv block 2
        TernaryDQTConv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        # Conv block 3
        TernaryDQTConv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
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
