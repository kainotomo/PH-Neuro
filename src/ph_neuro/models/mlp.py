"""Pre-built Hebbian MLP model.

A multi-layer perceptron built from :class:`~ph_neuro.layers.linear.TernaryHebbianLinear`
layers, with greedy layer-wise training support.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

from ph_neuro.layers.linear import TernaryHebbianLinear, ternary_sign


class HebbianMLP(nn.Module):
    """Multi-layer Hebbian MLP with ternary weights.

    Args:
        layer_sizes: Sequence of layer sizes, e.g. ``[784, 256, 128, 10]``.
        theta_upper: Hysteresis upper threshold (applied to all layers).
        theta_lower: Hysteresis lower threshold (applied to all layers).
        hebbian_rule: Hebbian rule for all layers (``'basic'``, ``'anti'``, etc.).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        theta_upper: float = 5.0,
        theta_lower: float = 1.0,
        hebbian_rule: str = "basic",
    ):
        super().__init__()
        self._layer_sizes = list(layer_sizes)
        layers: list[nn.Module] = []

        for i in range(len(layer_sizes) - 1):
            layers.append(
                TernaryHebbianLinear(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    theta_upper=theta_upper,
                    theta_lower=theta_lower,
                    hebbian_rule=hebbian_rule,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Forward pass through all layers with ternary activations.

        Args:
            x: Input tensor, shape ``(batch, in_features)``.

        Returns:
            Output tensor, shape ``(batch, out_features)``.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = ternary_sign(x)
        return x

    def get_layer(self, idx: int) -> TernaryHebbianLinear:
        """Get a specific layer by index."""
        return self.layers[idx]
