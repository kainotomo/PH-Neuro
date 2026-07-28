"""Pre-built model architectures for ternary Hebbian networks."""

from ph_neuro.models.cnn import HebbianCNN
from ph_neuro.models.mlp import HebbianMLP

__all__ = [
    "HebbianMLP",
    "HebbianCNN",
]
