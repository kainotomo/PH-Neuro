"""Core tensor representations for ternary weights and latent scores.

This module provides the fundamental data structures:
- :class:`TernaryTensor`: Storage for {-1, 0, +1} weights
- :class:`LatentScoreTensor`: Float scores paired with each ternary weight
- :func:`ternary_sign`: Activation function mapping any tensor to {-1, 0, +1}
"""

from ph_neuro.core.activation import ternary_sign
from ph_neuro.core.hebbian_rules import (
    anti_hebbian_update,
    bcm_update,
    hebbian_update,
    oja_update,
)
from ph_neuro.core.latent_scores import LatentScoreTensor
from ph_neuro.core.ternary_tensor import TernaryTensor

__all__ = [
    "TernaryTensor",
    "LatentScoreTensor",
    "ternary_sign",
    "hebbian_update",
    "anti_hebbian_update",
    "oja_update",
    "bcm_update",
]
