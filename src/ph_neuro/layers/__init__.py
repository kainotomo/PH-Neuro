"""Layer implementations for ternary Hebbian networks.

Provides PyTorch ``nn.Module`` subclasses that use ternary weights
and Hebbian updates instead of gradient-based learning.
"""

from ph_neuro.layers.attention import TernaryHebbianAttention
from ph_neuro.layers.conv import TernaryHebbianConv2d
from ph_neuro.layers.embedding import TernaryHebbianEmbedding
from ph_neuro.layers.linear import TernaryHebbianLinear

__all__ = [
    "TernaryHebbianLinear",
    "TernaryHebbianConv2d",
    "TernaryHebbianEmbedding",
    "TernaryHebbianAttention",
]
