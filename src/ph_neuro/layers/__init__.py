"""Layer implementations for ternary Hebbian and STE networks.

Provides PyTorch ``nn.Module`` subclasses that use ternary weights
with either Hebbian plasticity (no backprop) or STE backpropagation.
"""

from ph_neuro.layers.attention import TernaryHebbianAttention
from ph_neuro.layers.conv import TernaryHebbianConv2d
from ph_neuro.layers.embedding import TernaryHebbianEmbedding
from ph_neuro.layers.linear import TernaryHebbianLinear
from ph_neuro.layers.ste_linear import TernarySTELinear, ste_sign
from ph_neuro.layers.ste_conv import TernarySTEConv2d
from ph_neuro.layers.ste_hysteresis import (
    HysteresisSTEConv2d,
    HysteresisSTELinear,
    ste_sign_hysteresis,
)

__all__ = [
    "TernaryHebbianLinear",
    "TernaryHebbianConv2d",
    "TernaryHebbianEmbedding",
    "TernaryHebbianAttention",
    "TernarySTELinear",
    "TernarySTEConv2d",
    "ste_sign",
    "ste_sign_hysteresis",
    "HysteresisSTELinear",
    "HysteresisSTEConv2d",
]
