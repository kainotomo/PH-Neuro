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
from ph_neuro.layers.ste_dqt import TernaryDQTLinear, stochastic_round
from ph_neuro.layers.ste_dqt_transformer import (
    TernaryDQTRMSNorm,
    TernaryDQTLinear3D,
    TernaryDQTMultiheadAttention,
    TernaryDQTFeedForward,
    TernaryDQTTransformerBlock,
)
from ph_neuro.layers.ste_dqt_hysteresis import (
    TernaryDQTHysteresisLinear,
    hysteresis_stochastic_round,
)
from ph_neuro.layers.ste_lora import (
    TernarySTELoRALinear,
    count_lora_parameters,
    freeze_backbone,
    get_model_lora_state,
    iter_lora_layers,
    load_model_lora_state,
    reset_lora,
)
from ph_neuro.layers.ste_hysteresis import (
    HysteresisSTEConv2d,
    HysteresisSTELinear,
    ste_sign_hysteresis,
)
from ph_neuro.layers.fused_bn import (
    ElementWiseAffine1d,
    ElementWiseAffine2d,
    FusedTernaryConv2d,
    FusedTernaryLinear,
)

__all__ = [
    "TernaryHebbianLinear",
    "TernaryHebbianConv2d",
    "TernaryHebbianEmbedding",
    "TernaryHebbianAttention",
    "TernarySTELinear",
    "TernarySTEConv2d",
    "TernarySTELoRALinear",
    "iter_lora_layers",
    "freeze_backbone",
    "get_model_lora_state",
    "load_model_lora_state",
    "reset_lora",
    "count_lora_parameters",
    "ste_sign",
    "ste_sign_hysteresis",
    "HysteresisSTELinear",
    "HysteresisSTEConv2d",
    "TernaryDQTLinear",
    "stochastic_round",
    "TernaryDQTRMSNorm",
    "TernaryDQTLinear3D",
    "TernaryDQTMultiheadAttention",
    "TernaryDQTFeedForward",
    "TernaryDQTTransformerBlock",
    "TernaryDQTHysteresisLinear",
    "hysteresis_stochastic_round",
    "ElementWiseAffine1d",
    "ElementWiseAffine2d",
    "FusedTernaryLinear",
    "FusedTernaryConv2d",
]
