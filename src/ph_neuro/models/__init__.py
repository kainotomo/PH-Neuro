"""Pre-built model architectures for ternary Hebbian and STE networks."""

from ph_neuro.models.cnn import HebbianCNN
from ph_neuro.models.mlp import HebbianMLP
from ph_neuro.models.ste_models import hyst_ste_cnn, hyst_ste_mlp, ste_cnn, ste_mlp
from ph_neuro.models.fuse_bn import fuse_bn_layers
from ph_neuro.models.dqt_transformer import (
    DQTTransformer,
    dqt_gpt2,
    SMOKE_CONFIG,
    FULL_CONFIG,
    count_ternary_weights,
    count_parameters,
    count_float_parameters,
    model_summary,
)

__all__ = [
    "HebbianMLP",
    "HebbianCNN",
    "ste_mlp",
    "ste_cnn",
    "hyst_ste_mlp",
    "hyst_ste_cnn",
    "fuse_bn_layers",
    "DQTTransformer",
    "dqt_gpt2",
    "SMOKE_CONFIG",
    "FULL_CONFIG",
    "count_ternary_weights",
    "count_parameters",
    "count_float_parameters",
    "model_summary",
]
