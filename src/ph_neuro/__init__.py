"""PH-Neuro: Ternary Hebbian deep learning — no backpropagation."""

__version__ = "0.1.0.dev0"

from ph_neuro.core.activation import ternary_sign
from ph_neuro.core.hebbian_rules import (
    anti_hebbian_update,
    bcm_update,
    hebbian_update,
    oja_update,
)
from ph_neuro.core.latent_scores import LatentScoreTensor
from ph_neuro.core.ternary_tensor import TernaryTensor
from ph_neuro.layers.attention import TernaryHebbianAttention
from ph_neuro.layers.conv import TernaryHebbianConv2d
from ph_neuro.layers.embedding import TernaryHebbianEmbedding
from ph_neuro.layers.linear import TernaryHebbianLinear
from ph_neuro.models.cnn import HebbianCNN
from ph_neuro.models.mlp import HebbianMLP
from ph_neuro.training.data import get_cifar10_loaders, get_mnist_loaders
from ph_neuro.training.trainer import HebbianTrainer
from ph_neuro.utils.packing import pack_ternary, unpack_ternary
from ph_neuro.utils.popcount import popcount_matmul

__all__ = [
    # Version
    "__version__",
    # Core
    "TernaryTensor",
    "LatentScoreTensor",
    "ternary_sign",
    "hebbian_update",
    "anti_hebbian_update",
    "oja_update",
    "bcm_update",
    # Layers
    "TernaryHebbianLinear",
    "TernaryHebbianConv2d",
    "TernaryHebbianEmbedding",
    "TernaryHebbianAttention",
    # Models
    "HebbianMLP",
    "HebbianCNN",
    # Training
    "HebbianTrainer",
    "get_mnist_loaders",
    "get_cifar10_loaders",
    # Utils
    "pack_ternary",
    "unpack_ternary",
    "popcount_matmul",
]
