"""Training infrastructure for Hebbian networks."""

from ph_neuro.training.data import get_cifar10_loaders, get_mnist_loaders
from ph_neuro.training.tinystories import (
    get_tinystories_data,
    make_gpt2_tokenizer,
    make_synthetic_lm_loader,
    make_synthetic_stories,
    make_synthetic_token_sequences,
    pack_sequences,
    tokenize_texts,
)
from ph_neuro.training.ep import EPConfig, EquilibriumPropagationClassifier
from ph_neuro.training.greedy import LayerConfig, MultiLayerHebbianClassifier
from ph_neuro.training.neuromodulated import NeuromodulatedHebbianClassifier
from ph_neuro.training.supervised import SupervisedHebbianClassifier
from ph_neuro.training.trainer import HebbianTrainer

__all__ = [
    "HebbianTrainer",
    "SupervisedHebbianClassifier",
    "MultiLayerHebbianClassifier",
    "NeuromodulatedHebbianClassifier",
    "EquilibriumPropagationClassifier",
    "EPConfig",
    "LayerConfig",
    "get_mnist_loaders",
    "get_cifar10_loaders",
    "get_tinystories_data",
    "make_gpt2_tokenizer",
    "pack_sequences",
    "tokenize_texts",
    "make_synthetic_token_sequences",
    "make_synthetic_lm_loader",
    "make_synthetic_stories",
]
