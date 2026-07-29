"""Training infrastructure for Hebbian networks."""

from ph_neuro.training.data import get_cifar10_loaders, get_mnist_loaders
from ph_neuro.training.greedy import LayerConfig, MultiLayerHebbianClassifier
from ph_neuro.training.neuromodulated import NeuromodulatedHebbianClassifier
from ph_neuro.training.supervised import SupervisedHebbianClassifier
from ph_neuro.training.trainer import HebbianTrainer

__all__ = [
    "HebbianTrainer",
    "SupervisedHebbianClassifier",
    "MultiLayerHebbianClassifier",
    "NeuromodulatedHebbianClassifier",
    "LayerConfig",
    "get_mnist_loaders",
    "get_cifar10_loaders",
]
