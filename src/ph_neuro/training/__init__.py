"""Training infrastructure for Hebbian networks."""

from ph_neuro.training.data import get_cifar10_loaders, get_mnist_loaders
from ph_neuro.training.trainer import HebbianTrainer

__all__ = [
    "HebbianTrainer",
    "get_mnist_loaders",
    "get_cifar10_loaders",
]
