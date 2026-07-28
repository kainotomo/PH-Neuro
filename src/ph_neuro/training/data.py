"""Data loading utilities for Hebbian experiments.

Provides standardized data loaders for common datasets used in
PH-Neuro experiments (MNIST, CIFAR-10, etc.).
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


def get_mnist_loaders(
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders.

    Args:
        batch_size: Batch size for both loaders.
        root: Root directory for dataset storage.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of ``(train_loader, test_loader)``.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def get_cifar10_loaders(
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 train and test data loaders.

    Args:
        batch_size: Batch size for both loaders.
        root: Root directory for dataset storage.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of ``(train_loader, test_loader)``.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
