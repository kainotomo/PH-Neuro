"""Data loading utilities for Hebbian experiments.

Provides standardized data loaders for common datasets used in
PH-Neuro experiments (MNIST, CIFAR-10, etc.).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


# ── MNIST dataset helpers ──────────────────────────────────────────


def _get_mnist_transform() -> transforms.Compose:
    """Standard MNIST normalization transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


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
    transform = _get_mnist_transform()

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


# ── Split MNIST ────────────────────────────────────────────────────


def get_binary_mnist_loaders(
    class_a: int,
    class_b: int,
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST loaders for a binary classification task.

    Filters MNIST to only two digit classes and remaps labels to 0/1.

    Args:
        class_a: First digit class (remapped to label 0).
        class_b: Second digit class (remapped to label 1).
        batch_size: Batch size for both loaders.
        root: Root directory for dataset storage.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of ``(train_loader, test_loader)``, each yielding
        ``(images, labels)`` where labels are 0 or 1.
    """
    transform = _get_mnist_transform()

    # Build a TensorDataset from filtered data to avoid Subset index issues
    def _build_binary_dataset(dataset, a, b):
        data_list = []
        target_list = []
        for img, label in dataset:
            if label in (a, b):
                data_list.append(img.unsqueeze(0))
                target_list.append(0 if label == a else 1)
        data = torch.cat(data_list, dim=0)
        targets = torch.tensor(target_list, dtype=torch.long)
        return torch.utils.data.TensorDataset(data, targets)

    train_full = MNIST(root=root, train=True, download=True, transform=transform)
    test_full = MNIST(root=root, train=False, download=True, transform=transform)

    train_dataset = _build_binary_dataset(train_full, class_a, class_b)
    test_dataset = _build_binary_dataset(test_full, class_a, class_b)

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


def get_mnist_full_test_loader(
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
) -> DataLoader:
    """Get the full 10-class MNIST test loader.

    Useful for evaluating global accuracy after split MNIST training.

    Returns:
        DataLoader over the MNIST test set (all 10 classes).
    """
    transform = _get_mnist_transform()
    test_dataset = MNIST(root=root, train=False, download=True, transform=transform)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# ── Permuted MNIST ─────────────────────────────────────────────────


def _make_permutation(seed: int, n_pixels: int = 784) -> torch.Tensor:
    """Create a fixed random pixel permutation."""
    rng = torch.Generator().manual_seed(seed)
    return torch.randperm(n_pixels, generator=rng)


def _apply_permutation(images: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply a pixel permutation to batched flat images."""
    return images[:, perm]


class PermutedMNIST(MNIST):
    """MNIST with a fixed random pixel permutation applied.

    The permutation is applied deterministically based on a seed,
    making it reproducible across runs.
    """

    def __init__(
        self,
        perm_seed: int,
        root: str = "./data",
        train: bool = True,
        transform=None,
        **kwargs,
    ):
        super().__init__(root=root, train=train, transform=transform, **kwargs)
        self._perm = _make_permutation(perm_seed)

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        # img is (1, 28, 28) after ToTensor; flatten, permute, reshape
        img_flat = img.view(-1)
        img_permuted = _apply_permutation(img_flat.unsqueeze(0), self._perm).squeeze(0)
        img_permuted = img_permuted.view(1, 28, 28)
        return img_permuted, target


def get_permuted_mnist_loaders(
    perm_seed: int,
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST loaders with a fixed pixel permutation.

    Each image's pixels are rearranged according to a fixed random
    permutation determined by ``perm_seed``. The permutation is
    consistent across train and test splits.

    Args:
        perm_seed: Seed for the random pixel permutation.
        batch_size: Batch size for both loaders.
        root: Root directory for dataset storage.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of ``(train_loader, test_loader)``.
    """
    transform = _get_mnist_transform()

    train_dataset = PermutedMNIST(
        perm_seed=perm_seed, root=root, train=True, transform=transform, download=True
    )
    test_dataset = PermutedMNIST(
        perm_seed=perm_seed, root=root, train=False, transform=transform, download=True
    )

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


# ── CIFAR-10 ───────────────────────────────────────────────────────


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
