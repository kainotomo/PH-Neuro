"""Shared pytest fixtures for PH-Neuro tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the preferred compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_seed() -> None:
    """Fix random seed for reproducible tests."""
    torch.manual_seed(42)


@pytest.fixture
def small_ternary_tensor() -> torch.Tensor:
    """A small int8 tensor with values in {-1, 0, +1}."""
    return torch.tensor(
        [[1, 0, -1, 1], [0, -1, 1, 0], [-1, 1, 0, -1]],
        dtype=torch.int8,
    )


@pytest.fixture
def batch_ternary_input() -> torch.Tensor:
    """A batch of ternary input vectors."""
    return torch.tensor(
        [[1, 0, -1, 1], [0, 1, 0, -1], [1, -1, 0, 1], [0, 0, 1, -1]],
        dtype=torch.int8,
    )
