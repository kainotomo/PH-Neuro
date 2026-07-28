"""Tests for HebbianTrainer."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.layers.linear import TernaryHebbianLinear
from ph_neuro.models.mlp import HebbianMLP
from ph_neuro.training.trainer import HebbianTrainer


class TestHebbianTrainer:
    """Suite of tests for the Hebbian trainer."""

    @pytest.fixture
    def model(self):
        """A small 2-layer Hebbian MLP."""
        return HebbianMLP([4, 8, 2])

    @pytest.fixture
    def dataset(self):
        """A tiny synthetic dataset."""
        x = torch.randn(64, 4)
        y = torch.randint(0, 2, (64,))
        return TensorDataset(x, y)

    @pytest.fixture
    def loader(self, dataset):
        """A tiny data loader."""
        return DataLoader(dataset, batch_size=16)

    def test_create(self, model):
        """Creating the trainer should find all Hebbian layers."""
        trainer = HebbianTrainer(model, lr=0.01)
        assert len(trainer._hebbian_layers) == 2

    def test_fit_no_crash(self, model, loader):
        """Training for 1 epoch should not crash."""
        trainer = HebbianTrainer(model, lr=0.01, decay=1e-5)
        history = trainer.fit(loader, epochs=1)
        assert "loss" in history

    def test_evaluate(self, model, loader):
        """Evaluate should return a float accuracy."""
        trainer = HebbianTrainer(model, lr=0.01)
        acc = trainer.evaluate(loader)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_no_backward(self, model, loader):
        """Training should not have any .backward() calls."""
        trainer = HebbianTrainer(model, lr=0.01)
        # Monkey-patch backward to detect calls
        called = [False]

        original_backward = torch.Tensor.backward

        def tracking_backward(self, *args, **kwargs):
            called[0] = True
            return original_backward(self, *args, **kwargs)

        torch.Tensor.backward = tracking_backward  # type: ignore
        try:
            trainer.fit(loader, epochs=1)
            assert not called[0], ".backward() should never be called"
        finally:
            torch.Tensor.backward = original_backward  # type: ignore
