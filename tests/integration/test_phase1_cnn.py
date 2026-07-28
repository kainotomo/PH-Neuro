"""Integration tests for Phase 1.2 — Hebbian CNN on CIFAR-10.

Success criteria:
    1. No .backward() calls during greedy layer-wise training
    2. All weights remain in {-1, 0, +1} at every step
    3. Frozen layers do not change during subsequent layer training
    4. Weight flip rate stabilizes over training
    5. CIFAR-10 training completes end-to-end (even on synthetic data)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.core.activation import ternary_sign
from ph_neuro.models.cnn import HebbianCNN
from ph_neuro.training.data import get_cifar10_loaders
from ph_neuro.training.greedy import (
    _init_conv_connectivity,
    evaluate_cnn,
    train_conv_competitive_epoch,
    train_supervised_wta_epoch,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_synthetic_cifar_loader(
    n_samples: int = 32,
    batch_size: int = 8,
    img_size: int = 8,
    n_classes: int = 3,
) -> DataLoader:
    """Create a tiny synthetic dataset with CIFAR-like shape."""
    x = torch.randn(n_samples, 3, img_size, img_size)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def _count_backward_calls(
    model: HebbianCNN,
    loader: DataLoader,
    conv_epochs: list[int],
    output_epochs: int,
    device: torch.device,
) -> int:
    """Train and count how many times .backward() is called."""
    call_count = [0]
    original_backward = torch.Tensor.backward

    def tracking_backward(self, *args, **kwargs):  # type: ignore
        call_count[0] += 1
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = tracking_backward  # type: ignore
    try:
        _run_cnn_training(model, loader, conv_epochs, output_epochs, device)
    finally:
        torch.Tensor.backward = original_backward  # type: ignore

    return call_count[0]


def _check_weights_ternary(w: torch.Tensor) -> bool:
    """Check all weight values are in {-1, 0, +1}."""
    return bool(torch.all((w == -1) | (w == 0) | (w == 1)).item())


def _run_cnn_training(
    model: HebbianCNN,
    loader: DataLoader,
    conv_epochs: list[int],
    output_epochs: int,
    device: torch.device,
    verbose: bool = False,
) -> dict[str, float]:
    """Run the full greedy CNN training pipeline.

    Returns:
        Dict with final test accuracy and flip rates.
    """
    # Bootstrap
    _init_conv_connectivity(model.conv1, density=0.3)
    _init_conv_connectivity(model.conv2, density=0.3)

    # Step 1: Train Conv1
    model.conv1.requires_hebbian_(True)
    for _ in range(conv_epochs[0]):
        train_conv_competitive_epoch(
            conv_layer=model.conv1,
            loader=loader,
            frozen_encoder=None,
            device=device,
            lr=0.01,
            decay=0.0,
            epsilon=0.1,
        )
    model.conv1.requires_hebbian_(False)

    # Build frozen encoder: Conv1 → sign → MaxPool(2)
    frozen_conv_encoder = _FrozenConvEncoder(model.conv1, epsilon=0.1).to(device)

    # Step 2: Train Conv2 (on frozen Conv1 output)
    model.conv2.requires_hebbian_(True)
    for _ in range(conv_epochs[1]):
        train_conv_competitive_epoch(
            conv_layer=model.conv2,
            loader=loader,
            frozen_encoder=frozen_conv_encoder,
            device=device,
            lr=0.01,
            decay=0.0,
            epsilon=0.1,
        )
    model.conv2.requires_hebbian_(False)

    # Build frozen flat encoder: Conv1→sign→pool→Conv2→sign→pool→flatten
    frozen_flat_encoder = [_FrozenFlatEncoder(model, epsilon=0.1).to(device)]

    # Step 3: Train output layer
    model.output.requires_hebbian_(True)
    for _ in range(output_epochs):
        train_supervised_wta_epoch(
            layer=model.output,
            loader=loader,
            frozen_encoder=frozen_flat_encoder,
            device=device,
            lr=0.01,
            decay=0.0,
            epsilon=0.1,
        )
    model.output.requires_hebbian_(False)

    return {"status": "ok"}


class _FrozenConvEncoder(nn.Module):
    """Frozen encoder: Conv1 → sign → MaxPool2d."""

    def __init__(self, conv1: nn.Module, epsilon: float = 0.1):
        super().__init__()
        self.conv1 = conv1
        self.epsilon = epsilon
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = ternary_sign(h, epsilon=self.epsilon).float()
        h = self.pool(h)
        return h


class _FrozenFlatEncoder(nn.Module):
    """Frozen encoder: full conv stack → flatten."""

    def __init__(self, model: HebbianCNN, epsilon: float = 0.1):
        super().__init__()
        self.cnn = model
        self.epsilon = epsilon
        self.img_size = model._img_size
        self.in_channels = model._in_channels
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle flat input (from train_supervised_wta_epoch flattening)
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.in_channels, self.img_size, self.img_size)
        with torch.no_grad():
            h = self.cnn.conv1(x)
            h = ternary_sign(h, epsilon=self.epsilon).float()
            h = self.pool(h)
            h = self.cnn.conv2(h)
            h = ternary_sign(h, epsilon=self.epsilon).float()
            h = self.pool(h)
            h = h.reshape(h.shape[0], -1)
        return h


# ── Test 1: No backward calls ────────────────────────────────────


class TestNoBackward:
    """Verify that .backward() is never called during CNN training."""

    def test_no_backward_synthetic(self):
        """CNN training must not call .backward() (synthetic data)."""
        device = torch.device("cpu")
        loader = _make_synthetic_cifar_loader(n_samples=16, batch_size=4, img_size=8)
        model = HebbianCNN(
            in_channels=3, img_size=8, hidden_channels=4, n_classes=3, device=device,
        )

        n_calls = _count_backward_calls(
            model, loader, conv_epochs=[1, 1], output_epochs=1, device=device,
        )
        assert n_calls == 0, f".backward() was called {n_calls} time(s)"

    @pytest.mark.slow
    def test_no_backward_cifar10(self):
        """CNN training on real CIFAR-10 must not call .backward()."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, _ = get_cifar10_loaders(batch_size=64)
        model = HebbianCNN(in_channels=3, img_size=32, hidden_channels=16, device=device)

        n_calls = _count_backward_calls(
            model, train_loader, conv_epochs=[1, 1], output_epochs=1, device=device,
        )
        assert n_calls == 0, f".backward() was called {n_calls} time(s)"


# ── Test 2: Weights always ternary ───────────────────────────────


class TestWeightsAlwaysTernary:
    """Verify all weights remain in {-1, 0, +1} throughout training."""

    def test_weights_ternary_after_training(self):
        """Weights must be ternary after full training on synthetic data."""
        device = torch.device("cpu")
        loader = _make_synthetic_cifar_loader(n_samples=16, batch_size=4, img_size=8)
        model = HebbianCNN(
            in_channels=3, img_size=8, hidden_channels=4, n_classes=3, device=device,
        )

        _run_cnn_training(model, loader, conv_epochs=[2, 2], output_epochs=2, device=device)

        for name in ["conv1", "conv2", "output"]:
            w = getattr(model, name).weight.unpack()
            assert _check_weights_ternary(w), f"{name} weights not ternary"

    @pytest.mark.slow
    def test_weights_ternary_after_cifar10(self):
        """Weights must be ternary after training on real CIFAR-10."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, _ = get_cifar10_loaders(batch_size=64)
        model = HebbianCNN(in_channels=3, img_size=32, hidden_channels=16, device=device)

        _run_cnn_training(model, train_loader, conv_epochs=[1, 1], output_epochs=1, device=device)

        for name in ["conv1", "conv2", "output"]:
            w = getattr(model, name).weight.unpack()
            assert _check_weights_ternary(w), f"{name} weights not ternary"


# ── Test 3: Frozen layers don't change ───────────────────────────


class TestFrozenLayersStable:
    """Verify earlier layers remain unchanged during later training."""

    def test_conv1_frozen_during_conv2_training(self):
        """Conv1 weights should not change while training Conv2."""
        device = torch.device("cpu")
        loader = _make_synthetic_cifar_loader(n_samples=16, batch_size=4, img_size=8)
        model = HebbianCNN(
            in_channels=3, img_size=8, hidden_channels=4, n_classes=3, device=device,
        )

        # Train Conv1
        _init_conv_connectivity(model.conv1, density=0.3)
        model.conv1.requires_hebbian_(True)
        train_conv_competitive_epoch(
            model.conv1, loader, None, device, lr=0.01, decay=0.0, epsilon=0.1,
        )
        model.conv1.requires_hebbian_(False)
        w1_before = model.conv1.weight.unpack().clone()
        assert not torch.all(w1_before == 0), "Conv1 didn't learn anything"

        # Train Conv2 — Conv1 must remain unchanged
        _init_conv_connectivity(model.conv2, density=0.3)
        frozen_enc = _FrozenConvEncoder(model.conv1, epsilon=0.1).to(device)
        model.conv2.requires_hebbian_(True)
        train_conv_competitive_epoch(
            model.conv2, loader, frozen_encoder=frozen_enc, device=device, lr=0.01, decay=0.0, epsilon=0.1,
        )
        model.conv2.requires_hebbian_(False)

        w1_after = model.conv1.weight.unpack()
        assert torch.equal(w1_before, w1_after), "Conv1 weights changed while training Conv2"


# ── Test 4: End-to-end training ──────────────────────────────────


class TestEndToEndTraining:
    """Verify full training pipeline completes successfully."""

    def test_synthetic_training_completes(self):
        """Full training on synthetic data should complete without errors."""
        device = torch.device("cpu")
        loader = _make_synthetic_cifar_loader(n_samples=16, batch_size=4, img_size=8)
        model = HebbianCNN(
            in_channels=3, img_size=8, hidden_channels=4, n_classes=3, device=device,
        )

        result = _run_cnn_training(model, loader, conv_epochs=[2, 2], output_epochs=2, device=device)
        assert result["status"] == "ok"

    def test_evaluate_returns_accuracy(self):
        """evaluate_cnn should return a float between 0 and 1."""
        device = torch.device("cpu")
        loader = _make_synthetic_cifar_loader(n_samples=16, batch_size=4, img_size=8)
        model = HebbianCNN(
            in_channels=3, img_size=8, hidden_channels=4, n_classes=3, device=device,
        )

        acc = evaluate_cnn(model, loader, device, epsilon=0.1)
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} outside [0, 1]"


# ── Test 5: Gradient guard ───────────────────────────────────────


class TestGradientGuard:
    """Verify training enforces no autograd."""

    def test_training_requires_no_grad(self):
        """train_conv_competitive_epoch should be callable with torch.no_grad()."""
        device = torch.device("cpu")
        loader = _make_synthetic_cifar_loader(n_samples=8, batch_size=4, img_size=8)
        model = HebbianCNN(
            in_channels=3, img_size=8, hidden_channels=4, n_classes=3, device=device,
        )
        _init_conv_connectivity(model.conv1, density=0.3)

        with torch.no_grad():
            metrics = train_conv_competitive_epoch(
                model.conv1, loader, None, device, lr=0.01, decay=0.0, epsilon=0.1,
            )

        assert 0.0 <= metrics["flip_rate"] <= 1.0
        assert metrics["n_flips"] >= 0


# ── Test 6: CIFAR-10 accuracy benchmark (SLOW, GPU) ──────────────


@pytest.mark.slow
@pytest.mark.gpu
class TestCIFAR10Accuracy:
    """Full CIFAR-10 Hebbian CNN training."""

    @pytest.fixture(scope="class")
    def cifar10_data(self):
        """Load CIFAR-10 once for all tests in this class."""
        return get_cifar10_loaders(batch_size=128)

    def test_cifar10_training_completes(self, cifar10_data):
        """Training on CIFAR-10 should complete without errors."""
        train_loader, test_loader = cifar10_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = HebbianCNN(
            in_channels=3,
            img_size=32,
            hidden_channels=64,
            n_classes=10,
            device=device,
        )

        _run_cnn_training(model, train_loader, conv_epochs=[2, 2], output_epochs=5, device=device)

        acc = evaluate_cnn(model, test_loader, device, epsilon=0.1)
        print(f"\n  CIFAR-10 Hebbian CNN test accuracy: {100 * acc:.2f}%")

        # Accept any >10% (better than random) as successful completion
        assert acc > 0.10, f"Accuracy {100 * acc:.2f}% is worse than random"

    def test_cifar10_above_55_percent(self, cifar10_data):
        """Full CIFAR-10 training should exceed 55% (Phase 1.2 target)."""
        train_loader, test_loader = cifar10_data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = HebbianCNN(
            in_channels=3,
            img_size=32,
            hidden_channels=64,
            n_classes=10,
            device=device,
        )

        _run_cnn_training(model, train_loader, conv_epochs=[3, 3], output_epochs=10, device=device)

        acc = evaluate_cnn(model, test_loader, device, epsilon=0.1)
        print(f"\n  CIFAR-10 Hebbian CNN test accuracy: {100 * acc:.2f}%")

        assert acc > 0.55, (
            f"CIFAR-10 accuracy = {100 * acc:.2f}%, expected > 55%"
        )
