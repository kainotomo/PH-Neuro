"""Integration tests for Milestone M1.1 — DQT CNN on CIFAR-10.

Verifies the end-to-end DQT conv + linear pipeline on CIFAR-like inputs:
model construction, forward pass, single-batch overfitting (sanity check
that the DQT conv backward + stochastic rounding can actually learn), and
a short training loop where accuracy improves.

All tests run on CPU with synthetic data so they stay fast and do not
require CIFAR-10 to be downloaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.examples.run_m1_1_dqt_cifar10 import (
    ANNEAL_FRACTION,
    apply_dqt_rounding,
    train_dqt_cnn,
)
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
from ph_neuro.models.dqt_models import dqt_cnn

DEVICE = torch.device("cpu")


# ── Synthetic dataset helpers ──────────────────────────────────────


def _make_synthetic_cifar_loader(
    n_samples: int = 64,
    batch_size: int = 16,
    img_size: int = 16,
    n_classes: int = 5,
    separable: bool = False,
    pattern: bool = False,
) -> DataLoader:
    """Create a tiny synthetic dataset with CIFAR-like shape.

    Args:
        separable: If ``True``, build two easily-separable clusters (all-zeros
            vs all-ones) so the model can quickly overfit; otherwise random.
        pattern: If ``True``, give each class a distinct constant image
            (class k -> constant fill ``(k + 1) / (n_classes + 1)``) so the
            data is learnable within a few epochs.
    """
    if pattern:
        x = torch.zeros(n_samples, 3, img_size, img_size)
        y = torch.randint(0, n_classes, (n_samples,))
        for k in range(n_classes):
            mask = y == k
            x[mask] = (k + 1) / (n_classes + 1)
    elif separable:
        half = n_samples // 2
        x = torch.zeros(n_samples, 3, img_size, img_size)
        x[half:] = 1.0
        y = torch.cat(
            [torch.zeros(half, dtype=torch.long), torch.ones(n_samples - half, dtype=torch.long)]
        )
    else:
        x = torch.randn(n_samples, 3, img_size, img_size)
        y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def _count_dqt_layers(model: nn.Module) -> int:
    """Number of DQT weight-bearing layers in the model."""
    return sum(
        1
        for m in model.modules()
        if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear))
    )


def _check_ternary_invariants(model: nn.Module) -> bool:
    """All DQT weights must be int8 ternary in {-1, 0, +1}."""
    for m in model.modules():
        if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
            w = m.weight_ternary
            if w.dtype != torch.int8:
                return False
            if not bool(torch.all((w >= -1) & (w <= 1)).item()):
                return False
    return True


class TestDqtCnnBuild:
    """Model construction tests."""

    def test_dqt_cnn_build(self):
        """dqt_cnn() should build without error with correct layer count."""
        model = dqt_cnn(device=DEVICE)
        assert _count_dqt_layers(model) == 4, (
            "Expected 2 DQT conv + 2 DQT linear layers"
        )
        # Architecture sanity: 8192 -> 512 classifier (E021.3 final head —
        # E021/E021.2 temporarily used 256, reverted to 512 in E021.3)
        linear = model[9]  # index 8 is Flatten
        assert isinstance(linear, TernaryDQTLinear)
        assert linear.in_features == 8192
        assert linear.out_features == 512

    def test_dqt_cnn_forward(self):
        """Forward pass with dummy CIFAR-shaped data should produce logits."""
        model = dqt_cnn(device=DEVICE)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10), f"Got shape {out.shape}"
        assert out.requires_grad, "Logits should require grad for backprop"

    def test_dqt_cnn_ternary_invariants(self):
        """All DQT weights are int8 ternary right after construction."""
        model = dqt_cnn(device=DEVICE)
        assert _check_ternary_invariants(model)


class TestDqtCnnOverfit:
    """Single-batch overfitting sanity check."""

    def test_dqt_cnn_overfit_batch(self):
        """The DQT CNN should be able to fit a small separable batch."""
        torch.manual_seed(0)
        loader = _make_synthetic_cifar_loader(
            n_samples=64, batch_size=64, img_size=16, n_classes=2, separable=True,
        )
        x, y = next(iter(loader))
        model = dqt_cnn(img_size=16, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

        loss_first = None
        for step in range(60):
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            for m in model.modules():
                if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
                    m.apply_stochastic_rounding()
            if loss_first is None:
                loss_first = loss.item()
            scheduler.step()

        final_acc = (model(x).argmax(dim=1) == y).float().mean().item()
        assert loss_first is not None and loss.item() < loss_first, (
            f"Loss should decrease when overfitting (first={loss_first:.4f}, "
            f"final={loss.item():.4f})"
        )
        assert final_acc > 0.9, f"Should overfit separable batch, got {100 * final_acc:.1f}%"
        assert _check_ternary_invariants(model)


class TestDqtCnnTrainingLoop:
    """Short training-loop integration test."""

    def test_dqt_cnn_training_loop(self):
        """2 epochs should run without error and train accuracy should improve."""
        torch.manual_seed(1)
        train_loader = _make_synthetic_cifar_loader(
            n_samples=128, batch_size=16, img_size=16, n_classes=4, pattern=True,
        )
        # Test loader is a separate (different) random set — evaluate on it
        test_loader = _make_synthetic_cifar_loader(
            n_samples=64, batch_size=16, img_size=16, n_classes=4, pattern=True,
        )
        model = dqt_cnn(img_size=16, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

        results = train_dqt_cnn(
            model, train_loader, test_loader,
            optimizer, scheduler, DEVICE,
            epochs=2, max_patience=10, verbose=False,
        )

        assert results["epochs_trained"] == 2
        assert len(results["train_acc_history"]) == 2
        assert results["train_acc_history"][1] >= results["train_acc_history"][0], (
            "Train accuracy should improve across epochs"
        )
        assert 0.0 <= results["final_flip_rate"] <= 1.0
        assert results["weight_stats"]["pos_pct"] + \
            results["weight_stats"]["neg_pct"] + \
            results["weight_stats"]["zero_pct"] - 100.0 < 1e-4
        assert _check_ternary_invariants(model)

    def test_dqt_cnn_training_loop_stochastic_rounding(self):
        """Stochastic rounding keeps weights ternary after a full 2-epoch loop."""
        torch.manual_seed(2)
        loader = _make_synthetic_cifar_loader(
            n_samples=64, batch_size=16, img_size=16, n_classes=4,
        )
        model = dqt_cnn(img_size=16, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

        train_dqt_cnn(
            model, loader, loader, optimizer, scheduler, DEVICE,
            epochs=2, max_patience=10, verbose=False,
        )
        assert _check_ternary_invariants(model)

    def test_annealing_switch(self):
        """Training should anneal stochastic -> deterministic rounding correctly."""
        torch.manual_seed(3)
        loader = _make_synthetic_cifar_loader(
            n_samples=64, batch_size=16, img_size=16, n_classes=4,
        )
        model = dqt_cnn(img_size=16, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

        results = train_dqt_cnn(
            model, loader, loader, optimizer, scheduler, DEVICE,
            epochs=4, max_patience=10, verbose=False,
        )

        # int(4 * 0.85) = 3 → epochs 1-3 stochastic, epoch 4 deterministic
        assert results["anneal_start_epoch"] == int(4 * ANNEAL_FRACTION) == 3
        # After the final deterministic epoch, weight_ternary == sign(weight_float)
        for m in model.modules():
            if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
                expected = m.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
                assert torch.equal(m.weight_ternary, expected), (
                    "After the final deterministic epoch, weight_ternary should "
                    "equal sign(weight_float)"
                )
        assert _check_ternary_invariants(model)

    def test_annealing_switch_flag(self):
        """use_stochastic flag should control stochastic vs deterministic rounding."""
        torch.manual_seed(4)
        model = dqt_cnn(img_size=16, device=DEVICE)
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
                    m.weight_float.data.add_(0.5)

        # Deterministic: weight_ternary == sign(weight_float) exactly
        apply_dqt_rounding(model, use_stochastic=False)
        for m in model.modules():
            if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
                expected = m.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
                assert torch.equal(m.weight_ternary, expected), (
                    "Deterministic rounding must snap to sign(weight_float)"
                )

        # Stochastic: weights stay ternary int8 (values may differ from sign)
        apply_dqt_rounding(model, use_stochastic=True)
        for m in model.modules():
            if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
                w = m.weight_ternary
                assert w.dtype == torch.int8
                assert bool(torch.all((w >= -1) & (w <= 1)).item())
