"""Integration tests for Milestone M1.2 — DQT CNN on CIFAR-100.

Verifies the end-to-end DQT 3-conv + linear pipeline on CIFAR-100-shaped
inputs: model construction, forward pass, single-batch overfitting (sanity
check that the larger DQT conv backward + stochastic rounding can learn),
and a short training loop where accuracy beats chance (1% for 100 classes).

All tests run on CPU with synthetic data so they stay fast and do not
require CIFAR-100 to be downloaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.examples.run_m1_2_dqt_cifar100 import (
    ANNEAL_FRACTION,
    apply_dqt_rounding,
    train_dqt_cnn,
)
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
from ph_neuro.models.dqt_models import dqt_cnn_cifar100

DEVICE = torch.device("cpu")

# Layer layout of dqt_cnn_cifar100() as nn.Sequential:
#   0..3  Conv1 block (Conv→ReLU→BN→MaxPool)
#   4..7  Conv2 block
#   8..11 Conv3 block
#   12    Flatten
#   13    TernaryDQTLinear(flat→512)
#   14    ReLU
#   15    BatchNorm1d(512)
#   16    TernaryDQTLinear(512→n_classes)
FIRST_LINEAR_INDEX = 13


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


class TestDqtCnnCifar100Build:
    """Model construction tests."""

    def test_dqt_cnn_cifar100_build(self):
        """dqt_cnn_cifar100() should build with correct layer count/shapes."""
        model = dqt_cnn_cifar100(device=DEVICE)
        assert _count_dqt_layers(model) == 5, (
            "Expected 3 DQT conv + 2 DQT linear layers"
        )
        # First linear fan-in must be 4096 = 256 * (32//8)^2
        linear = model[FIRST_LINEAR_INDEX]
        assert isinstance(linear, TernaryDQTLinear)
        assert linear.in_features == 4096
        assert linear.out_features == 512
        # Output layer must have n_classes = 100
        out_linear = model[FIRST_LINEAR_INDEX + 3]
        assert isinstance(out_linear, TernaryDQTLinear)
        assert out_linear.out_features == 100
        # Conv channel progression 3 -> 64 -> 128 -> 256
        convs = [m for m in model if isinstance(m, TernaryDQTConv2d)]
        assert [c.in_channels for c in convs] == [3, 64, 128]
        assert [c.out_channels for c in convs] == [64, 128, 256]

    def test_dqt_cnn_cifar100_n_classes_parametric(self):
        """n_classes should be parametric (default 100)."""
        model = dqt_cnn_cifar100(n_classes=20, device=DEVICE)
        out_linear = model[FIRST_LINEAR_INDEX + 3]
        assert out_linear.out_features == 20

    def test_dqt_cnn_cifar100_forward(self):
        """Forward pass with dummy CIFAR-shaped data should produce logits."""
        model = dqt_cnn_cifar100(device=DEVICE)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 100), f"Got shape {out.shape}"
        assert out.requires_grad, "Logits should require grad for backprop"

    def test_dqt_cnn_cifar100_ternary_invariants(self):
        """All DQT weights are int8 ternary right after construction."""
        model = dqt_cnn_cifar100(device=DEVICE)
        assert _check_ternary_invariants(model)


class TestDqtCnnCifar100Overfit:
    """Single-batch overfitting sanity check."""

    def test_dqt_cnn_cifar100_overfit_batch(self):
        """The larger DQT CNN should be able to fit a small separable batch."""
        torch.manual_seed(0)
        loader = _make_synthetic_cifar_loader(
            n_samples=64, batch_size=64, img_size=16, n_classes=2, separable=True,
        )
        x, y = next(iter(loader))
        model = dqt_cnn_cifar100(img_size=16, device=DEVICE)
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


class TestDqtCnnCifar100TrainingLoop:
    """Short training-loop integration test."""

    def test_dqt_cnn_cifar100_training_loop(self):
        """2 epochs, 100 classes: test accuracy should beat chance (1%)."""
        torch.manual_seed(1)
        train_loader = _make_synthetic_cifar_loader(
            n_samples=400, batch_size=16, img_size=16, n_classes=100, pattern=True,
        )
        # Test loader is a separate (different) random set — evaluate on it
        test_loader = _make_synthetic_cifar_loader(
            n_samples=200, batch_size=16, img_size=16, n_classes=100, pattern=True,
        )
        model = dqt_cnn_cifar100(img_size=16, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

        results = train_dqt_cnn(
            model, train_loader, test_loader,
            optimizer, scheduler, DEVICE,
            epochs=2, max_patience=10, verbose=False,
        )

        # 100 classes -> chance = 1%; the pattern task must exceed it
        assert results["epochs_trained"] == 2
        assert len(results["train_acc_history"]) == 2
        assert results["best_accuracy"] > 0.01, (
            f"Test accuracy should beat chance (1%), got "
            f"{100 * results['best_accuracy']:.2f}%"
        )
        assert 0.0 <= results["final_flip_rate"] <= 1.0
        assert results["weight_stats"]["pos_pct"] + \
            results["weight_stats"]["neg_pct"] + \
            results["weight_stats"]["zero_pct"] - 100.0 < 1e-4
        assert _check_ternary_invariants(model)

    def test_dqt_cnn_cifar100_annealing(self):
        """Annealing logic should run (stochastic -> deterministic switch)."""
        torch.manual_seed(2)
        loader = _make_synthetic_cifar_loader(
            n_samples=64, batch_size=16, img_size=16, n_classes=10, pattern=True,
        )
        model = dqt_cnn_cifar100(img_size=16, device=DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

        results = train_dqt_cnn(
            model, loader, loader, optimizer, scheduler, DEVICE,
            epochs=2, max_patience=10, verbose=False,
        )

        assert results["anneal_start_epoch"] == int(2 * ANNEAL_FRACTION) == 1
        # After the final deterministic epoch, weight_ternary == sign(weight_float)
        for m in model.modules():
            if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear)):
                expected = m.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
                assert torch.equal(m.weight_ternary, expected), (
                    "After the final deterministic epoch, weight_ternary should "
                    "equal sign(weight_float)"
                )
        assert _check_ternary_invariants(model)

    def test_dqt_cnn_cifar100_rounding_modes(self):
        """use_stochastic flag should control stochastic vs deterministic rounding."""
        torch.manual_seed(3)
        model = dqt_cnn_cifar100(img_size=16, device=DEVICE)
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
