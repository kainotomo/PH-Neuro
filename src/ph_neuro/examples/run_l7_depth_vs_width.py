#!/usr/bin/env python3
"""L7: Depth vs Width Scaling — experiment runner.

Given a fixed parameter budget (~530K), systematically compares ternary
STE networks of varying depth (1-5 hidden layers) against equivalent
FP16 baselines on MNIST.

Usage:
    # Run ternary STE D=2 (3-layer MLP) on MNIST
    python -m ph_neuro.examples.run_l7_depth_vs_width \\
        --dataset mnist --depth 2 --weight-format ternary --seed 42

    # Run FP16 D=5 (6-layer MLP) on Fashion-MNIST
    python -m ph_neuro.examples.run_l7_depth_vs_width \\
        --dataset fashion --depth 5 --weight-format fp16 --seed 42

Output:
    JSON file per run::
        ``{output_dir}/results_{dataset}_{format}_d{depth}_seed{seed}.json``

See Also:
    ``aggregate_l7_results.py`` — collects and visualises all L7 results.
    ``run_l1_baseline_suite.py`` — L1 baseline suite (reference architecture).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header

warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")

# ── Depth configurations (all ~530K params) ─────────────────────────
#
# Formula: For D hidden layers of equal width w:
#   params = (D-1)w² + 794w   (with BatchNorm, no bias)
# Widths are rounded to nearest integer to match ~530K param budget.
#
# See plan for derivation.

DEPTH_CONFIGS: dict[int, list[int]] = {
    1: [784, 667, 10],
    2: [784, 432, 432, 10],
    3: [784, 353, 353, 353, 10],
    4: [784, 308, 308, 308, 308, 10],
    5: [784, 278, 278, 278, 278, 278, 10],
}

DATASET_CONFIGS = {
    "mnist": {
        "n_classes": 10,
        "epochs": 30,
        "loader_fn": "get_mnist_loaders",
    },
    "fashion": {
        "n_classes": 10,
        "epochs": 30,
        "loader_fn": "get_fashion_mnist_loaders",
    },
    "kmnist": {
        "n_classes": 10,
        "epochs": 30,
        "loader_fn": "get_kmnist_loaders",
    },
}


# ── Model builders ──────────────────────────────────────────────────


def _build_ternary_mlp(layer_sizes: list[int], device: torch.device) -> nn.Module:
    """Build a ternary STE MLP with given layer sizes.

    Uses ``ste_mlp()`` from ``ph_neuro.models.ste_models`` with
    BatchNorm enabled and flatten prepended for image inputs.
    """
    from ph_neuro.models.ste_models import ste_mlp

    return ste_mlp(layer_sizes, device=device)


def _build_fp16_mlp(layer_sizes: list[int], device: torch.device) -> nn.Module:
    """Build a standard FP16 MLP with given layer sizes.

    Architecture: ``Flatten + Linear + ReLU + BN + ... + Linear``
    Matching the ternary STE structure for fair comparison.
    """
    layers: list[nn.Module] = [nn.Flatten()]
    for i in range(len(layer_sizes) - 1):
        layers.append(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False)
        )
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
    return nn.Sequential(*layers).to(device)


# ── Data loading ────────────────────────────────────────────────────


def _get_loaders(dataset: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Get train and test loaders for a dataset."""
    from ph_neuro.training.data import (
        get_fashion_mnist_loaders,
        get_kmnist_loaders,
        get_mnist_loaders,
    )

    loader_map = {
        "mnist": get_mnist_loaders,
        "fashion": get_fashion_mnist_loaders,
        "kmnist": get_kmnist_loaders,
    }
    fn = loader_map[dataset]
    return fn(batch_size=batch_size)


# ── Training loop ───────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate a model on the test set.

    Returns:
        Test accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    device: torch.device,
    weight_format: str,
) -> dict:
    """Train a PyTorch model and return training results.

    Uses standard backpropagation with CrossEntropyLoss, optional
    cosine annealing scheduler, and early stopping (patience=10).

    Args:
        model: The model to train.
        train_loader: Training data loader.
        test_loader: Test data loader.
        optimizer: Optimizer (AdamW or None for Hebbian).
        scheduler: Optional LR scheduler.
        epochs: Number of training epochs.
        device: Device to train on.
        weight_format: ``"ternary"`` or ``"fp16"`` for weight stats.

    Returns:
        Dict with accuracy, loss, training time, weight stats.
    """
    model.train()
    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 10
    final_epoch = epochs

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"Train Acc: {100 * train_acc:5.2f}%  "
            f"Test Acc: {100 * test_acc:5.2f}%  "
            f"Loss: {total_loss / max(total, 1):.4f}  "
            f"LR: {lr:.2e}  "
            f"Time: {epoch_time:.1f}s"
        )

        if patience_counter >= max_patience:
            print(f"  ⏹️  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            final_epoch = epoch
            break

    total_time = time.time() - total_start

    # ── Compute weight statistics ───────────────────────────────────
    weight_stats = _compute_weight_stats(model, weight_format)

    return {
        "weight_format": weight_format,
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(test_acc),
        "training_time_seconds": float(total_time),
        "epochs_trained": final_epoch,
        **weight_stats,
    }


@torch.no_grad()
def _compute_weight_stats(model: nn.Module, weight_format: str) -> dict:
    """Extract weight statistics from a model.

    For ternary: sparsity (%), +1/0/-1 distribution.
    For FP16: mean, std, near-zero sparsity.
    """
    stats: dict = {
        "weight_sparsity_pct": 0.0,
        "weight_zero_pct": 0.0,
        "weight_pos_pct": 0.0,
        "weight_neg_pct": 0.0,
        "n_parameters": 0,
    }

    if weight_format == "ternary":
        total_w = 0
        total_zero = 0
        total_pos = 0
        total_neg = 0
        for module in model.modules():
            if hasattr(module, "ternary_weight"):
                w = module.ternary_weight().flatten()
                total_w += w.numel()
                total_zero += (w == 0).sum().item()
                total_pos += (w == 1).sum().item()
                total_neg += (w == -1).sum().item()
        if total_w > 0:
            stats["weight_zero_pct"] = 100.0 * total_zero / total_w
            stats["weight_pos_pct"] = 100.0 * total_pos / total_w
            stats["weight_neg_pct"] = 100.0 * total_neg / total_w
            stats["weight_sparsity_pct"] = stats["weight_zero_pct"]
            stats["n_parameters"] = total_w
    else:
        # FP16 — count near-zero weights as sparsity proxy
        total_w = 0
        total_small = 0
        for p in model.parameters():
            total_w += p.numel()
            total_small += (p.abs() < 0.01).sum().item()
        if total_w > 0:
            stats["weight_sparsity_pct"] = 100.0 * total_small / total_w
            stats["n_parameters"] = total_w

    # Add per-layer breakdown
    per_layer = _compute_per_layer_stats(model, weight_format)
    stats["per_layer"] = per_layer

    return stats


@torch.no_grad()
def _compute_per_layer_stats(model: nn.Module, weight_format: str) -> list[dict]:
    """Extract per-layer weight statistics.

    Returns:
        List of dicts, one per ternary/linear layer, with keys:
        ``layer_name``, ``n_params``, and format-specific stats.
    """
    per_layer: list[dict] = []

    if weight_format == "ternary":
        for name, module in model.named_modules():
            if hasattr(module, "ternary_weight"):
                w = module.ternary_weight().flatten()
                if w.numel() == 0:
                    continue
                per_layer.append({
                    "layer_name": name,
                    "n_params": w.numel(),
                    "zero_pct": 100.0 * (w == 0).sum().item() / w.numel(),
                    "pos_pct": 100.0 * (w == 1).sum().item() / w.numel(),
                    "neg_pct": 100.0 * (w == -1).sum().item() / w.numel(),
                })
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight is not None:
                w = module.weight.data.flatten()
                if w.numel() == 0:
                    continue
                per_layer.append({
                    "layer_name": name,
                    "n_params": w.numel(),
                    "mean": w.mean().item(),
                    "std": w.std().item(),
                    "near_zero_pct": 100.0 * (w.abs() < 0.01).sum().item() / w.numel(),
                })

    return per_layer


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L7: Depth vs Width Scaling — single run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to run on",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Number of hidden layers (1..5)",
    )
    parser.add_argument(
        "--weight-format",
        type=str,
        required=True,
        choices=["ternary", "fp16"],
        help="Weight format: ternary (STE) or fp16 (standard)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs (default: per-dataset)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="l7_results",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the L7 experiment."""
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Set seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    epochs = args.epochs or dataset_cfg["epochs"]
    layer_sizes = DEPTH_CONFIGS[args.depth]

    print_header(
        f"L7: Depth={args.depth} ({args.weight_format}) on {args.dataset.upper()} "
        f"(seed={args.seed})"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Layer sizes: {layer_sizes}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = _get_loaders(args.dataset, args.batch_size)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_test = len(test_loader.dataset)  # type: ignore[arg-type]
    print(f"Dataset: {n_train} train, {n_test} test samples")
    print()

    # ── Model ───────────────────────────────────────────────────────
    if args.weight_format == "ternary":
        model = _build_ternary_mlp(layer_sizes, device)
    else:
        model = _build_fp16_mlp(layer_sizes, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Layer sizes: {layer_sizes}")
    print()

    # ── Optimiser & scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # ── Train ───────────────────────────────────────────────────────
    results = train_and_evaluate(
        model, train_loader, test_loader,
        optimizer, scheduler, epochs, device,
        weight_format=args.weight_format,
    )

    # ── Build result dict ───────────────────────────────────────────
    result = {
        "experiment": "L7",
        "dataset": args.dataset,
        "weight_format": args.weight_format,
        "depth": args.depth,
        "layer_sizes": layer_sizes,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "n_parameters": n_params,
        **results,
    }

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    fname = (
        f"results_{args.dataset}_{args.weight_format}"
        f"_d{args.depth}_seed{args.seed}.json"
    )
    out_path = os.path.join(args.output_dir, fname)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print(f"  {'=' * 45}")
    print(f"  Best Test Acc: {100 * results['best_accuracy']:.2f}% "
          f"(epoch {results['best_epoch']})")
    print(f"  Final Test Acc: {100 * results['final_accuracy']:.2f}%")
    if results.get("weight_sparsity_pct", 0) > 0:
        print(f"  Weight sparsity: {results['weight_sparsity_pct']:.2f}%")
    print(f"  Training time: {results['training_time_seconds']:.1f}s")
    print(f"  {'=' * 45}")


if __name__ == "__main__":
    main()
