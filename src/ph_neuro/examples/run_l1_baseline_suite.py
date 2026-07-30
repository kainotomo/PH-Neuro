#!/usr/bin/env python3
"""L1: Ternary STE Baseline Suite — experiment runner.

Runs ONE variant × dataset combination and logs results as JSON.

Usage:
    # Run Ternary STE (V1) on MNIST
    python -m ph_neuro.examples.run_l1_baseline_suite \\
        --dataset mnist --variant v1 --epochs 30 --seed 42

    # Run FP16 (V2) on CIFAR-10
    python -m ph_neuro.examples.run_l1_baseline_suite \\
        --dataset cifar10 --variant v2 --epochs 100 --seed 42

    # Run all variants sequentially (bash)
    for variant in v1 v2 v3 v4 v5; do
        python -m ph_neuro.examples.run_l1_baseline_suite \\
            --dataset mnist --variant "$variant" --seed 42
    done

Output:
    JSON file per run: ``{output_dir}/results_{dataset}_{variant}_seed{seed}.json``

See Also:
    ``aggregate_l1_results.py`` — collects and visualizes all results.
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

# ── Dataset registry ────────────────────────────────────────────────

DATASETS = {
    "mnist": {
        "n_classes": 10,
        "arch": "mlp",
        "layer_sizes": [784, 512, 256, 10],
        "epochs": 30,
        "loader_fn": "get_mnist_loaders",
    },
    "fashion": {
        "n_classes": 10,
        "arch": "mlp",
        "layer_sizes": [784, 512, 256, 10],
        "epochs": 30,
        "loader_fn": "get_fashion_mnist_loaders",
    },
    "kmnist": {
        "n_classes": 10,
        "arch": "mlp",
        "layer_sizes": [784, 512, 256, 10],
        "epochs": 30,
        "loader_fn": "get_kmnist_loaders",
    },
    "cifar10": {
        "n_classes": 10,
        "arch": "cnn",
        "epochs": 100,
        "loader_fn": "get_cifar10_loaders",
    },
    "cifar100": {
        "n_classes": 100,
        "arch": "cnn",
        "epochs": 150,
        "loader_fn": "get_cifar100_loaders",
    },
}

# ── Training loop (shared across all variants) ──────────────────────


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    device: torch.device,
    variant: str,
) -> dict:
    """Train a model and return results.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        test_loader: Test data loader.
        optimizer: Optimizer.
        scheduler: Optional learning rate scheduler.
        epochs: Number of training epochs.
        device: Device to train on.
        variant: Variant ID (``v1``-``v5``) for logging.

    Returns:
        Dict with accuracy, loss, training time, etc.
    """
    model.train()
    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 10

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
            break

    total_time = time.time() - total_start

    # ── Compute weight statistics ───────────────────────────────────
    weight_stats = _compute_weight_stats(model, variant)

    return {
        "variant": variant,
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(test_acc),
        "training_time_seconds": float(total_time),
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        **weight_stats,
    }


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


def _compute_weight_stats(model: nn.Module, variant: str) -> dict:
    """Extract weight statistics from a model.

    For ternary variants (V1, V5): sparsity, +1/0/-1 distribution, flip rate.
    For float variants (V2, V3, V4): mean, std, sparsity of weights.
    """
    stats: dict = {
        "weight_sparsity_pct": 0.0,
        "weight_pos_pct": 0.0,
        "weight_neg_pct": 0.0,
        "weight_zero_pct": 0.0,
        "n_parameters": 0,
    }

    if variant in ("v1",):
        # Ternary STE layers
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

    elif variant == "v5":
        # Hebbian layers
        from ph_neuro.layers.linear import TernaryHebbianLinear

        total_w = 0
        total_zero = 0
        total_pos = 0
        total_neg = 0
        for module in model.modules():
            if isinstance(module, TernaryHebbianLinear):
                w = module.weight.unpack().flatten()
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
        # Float variants (V2, V3, V4)
        total_w = 0
        total_small = 0
        for p in model.parameters():
            total_w += p.numel()
            total_small += (p.abs() < 0.01).sum().item()
        if total_w > 0:
            stats["weight_sparsity_pct"] = 100.0 * total_small / total_w
            stats["n_parameters"] = total_w

    return stats


# ── Variant builders ────────────────────────────────────────────────


def _get_loaders(dataset: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Get train and test loaders for a dataset."""
    from ph_neuro.training.data import (
        get_cifar10_loaders,
        get_cifar100_loaders,
        get_fashion_mnist_loaders,
        get_kmnist_loaders,
        get_mnist_loaders,
    )

    loader_map = {
        "mnist": get_mnist_loaders,
        "fashion": get_fashion_mnist_loaders,
        "kmnist": get_kmnist_loaders,
        "cifar10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
    }
    fn = loader_map[dataset]
    return fn(batch_size=batch_size)


def build_model_and_optimizer(
    dataset: str,
    variant: str,
    device: torch.device,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Build model, optimizer, and scheduler for a given variant × dataset."""

    cfg = DATASETS[dataset]

    if variant == "v1":
        # Ternary STE — our method
        from ph_neuro.models.ste_models import ste_cnn, ste_mlp

        if cfg["arch"] == "mlp":
            model = ste_mlp(cfg["layer_sizes"], device=device)
        else:
            model = ste_cnn(n_classes=cfg["n_classes"], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif variant == "v2":
        # FP16 — float baseline (standard nn.Linear/nn.Conv2d)
        if cfg["arch"] == "mlp":
            layers: list[nn.Module] = [nn.Flatten()]
            for i in range(len(cfg["layer_sizes"]) - 1):
                layers.append(nn.Linear(cfg["layer_sizes"][i], cfg["layer_sizes"][i + 1]))
                if i < len(cfg["layer_sizes"]) - 2:
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.BatchNorm1d(cfg["layer_sizes"][i + 1]))
            model = nn.Sequential(*layers).to(device)
        else:
            flat_features = (2 * 64) * (32 // 4) * (32 // 4)  # CIFAR: 32//4=8 → 128*8*8=8192
            model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(flat_features, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Linear(512, cfg["n_classes"]),
            ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif variant == "v3":
        # INT8 QAT
        if cfg["arch"] == "mlp":
            from ph_neuro.training.qat_helpers import create_int8_qat_mlp

            model = create_int8_qat_mlp(cfg["layer_sizes"], device=device)
        else:
            from ph_neuro.training.qat_helpers import create_int8_qat_cnn

            model = create_int8_qat_cnn(n_classes=cfg["n_classes"], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif variant == "v4":
        # INT4 QAT
        if cfg["arch"] == "mlp":
            from ph_neuro.training.qat_helpers import create_int4_qat_mlp

            model = create_int4_qat_mlp(cfg["layer_sizes"], device=device)
        else:
            from ph_neuro.training.qat_helpers import create_int4_qat_cnn

            model = create_int4_qat_cnn(n_classes=cfg["n_classes"], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif variant == "v5":
        # Hebbian v1 — legacy baseline (single layer only)
        from ph_neuro.training.supervised import SupervisedHebbianClassifier

        in_features = (
            cfg["layer_sizes"][0] if cfg["arch"] == "mlp" else 3 * 32 * 32
        )
        classifier = SupervisedHebbianClassifier(
            in_features=in_features,
            out_features=cfg["n_classes"],
            theta_upper=1.0,
            theta_lower=0.3,
            device=device,
        )
        model = classifier
        optimizer = None  # Hebbian doesn't use an optimizer
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Cosine scheduler for all AdamW variants
    if optimizer is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"]
        )
    else:
        scheduler = None

    return model, optimizer, scheduler


# ── V5 (Hebbian v1) training ───────────────────────────────────────


def train_v5_hebbian(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> dict:
    """Train a Hebbian classifier (V5) without backprop.

    Uses the existing ``SupervisedHebbianClassifier`` infrastructure.
    """
    from ph_neuro.training.supervised import SupervisedHebbianClassifier

    classifier: SupervisedHebbianClassifier = model
    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        classifier.model.train()
        step_metrics: list[dict] = []

        for x, y in train_loader:
            with torch.no_grad():
                metrics = classifier.train_step(x, y, lr=0.01, decay=0.0, epsilon=0.1)
                step_metrics.append(metrics)

        test_acc = classifier.evaluate(test_loader, epsilon=0.1)
        avg_flip = sum(m["flip_rate"] for m in step_metrics) / max(len(step_metrics), 1)
        epoch_time = time.time() - epoch_start

        print(
            f"  Epoch {epoch:2d}/{epochs}  "
            f"Test Acc: {100 * test_acc:5.2f}%  "
            f"Flips: {100 * avg_flip:5.2f}%/step  "
            f"Time: {epoch_time:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

    total_time = time.time() - total_start

    weight_stats = _compute_weight_stats(classifier.model, "v5")
    return {
        "variant": "v5",
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(test_acc),
        "training_time_seconds": float(total_time),
        "epochs_trained": epoch,
        **weight_stats,
    }


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L1: Ternary STE Baseline Suite — single run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to run on",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["v1", "v2", "v3", "v4", "v5"],
        help="""Variant:
            v1=Ternary STE, v2=FP16, v3=INT8 QAT,
            v4=INT4 QAT, v5=Hebbian v1""",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs (default: per-dataset)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="l1_results",
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
    """Run the L1 experiment."""
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Set seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    dataset_cfg = DATASETS[args.dataset]
    epochs = args.epochs or dataset_cfg["epochs"]

    print_header(f"L1: {args.variant} on {args.dataset.upper()} (seed={args.seed})")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Architecture: {dataset_cfg['arch']}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = _get_loaders(args.dataset, args.batch_size)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_test = len(test_loader.dataset)  # type: ignore[arg-type]
    print(f"Dataset: {n_train} train, {n_test} test samples")
    print()

    # ── Model & training ────────────────────────────────────────────
    if args.variant == "v5":
        from ph_neuro.training.supervised import SupervisedHebbianClassifier

        model, optimizer, scheduler = build_model_and_optimizer(
            args.dataset, args.variant, device, args.lr, args.weight_decay
        )
        results = train_v5_hebbian(
            model, train_loader, test_loader, epochs, device
        )
    else:
        model, optimizer, scheduler = build_model_and_optimizer(
            args.dataset, args.variant, device, args.lr, args.weight_decay
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        print()

        results = train_and_evaluate(
            model, train_loader, test_loader,
            optimizer, scheduler, epochs, device,
            variant=args.variant,
        )

    # ── Build result dict ───────────────────────────────────────────
    result = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        **results,
    }

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"results_{args.dataset}_{args.variant}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print()
    print(f"Results saved to: {output_path}")
    print(
        f"Summary: {args.variant} on {args.dataset.upper()} — "
        f"Best Accuracy: {100 * result['best_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
