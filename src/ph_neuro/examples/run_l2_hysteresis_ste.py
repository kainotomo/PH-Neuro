#!/usr/bin/env python3
"""L2: Hysteresis-STE Ablation — experiment runner.

Runs a single Hysteresis-STE configuration on one dataset and logs
results as JSON. Supports sweeping over theta_upper and theta_lower
hyperparameters to study the accuracy-sparsity trade-off.

Usage:
    # Run with custom thresholds
    python -m ph_neuro.examples.run_l2_hysteresis_ste \\
        --dataset mnist --theta-upper 1.0 --theta-lower 0.3 --seed 42

    # Compare with standard STE (baseline, no hysteresis)
    python -m ph_neuro.examples.run_l2_hysteresis_ste \\
        --dataset mnist --control --seed 42

Output:
    JSON file: ``{output_dir}/results_{dataset}_th{theta_upper}_tl{theta_lower}_seed{seed}.json``
    or for control: ``{output_dir}/results_{dataset}_control_seed{seed}.json``

See Also:
    ``aggregate_l2_results.py`` — collects and visualises all L2 results.
    ``run_l1_baseline_suite.py`` — L1 baseline suite (standard STE).
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
}


# ── Training loop ───────────────────────────────────────────────────


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    device: torch.device,
    variant: str,
    theta_upper: float,
    theta_lower: float,
) -> dict:
    """Train a Hysteresis-STE model and return detailed results.

    Tracks accuracy, sparsity, flip rates, and hysteresis zone
    distribution per epoch.

    Args:
        model: The model to train (should contain HysteresisSTELinear layers).
        train_loader: Training data loader.
        test_loader: Test data loader.
        optimizer: Optimizer.
        scheduler: Optional learning rate scheduler.
        epochs: Number of training epochs.
        device: Device to train on.
        variant: Variant label (e.g. ``"hyst"`` or ``"control"``).
        theta_upper: Hysteresis upper threshold (for logging).
        theta_lower: Hysteresis lower threshold (for logging).

    Returns:
        Dict with accuracy, sparsity, flip rates, convergence metrics, etc.
    """
    from ph_neuro.layers.ste_hysteresis import HysteresisSTELinear

    model.train()
    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0

    # Per-epoch tracking
    epoch_accuracies: list[float] = []
    epoch_sparsities: list[float] = []
    epoch_flip_rates: list[float] = []
    epoch_hysteresis_zones: list[dict] = []

    # Snapshot weights before training (for flip rate calculation)
    prev_ternary_snapshot = _snapshot_ternary_weights(model)

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

        # ── Hysteresis-STE specific metrics ─────────────────────────
        current_ternary = _snapshot_ternary_weights(model)
        sparsity = _compute_sparsity(current_ternary)
        flip_rate = _compute_flip_rate(prev_ternary_snapshot, current_ternary)
        zones = _compute_hysteresis_zones(model, theta_upper, theta_lower)

        epoch_accuracies.append(test_acc)
        epoch_sparsities.append(sparsity)
        epoch_flip_rates.append(flip_rate)
        epoch_hysteresis_zones.append(zones)

        # Early stopping check
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        # Update snapshot for next epoch
        prev_ternary_snapshot = current_ternary

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"Train Acc: {100 * train_acc:5.2f}%  "
            f"Test Acc: {100 * test_acc:5.2f}%  "
            f"Sparsity: {sparsity:5.2f}%  "
            f"Flips: {100 * flip_rate:6.3f}%  "
            f"LR: {lr:.2e}  "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - total_start

    # ── Convergence speed: epochs to reach 95% of best accuracy ────
    target_acc = 0.95 * best_acc
    convergence_epoch = next(
        (i + 1 for i, acc in enumerate(epoch_accuracies) if acc >= target_acc),
        epochs,
    )

    # ── Final weight statistics ─────────────────────────────────────
    final_ternary = _snapshot_ternary_weights(model)
    final_sparsity = _compute_sparsity(final_ternary)
    final_dist = _compute_weight_distribution(final_ternary)

    # Average the last 5 epochs for stable metrics
    recent_flip_avg = (
        sum(epoch_flip_rates[-5:]) / min(len(epoch_flip_rates[-5:]), 5)
        if epoch_flip_rates
        else 0.0
    )
    recent_sparsity_avg = (
        sum(epoch_sparsities[-5:]) / min(len(epoch_sparsities[-5:]), 5)
        if epoch_sparsities
        else 0.0
    )

    return {
        "variant": variant,
        "theta_upper": theta_upper,
        "theta_lower": theta_lower,
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(test_acc),
        "training_time_seconds": float(total_time),
        "epochs_trained": epoch,
        # Sparsity metrics
        "final_weight_sparsity_pct": final_sparsity,
        "final_weight_pos_pct": final_dist["pos_pct"],
        "final_weight_neg_pct": final_dist["neg_pct"],
        "final_weight_zero_pct": final_dist["zero_pct"],
        "avg_sparsity_last_5_epochs_pct": recent_sparsity_avg,
        # Flip rate metrics
        "avg_flip_rate_last_5_epochs_pct": 100.0 * recent_flip_avg,
        "max_flip_rate_pct": 100.0 * max(epoch_flip_rates) if epoch_flip_rates else 0.0,
        # Hysteresis zone distribution (final epoch)
        "hysteresis_pct_above_upper": zones["pct_above_upper"],
        "hysteresis_pct_below_lower": zones["pct_below_lower"],
        "hysteresis_pct_in_gap": zones["pct_in_gap"],
        # Convergence
        "convergence_epoch_95pct": convergence_epoch,
        "total_parameters": final_ternary.numel() if final_ternary is not None else 0,
        # Per-epoch series (for detailed plotting)
        "epoch_accuracies": [float(a) for a in epoch_accuracies],
        "epoch_sparsities": [float(s) for s in epoch_sparsities],
        "epoch_flip_rates": [float(f) for f in epoch_flip_rates],
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


# ── Hysteresis-STE metric helpers ───────────────────────────────────


@torch.no_grad()
def _snapshot_ternary_weights(model: nn.Module) -> torch.Tensor | None:
    """Collect all ternary weights from HysteresisSTELinear layers.

    Returns:
        Concatenated 1D int8 tensor of all ternary weights, or None
        if no hysteresis layers are found.
    """
    from ph_neuro.layers.ste_hysteresis import HysteresisSTELinear

    parts: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, HysteresisSTELinear):
            parts.append(module.ternary_weight().flatten())
    if not parts:
        return None
    return torch.cat(parts)


def _compute_sparsity(weights: torch.Tensor | None) -> float:
    """Compute sparsity (fraction of weights at 0) as a percentage."""
    if weights is None or weights.numel() == 0:
        return 0.0
    return 100.0 * (weights == 0).sum().item() / weights.numel()


def _compute_flip_rate(
    before: torch.Tensor | None,
    after: torch.Tensor | None,
) -> float:
    """Compute fraction of weights that changed between two snapshots."""
    if before is None or after is None:
        return 0.0
    if before.numel() == 0:
        return 0.0
    return (before != after).sum().item() / before.numel()


def _compute_weight_distribution(weights: torch.Tensor | None) -> dict[str, float]:
    """Compute percentage of weights at +1, -1, and 0."""
    if weights is None or weights.numel() == 0:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "zero_pct": 0.0}
    total = weights.numel()
    zero = (weights == 0).sum().item()
    pos = (weights == 1).sum().item()
    neg = (weights == -1).sum().item()
    return {
        "pos_pct": 100.0 * pos / total,
        "neg_pct": 100.0 * neg / total,
        "zero_pct": 100.0 * zero / total,
    }


@torch.no_grad()
def _compute_hysteresis_zones(
    model: nn.Module,
    theta_upper: float,
    theta_lower: float,
) -> dict[str, float]:
    """Analyze hysteresis zone distribution across all layers.

    Classifies each latent score into:
    - ``above_upper``: |score| > theta_upper (strong, will be activated)
    - ``below_lower``: |score| < theta_lower (weak, will be deactivated)
    - ``in_gap``: between thresholds (protected by hysteresis)

    Returns:
        Dict with percentage of weights in each zone.
    """
    from ph_neuro.layers.ste_hysteresis import HysteresisSTELinear

    total = 0
    above = 0
    below = 0

    for module in model.modules():
        if isinstance(module, HysteresisSTELinear):
            scores = module.latent_scores.float().flatten()
            total += scores.numel()
            above += (scores.abs() > theta_upper).sum().item()
            below += (scores.abs() < theta_lower).sum().item()

    if total == 0:
        return {
            "pct_above_upper": 0.0,
            "pct_below_lower": 0.0,
            "pct_in_gap": 0.0,
        }

    pct_above = 100.0 * above / total
    pct_below = 100.0 * below / total
    return {
        "pct_above_upper": pct_above,
        "pct_below_lower": pct_below,
        "pct_in_gap": 100.0 - pct_above - pct_below,
    }


# ── Model builders ──────────────────────────────────────────────────


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


def build_hyst_model(
    dataset: str,
    theta_upper: float,
    theta_lower: float,
    device: torch.device,
    lr: float,
    weight_decay: float,
    control: bool = False,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Build a Hysteresis-STE model (or control STE model) and optimizer.

    Args:
        dataset: Dataset key (``mnist``, ``fashion``, ``kmnist``).
        theta_upper: Hysteresis upper threshold.
        theta_lower: Hysteresis lower threshold.
        device: Torch device.
        lr: Learning rate.
        weight_decay: Weight decay.
        control: If ``True``, use standard STE (no hysteresis) as baseline.

    Returns:
        Tuple of ``(model, optimizer, scheduler)``.
    """
    cfg = DATASETS[dataset]
    layer_sizes = cfg["layer_sizes"]

    if control:
        from ph_neuro.models.ste_models import ste_mlp

        model = ste_mlp(layer_sizes, device=device)
    else:
        from ph_neuro.models.ste_models import hyst_ste_mlp

        model = hyst_ste_mlp(
            layer_sizes,
            theta_upper=theta_upper,
            theta_lower=theta_lower,
            device=device,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    return model, optimizer, scheduler


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L2: Hysteresis-STE Ablation — single run",
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
        "--theta-upper",
        type=float,
        default=1.0,
        help="Hysteresis upper threshold (ignored if --control)",
    )
    parser.add_argument(
        "--theta-lower",
        type=float,
        default=0.3,
        help="Hysteresis lower threshold (ignored if --control)",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Run standard STE (no hysteresis) as control baseline",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs (default: per-dataset)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="l2_results",
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
    """Run the L2 experiment."""
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

    if args.control:
        variant = "control"
        label = "Standard STE (control)"
        theta_upper = 0.0  # not used
        theta_lower = 0.0  # not used
    else:
        variant = "hyst"
        label = f"Hysteresis-STE θ_u={args.theta_upper}, θ_l={args.theta_lower}"
        theta_upper = args.theta_upper
        theta_lower = args.theta_lower

    print_header(f"L2: {label} on {args.dataset.upper()} (seed={args.seed})")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Architecture: MLP {dataset_cfg['layer_sizes']}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = _get_loaders(args.dataset, args.batch_size)
    n_train = len(train_loader.dataset)  # type: ignore[arg-type]
    n_test = len(test_loader.dataset)  # type: ignore[arg-type]
    print(f"Dataset: {n_train} train, {n_test} test samples")
    print()

    # ── Model & training ────────────────────────────────────────────
    model, optimizer, scheduler = build_hyst_model(
        args.dataset,
        args.theta_upper,
        args.theta_lower,
        device,
        args.lr,
        args.weight_decay,
        control=args.control,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    results = train_and_evaluate(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs,
        device,
        variant=variant,
        theta_upper=theta_upper,
        theta_lower=theta_lower,
    )

    # ── Build result dict ───────────────────────────────────────────
    result = {
        "experiment": "L2",
        "dataset": args.dataset,
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

    if args.control:
        fname = f"results_{args.dataset}_control_seed{args.seed}.json"
    else:
        tu = str(args.theta_upper).replace(".", "_")
        tl = str(args.theta_lower).replace(".", "_")
        fname = f"results_{args.dataset}_th{tu}_tl{tl}_seed{args.seed}.json"

    output_path = os.path.join(args.output_dir, fname)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print()
    print(f"Results saved to: {output_path}")
    print(
        f"Summary: {label} on {args.dataset.upper()} — "
        f"Best Accuracy: {100 * result['best_accuracy']:.2f}%, "
        f"Sparsity: {result.get('final_weight_sparsity_pct', 0):.1f}%, "
        f"Flips: {result.get('avg_flip_rate_last_5_epochs_pct', 0):.3f}%/epoch"
    )


if __name__ == "__main__":
    main()
