#!/usr/bin/env python3
"""L8: Forgetting Baseline — experiment runner.

Measures catastrophic forgetting of standard SGD training (ternary STE
and FP16) on Split MNIST and Permuted MNIST. NO EWC, NO replay.

This is the **control experiment** for Track B (Continual Learning).
All subsequent CL experiments will be compared against these baselines.

Usage:
    # Ternary STE on Split MNIST (5 tasks)
    python -m ph_neuro.examples.run_l8_forgetting_baseline \\
        --protocol split --weight-format ternary --seed 42

    # FP16 on Permuted MNIST (10 tasks)
    python -m ph_neuro.examples.run_l8_forgetting_baseline \\
        --protocol permuted --weight-format fp16 --seed 42

    # Quick smoke test (2 tasks, 1 epoch each)
    python -m ph_neuro.examples.run_l8_forgetting_baseline \\
        --protocol split --weight-format ternary \\
        --epochs-per-task 1 --seed 42

Output:
    JSON file per run: ``{output_dir}/{protocol}_{weight_format}_seed{seed}.json``

See Also:
    ``aggregate_l8_results.py`` — collects and visualizes all results.
    ``backprop_baseline.py`` — legacy single-layer FP16 baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header

# ── Model builders ──────────────────────────────────────────────────


def _build_ternary_mlp(device: torch.device) -> nn.Module:
    """Build a 3-layer ternary STE MLP (784 → 512 → 256 → 10).

    Uses ``TernarySTELinear`` layers with ReLU + BatchNorm.
    Matches the L1 baseline suite architecture.
    """
    from ph_neuro.models.ste_models import ste_mlp

    return ste_mlp([784, 512, 256, 10], device=device)


def _build_fp16_mlp(device: torch.device) -> nn.Module:
    """Build a 3-layer FP16 MLP (784 → 512 → 256 → 10).

    Standard ``nn.Linear`` + ReLU + BatchNorm architecture,
    matching the ternary STE model structure for fair comparison.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Linear(256, 10),
    ).to(device)
    return model


# ── Weight statistics ───────────────────────────────────────────────


@torch.no_grad()
def _compute_ternary_weight_stats(model: nn.Module) -> dict[str, float]:
    """Extract ternary weight distribution statistics.

    Returns:
        Dict with sparsity_pct, pos_pct, neg_pct, zero_pct, n_parameters.
    """
    stats: dict[str, float] = {
        "weight_sparsity_pct": 0.0,
        "weight_pos_pct": 0.0,
        "weight_neg_pct": 0.0,
        "weight_zero_pct": 0.0,
        "n_parameters": 0.0,
    }
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
        stats["n_parameters"] = float(total_w)

    return stats


# ── Training function ───────────────────────────────────────────────


def train_task(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    task_idx: int,
    task_name: str,
) -> dict[str, float]:
    """Train model on a single task for a given number of epochs.

    Args:
        model: The model to train.
        loader: Training data loader for this task.
        optimizer: Optimizer (AdamW).
        epochs: Number of epochs to train.
        device: Torch device.
        task_idx: Index of this task (0-based) for logging.
        task_name: Human-readable task name for logging.

    Returns:
        Dict with metrics (final_loss, final_train_acc, training_time).
    """
    model.train()
    total_start = time.time()
    final_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        epoch_time = time.time() - epoch_start
        final_metrics = {
            "final_loss": total_loss / max(total, 1),
            "final_train_acc": train_acc,
            "training_time": time.time() - total_start,
        }

        print(
            f"    Task {task_idx + 1} ({task_name}), "
            f"Epoch {epoch:2d}/{epochs}  "
            f"Loss: {final_metrics['final_loss']:.4f}  "
            f"Train Acc: {100 * train_acc:5.2f}%  "
            f"Time: {epoch_time:.1f}s"
        )

    print(f"    -> Task {task_idx + 1} done.")
    return final_metrics


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L8: Forgetting Baseline — measure catastrophic forgetting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=["split", "permuted"],
        help="Continual learning protocol",
    )
    parser.add_argument(
        "--weight-format",
        type=str,
        required=True,
        choices=["ternary", "fp16"],
        help="Weight format: ternary (STE) or fp16 (standard float)",
    )
    parser.add_argument("--epochs-per-task", type=int, default=10, help="Epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of tasks (permuted only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="l8_results",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    """Run the L8 forgetting baseline experiment."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    # Set seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    protocol_name = 'Split' if args.protocol == 'split' else 'Permuted'
    title = f"L8: {args.weight_format.upper()} on {protocol_name} MNIST (seed={args.seed})"
    print_header(title)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Protocol: {args.protocol}")
    print(f"Weight format: {args.weight_format}")
    print(f"Epochs per task: {args.epochs_per_task}, Batch size: {args.batch_size}")
    print(f"AdamW: lr={args.lr}, weight_decay={args.weight_decay}")
    print()

    # ── Create tasks ──────────────────────────────────────────────
    from ph_neuro.training.continual import (
        create_permuted_mnist_tasks,
        create_split_mnist_tasks,
        make_backprop_predict_fn,
        run_continual_experiment,
    )
    from ph_neuro.training.data import get_mnist_full_test_loader

    if args.protocol == "split":
        tasks = create_split_mnist_tasks(batch_size=args.batch_size)
        global_test_loader = get_mnist_full_test_loader(batch_size=args.batch_size)
        print("Split MNIST tasks:")
    else:
        seeds = list(range(args.n_tasks))
        tasks = create_permuted_mnist_tasks(
            n_tasks=args.n_tasks, seeds=seeds, batch_size=args.batch_size
        )
        global_test_loader = None
        print("Permuted MNIST tasks:")

    for i, task in enumerate(tasks):
        n_train = len(task.train_loader.dataset)  # type: ignore[arg-type]
        print(f"  Task {i + 1}: {task.name} ({n_train} train samples)")
    print()

    # ── Build model & optimizer ───────────────────────────────────
    if args.weight_format == "ternary":
        model = _build_ternary_mlp(device)
    else:
        model = _build_fp16_mlp(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # ── Predict function ──────────────────────────────────────────
    # Device-aware wrapper: move data to device, return predictions on CPU
    base_predict_fn = make_backprop_predict_fn()

    def predict_fn(model, x):
        x = x.to(device)
        pred = base_predict_fn(model, x)
        return pred.cpu()

    # ── Train function ────────────────────────────────────────────
    def train_fn(model, task, task_idx):
        """Train on a single task."""
        return train_task(
            model,
            task.train_loader,
            optimizer,
            args.epochs_per_task,
            device,
            task_idx,
            task.name,
        )

    # ── Weight snapshot callback (ternary only) ───────────────────
    weight_snapshots: dict[int, dict] = {}

    if args.weight_format == "ternary":
        # Record initial weight stats
        weight_snapshots[-1] = _compute_ternary_weight_stats(model)

        def record_weight_fn(model, task_idx):
            stats = _compute_ternary_weight_stats(model)
            weight_snapshots[task_idx] = stats
            # Print weight summary
            sp = stats["weight_sparsity_pct"]
            pp = stats["weight_pos_pct"]
            np_ = stats["weight_neg_pct"]
            print(f"  Weights after task {task_idx + 1}: +1={pp:.1f}%  0={sp:.1f}%  -1={np_:.1f}%")
            return stats
    else:
        record_weight_fn = None

    # ── Run experiment ────────────────────────────────────────────
    print("Running continual learning experiment...")
    print("-" * 100)
    total_start = time.time()

    results = run_continual_experiment(
        model=model,
        tasks=tasks,
        train_fn=train_fn,
        predict_fn=predict_fn,
        global_test_loader=global_test_loader,
        record_weight_fn=record_weight_fn,
    )

    total_time = time.time() - total_start
    print("-" * 100)
    print(f"\nTotal time: {total_time:.1f}s")
    print()

    # ── Print summary ─────────────────────────────────────────────
    metrics = results["metrics"]
    print(f"Average accuracy: {100 * metrics['average_accuracy']:.2f}%")
    print(f"Average forgetting: {100 * metrics['average_forgetting']:.2f}%")
    print()

    if metrics["per_task_accuracy"]:
        print("Per-task final accuracy:")
        for i, acc in enumerate(metrics["per_task_accuracy"]):
            print(f"  Task {i + 1}: {100 * acc:.2f}%")
    print()

    if metrics["per_task_forgetting"]:
        print("Per-task forgetting:")
        for i, fgt in enumerate(metrics["per_task_forgetting"]):
            print(f"  Task {i + 1}: {100 * fgt:.2f}%")
    print()

    if results.get("global_accuracies"):
        print("Global accuracies (10-class):")
        for i, ga in enumerate(results["global_accuracies"]):
            print(f"  After task {i + 1}: {100 * ga:.2f}%")
    print()

    # Also print accuracy matrix in readable format
    print("Accuracy matrix (rows=tasks trained, cols=tasks evaluated):")
    acc_matrix = results["accuracy_matrix"]
    for i, row in enumerate(acc_matrix):
        formatted = "  ".join(f"{100 * v:5.2f}%" for v in row)
        print(f"  After task {i + 1}: {formatted}")

    # ── Build result dict ─────────────────────────────────────────
    result = {
        "experiment": "L8 Forgetting Baseline",
        "protocol": args.protocol,
        "weight_format": args.weight_format,
        "seed": args.seed,
        "device": str(device),
        "epochs_per_task": args.epochs_per_task,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "n_tasks": len(tasks),
        "n_parameters": n_params,
        "total_training_time_seconds": total_time,
        "accuracy_matrix": results["accuracy_matrix"],
        "per_task_accuracies": results.get("per_task_accuracies", {}),
        "global_accuracies": results.get("global_accuracies", []),
        "metrics": {
            "average_accuracy": metrics["average_accuracy"],
            "average_forgetting": metrics["average_forgetting"],
            "per_task_accuracy": metrics["per_task_accuracy"],
            "per_task_forgetting": metrics["per_task_forgetting"],
        },
        "training_metrics": results.get("training_metrics", []),
    }

    # Add weight snapshots (ternary only)
    if args.weight_format == "ternary":
        # Convert snapshots to serializable format
        serializable_snapshots: dict[str, dict] = {}
        for k, v in weight_snapshots.items():
            serializable_snapshots[str(k)] = v
        result["weight_snapshots"] = serializable_snapshots

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.protocol}_{args.weight_format}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print()
    print(f"Results saved to: {output_path}")
    print(
        f"Summary: {args.weight_format} on {args.protocol} MNIST — "
        f"Avg Forgetting: {100 * metrics['average_forgetting']:.2f}%, "
        f"Avg Accuracy: {100 * metrics['average_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
