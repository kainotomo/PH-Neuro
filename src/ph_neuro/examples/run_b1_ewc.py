#!/usr/bin/env python3
"""B1: EWC + Ternary STE — experiment runner.

Measures whether Elastic Weight Consolidation (EWC) reduces catastrophic
forgetting of ternary STE networks on Split MNIST and Permuted MNIST.

EWC regularizes the **latent scores** (the differentiable float parameters
of ``TernarySTELinear``) after each task by penalizing movement away from
the consolidated reference, weighted by the diagonal Fisher Information.

This is the **first Track B (Continual Learning)** experiment. It is
compared against the L8 control baseline (no EWC, no replay).

Usage:
    # EWC on Split MNIST (5 tasks), lambda=10
    python -m ph_neuro.examples.run_b1_ewc \\
        --protocol split --ewc-lambda 10 --seed 42

    # EWC on Permuted MNIST (10 tasks), online mode
    python -m ph_neuro.examples.run_b1_ewc \\
        --protocol permuted --ewc-lambda 10 --seed 42

    # Multi-task EWC (store per-task Fisher) instead of online
    python -m ph_neuro.examples.run_b1_ewc \\
        --protocol split --ewc-lambda 10 --no-online --seed 42

    # Quick smoke test (2 tasks, 1 epoch each)
    python -m ph_neuro.examples.run_b1_ewc \\
        --protocol split --ewc-lambda 10 --epochs-per-task 1 \\
        --fisher-samples 50 --seed 42

Output:
    JSON file per run: ``{output_dir}/{protocol}_ewc_lambda{L}_seed{seed}.json``

See Also:
    ``aggregate_b1_results.py`` — collects and visualizes all results.
    ``run_l8_forgetting_baseline.py`` — the control experiment (no EWC).
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
from ph_neuro.training.ewc import MultiTaskEWC, OnlineEWC

# Reuse the L8 model builder and weight-statistics helper so the B1 results
# are directly comparable with the L8 control baseline.
from ph_neuro.examples.run_l8_forgetting_baseline import (
    _build_ternary_mlp,
    _compute_ternary_weight_stats,
)


# ── Training function (with optional EWC penalty) ──────────────────


def train_task_ewc(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    task_idx: int,
    task_name: str,
    ewc_penalty_fn=None,
) -> dict[str, float]:
    """Train model on a single task for a given number of epochs.

    Identical to the L8 ``train_task`` except that an optional EWC penalty
    term is added to the loss before ``backward()``.

    Args:
        model: The model to train.
        loader: Training data loader for this task.
        optimizer: Optimizer (AdamW).
        epochs: Number of epochs to train.
        device: Torch device.
        task_idx: Index of this task (0-based) for logging.
        task_name: Human-readable task name for logging.
        ewc_penalty_fn: Optional callable ``penalty_fn(model) -> Tensor``
            returning the EWC regularization loss. If ``None``, no penalty
            is applied (equivalent to the L8 baseline).

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
            if ewc_penalty_fn is not None:
                loss = loss + ewc_penalty_fn(model)
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
        description="B1: EWC + Ternary STE — continual learning with EWC",
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
        "--ewc-lambda",
        type=float,
        default=10.0,
        help="EWC regularization strength (0 disables the penalty)",
    )
    parser.add_argument(
        "--fisher-samples",
        type=int,
        default=500,
        help="Number of batches to sample for the Fisher estimate per task",
    )
    parser.add_argument(
        "--online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use online EWC (accumulate one Fisher); else store per-task",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Fisher accumulation factor for online EWC",
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
        default="b1_results",
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
    """Run the B1 EWC experiment."""
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

    protocol_name = "Split" if args.protocol == "split" else "Permuted"
    ewc_mode = "Online" if args.online else "Multi-task"
    title = (
        f"B1: {ewc_mode} EWC (λ={args.ewc_lambda:g}) on "
        f"{protocol_name} MNIST (seed={args.seed})"
    )
    print_header(title)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Protocol: {args.protocol}")
    print(f"EWC: lambda={args.ewc_lambda:g}, online={args.online}, gamma={args.gamma:g}")
    print(f"Fisher samples: {args.fisher_samples}")
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
    model = _build_ternary_mlp(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # ── EWC manager ───────────────────────────────────────────────
    ewc = (
        OnlineEWC(model, gamma=args.gamma) if args.online else MultiTaskEWC(model)
    )

    def penalty_fn(m: nn.Module) -> torch.Tensor:
        """Return the EWC penalty (0 until the first task is consolidated)."""
        if not ewc.has_penalty():
            return torch.zeros((), device=next(m.parameters()).device, dtype=torch.float32)
        return ewc.penalty(m, args.ewc_lambda)

    # ── Predict function ──────────────────────────────────────────
    # Device-aware wrapper: move data to device, return predictions on CPU
    base_predict_fn = make_backprop_predict_fn()

    def predict_fn(model, x):
        x = x.to(device)
        pred = base_predict_fn(model, x)
        return pred.cpu()

    # ── Train function (trains then consolidates EWC) ────────────
    def train_fn(model, task, task_idx):
        """Train on a single task, then consolidate it into the EWC state."""
        metrics = train_task_ewc(
            model,
            task.train_loader,
            optimizer,
            args.epochs_per_task,
            device,
            task_idx,
            task.name,
            ewc_penalty_fn=penalty_fn,
        )
        # Consolidate the task AFTER training: estimate Fisher on the
        # just-trained model and snapshot reference latent scores.
        ewc.update(task.train_loader, args.fisher_samples, device)
        print(
            f"    -> EWC consolidated {ewc.n_tasks} task(s); "
            f"lambda={args.ewc_lambda:g}"
        )
        return metrics

    # ── Weight snapshot callback (ternary only) ───────────────────
    weight_snapshots: dict[int, dict] = {}
    weight_snapshots[-1] = _compute_ternary_weight_stats(model)

    def record_weight_fn(model, task_idx):
        stats = _compute_ternary_weight_stats(model)
        weight_snapshots[task_idx] = stats
        sp = stats["weight_sparsity_pct"]
        pp = stats["weight_pos_pct"]
        np_ = stats["weight_neg_pct"]
        print(
            f"  Weights after task {task_idx + 1}: "
            f"+1={pp:.1f}%  0={sp:.1f}%  -1={np_:.1f}%"
        )
        return stats

    # ── Run experiment ────────────────────────────────────────────
    print("Running continual learning experiment with EWC...")
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
        "experiment": "B1 EWC + Ternary STE",
        "protocol": args.protocol,
        "weight_format": "ternary",
        "ewc_lambda": args.ewc_lambda,
        "ewc_online": args.online,
        "ewc_gamma": args.gamma,
        "fisher_samples": args.fisher_samples,
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
        "weight_snapshots": {str(k): v for k, v in weight_snapshots.items()},
    }

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.protocol}_ewc_lambda{args.ewc_lambda:g}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print()
    print(f"Results saved to: {output_path}")
    print(
        f"Summary: EWC (λ={args.ewc_lambda:g}) on {args.protocol} MNIST — "
        f"Avg Forgetting: {100 * metrics['average_forgetting']:.2f}%, "
        f"Avg Accuracy: {100 * metrics['average_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
