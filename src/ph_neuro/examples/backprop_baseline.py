"""Backprop baseline for continual learning experiments — Phase 1.3.

Provides a standard backprop + SGD comparison against ternary Hebbian
continual learning. Uses the same architecture (784 → 10 linear layer)
but with float weights, SGD optimizer, and cross-entropy loss.

Expected results:
    - Split MNIST: >40% average forgetting
    - Permuted MNIST: >50% average forgetting

Usage:
    # Split MNIST
    python -m ph_neuro.examples.backprop_baseline --protocol split

    # Permuted MNIST
    python -m ph_neuro.examples.backprop_baseline --protocol permuted

    # Compare with Hebbian
    python -m ph_neuro.examples.split_mnist_continual
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header
from ph_neuro.training.continual import (
    create_permuted_mnist_tasks,
    create_split_mnist_tasks,
    evaluate_on_task,
    format_comparison_table,
    make_backprop_predict_fn,
    run_continual_experiment,
)
from ph_neuro.training.data import get_mnist_full_test_loader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backprop baseline for continual learning comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="split",
        choices=["split", "permuted"],
        help="Continual learning protocol",
    )
    parser.add_argument("--epochs-per-task", type=int, default=5, help="Epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of tasks (permuted only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    return parser.parse_args()


# ── Simple linear model ────────────────────────────────────────────


class LinearBackprop(nn.Module):
    """Single linear layer trained with backprop.

    Matches the architecture of ``TernaryHebbianLinear(784, 10)`` but
    uses float weights, SGD, and cross-entropy loss — providing a
    clean comparison against Hebbian learning.
    """

    def __init__(self, in_features: int = 784, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.linear(x_flat)


# ── Training helpers ────────────────────────────────────────────────


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch. Returns loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_flat = x.view(x.size(0), -1)

        optimizer.zero_grad()
        out = model(x_flat)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": correct / max(total, 1),
    }


def main() -> None:
    """Run the backprop continual learning baseline."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    protocol_name = args.protocol.capitalize()
    print_header(f"Backprop Baseline — {protocol_name} MNIST Continual Learning")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Protocol: {args.protocol}")
    print(f"Epochs per task: {args.epochs_per_task}, Batch size: {args.batch_size}")
    print(f"SGD: lr={args.lr}, momentum={args.momentum}")
    print()

    # ── Create tasks ──────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────
    model = LinearBackprop(in_features=784, out_features=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print(f"Model: LinearBackprop(784 -> 10)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} float weights")
    print()

    # ── Predict function ──────────────────────────────────────────
    predict_fn = make_backprop_predict_fn()

    # ── Train function ────────────────────────────────────────────
    def train_fn(model, task, task_idx):
        """Train on a single task with backprop."""
        start = time.time()
        for epoch in range(1, args.epochs_per_task + 1):
            metrics = train_epoch(model, task.train_loader, optimizer, device)
            print(
                f"    Task {task_idx + 1}, Epoch {epoch:2d}/{args.epochs_per_task}  "
                f"Loss: {metrics['loss']:.4f}  "
                f"Train acc: {100 * metrics['acc']:5.2f}%  "
                f"Time: {time.time() - start:.1f}s"
            )
        print(f"    -> Task {task_idx + 1} done.")
        return {"final_loss": metrics["loss"], "final_train_acc": metrics["acc"]}

    # ── Run experiment ────────────────────────────────────────────
    print("\nRunning continual learning experiment...")
    print("-" * 100)
    total_start = time.time()

    results = run_continual_experiment(
        model=model,
        tasks=tasks,
        train_fn=train_fn,
        predict_fn=predict_fn,
        global_test_loader=global_test_loader,
    )

    total_time = time.time() - total_start
    metrics = results["metrics"]

    # ── Results ───────────────────────────────────────────────────
    print()
    print_header("Results — Backprop Baseline")

    print("\nAccuracy matrix (rows=trained up to task, columns=evaluated on task):")
    print(f"{'':>12}", end="")
    for j in range(len(tasks)):
        print(f"{'Task ' + str(j + 1):>10}", end="")
    print()
    for i, row in enumerate(results["accuracy_matrix"]):
        print(f"{'After T' + str(i + 1):>12}", end="")
        for j, acc in enumerate(row):
            print(f"{100 * acc:>8.2f}% ", end="")
        print()

    print(f"\nAverage accuracy: {100 * metrics['average_accuracy']:.2f}%")
    print(f"Average forgetting: {100 * metrics['average_forgetting']:.2f}%")
    print(f"\nPer-task forgetting:")
    for j, f_val in enumerate(metrics["per_task_forgetting"]):
        print(f"  Task {j + 1}: {100 * f_val:.2f}%")

    if results["global_accuracies"]:
        print(f"\nGlobal 10-class accuracy progression:")
        for i, ga in enumerate(results["global_accuracies"]):
            print(f"  After task {i + 1}: {100 * ga:.2f}%")

    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}min)")

    # Interpretation
    avg_forget = metrics["average_forgetting"]
    if args.protocol == "split":
        if avg_forget > 0.40:
            print(f"\n✅ EXPECTED: Backprop forgetting {100 * avg_forget:.1f}% > 40% on split MNIST")
        else:
            print(f"\n⚠️  Backprop forgetting {100 * avg_forget:.1f}% < 40% (lower than expected)")
    else:
        if avg_forget > 0.50:
            print(f"\n✅ EXPECTED: Backprop forgetting {100 * avg_forget:.1f}% > 50% on permuted MNIST")
        else:
            print(f"\n⚠️  Backprop forgetting {100 * avg_forget:.1f}% < 50% (lower than expected)")

    # Save results
    suffix = args.protocol
    torch.save(
        {
            "accuracy_matrix": results["accuracy_matrix"],
            "metrics": metrics,
            "global_accuracies": results["global_accuracies"],
            "config": vars(args),
        },
        f"backprop_{suffix}_mnist_results.pt",
    )
    print(f"\nResults saved to backprop_{suffix}_mnist_results.pt")

    print(f"\nComparison command:")
    print(f"  # Load results and compare:")
    print(f"  hebb = torch.load('{suffix}_mnist_hebbian_results.pt')")
    print(f"  bp   = torch.load('backprop_{suffix}_mnist_results.pt')")
    print(f"  # Hebbian forgetting: {100 * avg_forget:.1f}%")
    print(f"  print(f'Backprop forgetting: {100 * avg_forget:.1f}%')")


if __name__ == "__main__":
    main()
