"""Permuted MNIST continual learning experiment — Phase 1.3.

Each task uses all 10 MNIST digits but with a different random pixel
permutation. Tests whether the network can learn 5 completely different
input-output mappings without interference.

Uses a single :class:`~ph_neuro.training.supervised.SupervisedHebbianClassifier`
(784 → 10) trained on each permutation sequentially.

Expected result:
    - Average forgetting <10% (target)
    - Backprop baseline >50% forgetting (run separately with
      :mod:`~ph_neuro.examples.backprop_baseline`)

Usage:
    python -m ph_neuro.examples.permuted_mnist_continual
    python -m ph_neuro.examples.permuted_mnist_continual --n-tasks 3 --epochs-per-task 10
"""

from __future__ import annotations

import argparse
import time

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.training.continual import (
    create_permuted_mnist_tasks,
    make_hebbian_predict_fn,
    run_continual_experiment,
)
from ph_neuro.training.supervised import SupervisedHebbianClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Permuted MNIST continual learning with ternary Hebbian",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of permutation tasks")
    parser.add_argument("--epochs-per-task", type=int, default=5, help="Epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Hebbian learning rate")
    parser.add_argument("--decay", type=float, default=0.0, help="Homeostatic decay rate")
    parser.add_argument("--theta-upper", type=float, default=5.0, help="Hysteresis upper threshold")
    parser.add_argument("--theta-lower", type=float, default=1.0, help="Hysteresis lower threshold")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Dead-zone for ternary_sign")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the permuted MNIST continual learning experiment."""
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print_header("PH-Neuro Phase 1.3 — Permuted MNIST Continual Learning (Hebbian)")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Tasks: {args.n_tasks}, Epochs per task: {args.epochs_per_task}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}, Decay: {args.decay}, Epsilon: {args.epsilon}")
    print(f"Theta upper: {args.theta_upper}, Theta lower: {args.theta_lower}")
    print()

    # ── Create tasks ──────────────────────────────────────────────
    print("Creating permuted MNIST tasks...")
    seeds = list(range(args.n_tasks))
    tasks = create_permuted_mnist_tasks(
        n_tasks=args.n_tasks, seeds=seeds, batch_size=args.batch_size
    )
    for i, task in enumerate(tasks):
        n_train = len(task.train_loader.dataset)  # type: ignore[arg-type]
        print(f"  Task {i + 1}: {task.name} ({n_train} train samples)")
    print()

    # ── Model ─────────────────────────────────────────────────────
    classifier = SupervisedHebbianClassifier(
        in_features=784,
        out_features=10,
        theta_upper=args.theta_upper,
        theta_lower=args.theta_lower,
        device=device,
    )
    print(f"Model: {classifier}")
    print(f"Weights: {classifier.model.weight.unpack().numel():,} ternary params")
    print()

    # ── Predict function ──────────────────────────────────────────
    predict_fn = make_hebbian_predict_fn(epsilon=args.epsilon)

    # ── Train function ────────────────────────────────────────────
    def train_fn(model, task, task_idx):
        """Train on a single permuted task."""
        metrics_list = []
        start = time.time()
        for epoch in range(1, args.epochs_per_task + 1):
            epoch_correct = 0
            epoch_total = 0
            for x, y in task.train_loader:
                with torch.no_grad():
                    step_metrics = model.train_step(
                        x, y, lr=args.lr, decay=args.decay, epsilon=args.epsilon
                    )
                    metrics_list.append(step_metrics)
                pred = model.predict(x, epsilon=args.epsilon)
                y_d = y.to(device)
                epoch_correct += (pred == y_d).sum().item()
                epoch_total += y_d.size(0)

            epoch_acc = epoch_correct / max(epoch_total, 1)
            avg_flip = sum(m["flip_rate"] for m in metrics_list[-10:]) / max(
                len(metrics_list[-10:]), 1
            )
            print(
                f"    Task {task_idx + 1}, Epoch {epoch:2d}/{args.epochs_per_task}  "
                f"Train acc: {100 * epoch_acc:5.2f}%  "
                f"Flips: {100 * avg_flip:5.2f}%/step  "
                f"Time: {time.time() - start:.1f}s"
            )

        w = model.get_weight_stats()
        print(f"    -> Task {task_idx + 1} done. Weights: "
              f"+{w['pos_pct']:.1f}% -{w['neg_pct']:.1f}% 0{w['zero_pct']:.1f}%")

        return {
            "avg_flip_rate": sum(m["flip_rate"] for m in metrics_list) / max(len(metrics_list), 1),
            "final_train_acc": epoch_acc,
        }

    # ── Record weight function ────────────────────────────────────
    def record_weight_fn(model, task_idx):
        """Record weight statistics after training a task."""
        base = model.get_weight_stats()
        w = model.model.weight.unpack()
        per_neuron = {}
        for i in range(10):
            n = w[i]
            total = n.numel()
            per_neuron[f"neuron_{i}"] = {
                "pos_pct": 100.0 * (n == 1).sum().item() / max(total, 1),
                "neg_pct": 100.0 * (n == -1).sum().item() / max(total, 1),
                "zero_pct": 100.0 * (n == 0).sum().item() / max(total, 1),
            }
        return {**base, "per_neuron": per_neuron}

    # ── Run experiment ────────────────────────────────────────────
    print("\nRunning continual learning experiment...")
    print("-" * 100)
    total_start = time.time()

    results = run_continual_experiment(
        model=classifier,
        tasks=tasks,
        train_fn=train_fn,
        predict_fn=predict_fn,
        record_weight_fn=record_weight_fn,
    )

    total_time = time.time() - total_start
    metrics = results["metrics"]

    # ── Results ───────────────────────────────────────────────────
    print()
    print_header("Results")

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
        print(f"  Task {j + 1} (seed={seeds[j]}): {100 * f_val:.2f}%")

    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}min)")

    # Interpretation
    avg_forget = metrics["average_forgetting"]
    if avg_forget < 0.10:
        print("✅ SUCCESS: Forgetting <10% (target met)")
    elif avg_forget < 0.15:
        print("⚠️  PARTIAL: Forgetting <15% but >10% (below target)")
    else:
        print(f"❌ FAILURE: Forgetting {100 * avg_forget:.1f}% > 10% (target not met)")

    print(f"\nTip: Compare with backprop baseline:")
    print(f"  python -m ph_neuro.examples.backprop_baseline --protocol permuted")

    torch.save(
        {
            "accuracy_matrix": results["accuracy_matrix"],
            "metrics": metrics,
            "config": vars(args),
            "weight_snapshots": results["weight_snapshots"],
        },
        "permuted_mnist_hebbian_results.pt",
    )
    print("\nResults saved to permuted_mnist_hebbian_results.pt")


if __name__ == "__main__":
    main()
