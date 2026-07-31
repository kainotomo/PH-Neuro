#!/usr/bin/env python3
"""B2: QLoRA + Frozen Ternary Backbone — experiment runner.

Measures whether a **frozen ternary STE backbone** augmented with per-task
**LoRA (Low-Rank Adaptation)** adapters can achieve continual learning
with **zero forgetting by design** on Split MNIST and Permuted MNIST.

Core idea (inspired by the TOM Accelerator, arXiv:2602.20662):

1. **Pre-train** a ternary STE MLP on full MNIST (10-class). The resulting
   ternary weights are then **frozen** — they never change again.
2. For **each task**, build a fresh LoRA adapter pair on every linear layer
   (``TernarySTELoRALinear``) using the frozen backbone, and train **only**
   the LoRA ``A``/``B`` matrices on that task's data.
3. Store each task's LoRA weights separately. Because the backbone never
   changes, earlier-task accuracy can never degrade → zero forgetting.

Two pre-training protocols:

- ``--pretrain full``  → pre-train on full MNIST for ``--epochs-pretrain``
  (default 10) epochs. Tests LoRA's ability to adapt a strong backbone.
- ``--pretrain task1`` → pre-train on full MNIST for just **1 epoch**
  (simulates limited initial data / a weakly-trained deployment backbone).

Usage::

    # Full-MNIST pre-trained backbone, LoRA rank 8, Split MNIST
    python -m ph_neuro.examples.run_b2_qlora \\
        --protocol split --pretrain full --lora-r 8 --seed 42

    # Task-1-limited backbone, LoRA rank 4, Permuted MNIST
    python -m ph_neuro.examples.run_b2_qlora \\
        --protocol permuted --pretrain task1 --lora-r 4 --seed 42

    # Quick smoke test (2 tasks, 1 epoch pre-train, 1 epoch per task)
    python -m ph_neuro.examples.run_b2_qlora \\
        --protocol split --pretrain task1 --lora-r 2 \\
        --epochs-per-task 1 --seed 42

Output:
    - ``{output_dir}/{protocol}_{pretrain}_qlora_r{r}_seed{seed}.json``
      per-run metrics (compatible with the B1/L8 result format).
    - ``{output_dir}/lora/backbone.pt`` — frozen backbone state dict.
    - ``{output_dir}/lora/task{k}.pt`` — per-task LoRA adapter state.

See Also:
    ``aggregate_b2_results.py`` — collects and visualizes all results.
    ``run_b1_ewc.py`` / ``run_l8_forgetting_baseline.py`` — comparison runs.
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

from ph_neuro.analysis.continual import evaluate_continual_learning
from ph_neuro.examples._utils import print_header
from ph_neuro.layers.ste_lora import (
    count_lora_parameters,
    freeze_backbone,
    get_model_lora_state,
    iter_lora_layers,
)
from ph_neuro.models.ste_models import ste_mlp_lora

# Reuse the L8 model builder and weight-statistics helper so the B2 results
# are directly comparable with the L8 control and B1 EWC baselines.
from ph_neuro.examples.run_l8_forgetting_baseline import (
    _build_ternary_mlp,
    _compute_ternary_weight_stats,
)

# Architecture shared by the backbone and the LoRA model (matches L1/L8/B1).
ARCH = [784, 512, 256, 10]


# ── Backbone helpers ───────────────────────────────────────────────


def _copy_backbone_state(plain_model: nn.Module, lora_model: nn.Module) -> None:
    """Copy backbone params (latent_scores, bias, BatchNorm) into a LoRA model.

    ``ste_mlp`` and ``ste_mlp_lora`` produce ``nn.Sequential`` modules with
    identical layer indices, so every backbone parameter name in the plain
    model's state dict also exists (with the same shape) in the LoRA model.
    LoRA-only keys (``*.lora_A``, ``*.lora_B``) are absent from the plain
    state dict and are therefore left untouched.

    Args:
        plain_model: Pre-trained ternary STE model (``ste_mlp``).
        lora_model: LoRA model (``ste_mlp_lora``) to load weights into.
    """
    plain_state = plain_model.state_dict()
    lora_state = lora_model.state_dict()
    for name, value in plain_state.items():
        if name in lora_state and lora_state[name].shape == value.shape:
            lora_state[name].copy_(value)


def train_backbone(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    label: str = "backbone",
) -> dict[str, float]:
    """Train the ternary backbone on full MNIST (BatchNorm active)."""
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
        train_acc = correct / max(total, 1)
        final_metrics = {
            "final_loss": total_loss / max(total, 1),
            "final_train_acc": train_acc,
            "training_time": time.time() - total_start,
        }
        print(
            f"  Pretrain {label}: Epoch {epoch:2d}/{epochs}  "
            f"Loss: {final_metrics['final_loss']:.4f}  "
            f"Train Acc: {100 * train_acc:5.2f}%  "
            f"Time: {time.time() - epoch_start:.1f}s"
        )
    print(f"  -> Pretrain {label} done.")
    return final_metrics


# ── LoRA training ──────────────────────────────────────────────────


def train_lora_task(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    task_idx: int,
    task_name: str,
) -> dict[str, float]:
    """Train LoRA adapters on a single task with the backbone frozen.

    The model is kept in ``eval()`` mode so BatchNorm running statistics
    (part of the frozen backbone) never update. Only LoRA parameters have
    ``requires_grad=True``, so ``backward()`` leaves the backbone untouched.

    Returns:
        Dict with metrics (final_loss, final_train_acc, training_time).
    """
    model.eval()
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
        train_acc = correct / max(total, 1)
        final_metrics = {
            "final_loss": total_loss / max(total, 1),
            "final_train_acc": train_acc,
            "training_time": time.time() - total_start,
        }
        print(
            f"    LoRA Task {task_idx + 1} ({task_name}), "
            f"Epoch {epoch:2d}/{epochs}  "
            f"Loss: {final_metrics['final_loss']:.4f}  "
            f"Train Acc: {100 * train_acc:5.2f}%  "
            f"Time: {time.time() - epoch_start:.1f}s"
        )
    print(f"    -> LoRA Task {task_idx + 1} done.")
    return final_metrics


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate classification accuracy on a loader (frozen BN)."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ── CLI ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="B2: QLoRA + Frozen Ternary Backbone — continual learning",
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
        "--pretrain",
        type=str,
        choices=["full", "task1"],
        default="full",
        help="full: pre-train on full MNIST for epochs-pretrain epochs; "
        "task1: pre-train for just 1 epoch (limited-data simulation)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=4,
        help="LoRA rank (number of low-rank dimensions)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA scaling constant (defaults to the rank r)",
    )
    parser.add_argument("--epochs-pretrain", type=int, default=10, help="Backbone pre-training epochs")
    parser.add_argument("--epochs-per-task", type=int, default=10, help="LoRA epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--n-tasks", type=int, default=10, help="Number of tasks (permuted only)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="b2_results",
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
    """Run the B2 QLoRA experiment."""
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

    # task1 protocol = limited-data simulation → 1 pre-training epoch.
    pretrain_epochs = args.epochs_pretrain
    if args.pretrain == "task1":
        pretrain_epochs = min(pretrain_epochs, 1)

    alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_r

    protocol_name = "Split" if args.protocol == "split" else "Permuted"
    title = (
        f"B2: QLoRA (r={args.lora_r}, α={alpha:g}) on frozen ternary backbone "
        f"| {protocol_name} MNIST | pretrain={args.pretrain} ({pretrain_epochs} ep) "
        f"| seed={args.seed}"
    )
    print_header(title)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Protocol: {args.protocol}")
    print(f"Pretrain: {args.pretrain} ({pretrain_epochs} epoch(s) on full MNIST)")
    print(f"LoRA: r={args.lora_r}, alpha={alpha:g}")
    print(f"LoRA epochs per task: {args.epochs_per_task}, Batch size: {args.batch_size}")
    print(f"AdamW: lr={args.lr}, weight_decay={args.weight_decay}")
    print()

    # ── Create tasks ──────────────────────────────────────────────
    from ph_neuro.training.continual import (
        create_permuted_mnist_tasks,
        create_split_mnist_tasks,
    )
    from ph_neuro.training.data import get_mnist_full_test_loader, get_mnist_loaders

    if args.protocol == "split":
        tasks = create_split_mnist_tasks(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        global_test_loader = get_mnist_full_test_loader(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        print("Split MNIST tasks:")
    else:
        seeds = list(range(args.n_tasks))
        tasks = create_permuted_mnist_tasks(
            n_tasks=args.n_tasks,
            batch_size=args.batch_size,
            seeds=seeds,
            num_workers=args.num_workers,
        )
        global_test_loader = get_mnist_full_test_loader(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        print("Permuted MNIST tasks:")
    for i, task in enumerate(tasks):
        print(f"  Task {i + 1}: {task.name}")

    # ── Phase 1: pre-train the ternary backbone ──────────────────
    print("\nPhase 1: Pre-training frozen ternary backbone")
    print("-" * 100)
    backbone = _build_ternary_mlp(device)
    pretrain_train_loader, _ = get_mnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    backbone_optimizer = torch.optim.AdamW(
        backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pretrain_metrics = train_backbone(
        backbone, pretrain_train_loader, backbone_optimizer, pretrain_epochs, device
    )

    backbone_test_acc = evaluate_model(backbone, global_test_loader, device)
    backbone_stats = _compute_ternary_weight_stats(backbone)
    n_params = int(backbone_stats["n_parameters"])
    n_lora_params = 0

    print(f"\n  Frozen backbone: full-MNIST test acc = {100 * backbone_test_acc:.2f}%")
    print(
        f"  Ternary weight distribution: +1={backbone_stats['weight_pos_pct']:.1f}%  "
        f"0={backbone_stats['weight_sparsity_pct']:.1f}%  "
        f"-1={backbone_stats['weight_neg_pct']:.1f}%"
    )

    # Save the frozen backbone for reproducibility
    lora_dir = os.path.join(args.output_dir, "lora")
    os.makedirs(lora_dir, exist_ok=True)
    torch.save(
        {k: v.clone() for k, v in backbone.state_dict().items()},
        os.path.join(lora_dir, "backbone.pt"),
    )

    # ── Phase 2: per-task LoRA training ──────────────────────────
    print("\nPhase 2: Per-task LoRA training (backbone frozen)")
    print("-" * 100)
    total_start = time.time()

    cross_matrix: list[list[float]] = []   # cross_matrix[i][j] = acc(adapter_i on task_j)
    self_accs: list[float] = []            # self_accs[i] = acc(adapter_i on task_i)
    global_accs: list[float] = []          # global_accs[i] = acc(adapter_i on 10-class test)
    training_metrics: list[dict[str, float]] = []
    adapter_params: list[int] = []

    for task_idx, task in enumerate(tasks):
        # Fresh LoRA model with the frozen backbone loaded
        model = ste_mlp_lora(ARCH, r=args.lora_r, alpha=alpha, device=device)
        _copy_backbone_state(backbone, model)
        freeze_backbone(model)
        model.eval()  # freeze BatchNorm running stats

        if task_idx == 0:
            n_lora_params = count_lora_parameters(model)
            print(f"\n  Backbone params: {n_params:,}  LoRA params (r={args.lora_r}): {n_lora_params:,}")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        metrics = train_lora_task(
            model,
            task.train_loader,
            optimizer,
            args.epochs_per_task,
            device,
            task_idx,
            task.name,
        )
        training_metrics.append(metrics)
        adapter_params.append(n_lora_params)

        # Save this task's LoRA adapter
        torch.save(
            get_model_lora_state(model),
            os.path.join(lora_dir, f"task{task_idx}.pt"),
        )

        # Evaluate adapter_{task_idx} on all tasks seen so far
        row: list[float] = []
        for eval_task_idx in range(task_idx + 1):
            loader = tasks[eval_task_idx].test_loaders.get(eval_task_idx)
            if loader is not None:
                acc = evaluate_model(model, loader, device)
            else:
                acc = 0.0
            row.append(acc)
        cross_matrix.append(row)
        self_accs.append(row[task_idx])

        if global_test_loader is not None:
            global_accs.append(evaluate_model(model, global_test_loader, device))

        formatted = "  ".join(f"{100 * v:5.2f}%" for v in row)
        print(
            f"  After LoRA task {task_idx + 1}: acc = {formatted}"
            f"  | global = {100 * global_accs[-1]:.2f}%"
        )

    total_time = time.time() - total_start
    print("-" * 100)
    print(f"\nTotal LoRA training time: {total_time:.1f}s")

    # ── Build standard CL accuracy matrix ────────────────────────
    # accuracy_matrix[i][j] = accuracy on task j after training task i.
    # Because adapters are per-task and independent, this equals the
    # self-accuracy of task j for every row i >= j → zero forgetting.
    accuracy_matrix: list[list[float]] = []
    for i in range(len(tasks)):
        accuracy_matrix.append([self_accs[j] for j in range(i + 1)])

    metrics = evaluate_continual_learning(accuracy_matrix)

    # ── Print summary ─────────────────────────────────────────────
    print()
    print(f"Average accuracy: {100 * metrics['average_accuracy']:.2f}%")
    print(f"Average forgetting: {100 * metrics['average_forgetting']:.2f}%")
    print()
    if metrics["per_task_accuracy"]:
        print("Per-task final accuracy:")
        for i, acc in enumerate(metrics["per_task_accuracy"]):
            print(f"  Task {i + 1}: {100 * acc:.2f}%")
    print()
    print("Cross-task accuracy matrix (rows=adapter, cols=task):")
    for i, row in enumerate(cross_matrix):
        formatted = "  ".join(f"{100 * v:5.2f}%" for v in row)
        print(f"  Adapter {i + 1}: {formatted}")
    print()
    if global_accs:
        print("Global 10-class accuracies (per adapter):")
        for i, ga in enumerate(global_accs):
            print(f"  Adapter {i + 1}: {100 * ga:.2f}%")

    # ── Build result dict ─────────────────────────────────────────
    result = {
        "experiment": "B2 QLoRA + Frozen Ternary Backbone",
        "protocol": args.protocol,
        "pretrain_protocol": args.pretrain,
        "weight_format": "ternary",
        "lora_rank": args.lora_r,
        "lora_alpha": alpha,
        "seed": args.seed,
        "device": str(device),
        "epochs_pretrain": pretrain_epochs,
        "epochs_per_task": args.epochs_per_task,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "n_tasks": len(tasks),
        "n_parameters": n_params,
        "n_lora_parameters": n_lora_params,
        "total_training_time_seconds": total_time,
        "backbone_test_accuracy": backbone_test_acc,
        "backbone_weight_stats": backbone_stats,
        "accuracy_matrix": accuracy_matrix,
        "cross_task_accuracy_matrix": cross_matrix,
        "per_task_accuracies": {
            f"after_task_{i}": row for i, row in enumerate(accuracy_matrix)
        },
        "global_accuracies": global_accs,
        "metrics": {
            "average_accuracy": metrics["average_accuracy"],
            "average_forgetting": metrics["average_forgetting"],
            "per_task_accuracy": metrics["per_task_accuracy"],
            "per_task_forgetting": metrics["per_task_forgetting"],
        },
        "training_metrics": training_metrics,
    }

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.protocol}_{args.pretrain}_qlora_r{args.lora_r}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print()
    print(f"Results saved to: {output_path}")
    print(
        f"Summary: QLoRA (r={args.lora_r}) on frozen ternary backbone "
        f"({args.pretrain} pretrain) — {protocol_name} MNIST: "
        f"Avg Forgetting: {100 * metrics['average_forgetting']:.2f}%, "
        f"Avg Accuracy: {100 * metrics['average_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
