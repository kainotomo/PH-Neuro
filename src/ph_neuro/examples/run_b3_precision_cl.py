#!/usr/bin/env python3
"""B3: Precision Comparison for Continual Learning — experiment runner.

Measures catastrophic forgetting of standard SGD training under four
weight precisions on Split MNIST and Permuted MNIST:

    - ``ternary``  → {-1, 0, +1} ternary weights (STE), PH-Neuro's method
    - ``int8``     → 8-bit fake-quantized weights (QAT + STE)
    - ``int4``     → 4-bit fake-quantized weights (QAT + STE)
    - ``fp16``     → full float16 precision (upper bound)

This replicates the core question of "When Less is More"
(arXiv:2512.18934 — quantization noise as implicit regularization for
continual learning) and extends the precision ladder down to **ternary**:
if stronger quantization reduces catastrophic forgetting, ternary (the
strongest quantization) should forget the least.

**Reuse:** this runner is a thin generalization of the L8 forgetting
baseline. It reuses the L8 model builders / training loop / task
infrastructure and adds the QAT model builders already provided by
``ph_neuro.training.qat_helpers``. The ``ternary`` and ``fp16`` runs
were already produced by L8 (``l8_results/``) with identical
hyperparameters and seeds — B3 only re-runs the two new precisions
(``int8``, ``int4``) into ``b3_results/``, and the aggregator merges
them with the L8 baselines.

Usage::

    # INT8 QAT on Split MNIST (5 tasks)
    python -m ph_neuro.examples.run_b3_precision_cl \\
        --protocol split --weight-format int8 --seed 42

    # INT4 QAT on Permuted MNIST (10 tasks)
    python -m ph_neuro.examples.run_b3_precision_cl \\
        --protocol permuted --weight-format int4 --seed 42

    # Quick smoke test (2 tasks, 1 epoch each)
    python -m ph_neuro.examples.run_b3_precision_cl \\
        --protocol split --weight-format int8 --epochs-per-task 1 --seed 42

Output:
    JSON file per run: ``{output_dir}/{protocol}_{weight_format}_seed{seed}.json``
    with the same schema as L8 so results are directly comparable.

See Also:
    ``aggregate_b3_results.py`` — collects B3 + L8 results into the
    four-way precision comparison table.
    ``run_l8_forgetting_baseline.py`` — the L8 control (ternary + fp16).
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

# Reuse the L8 FP16 model builder, weight-statistics helper and the
# shared training loop so B3 results are directly comparable with L8.
from ph_neuro.examples.run_l8_forgetting_baseline import (
    _build_fp16_mlp,
    _compute_ternary_weight_stats,
    train_task,
)

# Architecture shared by all four precisions (matches L1/L8/B1/B2).
ARCH = [784, 512, 256, 10]

# Precision label for headers / JSON.
WEIGHT_FORMATS = ("ternary", "fp16", "int8", "int4")


# ── Model builder ──────────────────────────────────────────────────


def build_model(
    weight_format: str,
    device: torch.device,
) -> nn.Module:
    """Build a 3-layer MLP (784→512→256→10) for the given weight precision.

    All four precisions use the identical architecture (hidden ReLU +
    BatchNorm), differing only in how the weights are stored/quantized.

    Args:
        weight_format: One of ``"ternary"``, ``"fp16"``, ``"int8"``, ``"int4"``.
        device: Torch device.

    Returns:
        An ``nn.Module`` producing 10-class logits.
    """
    if weight_format == "ternary":
        from ph_neuro.models.ste_models import ste_mlp

        return ste_mlp(ARCH, device=device)
    if weight_format == "fp16":
        return _build_fp16_mlp(device)
    if weight_format == "int8":
        from ph_neuro.training.qat_helpers import create_int8_qat_mlp

        return create_int8_qat_mlp(ARCH, device=device)
    if weight_format == "int4":
        from ph_neuro.training.qat_helpers import create_int4_qat_mlp

        return create_int4_qat_mlp(ARCH, device=device)
    raise ValueError(f"Unknown weight format: {weight_format!r} (expected {list(WEIGHT_FORMATS)})")


# ── Weight statistics ───────────────────────────────────────────────


@torch.no_grad()
def _compute_float_weight_stats(model: nn.Module) -> dict[str, float]:
    """Extract raw-float weight statistics for FP16 / QAT latent weights.

    Returns:
        Dict with ``weight_sparsity_pct`` (fraction with ``|w| < 0.01``),
        ``weight_mean_abs`` and ``n_parameters``.
    """
    total_w = 0
    total_small = 0
    total_abs = 0.0
    for p in model.parameters():
        total_w += p.numel()
        total_abs += float(p.abs().sum().item())
        total_small += int((p.abs() < 0.01).sum().item())
    return {
        "weight_sparsity_pct": 100.0 * total_small / max(total_w, 1),
        "weight_mean_abs": total_abs / max(total_w, 1),
        "n_parameters": float(total_w),
    }


@torch.no_grad()
def _compute_quant_weight_stats(model: nn.Module, num_bits: int) -> dict[str, float]:
    """Extract fake-quantized weight level distribution for QAT models.

    Applies the same per-tensor symmetric quantizer used in
    ``qat_helpers.fake_quantize_ste`` and counts how many weights round
    to each level (0 / positive / negative). Mirrors the ternary
    ``+1/0/-1`` distribution so sparsity is comparable across precisions.

    Args:
        model: QAT MLP (``_QuantizedLinear`` layers carry ``num_bits``).
        num_bits: 8 (INT8) or 4 (INT4).

    Returns:
        Dict with ``weight_zero_pct``, ``weight_pos_pct``,
        ``weight_neg_pct``, ``weight_sparsity_pct`` (= zero pct) and
        ``n_parameters``.
    """
    qmax = 2 ** (num_bits - 1) - 1
    total_w = 0
    total_zero = 0
    total_pos = 0
    total_neg = 0

    for module in model.modules():
        if getattr(module, "num_bits", None) != num_bits:
            continue
        w = module.weight.flatten()
        if w.numel() == 0:
            continue
        abs_max = w.abs().max().clamp(min=1e-8)
        scale = abs_max / float(qmax)
        wq = torch.round(w / scale)
        total_w += w.numel()
        total_zero += int((wq == 0).sum().item())
        total_pos += int((wq > 0).sum().item())
        total_neg += int((wq < 0).sum().item())

    stats: dict[str, float] = {
        "weight_zero_pct": 0.0,
        "weight_pos_pct": 0.0,
        "weight_neg_pct": 0.0,
        "weight_sparsity_pct": 0.0,
        "n_parameters": float(total_w),
    }
    if total_w > 0:
        stats["weight_zero_pct"] = 100.0 * total_zero / total_w
        stats["weight_pos_pct"] = 100.0 * total_pos / total_w
        stats["weight_neg_pct"] = 100.0 * total_neg / total_w
        stats["weight_sparsity_pct"] = stats["weight_zero_pct"]
    return stats


def _compute_weight_stats(model: nn.Module, weight_format: str) -> dict[str, float]:
    """Unified per-format weight statistics recorder."""
    if weight_format == "ternary":
        return _compute_ternary_weight_stats(model)
    if weight_format in ("int8", "int4"):
        num_bits = 8 if weight_format == "int8" else 4
        return _compute_quant_weight_stats(model, num_bits)
    return _compute_float_weight_stats(model)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="B3: Precision Comparison for Continual Learning — "
        "ternary (STE) vs INT8 (QAT) vs INT4 (QAT) vs FP16",
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
        choices=list(WEIGHT_FORMATS),
        help="Weight precision: ternary (STE), fp16, int8 (QAT), int4 (QAT)",
    )
    parser.add_argument("--epochs-per-task", type=int, default=10, help="Epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--n-tasks", type=int, default=5, help="Number of tasks (permuted only)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 = deterministic)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="b3_results",
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
    """Run the B3 precision comparison experiment."""
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
    title = (
        f"B3: {args.weight_format.upper()} on {protocol_name} MNIST "
        f"(seed={args.seed}) — precision comparison"
    )
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
        global_test_loader = None
        print("Permuted MNIST tasks:")

    for i, task in enumerate(tasks):
        n_train = len(task.train_loader.dataset)  # type: ignore[arg-type]
        print(f"  Task {i + 1}: {task.name} ({n_train} train samples)")
    print()

    # ── Build model & optimizer ───────────────────────────────────
    model = build_model(args.weight_format, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
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

    # ── Weight snapshot callback (all formats) ────────────────────
    weight_snapshots: dict[int, dict] = {}
    weight_snapshots[-1] = _compute_weight_stats(model, args.weight_format)

    def record_weight_fn(model, task_idx):
        stats = _compute_weight_stats(model, args.weight_format)
        weight_snapshots[task_idx] = stats
        # Compact per-format weight summary line.
        # QAT stats carry quantized +/-/0 level distribution (like ternary);
        # FP16 falls back to the small-magnitude fraction.
        sp = stats["weight_sparsity_pct"]
        pp = stats.get("weight_pos_pct", 0.0)
        np_ = stats.get("weight_neg_pct", 0.0)
        if args.weight_format == "ternary":
            print(f"  Weights after task {task_idx + 1}: +1={pp:.1f}%  0={sp:.1f}%  -1={np_:.1f}%")
        else:
            print(f"  Weights after task {task_idx + 1}: +lvl={pp:.1f}%  0={sp:.1f}%  -lvl={np_:.1f}%")
        return stats

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

    # ── Build result dict (L8-compatible schema) ──────────────────
    result = {
        "experiment": "B3 Precision Comparison",
        "protocol": args.protocol,
        "weight_format": args.weight_format,
        "seed": args.seed,
        "device": str(device),
        "epochs_per_task": args.epochs_per_task,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
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
