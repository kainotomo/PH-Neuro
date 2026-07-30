#!/usr/bin/env python3
"""L5: BatchNorm Fusion — experiment runner.

Trains a ternary STE model with BatchNorm, fuses BN into the preceding
ternary layers, then verifies output equivalence and benchmarks speedup.

Usage:
    # MLP on MNIST (default)
    python -m ph_neuro.examples.run_l5_bn_fusion

    # CNN on CIFAR-10
    python -m ph_neuro.examples.run_l5_bn_fusion \\
        --dataset cifar10 --arch cnn --epochs 50 --seed 42

    # Quick smoke test
    python -m ph_neuro.examples.run_l5_bn_fusion \\
        --dataset mnist --epochs 3 --benchmark-batches 10

Output:
    JSON file: ``{output_dir}/l5_bn_fusion_{dataset}_{arch}_seed{seed}.json``

See Also:
    ``fuse_bn_layers()`` in :mod:`ph_neuro.models.fuse_bn`
    ``FusedTernaryLinear`` / ``FusedTernaryConv2d`` in :mod:`ph_neuro.layers.fused_bn`
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
from ph_neuro.models.fuse_bn import fuse_bn_layers
from ph_neuro.models.ste_models import ste_cnn, ste_mlp

# ── Dataset registry ────────────────────────────────────────────────

DATASETS = {
    "mnist": {
        "n_classes": 10,
        "arch": "mlp",
        "layer_sizes": [784, 512, 256, 10],
        "epochs": 30,
        "batch_size": 128,
        "loader_fn": "get_mnist_loaders",
        "input_shape": (1, 28, 28),
    },
    "fashion": {
        "n_classes": 10,
        "arch": "mlp",
        "layer_sizes": [784, 512, 256, 10],
        "epochs": 30,
        "batch_size": 128,
        "loader_fn": "get_fashion_mnist_loaders",
        "input_shape": (1, 28, 28),
    },
    "cifar10": {
        "n_classes": 10,
        "arch": "cnn",
        "epochs": 100,
        "batch_size": 64,
        "loader_fn": "get_cifar10_loaders",
        "input_shape": (3, 32, 32),
    },
}


def _get_loaders(dataset: str, batch_size: int | None = None) -> tuple[DataLoader, DataLoader]:
    """Get train and test data loaders."""
    from ph_neuro.training.data import (
        get_cifar10_loaders,
        get_fashion_mnist_loaders,
        get_mnist_loaders,
    )

    loader_map = {
        "mnist": get_mnist_loaders,
        "fashion": get_fashion_mnist_loaders,
        "cifar10": get_cifar10_loaders,
    }
    cfg = DATASETS[dataset]
    bs = batch_size or cfg["batch_size"]
    fn = loader_map[dataset]
    return fn(batch_size=bs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy on a data loader.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    x: torch.Tensor,
    n_batches: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Benchmark median inference time over multiple forward passes.

    Args:
        model: Model in eval mode.
        x: Input tensor (reused for all iterations).
        n_batches: Number of forward passes to time.
        warmup: Number of warmup passes (excluded from timing).

    Returns:
        Dict with ``mean_time_ms``, ``median_time_ms``, ``std_time_ms``,
        ``min_time_ms``, ``max_time_ms``.
    """
    model.eval()

    # Warmup
    for _ in range(warmup):
        _ = model(x)

    # Timed runs
    times: list[float] = []
    for _ in range(n_batches):
        start = time.perf_counter()
        _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    times_t = torch.tensor(times)
    return {
        "mean_time_ms": float(times_t.mean()),
        "median_time_ms": float(times_t.median()),
        "std_time_ms": float(times_t.std()),
        "min_time_ms": float(times_t.min()),
        "max_time_ms": float(times_t.max()),
        "n_batches": n_batches,
    }


def compute_output_diff(
    original: nn.Module,
    fused: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> dict[str, float]:
    """Compute statistics of output differences between original and fused.

    Args:
        original: Original (unfused) model.
        fused: Fused model.
        loader: Data loader.
        device: Torch device.
        n_batches: Number of batches to compare.

    Returns:
        Dict with ``mse``, ``max_abs_diff``, ``mean_abs_diff``,
        ``n_samples_checked``, ``n_batches_checked``.
    """
    original.eval()
    fused.eval()

    total_mse = 0.0
    total_max_diff = 0.0
    total_mean_diff = 0.0
    n_checked = 0
    n_batches_checked = 0

    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)

        with torch.no_grad():
            out_orig = original(x)
            out_fused = fused(x)

        diff = (out_fused - out_orig).abs()
        total_mse += (diff ** 2).mean().item()
        total_max_diff = max(total_max_diff, diff.max().item())
        total_mean_diff += diff.mean().item()
        n_checked += x.size(0)
        n_batches_checked += 1

    return {
        "mse": total_mse / max(n_batches_checked, 1),
        "max_abs_diff": total_max_diff,
        "mean_abs_diff": total_mean_diff / max(n_batches_checked, 1),
        "n_samples_checked": n_checked,
        "n_batches_checked": n_batches_checked,
    }


def count_bn_layers(model: nn.Module) -> int:
    """Count BatchNorm layers in a model."""
    return sum(
        1 for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
    )


def count_fused_layers(model: nn.Module) -> dict[str, int]:
    """Count fused vs remaining layers."""
    from ph_neuro.layers.fused_bn import FusedTernaryConv2d, FusedTernaryLinear

    n_fused_linear = 0
    n_fused_conv = 0
    for m in model.modules():
        if isinstance(m, FusedTernaryLinear):
            n_fused_linear += 1
        if isinstance(m, FusedTernaryConv2d):
            n_fused_conv += 1

    return {
        "n_fused_linear_layers": n_fused_linear,
        "n_fused_conv_layers": n_fused_conv,
        "n_total_fused": n_fused_linear + n_fused_conv,
    }


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L5: BatchNorm Fusion for Ternary STE Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(DATASETS.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["mlp", "cnn"],
        help="Architecture (auto-detected from dataset if not specified)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (dataset default if not set)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (dataset default if not set)")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=200,
        help="Number of forward passes for inference benchmarking",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="l5_results",
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
    """Run the L5 BN fusion experiment."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cfg = DATASETS[args.dataset]
    arch = args.arch or cfg["arch"]
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]

    title = f"L5: BN Fusion — {arch.upper()} on {args.dataset.upper()} (seed={args.seed})"
    print_header(title)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {arch}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"AdamW: lr={args.lr}, weight_decay={args.weight_decay}")
    print()

    # ── Data loaders ──────────────────────────────────────────────
    train_loader, test_loader = _get_loaders(args.dataset, batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")
    print()

    # ── Build model ───────────────────────────────────────────────
    if arch == "mlp":
        model = ste_mlp(cfg["layer_sizes"], batch_norm=True, device=device)
    else:
        model = ste_cnn(in_channels=3, img_size=32, n_classes=cfg["n_classes"], device=device)

    n_params = sum(p.numel() for p in model.parameters())
    n_bn_before = count_bn_layers(model)
    print(f"Model parameters: {n_params:,}")
    print(f"BatchNorm layers: {n_bn_before}")
    print()

    # ── Train ─────────────────────────────────────────────────────
    print("Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    train_start = time.time()
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
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

        scheduler.step()
        train_acc = correct / max(total, 1)
        test_acc = evaluate(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
        epoch_time = time.time() - epoch_start

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"Train Acc: {100 * train_acc:5.2f}%  "
            f"Test Acc: {100 * test_acc:5.2f}%  "
            f"Loss: {total_loss / max(total, 1):.4f}  "
            f"Time: {epoch_time:.1f}s"
        )

    train_time = time.time() - train_start
    print(f"\n  Training complete: {train_time:.1f}s, Best test acc: {100 * best_test_acc:.2f}%")
    print()

    # ── Fuse BN ───────────────────────────────────────────────────
    print("Fusing BatchNorm layers...")
    model.eval()

    fuse_start = time.time()
    fused_model = fuse_bn_layers(model, inplace=False)
    fuse_time = time.time() - fuse_start

    n_bn_after = count_bn_layers(fused_model)
    fused_counts = count_fused_layers(fused_model)

    print(f"  Fusion time: {fuse_time * 1000:.1f}ms")
    print(f"  BN layers before: {n_bn_before} → after: {n_bn_after}")
    print(f"  Fused layer counts: {fused_counts}")
    print()

    # ── Verify output equivalence ──────────────────────────────────
    print("Verifying output equivalence...")
    diff_stats = compute_output_diff(model, fused_model, test_loader, device, n_batches=20)

    print(f"  MSE:              {diff_stats['mse']:.2e}")
    print(f"  Mean abs diff:    {diff_stats['mean_abs_diff']:.2e}")
    print(f"  Max abs diff:     {diff_stats['max_abs_diff']:.2e}")
    print(f"  Samples checked:  {diff_stats['n_samples_checked']}")
    print()

    # ── Benchmark ─────────────────────────────────────────────────
    print("Benchmarking inference speed...")
    # Get a sample batch for benchmarking
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x[:batch_size].to(device)

    print(f"  Unfused model ({n_bn_before} BN layers)...")
    unfused_bench = benchmark_inference(model, sample_x, n_batches=args.benchmark_batches)
    print(f"    Median: {unfused_bench['median_time_ms']:.3f} ms  "
          f"Mean: {unfused_bench['mean_time_ms']:.3f} ms  "
          f"Std: {unfused_bench['std_time_ms']:.3f} ms")

    print(f"  Fused model (0 BN layers)...")
    fused_bench = benchmark_inference(fused_model, sample_x, n_batches=args.benchmark_batches)
    print(f"    Median: {fused_bench['median_time_ms']:.3f} ms  "
          f"Mean: {fused_bench['mean_time_ms']:.3f} ms  "
          f"Std: {fused_bench['std_time_ms']:.3f} ms")

    speedup = unfused_bench["median_time_ms"] / max(fused_bench["median_time_ms"], 1e-9)
    print(f"\n  Speedup (median): {speedup:.2f}x")
    print()

    # ── Accuracy comparison ───────────────────────────────────────
    print("Comparing accuracy...")
    acc_unfused = evaluate(model, test_loader, device)
    acc_fused = evaluate(fused_model, test_loader, device)
    print(f"  Unfused accuracy: {100 * acc_unfused:.2f}%")
    print(f"  Fused accuracy:   {100 * acc_fused:.2f}%")
    print(f"  Difference:       {100 * (acc_fused - acc_unfused):.4f}pp")
    print()

    # ── Save results ──────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "experiment": "L5 BN Fusion",
        "dataset": args.dataset,
        "architecture": arch,
        "seed": args.seed,
        "device": str(device),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_time_seconds": train_time,
            "best_test_accuracy": best_test_acc,
            "n_parameters": n_params,
        },
        "fusion": {
            "fusion_time_ms": fuse_time * 1000,
            "bn_layers_before": n_bn_before,
            "bn_layers_after": n_bn_after,
            "fused_layer_counts": fused_counts,
        },
        "output_equivalence": diff_stats,
        "benchmark_unfused": unfused_bench,
        "benchmark_fused": fused_bench,
        "speedup_median_x": speedup,
        "accuracy": {
            "unfused": acc_unfused,
            "fused": acc_fused,
            "difference_pp": acc_fused - acc_unfused,
        },
    }

    output_path = os.path.join(
        args.output_dir,
        f"l5_bn_fusion_{args.dataset}_{arch}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Dataset:          {args.dataset} ({arch})")
    print(f"  Model params:     {n_params:,}")
    print(f"  BN layers fused:  {n_bn_before} → {n_bn_after}")
    print(f"  Output MSE:       {diff_stats['mse']:.2e}")
    print(f"  Max abs diff:     {diff_stats['max_abs_diff']:.2e}")
    print(f"  Fused accuracy:   {100 * acc_fused:.2f}% (unfused: {100 * acc_unfused:.2f}%)")
    print(f"  Speedup (median): {speedup:.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
