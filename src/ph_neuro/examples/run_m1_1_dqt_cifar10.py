#!/usr/bin/env python3
"""Milestone M1.1 — DQT CNN on CIFAR-10 (>80% accuracy GO/NO-GO).

First demonstration of Direct Quantized Training (DQT) on CONVOLUTIONAL
layers. Uses :class:`TernaryDQTConv2d` (int8 ternary weights + stochastic
rounding, no latent float scores) for the conv blocks and
:class:`TernaryDQTLinear` for the classifier — mirroring the ``ste_cnn()``
architecture so results are directly comparable to the E009/L1 STE baseline
(72.2-72.75% on CIFAR-10).

The critical DQT mechanic: after EVERY ``optimizer.step()`` we call
``apply_stochastic_rounding()`` on every DQT layer to discretize the float
accumulation buffer into int8 ternary weights.

Usage::

    python -m ph_neuro.examples.run_m1_1_dqt_cifar10 \\
        --lr 0.01 --epochs 100 --seed 42 --batch-size 128

Output:
    JSON file: ``{output_dir}/results_dqt_cifar10_lr{lr}_seed{seed}.json``
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
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.training.data import get_cifar10_loaders

warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")

DQT_LAYERS = (TernaryDQTConv2d, TernaryDQTLinear)


# ── Helpers ─────────────────────────────────────────────────────────


def is_dqt_module(module: nn.Module) -> bool:
    """Whether a module is a DQT weight-bearing layer."""
    return isinstance(module, DQT_LAYERS)


@torch.no_grad()
def apply_stochastic_rounding(model: nn.Module) -> float:
    """Apply DQT stochastic rounding to every DQT layer.

    Returns:
        Mean flip rate across all DQT layers.
    """
    flips: list[float] = []
    for module in model.modules():
        if is_dqt_module(module):
            flips.append(module.apply_stochastic_rounding()["flip_rate"])
    return sum(flips) / max(len(flips), 1) if flips else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate test accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def compute_weight_stats(model: nn.Module) -> dict[str, float]:
    """Aggregate ternary weight stats across all DQT layers.

    Returns:
        Dict with ``pos_pct``, ``neg_pct``, ``zero_pct`` (sparsity),
        and ``n_ternary_weights``.
    """
    total = zeros = pos = neg = 0
    for module in model.modules():
        if is_dqt_module(module):
            w = module.weight_ternary
            n = w.numel()
            total += n
            zeros += (w == 0).sum().item()
            pos += (w == 1).sum().item()
            neg += (w == -1).sum().item()
    if total == 0:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "zero_pct": 0.0, "n_ternary_weights": 0}
    return {
        "pos_pct": 100.0 * pos / total,
        "neg_pct": 100.0 * neg / total,
        "zero_pct": 100.0 * zeros / total,
        "n_ternary_weights": int(total),
    }


# ── Training Loop ───────────────────────────────────────────────────


def train_dqt_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epochs: int,
    max_patience: int = 15,
    verbose: bool = True,
) -> dict:
    """Train a DQT CNN on CIFAR-10.

    Key DQT difference from STE: after each ``optimizer.step()`` we call
    ``apply_stochastic_rounding()`` on every DQT layer to discretize the
    float accumulation buffer into int8 ternary weights.

    Returns:
        Dict with per-epoch histories and final metrics:
        - ``best_accuracy``, ``best_epoch``, ``final_accuracy``
        - ``epochs_trained``, ``training_time_seconds``
        - ``train_acc_history``, ``test_acc_history``, ``loss_history``,
          ``lr_history``, ``flip_history``, ``epoch_times``
    """
    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0
    final_acc = 0.0
    patience = 0
    epochs_trained = 0

    history: dict[str, list] = {
        "train_acc": [],
        "test_acc": [],
        "loss": [],
        "lr": [],
        "flip": [],
        "epoch_time": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_samples = 0
        correct = 0
        total = 0
        epoch_flips = 0.0
        n_steps = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            # ── DQT: stochastic rounding after EVERY optimizer step ──
            epoch_flips += apply_stochastic_rounding(model)
            n_steps += 1

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / max(total, 1)
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        epochs_trained = epoch
        lr = optimizer.param_groups[0]["lr"]
        avg_flip = epoch_flips / max(n_steps, 1)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        final_acc = test_acc

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(total_loss / max(n_samples, 1))
        history["lr"].append(lr)
        history["flip"].append(avg_flip)
        history["epoch_time"].append(epoch_time)

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"Train: {100 * train_acc:5.2f}%  "
                f"Test: {100 * test_acc:5.2f}%  "
                f"Loss: {total_loss / max(n_samples, 1):.4f}  "
                f"Flip: {avg_flip:.4f}  "
                f"LR: {lr:.2e}  "
                f"Time: {epoch_time:.1f}s"
            )

        if patience >= max_patience:
            print(f"  ⏹️  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    total_time = time.time() - total_start

    # Flip rate after convergence: average of the last 5 epochs
    final_flip_rate = (
        sum(history["flip"][-5:]) / min(5, len(history["flip"]))
        if history["flip"]
        else 0.0
    )

    return {
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(final_acc),
        "epochs_trained": epochs_trained,
        "training_time_seconds": float(total_time),
        "final_flip_rate": float(final_flip_rate),
        "weight_stats": compute_weight_stats(model),
        "train_acc_history": [float(a) for a in history["train_acc"]],
        "test_acc_history": [float(a) for a in history["test_acc"]],
        "loss_history": [float(a) for a in history["loss"]],
        "lr_history": [float(a) for a in history["lr"]],
        "flip_history": [float(a) for a in history["flip"]],
        "epoch_times": [float(a) for a in history["epoch_time"]],
    }


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone M1.1: DQT CNN on CIFAR-10 (GO/NO-GO >80%)"
    )
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (DQT best: 0.01)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--output-dir", default="m1_1_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print_header(
        f"M1.1 DQT CNN CIFAR-10 (GO/NO-GO >80%): "
        f"lr={args.lr}, {args.epochs}ep, seed={args.seed}"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(
        f"CIFAR-10: {len(train_loader.dataset)} train, "
        f"{len(test_loader.dataset)} test samples"  # type: ignore[arg-type]
    )
    print()

    # ── Model ───────────────────────────────────────────────────────
    model = dqt_cnn(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (float buffers + BN): {n_params:,}")
    print(f"  Ternary weights: {sum(m.weight_ternary.numel() for m in model.modules() if is_dqt_module(m)):,}")
    print(f"  Conv layers: TernaryDQTConv2d(3→64) → TernaryDQTConv2d(64→128) "
          f"(int8 ternary, no latent scores)")
    print()

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Train ───────────────────────────────────────────────────────
    print("Training...")
    print()
    results = train_dqt_cnn(
        model, train_loader, test_loader,
        optimizer, scheduler, device,
        epochs=args.epochs, max_patience=args.patience,
    )

    # ── Peak GPU memory ─────────────────────────────────────────────
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    # ── Build result dict ───────────────────────────────────────────
    result = {
        "experiment": "m1_1_dqt_cifar10",
        "dataset": "cifar10",
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "architecture": "DQT CNN: Conv(3->64)->Pool->Conv(64->128)->Pool->FC(8192->512)->FC(512->10)",
        "method": "Direct Quantized Training (DQT) with stochastic rounding (conv + linear)",
        "n_parameters": n_params,
        "peak_gpu_memory_mb": peak_mem_mb,
        "ste_baseline_accuracy": 0.7275,  # E009/L1 CIFAR-10 ternary STE
        **results,
    }

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"results_dqt_cifar10_lr{args.lr}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────
    ws = result["weight_stats"]
    print()
    print_header("Results Summary")
    print(f"  Best Test Accuracy:  {100 * result['best_accuracy']:.2f}%  (epoch {result['best_epoch']})")
    print(f"  Final Test Accuracy: {100 * result['final_accuracy']:.2f}%")
    print(f"  Training Time:       {result['training_time_seconds']:.1f}s")
    print(f"  Weight +1:           {ws['pos_pct']:.1f}%")
    print(f"  Weight -1:           {ws['neg_pct']:.1f}%")
    print(f"  Weight 0 (sparsity): {ws['zero_pct']:.1f}%")
    print(f"  Final Flip Rate:     {result['final_flip_rate']:.4f}")
    if peak_mem_mb > 0:
        print(f"  Peak GPU Memory:     {peak_mem_mb:.1f} MB")
    print()
    print(f"  STE Baseline (E009): 72.75%")
    verdict = "GO ✅ (>80%)" if result["best_accuracy"] > 0.80 else "NO-GO 🔴 (<=80%)"
    print(f"  M1.1 Verdict:        {100 * result['best_accuracy']:.2f}%  →  {verdict}")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
