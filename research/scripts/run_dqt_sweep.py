#!/usr/bin/env python3
"""DQT hyperparameter sweep — test fallback strategies for convergence.

Usage:
    python scripts/run_dqt_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.utils.optimizers import make_adamw


def get_mnist_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    from torchvision import transforms
    from torchvision.datasets import MNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_dqt_mlp(layer_sizes: list[int], device: torch.device, init_std: float = 0.1) -> nn.Sequential:
    """Build DQT MLP with configurable init std."""
    layers: list[nn.Module] = [nn.Flatten()]
    sizes = list(layer_sizes)
    for i in range(len(sizes) - 1):
        dqt = TernaryDQTLinear(sizes[i], sizes[i + 1], bias=False)
        # Override initialization
        nn.init.normal_(dqt.weight_float, mean=0.0, std=init_std)
        # Re-initialize ternary weights
        from ph_neuro.layers.ste_dqt import stochastic_round
        dqt.weight_ternary = stochastic_round(dqt.weight_float.data)
        layers.append(dqt)
        if i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
    return nn.Sequential(*layers).to(device)


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_one_config(
    name: str,
    lr: float,
    epochs: int,
    batch_size: int,
    init_std: float,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train one DQT configuration."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    model = build_dqt_mlp([784, 512, 256, 10], device, init_std=init_std)
    # OPT-2: 8-bit AdamW (states 8→2 B/param), falls back to fp32.
    optimizer = make_adamw(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"  LR={lr}, epochs={epochs}, batch={batch_size}, init_std={init_std}")
    print(f"{'='*60}")

    best_acc = 0.0
    best_epoch = 0
    total_start = time.time()
    patience = 0
    max_patience = max(15, epochs // 3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            for m in model.modules():
                if isinstance(m, TernaryDQTLinear):
                    m.apply_stochastic_rounding()

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

        # Recompute train accuracy properly
        train_acc = _compute_train_acc(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            # Compute weight stats
            total_w = total_zero = total_pos = total_neg = 0
            for m in model.modules():
                if isinstance(m, TernaryDQTLinear):
                    w = m.weight_ternary
                    n = w.numel()
                    total_w += n
                    total_zero += (w == 0).sum().item()
                    total_pos += (w == 1).sum().item()
                    total_neg += (w == -1).sum().item()
            sparsity = 100 * total_zero / max(total_w, 1)

            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"Train: {100*train_acc:.2f}%  "
                f"Test: {100*test_acc:.2f}%  "
                f"Best: {100*best_acc:.2f}% (ep {best_epoch})  "
                f"Sparsity: {sparsity:.1f}%"
            )

        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    total_time = time.time() - total_start

    # Final weight stats
    total_w = total_zero = total_pos = total_neg = 0
    for m in model.modules():
        if isinstance(m, TernaryDQTLinear):
            w = m.weight_ternary
            n = w.numel()
            total_w += n
            total_zero += (w == 0).sum().item()
            total_pos += (w == 1).sum().item()
            total_neg += (w == -1).sum().item()

    return {
        "name": name,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "init_std": init_std,
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "training_time_seconds": float(total_time),
        "weight_sparsity_pct": 100 * total_zero / max(total_w, 1),
        "weight_pos_pct": 100 * total_pos / max(total_w, 1),
        "weight_neg_pct": 100 * total_neg / max(total_w, 1),
    }


@torch.no_grad()
def _compute_train_acc(model, loader, device):
    """Quick train accuracy (on first 2000 samples only for speed)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
        if total >= 2000:
            break
    return correct / max(total, 1)


def main():
    configs = [
        # Baseline (already run)
        # {"name": "baseline_lr0.001_ep30", "lr": 0.001, "epochs": 30, "batch_size": 128, "init_std": 0.1},
        
        # Fallback 1: More epochs
        {"name": "more_epochs_lr0.001_ep60", "lr": 0.001, "epochs": 60, "batch_size": 128, "init_std": 0.1},
        
        # Fallback 2: Higher LR
        {"name": "high_lr0.01_ep30", "lr": 0.01, "epochs": 30, "batch_size": 128, "init_std": 0.1},
        {"name": "high_lr0.01_ep60", "lr": 0.01, "epochs": 60, "batch_size": 128, "init_std": 0.1},
        
        # Fallback 3: Bigger initial weights (reduce initial sparsity)
        {"name": "big_init_lr0.001_ep30", "lr": 0.001, "epochs": 30, "batch_size": 128, "init_std": 0.5},
        {"name": "big_init_lr0.001_ep60", "lr": 0.001, "epochs": 60, "batch_size": 128, "init_std": 0.5},
        
        # Fallback 4: Smaller batch + higher LR
        {"name": "small_batch_lr0.01_ep60", "lr": 0.01, "epochs": 60, "batch_size": 64, "init_std": 0.3},
    ]

    results = []
    for cfg in configs:
        result = train_one_config(**cfg)
        results.append(result)
        print(f"\n  >>> {cfg['name']}: Best={100*result['best_accuracy']:.2f}%, "
              f"Sparsity={result['weight_sparsity_pct']:.1f}%, "
              f"Time={result['training_time_seconds']:.0f}s")

    # Save all results
    os.makedirs("dqt_results", exist_ok=True)
    with open("dqt_results/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("  SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'Config':<35} {'Best Acc':>8} {'Sparsity':>9} {'Time':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<35} {100*r['best_accuracy']:>7.2f}% {r['weight_sparsity_pct']:>8.1f}% {r['training_time_seconds']:>6.0f}s")


if __name__ == "__main__":
    main()
