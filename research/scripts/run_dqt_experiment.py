#!/usr/bin/env python3
"""Direct Quantized Training (DQT) — pilot experiment on MNIST.

Trains a ternary MLP using DQT with stochastic rounding (no latent float
scores). Compares against the L1 STE baseline (98.0% accuracy).

Usage:
    python scripts/run_dqt_experiment.py

Output:
    dqt_results/results_mnist_seed42.json
"""

from __future__ import annotations

import json
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")


# ── Helpers ─────────────────────────────────────────────────────────


def print_header(title: str) -> None:
    """Print a section header."""
    width = min(len(title) + 4, 80)
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def get_mnist_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders."""
    from torchvision import transforms
    from torchvision.datasets import MNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
    )
    return train_loader, test_loader


# ── DQT Model Builder ───────────────────────────────────────────────


def build_dqt_mlp(
    layer_sizes: list[int],
    device: torch.device,
) -> nn.Sequential:
    """Build an MLP with TernaryDQTLinear layers.

    Architecture matches the L1 STE baseline:
        Flatten → DQTLinear(784,512) → ReLU → BN →
        DQTLinear(512,256) → ReLU → BN →
        DQTLinear(256,10)

    Args:
        layer_sizes: Layer widths, e.g. [784, 512, 256, 10].
        device: Torch device.

    Returns:
        nn.Sequential model with DQT layers.
    """
    from ph_neuro.layers.ste_dqt import TernaryDQTLinear

    layers: list[nn.Module] = [nn.Flatten()]
    sizes = list(layer_sizes)

    for i in range(len(sizes) - 1):
        layers.append(TernaryDQTLinear(sizes[i], sizes[i + 1], bias=False))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))

    model = nn.Sequential(*layers)
    return model.to(device)


# ── Training Loop ───────────────────────────────────────────────────


def train_dqt(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    device: torch.device,
) -> dict:
    """Train a DQT model.

    Key difference from STE: after each optimizer.step(), we call
    apply_stochastic_rounding() on every TernaryDQTLinear layer to
    discretize the float accumulation buffer into ternary int8 weights.

    Args:
        model: The DQT model.
        train_loader: Training data loader.
        test_loader: Test data loader.
        optimizer: AdamW optimizer (tracks weight_float parameters).
        scheduler: Optional LR scheduler.
        epochs: Number of training epochs.
        device: Device.

    Returns:
        Dict with training results.
    """
    from ph_neuro.layers.ste_dqt import TernaryDQTLinear

    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0
    final_acc = 0.0
    patience_counter = 0
    max_patience = 10

    # Collect flip rates per epoch
    flip_history: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_flips = 0.0
        n_dqt_layers = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            # ── DQT: apply stochastic rounding after optimizer step ──
            for module in model.modules():
                if isinstance(module, TernaryDQTLinear):
                    stats = module.apply_stochastic_rounding()
                    epoch_flips += stats["flip_rate"]
                    n_dqt_layers += 1

            total_loss += loss.item() * x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += x.size(0)

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / max(total, 1)
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        avg_flip = epoch_flips / max(n_dqt_layers, 1) if n_dqt_layers > 0 else 0.0
        flip_history.append(avg_flip)

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        final_acc = test_acc

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"Train: {100 * train_acc:5.2f}%  "
            f"Test: {100 * test_acc:5.2f}%  "
            f"Loss: {total_loss / max(total, 1):.4f}  "
            f"Flip: {avg_flip:.4f}  "
            f"LR: {lr:.2e}  "
            f"Time: {epoch_time:.1f}s"
        )

        if patience_counter >= max_patience:
            print(f"  ⏹️  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    total_time = time.time() - total_start

    # ── Weight statistics ───────────────────────────────────────────
    weight_stats = _compute_dqt_weight_stats(model)

    # ── Flip rate after convergence (last 5 epochs average) ─────────
    final_flip_rate = (
        sum(flip_history[-5:]) / min(5, len(flip_history))
        if flip_history
        else 0.0
    )

    return {
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(final_acc),
        "training_time_seconds": float(total_time),
        "epochs_trained": epoch,
        "final_flip_rate": float(final_flip_rate),
        "flip_history": [float(f) for f in flip_history],
        **weight_stats,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def _compute_dqt_weight_stats(model: nn.Module) -> dict:
    """Extract weight statistics from a DQT model."""
    from ph_neuro.layers.ste_dqt import TernaryDQTLinear

    total_w = 0
    total_zero = 0
    total_pos = 0
    total_neg = 0

    for module in model.modules():
        if isinstance(module, TernaryDQTLinear):
            w = module.weight_ternary.flatten()
            n = w.numel()
            total_w += n
            total_zero += (w == 0).sum().item()
            total_pos += (w == 1).sum().item()
            total_neg += (w == -1).sum().item()

    stats = {
        "n_parameters": total_w,
        "weight_zero_pct": 0.0,
        "weight_pos_pct": 0.0,
        "weight_neg_pct": 0.0,
        "weight_sparsity_pct": 0.0,
    }

    if total_w > 0:
        stats["weight_zero_pct"] = 100.0 * total_zero / total_w
        stats["weight_pos_pct"] = 100.0 * total_pos / total_w
        stats["weight_neg_pct"] = 100.0 * total_neg / total_w
        stats["weight_sparsity_pct"] = stats["weight_zero_pct"]

    return stats


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    """Run the DQT pilot experiment."""
    # ── Configuration (matching L1 baseline) ─────────────────────────
    seed = 42
    batch_size = 128
    epochs = 30
    lr = 0.001
    weight_decay = 1e-4
    layer_sizes = [784, 512, 256, 10]
    output_dir = "dqt_results"
    dataset_name = "mnist"

    # ── Setup ────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        # Make stochastic rounding deterministic for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print_header("DQT Pilot Experiment: Ternary MLP on MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {layer_sizes}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, WD: {weight_decay}")
    print(f"Seed: {seed}")
    print()

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")  # type: ignore[arg-type]
    print()

    # ── Model ────────────────────────────────────────────────────────
    model = build_dqt_mlp(layer_sizes, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"  (float accumulation buffers only — ternary weights stored as int8)")
    print()

    # ── Optimizer ────────────────────────────────────────────────────
    # Only weight_float and bias are nn.Parameters
    # OPT-2: 8-bit AdamW (states 8→2 B/param) with an fp32 fallback.
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Train ────────────────────────────────────────────────────────
    print("Training...")
    print()
    results = train_dqt(
        model, train_loader, test_loader,
        optimizer, scheduler, epochs, device,
    )

    # ── Build result dict ────────────────────────────────────────────
    result = {
        "experiment": "dqt",
        "dataset": dataset_name,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "layer_sizes": layer_sizes,
        "method": "Direct Quantized Training (DQT) with stochastic rounding",
        **results,
    }

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"results_{dataset_name}_seed{seed}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print_header("Results Summary")
    print(f"  Best Test Accuracy:  {100 * result['best_accuracy']:.2f}%  (epoch {result['best_epoch']})")
    print(f"  Final Test Accuracy: {100 * result['final_accuracy']:.2f}%")
    print(f"  Training Time:       {result['training_time_seconds']:.1f}s")
    print(f"  Weight Sparsity:     {result['weight_sparsity_pct']:.1f}%")
    print(f"  Weight +1:           {result['weight_pos_pct']:.1f}%")
    print(f"  Weight -1:           {result['weight_neg_pct']:.1f}%")
    print(f"  Weight 0:            {result['weight_zero_pct']:.1f}%")
    print(f"  Final Flip Rate:     {result['final_flip_rate']:.4f}")
    print()
    print(f"  L1 STE Baseline:     98.17% (best) / 97.97% (final)")
    print(f"  DQT (this run):      {100 * result['best_accuracy']:.2f}% (best) / {100 * result['final_accuracy']:.2f}% (final)")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
