#!/usr/bin/env python3
"""DQT + Hysteresis-STE: combined experiment on MNIST.

Combines two previously validated techniques:
- **DQT (E017)**: ternary weights stored as int8, float buffer only for
  gradient accumulation, updated via stochastic rounding (no latent scores).
- **Hysteresis-STE (E016/L2)**: dual-threshold hysteresis
  (``theta_upper``/``theta_lower``) as a sparsity regularizer.

Research question: can the combination give >97% accuracy with >90% sparsity
AND without latent float scores (4.5x less training memory)?

Usage:
    # Default (DQT best hyperparams + L2 best thresholds)
    python -m ph_neuro.examples.run_dqt_hysteresis --output-dir dqt_hysteresis_results

    # Custom hyperparameters
    python -m ph_neuro.examples.run_dqt_hysteresis \\
        --lr 0.01 --epochs 60 --batch-size 128 --init-std 0.1 \\
        --theta-upper 0.3 --theta-lower 0.15 --seed 42

    # Deadzone ablation: also stochastic-round the hysteresis gap
    python -m ph_neuro.examples.run_dqt_hysteresis --explore-gap

Output:
    JSON file: ``{output_dir}/results_mnist_seed{seed}.json``
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
from ph_neuro.utils.optimizers import make_adamw

warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")


# ── Helpers ─────────────────────────────────────────────────────────


def get_mnist_loaders(batch_size: int = 128, num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


def build_dqt_hysteresis_mlp(
    layer_sizes: list[int],
    device: torch.device,
    theta_upper: float = 0.3,
    theta_lower: float = 0.15,
    explore_gap: bool = False,
    init_std: float = 0.1,
) -> nn.Sequential:
    """Build an MLP with TernaryDQTHysteresisLinear layers.

    Architecture matches L1 / L2 / DQT baselines:
        Flatten → DQTHystLinear(784,512) → ReLU → BN →
        DQTHystLinear(512,256) → ReLU → BN →
        DQTHystLinear(256,10)

    Args:
        layer_sizes: Layer widths, e.g. [784, 512, 256, 10].
        device: Torch device.
        theta_upper: Hysteresis upper threshold.
        theta_lower: Hysteresis lower threshold.
        explore_gap: Whether to stochastic-round the hysteresis gap.
        init_std: Init std for the float accumulation buffers.

    Returns:
        nn.Sequential model.
    """
    from ph_neuro.layers.ste_dqt_hysteresis import TernaryDQTHysteresisLinear

    layers: list[nn.Module] = [nn.Flatten()]
    sizes = list(layer_sizes)

    for i in range(len(sizes) - 1):
        layers.append(
            TernaryDQTHysteresisLinear(
                sizes[i],
                sizes[i + 1],
                theta_upper=theta_upper,
                theta_lower=theta_lower,
                explore_gap=explore_gap,
                bias=False,
                init_std=init_std,
            )
        )
        if i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))

    model = nn.Sequential(*layers)
    return model.to(device)


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


# ── Training Loop ───────────────────────────────────────────────────


def train_dqt_hysteresis(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epochs: int,
    device: torch.device,
) -> dict:
    """Train a DQT+Hysteresis model.

    After each ``optimizer.step()`` we call ``apply_stochastic_rounding()``
    on every TernaryDQTHysteresisLinear layer to discretise the float
    accumulation buffer into ternary int8 weights via the hysteresis-gated
    stochastic rounding rule.

    Returns:
        Dict with training results (accuracy, sparsity, flip rates, time).
    """
    from ph_neuro.layers.ste_dqt_hysteresis import TernaryDQTHysteresisLinear

    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0
    final_acc = 0.0
    patience_counter = 0
    max_patience = 10

    # Per-epoch tracking
    epoch_accuracies: list[float] = []
    epoch_sparsities: list[float] = []
    epoch_flip_rates: list[float] = []

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

            # ── Combined DQT: hysteresis + stochastic rounding ──
            for module in model.modules():
                if isinstance(module, TernaryDQTHysteresisLinear):
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
        sparsity = _compute_weight_sparsity(model)
        epoch_accuracies.append(test_acc)
        epoch_sparsities.append(sparsity)
        epoch_flip_rates.append(avg_flip)

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
            f"Sparsity: {sparsity:5.2f}%  "
            f"Flip: {avg_flip:.4f}  "
            f"LR: {lr:.2e}  "
            f"Time: {epoch_time:.1f}s"
        )

        if patience_counter >= max_patience:
            print(f"  ⏹️  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    total_time = time.time() - total_start

    # ── Weight statistics ───────────────────────────────────────────
    weight_stats = _compute_dqt_hysteresis_weight_stats(model)

    # ── Flip rate after convergence (last 5 epochs average) ─────────
    final_flip_rate = (
        sum(epoch_flip_rates[-5:]) / min(5, len(epoch_flip_rates))
        if epoch_flip_rates
        else 0.0
    )
    final_sparsity = (
        sum(epoch_sparsities[-5:]) / min(5, len(epoch_sparsities))
        if epoch_sparsities
        else 0.0
    )

    return {
        "best_accuracy": float(best_acc),
        "best_epoch": best_epoch,
        "final_accuracy": float(final_acc),
        "training_time_seconds": float(total_time),
        "epochs_trained": epoch,
        "final_flip_rate": float(final_flip_rate),
        "final_sparsity_pct": float(final_sparsity),
        "epoch_accuracies": [float(a) for a in epoch_accuracies],
        "epoch_sparsities": [float(s) for s in epoch_sparsities],
        "epoch_flip_rates": [float(f) for f in epoch_flip_rates],
        **weight_stats,
    }


# ── Weight statistics helpers ───────────────────────────────────────


def _compute_weight_sparsity(model: nn.Module) -> float:
    """Weight sparsity (percent zero) across all ternary layers."""
    from ph_neuro.layers.ste_dqt_hysteresis import TernaryDQTHysteresisLinear

    total_w = 0
    total_zero = 0
    for module in model.modules():
        if isinstance(module, TernaryDQTHysteresisLinear):
            w = module.weight_ternary.flatten()
            total_w += w.numel()
            total_zero += (w == 0).sum().item()
    return 100.0 * total_zero / max(total_w, 1)


def _compute_dqt_hysteresis_weight_stats(model: nn.Module) -> dict:
    """Extract weight statistics from a DQT+Hysteresis model."""
    from ph_neuro.layers.ste_dqt_hysteresis import TernaryDQTHysteresisLinear

    total_w = 0
    total_zero = 0
    total_pos = 0
    total_neg = 0
    n_layers = 0

    for module in model.modules():
        if isinstance(module, TernaryDQTHysteresisLinear):
            w = module.weight_ternary.flatten()
            n = w.numel()
            total_w += n
            total_zero += (w == 0).sum().item()
            total_pos += (w == 1).sum().item()
            total_neg += (w == -1).sum().item()
            n_layers += 1

    stats = {
        "n_parameters": total_w,
        "n_layers": n_layers,
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DQT + Hysteresis-STE combined experiment"
    )
    parser.add_argument(
        "--dataset", default="mnist", choices=["mnist"], help="Dataset"
    )
    parser.add_argument(
        "--layer-sizes", default="784,512,256,10",
        help="Comma-separated layer sizes",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--init-std", type=float, default=0.1, help="Init std for float buffers"
    )
    parser.add_argument(
        "--theta-upper", type=float, default=0.3,
        help="Hysteresis upper threshold",
    )
    parser.add_argument(
        "--theta-lower", type=float, default=0.15,
        help="Hysteresis lower threshold",
    )
    parser.add_argument(
        "--explore-gap", action="store_true",
        help="Stochastic-round the hysteresis gap",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader workers"
    )
    parser.add_argument("--output-dir", default="dqt_hysteresis_results", help="Output directory")
    return parser.parse_args()


def main() -> None:
    """Run the DQT+Hysteresis combined experiment."""
    args = parse_args()

    # ── Configuration ───────────────────────────────────────────────
    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    layer_sizes = [int(s) for s in args.layer_sizes.split(",")]
    theta_upper = args.theta_upper
    theta_lower = args.theta_lower
    explore_gap = args.explore_gap
    init_std = args.init_std
    output_dir = args.output_dir
    dataset_name = args.dataset

    # ── Setup ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print_header("DQT + Hysteresis-STE Combined Experiment: Ternary MLP on MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {layer_sizes}")
    print(
        f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, "
        f"WD: {weight_decay}, Init std: {init_std}"
    )
    print(f"θ_upper: {theta_upper}, θ_lower: {theta_lower}, explore_gap: {explore_gap}")
    print(f"Seed: {seed}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size, num_workers=args.num_workers
    )
    print(
        f"MNIST: {len(train_loader.dataset)} train, "
        f"{len(test_loader.dataset)} test samples"  # type: ignore[arg-type]
    )
    print()

    # ── Model ───────────────────────────────────────────────────────
    model = build_dqt_hysteresis_mlp(
        layer_sizes,
        device,
        theta_upper=theta_upper,
        theta_lower=theta_lower,
        explore_gap=explore_gap,
        init_std=init_std,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(
        "  (float accumulation buffers only — ternary weights stored as int8, "
        "no latent scores)"
    )
    print()

    # ── Optimizer ───────────────────────────────────────────────────
    # OPT-2: 8-bit AdamW (states 8→2 B/param), falls back to fp32.
    optimizer = make_adamw(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Train ───────────────────────────────────────────────────────
    print("Training...")
    print()
    results = train_dqt_hysteresis(
        model, train_loader, test_loader,
        optimizer, scheduler, epochs, device,
    )

    # ── Build result dict ───────────────────────────────────────────
    result = {
        "experiment": "dqt_hysteresis",
        "dataset": dataset_name,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "init_std": init_std,
        "theta_upper": theta_upper,
        "theta_lower": theta_lower,
        "explore_gap": explore_gap,
        "layer_sizes": layer_sizes,
        "method": "DQT (stochastic rounding) + Hysteresis-STE (dual threshold), no latent scores",
        **results,
    }

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    tag = f"th{theta_upper}_tl{theta_lower}"
    if explore_gap:
        tag += "_gap"
    output_path = os.path.join(
        output_dir, f"results_{dataset_name}_{tag}_seed{seed}.json"
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print_header("Results Summary")
    print(
        f"  Best Test Accuracy:  {100 * result['best_accuracy']:.2f}%  "
        f"(epoch {result['best_epoch']})"
    )
    print(f"  Final Test Accuracy: {100 * result['final_accuracy']:.2f}%")
    print(f"  Training Time:       {result['training_time_seconds']:.1f}s")
    print(f"  Weight Sparsity:     {result['weight_sparsity_pct']:.1f}%")
    print(f"  Weight +1:           {result['weight_pos_pct']:.1f}%")
    print(f"  Weight -1:           {result['weight_neg_pct']:.1f}%")
    print(f"  Weight 0:            {result['weight_zero_pct']:.1f}%")
    print(f"  Final Flip Rate:     {result['final_flip_rate']:.4f}")
    print()
    print("  L1 STE Baseline:      98.17% | sparsity 0%   | latent scores: yes")
    print("  Hysteresis (L2):      97.92% | sparsity 95%  | latent scores: yes")
    print("  DQT (E017):           98.23% | sparsity 56%  | latent scores: no")
    print(f"  DQT+Hysteresis (this): {100 * result['best_accuracy']:.2f}% | "
          f"sparsity {result['weight_sparsity_pct']:.1f}% | latent scores: no")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
