"""CIFAR-10 Hebbian CNN experiment — Phase 1.2.

Trains a :class:`~ph_neuro.models.cnn.HebbianCNN` with greedy layer-wise
Hebbian learning on CIFAR-10.

Architecture (default):
    ``Conv(3→64, 3×3) → sign → MaxPool2d(2)
    → Conv(64→128, 3×3) → sign → MaxPool2d(2)
    → Flatten → Linear(8192→10)``

Training strategy (greedy layer-wise):
    - **Conv1**: unsupervised per-position competitive Hebbian
    - **Conv2**: unsupervised per-position competitive Hebbian (on Conv1's output)
    - **Output Linear**: supervised WTA Hebbian (from Phase 0)

Key features:
    - No ``.backward()`` calls anywhere
    - All weights remain in {-1, 0, +1} at every step
    - Per-layer hyperparameters (lr, epochs, theta values)

Usage:
    python -m ph_neuro.examples.cifar10_cnn
    python -m ph_neuro.examples.cifar10_cnn --conv-epochs 5 5 --lr 0.005 0.005 0.01
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.core.activation import ternary_sign
from ph_neuro.examples._utils import print_header
from ph_neuro.models.cnn import HebbianCNN
from ph_neuro.training.data import get_cifar10_loaders
from ph_neuro.training.greedy import (
    _init_conv_connectivity,
    evaluate_cnn,
    train_conv_class_guided_epoch,
    train_conv_competitive_epoch,
    train_supervised_wta_epoch,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Hebbian CNN on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument(
        "--hidden-channels", type=int, default=64,
        help="Number of channels in the first conv layer. Second gets 2× this.",
    )
    parser.add_argument(
        "--n-classes", type=int, default=10,
        help="Number of output classes.",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--conv-epochs", type=int, nargs="+", default=[5, 5],
        help="Epochs for conv layers (conv1 conv2).",
    )
    parser.add_argument(
        "--output-epochs", type=int, default=10,
        help="Epochs for output linear layer.",
    )
    parser.add_argument(
        "--lr", type=float, nargs="+", default=[0.01, 0.01, 0.01],
        help="Learning rates (conv1 conv2 output).",
    )
    parser.add_argument("--decay", type=float, default=0.0,
                        help="Homeostatic decay rate")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Dead-zone for ternary_sign")
    parser.add_argument(
        "--class-guided", action="store_true", default=True,
        help="Use class-guided Hebbian for conv layers (assigns filters to classes)",
    )
    parser.add_argument(
        "--competitive", action="store_true", default=False,
        help="Use competitive Hebbian for conv layers (unsupervised, per-position WTA)",
    )

    # Theta
    parser.add_argument(
        "--theta-upper", type=float, default=2.0,
        help="Hysteresis upper threshold for conv layers",
    )
    parser.add_argument(
        "--theta-lower", type=float, default=0.5,
        help="Hysteresis lower threshold for conv layers",
    )
    parser.add_argument(
        "--output-theta-upper", type=float, default=1.0,
        help="Hysteresis upper threshold for output layer",
    )
    parser.add_argument(
        "--output-theta-lower", type=float, default=0.3,
        help="Hysteresis lower threshold for output layer",
    )

    # Misc
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch progress")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def compute_weight_stats(model: HebbianCNN, device: torch.device) -> dict[str, dict[str, float]]:
    """Compute weight statistics for all layers of a HebbianCNN.

    Returns:
        Nested dict: ``{layer_name: {"pos_pct", "neg_pct", "zero_pct", "n_weights"}}``
    """
    stats = {}
    for name in ["conv1", "conv2", "output"]:
        layer = getattr(model, name)
        w = layer.weight.unpack()
        total = w.numel()
        stats[name] = {
            "pos_pct": 100.0 * (w == 1).sum().item() / max(total, 1),
            "neg_pct": 100.0 * (w == -1).sum().item() / max(total, 1),
            "zero_pct": 100.0 * (w == 0).sum().item() / max(total, 1),
            "n_weights": total,
        }
    return stats


def main() -> None:
    """Run the CIFAR-10 Hebbian CNN experiment."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print_header("PH-Neuro Phase 1.2 — Hebbian CNN on CIFAR-10")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print()

    # ── Model ─────────────────────────────────────────────────────
    model = HebbianCNN(
        in_channels=3,
        img_size=32,
        hidden_channels=args.hidden_channels,
        n_classes=args.n_classes,
        theta_upper=args.theta_upper,
        theta_lower=args.theta_lower,
        output_theta_upper=args.output_theta_upper,
        output_theta_lower=args.output_theta_lower,
        device=device,
    )

    total_params = sum(
        layer.weight.unpack().numel()
        for layer in [model.conv1, model.conv2, model.output]
    )
    print(f"Architecture:")
    print(f"  Conv1: {model.conv1}")
    print(f"  Conv2: {model.conv2}")
    print(f"  Output: {model.output}")
    print(f"  Total ternary weights: {total_params:,}")
    print(f"  Estimated size (int8): {total_params / 1e6:.2f} MB")
    print()

    # ── Data ──────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    print(f"CIFAR-10: {len(train_loader.dataset):,} train, "
          f"{len(test_loader.dataset):,} test samples")
    print()

    # ── Bootstrap conv layers with sparse random weights ──────────
    print("Bootstrapping conv layers with sparse random weights...")
    _init_conv_connectivity(model.conv1, density=0.1)
    _init_conv_connectivity(model.conv2, density=0.1)
    print()

    # ── Step 1: Train Conv1 (class-guided or competitive Hebbian) ──
    conv_rule = "competitive" if args.competitive else "class-guided"
    print("=" * 70)
    print(f"  Step 1: Training Conv1 (unsupervised, {conv_rule} Hebbian)")
    print("=" * 70)

    model.conv1.requires_hebbian_(True)
    lr_conv1, lr_conv2, lr_output = args.lr[0], args.lr[1] if len(args.lr) > 1 else args.lr[0], args.lr[-1]

    for epoch in range(1, args.conv_epochs[0] + 1):
        if args.competitive:
            metrics = train_conv_competitive_epoch(
                conv_layer=model.conv1,
                loader=train_loader,
                frozen_encoder=None,
                device=device,
                lr=lr_conv1,
                decay=args.decay,
                epsilon=args.epsilon,
            )
        else:
            metrics = train_conv_class_guided_epoch(
                conv_layer=model.conv1,
                loader=train_loader,
                frozen_encoder=None,
                device=device,
                lr=lr_conv1,
                decay=args.decay,
                epsilon=args.epsilon,
                n_classes=args.n_classes,
            )
        if not args.quiet:
            print(
                f"  Conv1 Epoch {epoch:2d}/{args.conv_epochs[0]}  "
                f"Flips: {100 * metrics['flip_rate']:6.3f}%/step"
            )

    model.conv1.requires_hebbian_(False)

    # ── Step 2: Train Conv2 (class-guided or competitive Hebbian) ──
    print()
    print("=" * 70)
    print(f"  Step 2: Training Conv2 (on frozen Conv1 output, {conv_rule} Hebbian)")
    print("=" * 70)

    # Build a frozen encoder from conv1 + pool + sign
    frozen_encoder = _FrozenConvEncoder(model.conv1, epsilon=args.epsilon).to(device)

    model.conv2.requires_hebbian_(True)
    for epoch in range(1, args.conv_epochs[1] + 1):
        if args.competitive:
            metrics = train_conv_competitive_epoch(
                conv_layer=model.conv2,
                loader=train_loader,
                frozen_encoder=frozen_encoder,
                device=device,
                lr=lr_conv2,
                decay=args.decay,
                epsilon=args.epsilon,
            )
        else:
            metrics = train_conv_class_guided_epoch(
                conv_layer=model.conv2,
                loader=train_loader,
                frozen_encoder=frozen_encoder,
                device=device,
                lr=lr_conv2,
                decay=args.decay,
                epsilon=args.epsilon,
                n_classes=args.n_classes,
            )
        if not args.quiet:
            print(
                f"  Conv2 Epoch {epoch:2d}/{args.conv_epochs[1]}  "
                f"Flips: {100 * metrics['flip_rate']:6.3f}%/step"
            )

    model.conv2.requires_hebbian_(False)

    # ── Step 3: Train output layer (supervised WTA) ──────────────
    print()
    print("=" * 70)
    print("  Step 3: Training output Linear (supervised WTA Hebbian)")
    print("=" * 70)

    # Build a frozen encoder: conv1 → sign → pool → conv2 → sign → pool → flatten
    frozen_flat_encoder = [_FrozenFlatEncoder(model, epsilon=args.epsilon).to(device)]

    model.output.requires_hebbian_(True)
    for epoch in range(1, args.output_epochs + 1):
        metrics = train_supervised_wta_epoch(
            layer=model.output,
            loader=train_loader,
            frozen_encoder=frozen_flat_encoder,
            device=device,
            lr=lr_output,
            decay=args.decay,
            epsilon=args.epsilon,
        )
        if not args.quiet:
            print(
                f"  Output Epoch {epoch:2d}/{args.output_epochs}  "
                f"Acc: {100 * metrics['accuracy']:5.2f}%  "
                f"Flips: {100 * metrics['flip_rate']:6.3f}%/step"
            )

    model.output.requires_hebbian_(False)

    # ── Final evaluation ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Final Evaluation")
    print("=" * 70)

    test_acc = evaluate_cnn(model, test_loader, device, epsilon=args.epsilon)
    train_acc = evaluate_cnn(model, train_loader, device, epsilon=args.epsilon)

    print(f"\n  Train accuracy: {100 * train_acc:.2f}%")
    print(f"  Test accuracy:  {100 * test_acc:.2f}%")
    print()

    # ── Weight statistics ─────────────────────────────────────────
    stats = compute_weight_stats(model, device)

    print(f"  {'Layer':<12} {'+1%':>8} {'-1%':>8} {'0%':>8} {'Weights':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for name in ["conv1", "conv2", "output"]:
        s = stats[name]
        print(f"  {name:<12} {s['pos_pct']:>7.2f}% {s['neg_pct']:>7.2f}% "
              f"{s['zero_pct']:>7.2f}% {s['n_weights']:>12,}")
    print()

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("  ── Summary ──")
    print(f"  Architecture: Conv({3}→{args.hidden_channels}) → Conv({args.hidden_channels}"
          f"→{2*args.hidden_channels}) → Linear({model.output._in_features}→10)")
    print(f"  Total ternary weights: {total_params:,}")
    print(f"  Test accuracy: {100 * test_acc:.2f}%")
    print(f"  Target: >55% ({'✅ PASS' if test_acc > 0.55 else '❌ BELOW TARGET'})")

    baseline_msg = (
        f"  Single Layer Hebbian (Phase 0): 88.4% MNIST\n"
        f"  Multi Layer Hebbian (Phase 1.1): 87.9% MNIST\n"
        f"  Backprop (same arch, float):    ~88% CIFAR-10 (estimate)\n"
        f"  Float Hebbian (same arch):      ~75% CIFAR-10 (estimate)\n"
        f"  SoftHebb (Journé et al. 2023):  80.3% CIFAR-10\n"
        f"  Random ternary weights:         ~10% CIFAR-10"
    )
    print(f"\n{baseline_msg}")


class _FrozenConvEncoder(torch.nn.Module):
    """Frozen encoder: Conv1 → sign → MaxPool2d.

    Used for greedy layer 2 training. Runs conv1 with frozen weights,
    then applies ternary_sign and MaxPool.
    """

    def __init__(self, conv1: torch.nn.Module, epsilon: float = 0.1):
        super().__init__()
        self.conv1 = conv1
        self.epsilon = epsilon
        self.pool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = ternary_sign(h, epsilon=self.epsilon).float()
        h = self.pool(h)
        return h


class _FrozenFlatEncoder(torch.nn.Module):
    """Frozen encoder: full conv stack → flatten.

    Used for output layer training. Accepts flat or spatial input,
    reshapes internally, runs conv1→sign→pool→conv2→sign→pool→flatten
    with all weights frozen, returns a flat vector for the linear output layer.
    """

    def __init__(self, model: HebbianCNN, epsilon: float = 0.1):
        super().__init__()
        self.cnn = model
        self.epsilon = epsilon
        self.img_size = model._img_size
        self.in_channels = model._in_channels
        self.pool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle flat input (from train_supervised_wta_epoch flattening)
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.in_channels, self.img_size, self.img_size)
        with torch.no_grad():
            h = self.cnn.conv1(x)
            h = ternary_sign(h, epsilon=self.epsilon).float()
            h = self.pool(h)

            h = self.cnn.conv2(h)
            h = ternary_sign(h, epsilon=self.epsilon).float()
            h = self.pool(h)

            h = h.reshape(h.shape[0], -1)
        return h


if __name__ == "__main__":
    main()
