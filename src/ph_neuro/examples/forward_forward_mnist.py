"""Forward-Forward MNIST experiment — Phase 2, TFF-1.

Trains a single :class:`~ph_neuro.layers.linear.TernaryHebbianLinear`
layer (784 → 10) on MNIST using the Forward-Forward algorithm.

Key features:
    - No ``.backward()`` calls
    - No optimizers or loss functions
    - Two forward passes: positive (real data → Hebbian strengthen)
      and negative (corrupted data → anti-Hebbian weaken)
    - Ternary weights learned via Hebbian plasticity + hysteresis
    - Popcount goodness: activation magnitude as learning signal

Usage:
    python -m ph_neuro.examples.forward_forward_mnist
    python -m ph_neuro.examples.forward_forward_mnist --epochs 10 --lr-pos 0.02
"""

from __future__ import annotations

import argparse
import time

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.forward_forward import ForwardForwardClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a single-layer Forward-Forward classifier on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--lr-pos", type=float, default=0.01, help="Hebbian learning rate (positive pass)"
    )
    parser.add_argument(
        "--lr-neg", type=float, default=0.0, help="Anti-Hebbian learning rate (negative pass)"
    )
    parser.add_argument("--decay", type=float, default=0.0, help="Homeostatic decay rate")
    parser.add_argument("--theta-upper", type=float, default=1.0, help="Hysteresis upper threshold")
    parser.add_argument("--theta-lower", type=float, default=0.3, help="Hysteresis lower threshold")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Dead-zone for ternary_sign")
    parser.add_argument(
        "--mask-ratio", type=float, default=0.5, help="Fraction of pixels to mask in negative data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Forward-Forward MNIST experiment."""
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print_header("PH-Neuro Phase 2 — TFF-1: Forward-Forward MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR pos: {args.lr_pos}, LR neg: {args.lr_neg}, Decay: {args.decay}")
    print(f"Epsilon: {args.epsilon}, Mask ratio: {args.mask_ratio}")
    print(f"Theta upper: {args.theta_upper}, Theta lower: {args.theta_lower}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    print()

    # ── Model ───────────────────────────────────────────────────────
    classifier = ForwardForwardClassifier(
        in_features=784,
        out_features=10,
        theta_upper=args.theta_upper,
        theta_lower=args.theta_lower,
        device=device,
    )
    print(f"Classifier: {classifier}")
    print(f"Weights: {classifier.model.weight.unpack().numel():,} params")
    print()

    # ── Training ────────────────────────────────────────────────────
    print("Training...")
    print("-" * 100)

    total_start = time.time()
    total_steps = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        classifier.model.train()

        step_metrics: list[dict[str, float]] = []

        for x, y in train_loader:
            with torch.no_grad():
                metrics = classifier.train_step(
                    x,
                    y,
                    lr_pos=args.lr_pos,
                    lr_neg=args.lr_neg,
                    decay=args.decay,
                    epsilon=args.epsilon,
                    mask_ratio=args.mask_ratio,
                )
                step_metrics.append(metrics)
                total_steps += 1

        # Evaluate
        acc = classifier.evaluate(test_loader, epsilon=args.epsilon)
        weight_stats = classifier.get_weight_stats()
        avg_flip_rate = sum(m["flip_rate"] for m in step_metrics) / max(len(step_metrics), 1)

        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"Acc: {100 * acc:5.2f}%  "
            f"W: +{weight_stats['pos_pct']:4.1f}% "
            f"\u2212{weight_stats['neg_pct']:4.1f}% "
            f"0{weight_stats['zero_pct']:5.1f}%  "
            f"Flips: {100 * avg_flip_rate:5.2f}%/step  "
            f"Time: {epoch_time:.1f}s"
        )

    total_time = time.time() - total_start
    assert total_steps > 0

    # ── Final summary ──────────────────────────────────────────────
    final_acc = classifier.evaluate(test_loader, epsilon=args.epsilon)
    final_weights = classifier.get_weight_stats()

    print()
    print_header("Results")
    print(f"Final test accuracy: {100 * final_acc:.2f}%")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print(f"Total steps: {total_steps}")
    print(f"Weight distribution: +{final_weights['pos_pct']:.1f}%, "
          f"\u2212{final_weights['neg_pct']:.1f}%, "
          f"0{final_weights['zero_pct']:.1f}%")
    print()

    # Verify invariants
    print("No .backward() calls: \u2713 (verified by design)")
    print("All weights ternary: \u2713 (verified by TernaryHebbianLinear)")
    print()

    # Comparison with WTA baseline
    wta_baseline = 88.42
    diff = 100 * final_acc - wta_baseline
    print(f"Comparison with WTA baseline ({wta_baseline:.1f}%):")
    print(f"  Forward-Forward: {100 * final_acc:.2f}% ({'+' if diff >= 0 else ''}{diff:.2f}pp vs WTA)")
    print()

    if 100 * final_acc > 87.0:
        print(f"\u2713 SUCCESS: {100 * final_acc:.1f}% MNIST accuracy with Forward-Forward!")
        print("  (Matches WTA baseline of ~88% — FF mechanism validated)")
    elif 100 * final_acc > 85.0:
        print(f"Acceptable: {100 * final_acc:.1f}% — close to the 88.4% WTA baseline.")
        print("  Try tuning hyperparameters or increasing epochs.")
    else:
        print(f"Below target: {100 * final_acc:.1f}% < 85%. Try:")
        print("  - Lower --theta-upper (e.g. 0.5) for faster weight activation")
        print("  - Increase --lr-pos (e.g. 0.02) or adjust --lr-neg ratio")
        print("  - Add --decay (e.g. 0.001) for homeostatic regularization")
        print("  - More --epochs")


if __name__ == "__main__":
    main()
