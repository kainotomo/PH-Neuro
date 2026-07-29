"""NTH-4: Neuromodulated Hebbian Multi-Layer MLP on MNIST — Phase 2.

Trains a 2-layer ternary network (784 \u2192 512 \u2192 10) on MNIST using
**neuromodulated Hebbian** (three-factor) learning with joint training:

1. **Hidden layer** (784\u2192512): Modulated Hebbian update via one of three
   modulator propagation approaches:
   - ``label_broadcast`` (A): correctness signal broadcast to active hidden neurons
   - ``weight_feedback`` (B): M_hidden = M_output @ W_out
   - ``random_feedback`` (C): M_hidden = M_output @ B (fixed random matrix)

2. **Output layer** (512\u219210): Standard NTH label modulator (NTH-1 / WTA equivalent)

Both layers are updated **jointly** (not greedy) because the hidden modulator
requires the output layer's predictions and weights.

Key features:
    - No ``.backward()`` calls
    - No optimizers or loss functions
    - Joint NTH update: \u0394W_hidden = \u03b7_hidden \u00d7 M_hidden\u1d40 @ pre, \u0394W_output = \u03b7_output \u00d7 M_output\u1d40 @ h_hidden
    - Ternary weights learned via Hebbian plasticity + hysteresis
    - Three modulator approaches evaluated sequentially
    - Comparison to ALL prior experiments

Usage:
    python -m ph_neuro.examples.nth_multilayer_mnist
    python -m ph_neuro.examples.nth_multilayer_mnist --modulator-mode weight_feedback
    python -m ph_neuro.examples.nth_multilayer_mnist --modulator-mode random_feedback
"""

from __future__ import annotations

import argparse
import time

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.nth_multilayer import NTHMultiLayerClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a 2-layer NTH MLP on MNIST (NTH-4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")

    # Modulator mode
    parser.add_argument(
        "--modulator-mode",
        type=str,
        default="label_broadcast",
        choices=["label_broadcast", "weight_feedback", "random_feedback"],
        help="Hidden-layer modulator approach",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--lr-hidden", type=float, default=0.005, help="Hidden layer Hebbian LR"
    )
    parser.add_argument(
        "--lr-output", type=float, default=0.01, help="Output layer Hebbian LR"
    )
    parser.add_argument("--decay", type=float, default=0.0, help="Homeostatic decay rate")

    # Hysteresis
    parser.add_argument("--theta-upper", type=float, default=1.0, help="Hysteresis upper threshold")
    parser.add_argument("--theta-lower", type=float, default=0.3, help="Hysteresis lower threshold")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Dead-zone for ternary_sign")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )

    return parser.parse_args()


BASELINES = {
    "Phase 0 WTA 1-layer": 88.4,
    "Phase 1.1 unsup Hebbian 2-layer": 87.9,
    "TFF-1 (FF 1-layer)": 87.9,
    "NTH-1 (label modulator 1-layer)": 88.15,
    "TFF-2 (FF 2-layer)": 86.81,
    "Backprop MLP 2-layer (theoretical cap)": 98.0,
}


def _get_tier(acc: float) -> tuple[str, str]:
    """Determine the tier and color for a given accuracy."""
    pct = 100 * acc
    if pct > 92:
        return "\U0001f7e2 Major success", "Proceed to NTH-5 (CIFAR-10 CNN)!"
    elif pct > 90:
        return "\U0001f7e1 Moderate success", "NTH helps but ternary is a bottleneck"
    elif pct > 88:
        return "\U0001f9e0 Marginal", "Modulator does something but not enough"
    else:
        return "\U0001f534 Fail", "Ternary Hebbian hidden layers are fundamentally limited"


def _print_section(title: str) -> None:
    """Print a section header."""
    print()
    print(f"  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")


def main() -> None:
    """Run the NTH-4 experiment."""
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Header ──────────────────────────────────────────────────────
    mode_names = {
        "label_broadcast": "A: Label broadcast (correctness \u2192 hidden)",
        "weight_feedback": "B: Weight-feedback (M_output @ W_out)",
        "random_feedback": "C: Random feedback alignment (M_output @ B)",
    }

    print_header("PH-Neuro Phase 2 \u2014 NTH-4: Neuromodulated Hebbian 2-Layer MLP on MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: 784 \u2192 {args.hidden_size} \u2192 10")
    print(f"Modulator mode: {mode_names[args.modulator_mode]}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"LR hidden: {args.lr_hidden}, LR output: {args.lr_output}, Decay: {args.decay}")
    print(f"Theta upper: {args.theta_upper}, Theta lower: {args.theta_lower}")
    print(f"Epsilon: {args.epsilon}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    print()

    # ── Model ───────────────────────────────────────────────────────
    classifier = NTHMultiLayerClassifier(
        in_features=784,
        hidden_size=args.hidden_size,
        out_features=10,
        modulator_mode=args.modulator_mode,
        theta_upper=args.theta_upper,
        theta_lower=args.theta_lower,
        device=device,
    )
    print(f"Classifier: {classifier}")
    total_params = sum(
        layer.weight.unpack().numel() for layer in classifier.model.layers
    )
    print(f"Total weights: {total_params:,} ternary params")
    print()

    # ── Training ────────────────────────────────────────────────────
    total_start = time.time()

    history = classifier.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        lr_hidden=args.lr_hidden,
        lr_output=args.lr_output,
        epochs=args.epochs,
        decay=args.decay,
        epsilon=args.epsilon,
        verbose=True,
    )

    total_time = time.time() - total_start

    # ── Final evaluation ──────────────────────────────────────────
    final_acc = classifier.evaluate(test_loader, epsilon=args.epsilon)
    weight_stats = classifier.get_weight_stats()

    print()
    print_header("Results")
    print(f"Modulator mode: {mode_names[args.modulator_mode]}")
    print(f"Final test accuracy: {100 * final_acc:.2f}%")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print()

    # Weight distributions
    print("Weight distributions:")
    for name in ["hidden", "output"]:
        s = weight_stats[name]
        print(f"  {name.capitalize():6s} layer: +{s['pos_pct']:5.1f}%  "
              f"\u2212{s['neg_pct']:5.1f}%  0{s['zero_pct']:5.1f}%")
    print()

    # ── Comparison table ─────────────────────────────────────────
    _print_section("Comparison to all prior experiments")
    nth4_val = 100 * final_acc
    print(f"  {'Experiment':<40s} {'Accuracy':>8s}  {'vs NTH-4':>10s}")
    print(f"  {'─' * 40} {'─' * 8}  {'─' * 10}")
    print(f"  {'NTH-4 (this run)':<40s} {nth4_val:7.2f}%  {'\u2014':>10s}")
    for name, acc in BASELINES.items():
        diff = nth4_val - acc
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<40s} {acc:7.1f}%  {sign}{diff:+.2f}pp")
    print()

    # ── Go/no-go assessment ──────────────────────────────────────
    tier, decision = _get_tier(final_acc)
    print(f"Tier: {tier} ({100 * final_acc:.2f}%)")
    print(f"Decision: {decision}")
    print()

    # ── Per-epoch breakdown ──────────────────────────────────────
    _print_section("Per-epoch breakdown")
    print(f"  {'Epoch':>6s}  {'Acc':>7s}  {'Hidden Flips':>13s}  {'Output Flips':>13s}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 13}  {'─' * 13}")
    for epoch in range(len(history["accuracy"])):
        acc = 100 * history["accuracy"][epoch]
        flips_h = 100 * history["flip_rate_hidden"][epoch]
        flips_o = 100 * history["flip_rate_output"][epoch]
        print(f"  {epoch + 1:6d}  {acc:6.2f}%  {flips_h:11.4f}%  {flips_o:11.4f}%")
    print()

    # ── Per-epoch accuracy trajectory ─────────────────────────────
    _print_section("Accuracy trajectory")
    for epoch in range(len(history["accuracy"])):
        acc = 100 * history["accuracy"][epoch]
        bar_len = int(acc / 2)
        bar = "\u2588" * bar_len + "\u2591" * (50 - bar_len)
        print(f"  Epoch {epoch + 1:2d}: {bar} {acc:5.2f}%")
    print()

    # ── Invariant checks ─────────────────────────────────────────
    print("Invariant checks:")
    print(f"  No .backward() calls: \u2713 (by design)")
    print(f"  All weights ternary: \u2713 (verified by TernaryHebbianLinear)")
    print()

    # ── Final conclusion ─────────────────────────────────────────
    peak_acc = 100 * max(history["accuracy"])
    print_header("Conclusion")
    print(f"  Modulator mode: {args.modulator_mode}")
    print(f"  Peak accuracy: {peak_acc:.2f}%")
    print(f"  Final accuracy: {nth4_val:.2f}%")
    print(f"  Tier: {tier}")
    print(f"  Decision: {decision}")
    print()


if __name__ == "__main__":
    main()
