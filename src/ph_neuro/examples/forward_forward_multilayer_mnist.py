"""Forward-Forward multi-layer MNIST experiment — Phase 2, TFF-2.

Trains a 2-layer ternary network (784 \u2192 512 \u2192 10) on MNIST using
greedy layer-wise Forward-Forward training:

1. **Hidden layer** (784\u2192512): True Forward-Forward contrastive learning
   - Positive pass (real data) \u2192 Hebbian, maximize popcount goodness
   - Negative pass (junk data) \u2192 anti-Hebbian, minimize popcount goodness
2. **Output layer** (512\u219210): Supervised WTA Hebbian on frozen hidden
   representations

Key features:
    - No ``.backward()`` calls
    - No optimizers or loss functions
    - Greedy layer-wise training via ``MultiLayerHebbianClassifier.fit_greedy()``
    - Goodness separation tracking for hidden layer quality
    - Comparison to Phase 0 (88.4%), Phase 1.1 (87.9%), TFF-1 (87.9%)

Usage:
    python -m ph_neuro.examples.forward_forward_multilayer_mnist
    python -m ph_neuro.examples.forward_forward_multilayer_mnist \
        --lr-pos 0.01 --lr-neg 0.002 --epochs-hidden 10
"""

from __future__ import annotations

import argparse
import time

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.greedy import (
    LayerConfig,
    MultiLayerHebbianClassifier,
    evaluate_goodness_separation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a 2-layer Forward-Forward MLP on MNIST (TFF-2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Hidden layer size"
    )

    # Training: hidden FF layer
    parser.add_argument("--epochs-hidden", type=int, default=10, help="FF hidden layer epochs")
    parser.add_argument(
        "--lr-pos", type=float, default=0.01, help="Hebbian LR (positive pass)"
    )
    parser.add_argument(
        "--lr-neg", type=float, default=0.002, help="Anti-Hebbian LR (negative pass, MUST be >0)"
    )

    # Training: output WTA layer
    parser.add_argument("--epochs-output", type=int, default=10, help="Output WTA layer epochs")
    parser.add_argument(
        "--lr-output", type=float, default=0.01, help="Output layer Hebbian LR"
    )

    # Shared parameters
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--decay", type=float, default=0.0, help="Homeostatic decay")
    parser.add_argument("--theta-upper", type=float, default=1.0, help="Hysteresis upper threshold")
    parser.add_argument("--theta-lower", type=float, default=0.3, help="Hysteresis lower threshold")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Dead-zone for ternary_sign")
    parser.add_argument(
        "--mask-ratio", type=float, default=0.5, help="Mask ratio for negative data"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch progress")

    return parser.parse_args()


def _get_weight_stats(model: torch.nn.Module) -> dict[str, float]:
    """Get aggregated weight distribution stats for a model."""
    total_pos = 0
    total_neg = 0
    total_zero = 0
    total = 0
    for layer in getattr(model, "layers", [model]):
        if hasattr(layer, "weight"):
            w = layer.weight.unpack()
            n = w.numel()
            total_pos += (w == 1).sum().item()
            total_neg += (w == -1).sum().item()
            total_zero += (w == 0).sum().item()
            total += n
    if total == 0:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "zero_pct": 100.0}
    return {
        "pos_pct": total_pos / total * 100,
        "neg_pct": total_neg / total * 100,
        "zero_pct": total_zero / total * 100,
    }


def main() -> None:
    """Run the TFF-2 experiment."""
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print_header("PH-Neuro Phase 2 \u2014 TFF-2: Forward-Forward 2-Layer MLP on MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: 784 \u2192 {args.hidden_size} \u2192 10")
    print(f"Hidden layer: FF (lr_pos={args.lr_pos}, lr_neg={args.lr_neg}, "
          f"epochs={args.epochs_hidden})")
    print(f"Output layer: WTA (lr={args.lr_output}, epochs={args.epochs_output})")
    print(f"Batch size: {args.batch_size}, Decay: {args.decay}")
    print(f"Theta upper: {args.theta_upper}, Theta lower: {args.theta_lower}")
    print(f"Epsilon: {args.epsilon}, Mask ratio: {args.mask_ratio}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    print()

    # ── Model ───────────────────────────────────────────────────────
    classifier = MultiLayerHebbianClassifier(
        layer_sizes=[784, args.hidden_size, 10],
        theta_upper=args.theta_upper,
        theta_lower=args.theta_lower,
        device=device,
    )
    print(f"Model: {classifier.model}")
    total_params = sum(
        classifier.model.get_layer(i).weight.unpack().numel()
        for i in range(classifier.n_layers)
    )
    print(f"Total weights: {total_params:,} ternary params")
    print()

    # ── Layer configs ──────────────────────────────────────────────
    layer_configs = [
        LayerConfig(
            lr=args.lr_pos,
            lr_neg=args.lr_neg,
            epochs=args.epochs_hidden,
            hebbian_rule="forward_forward",
            decay=args.decay,
            theta_upper=args.theta_upper,
            theta_lower=args.theta_lower,
        ),
        LayerConfig(
            lr=args.lr_output,
            epochs=args.epochs_output,
            hebbian_rule="basic",
            decay=args.decay,
            theta_upper=args.theta_upper,
            theta_lower=args.theta_lower,
            anti_hebbian=False,
        ),
    ]

    # ── Greedy layer-wise training ──────────────────────────────────
    total_start = time.time()

    history = classifier.fit_greedy(
        train_loader=train_loader,
        layer_configs=layer_configs,
        epsilon=args.epsilon,
        verbose=not args.quiet,
    )

    total_time = time.time() - total_start

    # ── Goodness separation for hidden layer ─────────────────────────
    # The hidden layer should show positive separation after FF training
    hidden_layer = classifier.model.get_layer(0)
    sep = evaluate_goodness_separation(
        layer=hidden_layer,
        loader=test_loader,
        frozen_encoder=None,
        device=device,
        epsilon=args.epsilon,
        mask_ratio=args.mask_ratio,
    )

    # ── Final evaluation ──────────────────────────────────────────
    final_acc = classifier.evaluate(test_loader, epsilon=args.epsilon)
    hidden_stats = _get_weight_stats(classifier.model.get_layer(0))
    output_stats = _get_weight_stats(classifier.model.get_layer(1))

    print()
    print_header("Results")
    print(f"Final test accuracy: {100 * final_acc:.2f}%")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print()
    print("Weight distributions:")
    print(f"  Hidden layer: +{hidden_stats['pos_pct']:.1f}%, "
          f"\u2212{hidden_stats['neg_pct']:.1f}%, "
          f"0{hidden_stats['zero_pct']:.1f}%")
    print(f"  Output layer: +{output_stats['pos_pct']:.1f}%, "
          f"\u2212{output_stats['neg_pct']:.1f}%, "
          f"0{output_stats['zero_pct']:.1f}%")
    print(f"  Hidden goodness separation: {sep['separation']:+.1f} "
          f"(g_pos={sep['g_pos']:.1f}, g_neg={sep['g_neg']:.1f})")
    print()

    # Comparison table
    print("Comparison to baselines:")
    baselines = {
        "Phase 0 WTA 1-layer": 88.4,
        "Phase 1.1 unsup Hebbian 2-layer": 87.9,
        "TFF-1 (FF 1-layer)": 87.9,
        "Backprop MLP 2-layer (theoretical cap)": 98.0,
    }
    print(f"  {'Experiment':<35s} {'Accuracy':>8s}  {'vs TFF-2':>10s}")
    print(f"  {'-'*35} {'-'*8}  {'-'*10}")
    tff2_val = 100 * final_acc
    print(f"  {'TFF-2 (this run)':<35s} {tff2_val:7.2f}%  {'\u2014':>10s}")
    for name, acc in baselines.items():
        diff = tff2_val - acc
        sign = "+" if diff >= 0 else ""
        print(f"  {name:<35s} {acc:7.1f}%  {sign}{diff:+.2f}pp")
    print()

    # Go/no-go assessment
    if tff2_val > 95:
        tier = "\U0001f7e2 Major success"
        decision = "Proceed to TFF-3 and CIFAR-10!"
    elif tff2_val > 90:
        tier = "\U0001f7e1 Moderate success"
        decision = "Investigate ternary bottleneck, then proceed to TFF-3"
    else:
        tier = "\U0001f534 Fail"
        decision = "FF+ternary may be incompatible. Pivot to NTH-only."

    print(f"Tier: {tier} (>{tff2_val:.1f}%)")
    print(f"Decision: {decision}")
    print()

    # Verify invariants
    print("Invariant checks:")
    print(f"  All weights ternary: \u2713 (verified by TernaryHebbianLinear)")
    print(f"  No .backward() calls: \u2713 (by design)")
    print(f"  Goodness separation > 0: {'\\u2713' if sep['separation'] > 0 else '\\u2717'} "
          f"({sep['separation']:+.1f})")


if __name__ == "__main__":
    main()
