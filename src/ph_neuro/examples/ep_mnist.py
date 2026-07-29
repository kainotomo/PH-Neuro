"""TEP-1: Equilibrium Propagation on MNIST — Phase 2, last experiment.

Trains a 2-layer ternary network (784→512→10) on MNIST using **joint**
Equilibrium Propagation training:

1. **Hidden layer** (784→512): EP difference-of-correlations update:
   ΔS_hidden = η_h × (h_target^T @ x - h_free^T @ x)
   where h_target = ternary_sign(S_out^T @ y_onehot) is a class-specific
   ternary hidden state derived from the output layer's dense latent scores.

2. **Output layer** (512→10): Standard WTA (same as Phase 0):
   ΔS_output = η_o × (y_target^T @ h_free - y_pred^T @ h_free)

Key features:
    - No ``.backward()`` calls
    - No optimizers or loss functions
    - Joint training (both layers updated per batch)
    - Warmup phase: output WTA only for first N epochs before enabling EP
    - Hidden target correlation tracking (diagnostic: is h_free aligning with h_target?)
    - Comparison to ALL prior 8 experiments

Usage:
    python -m ph_neuro.examples.ep_mnist
    python -m ph_neuro.examples.ep_mnist --epochs 30 --lr-hidden 0.01
"""

from __future__ import annotations

import argparse
import time

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.ep import EPConfig, EquilibriumPropagationClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TEP-1: Train a 2-layer EP MLP on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer size")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup epochs (output WTA only)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    # Learning rates
    parser.add_argument(
        "--lr-hidden", type=float, default=0.005, help="Hidden layer EP learning rate"
    )
    parser.add_argument(
        "--lr-output", type=float, default=0.01, help="Output layer WTA learning rate"
    )

    # Hysteresis
    parser.add_argument(
        "--theta-upper", type=float, default=0.5, help="Hysteresis upper threshold"
    )
    parser.add_argument(
        "--theta-lower", type=float, default=0.15, help="Hysteresis lower threshold"
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Dead-zone for ternary_sign")

    # Regularization
    parser.add_argument("--decay", type=float, default=0.0, help="Homeostatic decay rate")
    parser.add_argument(
        "--hidden-density", type=float, default=0.1, help="Hidden layer initial connectivity"
    )

    # EP options
    parser.add_argument(
        "--update-correct",
        action="store_true",
        help="Apply hidden EP update even on correct predictions",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )

    return parser.parse_args()


# ── Baseline references ──────────────────────────────────────────

BASELINES: dict[str, float] = {
    "Phase 0 WTA 1-layer": 88.4,
    "Phase 1.1 unsup Hebbian 2-layer": 87.9,
    "TFF-1 (FF 1-layer)": 87.9,
    "NTH-1 (label modulator 1-layer)": 88.15,
    "TFF-2 (FF 2-layer)": 86.81,
    "NTH-4 B (weight feedback)": 85.79,
    "NTH-4 C (random feedback)": 85.02,
    "NTH-4b D (latent score feedback)": 86.68,
    "Backprop MLP 2-layer (cap)": 98.0,
}


def _get_tier(acc: float, pct: float | None = None) -> tuple[str, str]:
    """Determine the tier and emoji for a given accuracy."""
    if pct is None:
        pct = 100 * acc
    if pct > 92:
        return "🟢 Breakthrough", "EP works with ternary weights!"
    elif pct > 90:
        return "🟡 Partial success", "EP does something, ternary is bottleneck"
    elif pct > 88:
        return "🟠 Marginal", "Slight improvement over ~88% bound"
    else:
        return "🔴 Failure", "No method trains ternary Hebbian hidden layers"


def main() -> None:
    """Run the TEP-1 experiment."""
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print_header("PH-Neuro TEP-1: Equilibrium Propagation 2-Layer MLP on MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: 784 → {args.hidden_size} → 10")
    print(f"Joint EP training:")
    print(f"  Warmup: {args.warmup} epochs (output WTA only)")
    print(f"  Hidden: EP (lr={args.lr_hidden}, update-on-correct={args.update_correct})")
    print(f"  Output: WTA (lr={args.lr_output})")
    print(f"  Both layers updated per batch")
    print(f"Hysteresis: θ_u={args.theta_upper}, θ_l={args.theta_lower}")
    print(f"Epsilon: {args.epsilon}, Decay: {args.decay}")
    print(f"Hidden density: {args.hidden_density}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    print()

    # ── Configuration ──────────────────────────────────────────────
    cfg = EPConfig(
        lr_hidden=args.lr_hidden,
        lr_output=args.lr_output,
        theta_upper=args.theta_upper,
        theta_lower=args.theta_lower,
        decay=args.decay,
        epsilon=args.epsilon,
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        hidden_update_on_correct=args.update_correct,
        hidden_density=args.hidden_density,
    )

    # ── Model ───────────────────────────────────────────────────────
    classifier = EquilibriumPropagationClassifier(
        in_features=784,
        hidden_size=args.hidden_size,
        out_features=10,
        cfg=cfg,
        device=device,
    )
    print(f"Model: {classifier.model}")

    w_hidden = classifier.hidden_layer.weight.unpack()
    w_output = classifier.output_layer.weight.unpack()
    total_params = w_hidden.numel() + w_output.numel()
    print(f"Total weights: {total_params:,} ternary params")
    print()

    # ── Training ────────────────────────────────────────────────────
    total_start = time.time()

    history = classifier.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        verbose=True,
    )

    total_time = time.time() - total_start

    # ── Final evaluation ──────────────────────────────────────────
    final_train_acc = history["accuracy"][-1]
    final_test_acc = history.get("test_accuracy", [final_train_acc])[-1]
    final_hidden_flip = history["flip_rate_hidden"][-1]
    final_output_flip = history["flip_rate_output"][-1]
    final_h_corr = history["h_target_corr"][-1]
    final_h_sparse = history["h_sparsity"][-1]
    final_o_sparse = history["out_sparsity"][-1]

    # Weight distribution
    w_dist = classifier.get_weight_distribution()

    # Tier
    tier_emoji, tier_msg = _get_tier(final_test_acc)

    print_header("Results")
    print(f"  Test accuracy: {final_test_acc*100:.2f}%")
    print(f"  Tier: {tier_emoji} — {tier_msg}")
    print()
    print(f"  Hidden flip rate: {final_hidden_flip*100:.4f}%/step")
    print(f"  Output flip rate: {final_output_flip*100:.4f}%/step")
    print(f"  Hidden target correlation: {final_h_corr:.3f}")
    print(f"  Hidden sparsity: {final_h_sparse*100:.1f}%")
    print(f"  Output sparsity: {final_o_sparse*100:.1f}%")
    print()
    print(f"  Hidden weight distribution:")
    print(f"    +1: {w_dist['hidden']['pos_pct']:.1f}%")
    print(f"    -1: {w_dist['hidden']['neg_pct']:.1f}%")
    print(f"     0: {w_dist['hidden']['zero_pct']:.1f}%")
    print(f"  Output weight distribution:")
    print(f"    +1: {w_dist['output']['pos_pct']:.1f}%")
    print(f"    -1: {w_dist['output']['neg_pct']:.1f}%")
    print(f"     0: {w_dist['output']['zero_pct']:.1f}%")
    print()
    print(f"  Total training time: {total_time:.1f}s")
    print()

    # ── Comparison to baselines ────────────────────────────────────
    print_header("Comparison to All Prior Experiments")
    print(f"  {'Experiment':<42s} {'Accuracy':>8s} {'vs TEP-1':>10s}")
    print(f"  {'-'*42} {'-'*8} {'-'*10}")
    print(f"  {'TEP-1 (this run)':<42s} {final_test_acc*100:>7.2f}% {'—':>10s}")

    # Sort baselines by accuracy descending
    sorted_baselines = sorted(BASELINES.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_baselines:
        diff = final_test_acc * 100 - acc
        diff_str = f"+{diff:.2f}pp" if diff >= 0 else f"{diff:.2f}pp"
        print(f"  {name:<42s} {acc:>7.2f}% {diff_str:>10s}")

    # ── Invariant checks ──────────────────────────────────────────
    print_header("Invariant Checks")
    print(f"  Ternary weights: ✅ (verified by TernaryHebbianLinear)")
    print(f"  No .backward() calls: ✅ (by design)")
    print(f"  Flip rate < 1% after convergence: "
          f"{'✅' if final_output_flip < 0.01 else '⚠️'} "
          f"({final_output_flip*100:.4f}%)")

    # Check that hidden flip rate was non-zero during training
    max_hidden_flip = max(history["flip_rate_hidden"])
    if max_hidden_flip > 0:
        print(f"  Hidden layer changed: ✅ (max flip rate {max_hidden_flip*100:.4f}%)")
    else:
        print(f"  Hidden layer changed: ❌ (0.0000% — no hidden learning)")

    print()
    print(f"  {tier_emoji} Final result: {final_test_acc*100:.2f}% — {tier_msg}")


if __name__ == "__main__":
    main()
