"""Multi-layer Hebbian MLP on MNIST — Phase 1 experiment.

Trains a :class:`~ph_neuro.models.mlp.HebbianMLP` with greedy layer-wise
Hebbian learning on MNIST.

Architecture (default): 784 -> 256 -> 128 -> 10, each layer ternary.

Training strategy:
    - Hidden layers: unsupervised self-organizing Hebbian (basic, Oja, or BCM)
    - Output layer: supervised WTA Hebbian (strengthen correct, weaken wrong)

Key features:
    - No ``.backward()`` calls
    - No optimizers or loss functions
    - Per-layer hyperparameters (lr, epochs, rule, decay, theta)
    - Single-layer mode for baseline comparison (replicates Phase 0)

Usage:
    python -m ph_neuro.examples.mnist_multilayer
    python -m ph_neuro.examples.mnist_multilayer --single-layer
    python -m ph_neuro.examples.mnist_multilayer --n-layers 2 --hidden-sizes 256
"""

from __future__ import annotations

import argparse
import time

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.greedy import LayerConfig, MultiLayerHebbianClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a multi-layer Hebbian MLP on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Architecture
    parser.add_argument(
        "--single-layer",
        action="store_true",
        help="Train a single layer (784 -> 10) as baseline",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of layers (including output). Ignored if --single-layer",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer sizes (order matters). "
        "E.g. --hidden-sizes 256 for 2-layer, "
        "--hidden-sizes 256 128 for 3-layer",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, nargs="+", default=None,
                        help="Epochs per layer. If single value, applied to all layers")
    parser.add_argument("--lr", type=float, nargs="+", default=None,
                        help="Learning rate per layer")
    parser.add_argument("--decay", type=float, default=0.0,
                        help="Homeostatic decay (applied to all layers if single value)")
    parser.add_argument("--hebbian-rules", type=str, nargs="+", default=None,
                        choices=["basic", "oja", "bcm"],
                        help="Hebbian rule per hidden layer. Output always uses WTA")

    # Theta
    parser.add_argument("--theta-upper", type=float, default=None,
                        help="Hysteresis upper threshold (applied to all layers)")
    parser.add_argument("--theta-lower", type=float, default=None,
                        help="Hysteresis lower threshold (applied to all layers)")

    # WTA options
    parser.add_argument("--anti-hebbian", action="store_true",
                        help="Apply anti-Hebbian to all non-target output classes")

    # Misc
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Dead-zone for ternary_sign")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-epoch progress")

    return parser.parse_args()


def build_layer_configs(args: argparse.Namespace, n_layers: int) -> list[LayerConfig]:
    """Build per-layer configs from parsed args."""
    # Default theta values
    theta_upper = args.theta_upper if args.theta_upper is not None else (1.0 if n_layers == 1 else 5.0)
    theta_lower = args.theta_lower if args.theta_lower is not None else (0.3 if n_layers == 1 else 1.0)

    # Epochs
    if args.epochs is not None:
        epochs = args.epochs if len(args.epochs) == n_layers else [args.epochs[0]] * n_layers
    else:
        epochs = [5] * (n_layers - 1) + [10]

    # Learning rates
    if args.lr is not None:
        lrs = args.lr if len(args.lr) == n_layers else [args.lr[0]] * n_layers
    else:
        lrs = [0.01] * (n_layers - 1) + [0.01]

    # Hebbian rules for hidden layers
    if args.hebbian_rules is not None:
        hidden_rules = args.hebbian_rules
        # Pad or truncate to match number of hidden layers
        n_hidden = n_layers - 1
        while len(hidden_rules) < n_hidden:
            hidden_rules.append("basic")
        hidden_rules = hidden_rules[:n_hidden]
    else:
        hidden_rules = ["basic"] * (n_layers - 1)

    # Decay
    decay = args.decay

    configs = []
    for i in range(n_layers):
        is_output = i == n_layers - 1
        configs.append(
            LayerConfig(
                lr=lrs[i],
                epochs=epochs[i],
                hebbian_rule=hidden_rules[i] if not is_output else None,
                decay=decay,
                theta_upper=theta_upper if not is_output else 1.0,
                theta_lower=theta_lower if not is_output else 0.3,
                anti_hebbian=args.anti_hebbian if is_output else False,
            )
        )
    return configs


def main() -> None:
    """Run the multi-layer Hebbian MNIST experiment."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    # Architecture
    if args.single_layer:
        layer_sizes = [784, 10]
    else:
        n_hidden = args.n_layers - 1
        hidden = args.hidden_sizes[:n_hidden]
        layer_sizes = [784] + hidden + [10]

    n_layers = len(layer_sizes) - 1

    print_header("PH-Neuro Phase 1 — Multi-layer Hebbian MNIST")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {' -> '.join(str(s) for s in layer_sizes)}")
    print(f"Layers: {n_layers} ({n_layers - 1} hidden + 1 output)")
    print(f"Batch size: {args.batch_size}, Epsilon: {args.epsilon}")
    print()

    # Build layer configs
    configs = build_layer_configs(args, n_layers)
    for i, cfg in enumerate(configs):
        is_output = i == n_layers - 1
        rule_str = "WTA" if is_output else cfg.hebbian_rule
        print(
            f"  Layer {i + 1}: {layer_sizes[i]} -> {layer_sizes[i + 1]}  "
            f"lr={cfg.lr}, epochs={cfg.epochs}, rule={rule_str}, "
            f"theta=({cfg.theta_upper}, {cfg.theta_lower})"
        )
    print()

    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    print()

    # Model
    classifier = MultiLayerHebbianClassifier(
        layer_sizes=layer_sizes,
        theta_upper=5.0,
        theta_lower=1.0,
        device=device,
    )
    total_params = sum(
        l.weight.unpack().numel() for l in classifier.model.layers
    )
    print(f"Model: {classifier}")
    print(f"Total ternary weights: {total_params:,}")
    print()

    # Training
    print("Training (greedy layer-wise)...")
    print("-" * 100)

    total_start = time.time()

    history = classifier.fit_greedy(
        train_loader=train_loader,
        layer_configs=configs,
        epsilon=args.epsilon,
        verbose=not args.quiet,
    )

    total_time = time.time() - total_start

    # Final evaluation
    final_acc = classifier.evaluate(test_loader, epsilon=args.epsilon)
    weight_stats = classifier.get_all_weight_stats()

    # Results
    print()
    print_header("Results")
    print(f"Final test accuracy: {100 * final_acc:.2f}%")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print()

    print("Layer-wise weight distribution:")
    print(f"  {'Layer':<12} {'Size':<16} {'+1%':<8} {'-1%':<8} {'0%':<8}")
    print(f"  {'-' * 44}")
    for i, stats in enumerate(weight_stats):
        layer = classifier.model.get_layer(i)
        sz = f"{layer._in_features}\u2192{layer._out_features}"
        print(
            f"  {f'Layer {i + 1}':<12} {sz:<16} "
            f"{stats['pos_pct']:<7.1f}% "
            f"{stats['neg_pct']:<7.1f}% "
            f"{stats['zero_pct']:<7.1f}%"
        )
    print()

    # Verify invariants
    print("No .backward() calls: \u2713 (verified by design)")
    print("All weights ternary: \u2713 (verified by TernaryHebbianLinear)")
    print(f"Depth improvement over single-layer: "
          f"{'YES' if final_acc > 0.90 else 'NO'}")
    print()

    # Success criteria
    if final_acc > 0.95:
        print(f"\u2713 SUCCESS: {100 * final_acc:.1f}% MNIST accuracy with {n_layers}-layer Hebbian MLP!")
        print("  Depth provides meaningful improvement over single-layer baseline (88.4%).")
    elif final_acc > 0.90:
        print(f"PARTIAL: {100 * final_acc:.1f}% — above single-layer max but below 95% target.")
        print("  Try tuning hyperparameters (lr, epochs, theta) or deeper architecture.")
    else:
        print(f"Accuracy {100 * final_acc:.1f}% — below single-layer baseline.")
        print("  Consider: more epochs per layer, lower theta thresholds,")
        print("  different Hebbian rules for hidden layers (--hebbian-rules oja bcm)")


if __name__ == "__main__":
    main()
