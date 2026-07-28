"""Ablation runner for multi-layer Hebbian MLP on MNIST.

Systematically sweeps over the 5 ablation axes from Phase 1:

1. Depth: 1-layer, 2-layer, 3-layer
2. Anti-Hebbian: on/off for output layer
3. Homeostatic decay: 0, 1e-5, 1e-4 for hidden layers
4. Theta thresholds: default, low, high per layer
5. Hebbian variants for hidden layers: basic, Oja, BCM

All runs use the same data and evaluation protocol. Results are
printed as a formatted table for easy comparison.

Usage:
    python -m ph_neuro.examples.ablation_multilayer
    python -m ph_neuro.examples.ablation_multilayer --quick  # fewer combos
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch

from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.greedy import LayerConfig, MultiLayerHebbianClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for multi-layer Hebbian MLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run a reduced set of ablation combos")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# ── Ablation space ───────────────────────────────────────────────

def get_ablation_configs(quick: bool = False) -> list[dict]:
    """Generate all ablation configurations.

    Each config is a dict with:
        - name: short description
        - params: dict of parameters (for display)
        - layer_sizes: list of layer sizes
        - configs: list of LayerConfig
    """
    base_hidden = dict(lr=0.01, epochs=5, decay=0.0, theta_upper=5.0, theta_lower=1.0)
    base_output = dict(lr=0.01, epochs=10, decay=0.0, theta_upper=1.0, theta_lower=0.3)

    configs = []

    # ── Depth ablation ────────────────────────────────────────
    for depth, hidden_sizes in [
        ("1-layer", []),
        ("2-layer", [256]),
        ("3-layer", [256, 128]),
    ]:
        n = len(hidden_sizes) + 1  # hidden + output
        layer_cfgs = []
        for i in range(n):
            is_output = i == n - 1
            if is_output:
                layer_cfgs.append(LayerConfig(**base_output, anti_hebbian=False))
            else:
                layer_cfgs.append(LayerConfig(**base_hidden, hebbian_rule="basic"))
        configs.append({
            "name": f"Depth: {depth}",
            "params": {"depth": depth},
            "layer_sizes": [784] + hidden_sizes + [10],
            "configs": layer_cfgs,
        })

    if quick:
        return configs

    # ── Anti-Hebbian ablation (3-layer) ───────────────────────
    for anti in [False, True]:
        n = 3
        layer_cfgs = []
        for i in range(n):
            is_output = i == n - 1
            if is_output:
                layer_cfgs.append(LayerConfig(**base_output, anti_hebbian=anti))
            else:
                layer_cfgs.append(LayerConfig(**base_hidden, hebbian_rule="basic"))
        configs.append({
            "name": f"Anti-Hebbian: {anti}",
            "params": {"anti_hebbian": anti},
            "layer_sizes": [784, 256, 128, 10],
            "configs": layer_cfgs,
        })

    # ── Decay ablation (3-layer) ──────────────────────────────
    for decay in [0.0, 1e-5, 1e-4]:
        n = 3
        layer_cfgs = []
        for i in range(n):
            is_output = i == n - 1
            cfg = dict(base_output if is_output else base_hidden)
            cfg["decay"] = decay
            layer_cfgs.append(LayerConfig(**cfg))
        configs.append({
            "name": f"Decay: {decay}",
            "params": {"decay": decay},
            "layer_sizes": [784, 256, 128, 10],
            "configs": layer_cfgs,
        })

    # ── Theta ablation (3-layer) ──────────────────────────────
    for theta_label, th_u_h, th_l_h, th_u_o, th_l_o in [
        ("Theta: low", 2.0, 0.3, 0.5, 0.1),
        ("Theta: default", 5.0, 1.0, 1.0, 0.3),
        ("Theta: high", 10.0, 5.0, 3.0, 1.0),
    ]:
        n = 3
        layer_cfgs = []
        for i in range(n):
            is_output = i == n - 1
            cfg = dict(base_output if is_output else base_hidden)
            cfg["theta_upper"] = th_u_o if is_output else th_u_h
            cfg["theta_lower"] = th_l_o if is_output else th_l_h
            layer_cfgs.append(LayerConfig(**cfg))
        configs.append({
            "name": f"Theta: {theta_label}",
            "params": {"th_u": th_u_h, "th_l": th_l_h},
            "layer_sizes": [784, 256, 128, 10],
            "configs": layer_cfgs,
        })

    # ── Hebbian variant ablation (3-layer) ────────────────────
    for rule in ["basic", "oja", "bcm"]:
        n = 3
        layer_cfgs = []
        for i in range(n):
            is_output = i == n - 1
            if is_output:
                layer_cfgs.append(LayerConfig(**base_output, anti_hebbian=False))
            else:
                layer_cfgs.append(LayerConfig(**base_hidden, hebbian_rule=rule))
        configs.append({
            "name": f"Hidden rule: {rule}",
            "params": {"hidden_rule": rule},
            "layer_sizes": [784, 256, 128, 10],
            "configs": layer_cfgs,
        })

    return configs


def print_results_table(results: list[dict]) -> None:
    """Print ablation results as a formatted table."""
    print()
    print(f"  {'Ablation':<30} {'Test Acc':<10} {'Params':<12} {'Time':<10}")
    print(f"  {'-' * 62}")
    for r in results:
        acc_str = f"{100 * r['accuracy']:.2f}%" if r["accuracy"] is not None else "  N/A"
        print(
            f"  {r['name']:<30} {acc_str:<10} "
            f"{r['n_params']:<12,} {r['time']:<10.1f}"
        )
    print()


def main() -> None:
    """Run all ablations."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    print("PH-Neuro Phase 1 — Ablation: Multi-layer Hebbian MLP on MNIST")
    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")
    print()

    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    ablated_configs = get_ablation_configs(quick=args.quick)

    print(f"Running {len(ablated_configs)} ablation experiments...")
    print()

    results = []

    for idx, cfg in enumerate(ablated_configs):
        print(f"  [{idx + 1}/{len(ablated_configs)}] {cfg['name']} "
              f"({' -> '.join(str(s) for s in cfg['layer_sizes'])})")

        classifier = MultiLayerHebbianClassifier(
            layer_sizes=cfg["layer_sizes"],
            device=device,
        )

        total_params = sum(
            l.weight.unpack().numel() for l in classifier.model.layers
        )

        start_time = time.time()

        try:
            classifier.fit_greedy(
                train_loader=train_loader,
                layer_configs=cfg["configs"],
                epsilon=args.epsilon,
                verbose=False,
            )

            acc = classifier.evaluate(test_loader, epsilon=args.epsilon)
            elapsed = time.time() - start_time

        except Exception as e:
            print(f"    ERROR: {e}")
            acc = None
            elapsed = time.time() - start_time

        results.append({
            "name": cfg["name"],
            "accuracy": acc,
            "n_params": total_params,
            "time": elapsed,
        })

        acc_str = f"{100 * acc:.2f}%" if acc is not None else "ERROR"
        print(f"    Accuracy: {acc_str}  Time: {elapsed:.1f}s")
        print()

    # Results table
    print("=" * 70)
    print("  ABLATION RESULTS")
    print("=" * 70)
    print_results_table(results)

    # Summary
    best = max(results, key=lambda r: r["accuracy"] if r["accuracy"] is not None else 0)
    print(f"Best: {best['name']} — {100 * best['accuracy']:.2f}%")

    if best["accuracy"] and best["accuracy"] > 0.95:
        print(f"\n\u2713 SUCCESS: Best ablation exceeds 95% accuracy target!")
    elif best["accuracy"] and best["accuracy"] > 0.90:
        print(f"\nPARTIAL: Best ablation exceeds single-layer max (88.4%).")
    else:
        print(f"\nBest ablation did not exceed single-layer baseline.")


if __name__ == "__main__":
    main()
