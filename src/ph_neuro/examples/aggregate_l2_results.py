#!/usr/bin/env python3
"""L2 Results Aggregator — collects Hysteresis-STE ablation results.

Produces comparison tables for accuracy, sparsity, and flip rate
across theta_upper × theta_lower combinations, alongside standard
STE (control) baselines.

Usage:
    # Aggregate all results
    python -m ph_neuro.examples.aggregate_l2_results --results-dir l2_results

    # Output as CSV
    python -m ph_neuro.examples.aggregate_l2_results --results-dir l2_results --output summary.csv

Output:
    - Console: formatted comparison tables grouped by dataset
    - CSV: optionally written to file
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# ── Config ──────────────────────────────────────────────────────────

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion": "Fashion-MNIST",
    "kmnist": "KMNIST",
}


# ── Loading ─────────────────────────────────────────────────────────


def load_results(results_dir: str) -> list[dict]:
    """Load all L2 result JSON files from a directory.

    Returns:
        List of result dicts.
    """
    results: list[dict] = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith("results_"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        results.append(data)
    return results


# ── Table format helpers ────────────────────────────────────────────


def _fmt(val: float, decimals: int = 2) -> str:
    """Format a float with given decimal places."""
    return f"{val:.{decimals}f}"


def _fmt_pct(val: float, decimals: int = 2) -> str:
    """Format a percentage value."""
    return f"{val:.{decimals}f}%"


# ── Per-dataset tables ──────────────────────────────────────────────


def print_accuracy_table(results: list[dict], dataset: str) -> None:
    """Print accuracy table: theta_upper × theta_lower."""
    print(f"\n{'=' * 72}")
    print(f"  {DATASET_LABELS.get(dataset, dataset.upper())}")
    print(f"{'=' * 72}")

    # Separate control vs hysteresis
    control_acc: float | None = None
    hyst_results: dict[tuple[float, float], float] = {}

    for r in results:
        if r.get("dataset") != dataset:
            continue
        if r.get("variant") == "control":
            control_acc = r.get("best_accuracy", 0.0)
        else:
            tu = r.get("theta_upper", 0.0)
            tl = r.get("theta_lower", 0.0)
            hyst_results[(tu, tl)] = r.get("best_accuracy", 0.0)

    # Collect unique thresholds
    thetas_upper = sorted(set(tu for tu, _ in hyst_results))
    thetas_lower = sorted(set(tl for _, tl in hyst_results))

    # ── Accuracy table ──────────────────────────────────────────────
    print(f"\n  Accuracy (%):  θ_lower →")
    header = "  θ_upper " + " ".join(f"θ_l={tl:<5}" for tl in thetas_lower)
    print(f"  {'─' * len(header)}")
    print(header)

    for tu in thetas_upper:
        row = f"  {tu:<8}"
        for tl in thetas_lower:
            if (tu, tl) in hyst_results:
                row += f"  {_fmt(100 * hyst_results[(tu, tl)]):>5}"
            else:
                row += f"  {' — ':>5}"
        print(row)

    if control_acc is not None:
        print(f"\n  Standard STE (control): {_fmt(100 * control_acc)}%")

    # ── Sparsity table ──────────────────────────────────────────────
    print(f"\n  Final Sparsity (%):  θ_lower →")
    print(f"  {'─' * len(header)}")
    print(header)

    hyst_sparsity: dict[tuple[float, float], float] = {}
    control_sparsity: float | None = None

    for r in results:
        if r.get("dataset") != dataset:
            continue
        if r.get("variant") == "control":
            control_sparsity = r.get("final_weight_sparsity_pct", 0.0)
        else:
            tu = r.get("theta_upper", 0.0)
            tl = r.get("theta_lower", 0.0)
            hyst_sparsity[(tu, tl)] = r.get("final_weight_sparsity_pct", 0.0)

    for tu in thetas_upper:
        row = f"  {tu:<8}"
        for tl in thetas_lower:
            if (tu, tl) in hyst_sparsity:
                row += f"  {_fmt(hyst_sparsity[(tu, tl)]):>5}"
            else:
                row += f"  {' — ':>5}"
        print(row)

    if control_sparsity is not None:
        print(f"\n  Standard STE (control): {_fmt(control_sparsity)}%")

    # ── Flip rate table ─────────────────────────────────────────────
    print(f"\n  Avg Flip Rate (%/epoch, last 5):  θ_lower →")
    print(f"  {'─' * len(header)}")
    print(header)

    hyst_flips: dict[tuple[float, float], float] = {}
    control_flips: float | None = None

    for r in results:
        if r.get("dataset") != dataset:
            continue
        if r.get("variant") == "control":
            control_flips = r.get("avg_flip_rate_last_5_epochs_pct", 0.0)
        else:
            tu = r.get("theta_upper", 0.0)
            tl = r.get("theta_lower", 0.0)
            hyst_flips[(tu, tl)] = r.get("avg_flip_rate_last_5_epochs_pct", 0.0)

    for tu in thetas_upper:
        row = f"  {tu:<8}"
        for tl in thetas_lower:
            if (tu, tl) in hyst_flips:
                row += f"  {_fmt(hyst_flips[(tu, tl)], 3):>5}"
            else:
                row += f"  {' — ':>5}"
        print(row)

    if control_flips is not None:
        print(f"\n  Standard STE (control): {_fmt(control_flips, 3)}%/epoch")

    # ── Hysteresis zone table ───────────────────────────────────────
    print(f"\n  Hysteresis Zone — % in Gap:  θ_lower →")
    print(f"  {'─' * len(header)}")
    print(header)

    hyst_gap: dict[tuple[float, float], float] = {}

    for r in results:
        if r.get("dataset") != dataset:
            continue
        if r.get("variant") != "control":
            tu = r.get("theta_upper", 0.0)
            tl = r.get("theta_lower", 0.0)
            hyst_gap[(tu, tl)] = r.get("hysteresis_pct_in_gap", 0.0)

    for tu in thetas_upper:
        row = f"  {tu:<8}"
        for tl in thetas_lower:
            if (tu, tl) in hyst_gap:
                row += f"  {_fmt(hyst_gap[(tu, tl)]):>5}"
            else:
                row += f"  {' — ':>5}"
        print(row)

    # ── Convergence speed ───────────────────────────────────────────
    print(f"\n  Convergence (epochs to 95% best):  θ_lower →")
    print(f"  {'─' * len(header)}")
    print(header)

    hyst_conv: dict[tuple[float, float], int] = {}
    control_conv: int | None = None

    for r in results:
        if r.get("dataset") != dataset:
            continue
        if r.get("variant") == "control":
            control_conv = r.get("convergence_epoch_95pct", 0)
        else:
            tu = r.get("theta_upper", 0.0)
            tl = r.get("theta_lower", 0.0)
            hyst_conv[(tu, tl)] = r.get("convergence_epoch_95pct", 0)

    for tu in thetas_upper:
        row = f"  {tu:<8}"
        for tl in thetas_lower:
            if (tu, tl) in hyst_conv:
                row += f"  {hyst_conv[(tu, tl)]:>5}"
            else:
                row += f"  {' — ':>5}"
        print(row)

    if control_conv is not None:
        print(f"\n  Standard STE (control): {control_conv} epochs")

    # ── Best trade-off summary ──────────────────────────────────────
    print(f"\n  Best Accuracy-Sparsity Trade-off:")
    best_tradeoff: list[tuple[float, float, float, float]] = []
    for (tu, tl), acc in hyst_results.items():
        sp = hyst_sparsity.get((tu, tl), 0.0)
        best_tradeoff.append((acc, sp, tu, tl))

    best_tradeoff.sort(key=lambda x: -x[0])
    print(f"  {'θ_upper':<8} {'θ_lower':<8} {'Accuracy':<10} {'Sparsity':<10}")
    print(f"  {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10}")
    for acc, sp, tu, tl in best_tradeoff[:3]:
        print(f"  {tu:<8.1f} {tl:<8.2f} {100*acc:<10.2f}% {sp:<10.2f}%")

    # Highest sparsity configurations (with non-trivial accuracy)
    print(f"\n  Highest Sparsity (accuracy >= 95% of best):")
    best_acc = max(hyst_results.values()) if hyst_results else 0.0
    threshold = 0.95 * best_acc
    high_sparsity = [
        (acc, sp, tu, tl)
        for (tu, tl), acc in hyst_results.items()
        if acc >= threshold
    ]
    high_sparsity.sort(key=lambda x: -x[1])
    for acc, sp, tu, tl in high_sparsity[:3]:
        print(f"  θ_u={tu:.1f} θ_l={tl:.2f} → Acc: {100*acc:.2f}%, Sparsity: {sp:.2f}%")

    print()


# ── Summary across datasets ─────────────────────────────────────────


def print_cross_dataset_summary(results: list[dict]) -> None:
    """Print a compact summary comparing all datasets."""
    print(f"\n{'=' * 72}")
    print(f"  Cross-Dataset Summary")
    print(f"{'=' * 72}")

    # Group control results by dataset
    control_by_dataset: dict[str, dict] = {}
    for r in results:
        if r.get("variant") == "control":
            control_by_dataset[r.get("dataset", "")] = r

    # Best hysteresis config per dataset
    hyst_best: dict[str, tuple[dict, float, float]] = {}
    for r in results:
        if r.get("variant") != "control":
            ds = r.get("dataset", "")
            acc = r.get("best_accuracy", 0.0)
            if ds not in hyst_best or acc > hyst_best[ds][0].get("best_accuracy", 0.0):
                hyst_best[ds] = (r, r.get("theta_upper", 0.0), r.get("theta_lower", 0.0))

    print(f"\n{'Dataset':<15} {'Method':<30} {'Accuracy':<12} {'Sparsity':<10} {'Flips':<10}")
    print(f"{'─' * 15} {'─' * 30} {'─' * 12} {'─' * 10} {'─' * 10}")

    for ds in sorted(set(r.get("dataset", "") for r in results)):
        # Control
        if ds in control_by_dataset:
            c = control_by_dataset[ds]
            print(
                f"{DATASET_LABELS.get(ds, ds):<15} "
                f"{'Standard STE (control)':<30} "
                f"{_fmt_pct(100 * c.get('best_accuracy', 0.0)):<12} "
                f"{_fmt_pct(c.get('final_weight_sparsity_pct', 0.0)):<10} "
                f"{_fmt(c.get('avg_flip_rate_last_5_epochs_pct', 0.0), 3):<10}"
            )

        # Best hysteresis
        if ds in hyst_best:
            r, tu, tl = hyst_best[ds]
            print(
                f"{'':15} "
                f"Hysteresis-STE θ_u={tu}, θ_l={tl:<8.2f}"
                f"{_fmt_pct(100 * r.get('best_accuracy', 0.0)):<12} "
                f"{_fmt_pct(r.get('final_weight_sparsity_pct', 0.0)):<10} "
                f"{_fmt(r.get('avg_flip_rate_last_5_epochs_pct', 0.0), 3):<10}"
            )

        # Hysteresis with highest sparsity (>= 95% of best acc)
        ds_hyst = [r for r in results if r.get("dataset") == ds and r.get("variant") != "control"]
        if ds_hyst:
            best_ds_acc = max(r.get("best_accuracy", 0.0) for r in ds_hyst)
            threshold = 0.95 * best_ds_acc
            sparsity_results = [
                r for r in ds_hyst if r.get("best_accuracy", 0.0) >= threshold
            ]
            if sparsity_results:
                sparsest = max(sparsity_results, key=lambda r: r.get("final_weight_sparsity_pct", 0.0))
                print(
                    f"{'':15} "
                    f"Sparsest (θ_u={sparsest.get('theta_upper')}, θ_l={sparsest.get('theta_lower'):<8.2f})"
                    f"{_fmt_pct(100 * sparsest.get('best_accuracy', 0.0)):<12} "
                    f"{_fmt_pct(sparsest.get('final_weight_sparsity_pct', 0.0)):<10} "
                    f"{_fmt(sparsest.get('avg_flip_rate_last_5_epochs_pct', 0.0), 3):<10}"
                )

        print()

    print()


# ── CSV export ──────────────────────────────────────────────────────


def export_csv(results: list[dict], output_path: str) -> None:
    """Export all results to a CSV file."""
    import csv

    # Determine all keys present across all results
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.keys())

    # Order keys: core metrics first, then others
    core_keys = [
        "experiment", "dataset", "variant",
        "theta_upper", "theta_lower",
        "seed", "epochs", "batch_size", "learning_rate",
        "best_accuracy", "best_epoch", "final_accuracy",
        "final_weight_sparsity_pct",
        "final_weight_pos_pct", "final_weight_neg_pct", "final_weight_zero_pct",
        "avg_sparsity_last_5_epochs_pct",
        "avg_flip_rate_last_5_epochs_pct", "max_flip_rate_pct",
        "hysteresis_pct_above_upper", "hysteresis_pct_below_lower", "hysteresis_pct_in_gap",
        "convergence_epoch_95pct", "total_parameters",
        "training_time_seconds", "epochs_trained",
    ]
    remaining = sorted(all_keys - set(core_keys))
    ordered_keys = [k for k in core_keys if k in all_keys] + remaining

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV exported to: {output_path}")


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L2 Results Aggregator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="l2_results",
        help="Directory containing L2 result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    """Run the L2 results aggregator."""
    args = parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}")
        print("Run L2 experiments first with scripts/run_l2_ablation.sh")
        return

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result files from {args.results_dir}")
    print(f"  Datasets: {sorted(set(r.get('dataset', '?') for r in results))}")
    print(f"  Variants: {sorted(set(r.get('variant', '?') for r in results))}")

    # Print tables per dataset
    for ds in sorted(set(r.get("dataset", "") for r in results)):
        print_accuracy_table(results, ds)

    # Print cross-dataset summary
    print_cross_dataset_summary(results)

    # Optional CSV export
    if args.output:
        export_csv(results, args.output)


if __name__ == "__main__":
    main()
