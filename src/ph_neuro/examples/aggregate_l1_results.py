#!/usr/bin/env python3
"""L1 Results Aggregator — collects JSON results and produces comparison tables.

Usage:
    # Aggregate all results from the output directory
    python -m ph_neuro.examples.aggregate_l1_results --results-dir l1_results

    # Produce comparison table as CSV
    python -m ph_neuro.examples.aggregate_l1_results --results-dir l1_results --output summary.csv

Output:
    - Console: formatted comparison tables
    - CSV: optionally written to file
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


# ── Config ──────────────────────────────────────────────────────────

VARIANT_LABELS = {
    "v1": "Ternary STE (ours)",
    "v2": "FP16",
    "v3": "INT8 QAT",
    "v4": "INT4 QAT",
    "v5": "Hebbian v1",
}

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
}


# ── Loading ─────────────────────────────────────────────────────────


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all JSON result files from a directory.

    Returns:
        Dict: ``{(dataset, variant, seed): result_dict}``
    """
    results: dict = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        key = (data["dataset"], data["variant"], data["seed"])
        results[key] = data
    return results


# ── Aggregation ─────────────────────────────────────────────────────


def aggregate_by_dataset_variant(results: dict) -> dict:
    """Aggregate results by (dataset, variant), computing mean and std.

    Returns:
        Nested dict: ``{dataset: {variant: {metric: value}}}``
    """
    grouped: dict = defaultdict(lambda: defaultdict(list))

    for (dataset, variant, _seed), data in results.items():
        grouped[dataset][variant].append(data)

    aggregated: dict = defaultdict(dict)

    for dataset in sorted(grouped.keys()):
        for variant in sorted(grouped[dataset].keys()):
            entries = grouped[dataset][variant]
            accs = [e["best_accuracy"] for e in entries]
            times = [e["training_time_seconds"] for e in entries]

            # Build per-variant summary
            summary = {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "accuracy_max": float(np.max(accs)),
                "time_mean": float(np.mean(times)),
                "n_runs": len(entries),
            }

            # Weight stats (from first run)
            first = entries[0]
            if "weight_sparsity_pct" in first:
                summary["weight_sparsity_pct"] = first["weight_sparsity_pct"]
            if "weight_zero_pct" in first:
                summary["weight_zero_pct"] = first["weight_zero_pct"]
            if "weight_pos_pct" in first:
                summary["weight_pos_pct"] = first["weight_pos_pct"]
            if "weight_neg_pct" in first:
                summary["weight_neg_pct"] = first["weight_neg_pct"]
            if "n_parameters" in first:
                summary["n_parameters"] = first["n_parameters"]

            aggregated[dataset][variant] = summary

    return dict(aggregated)


# ── Formatting ──────────────────────────────────────────────────────


def print_comparison_table(aggregated: dict) -> None:
    """Print a Markdown-style comparison table."""
    datasets = sorted(aggregated.keys())
    variants = ["v1", "v2", "v3", "v4", "v5"]

    # Header
    print(f"{'Dataset':<14}", end="")
    for v in variants:
        print(f"  {VARIANT_LABELS[v]:>24}", end="")
    print()

    print(f"{'':-<14}", end="")
    for _ in variants:
        print(f"  {'':-<24}", end="")
    print()

    for ds in datasets:
        print(f"{DATASET_LABELS[ds]:<14}", end="")
        for v in variants:
            if v in aggregated.get(ds, {}):
                s = aggregated[ds][v]
                mean = s["accuracy_mean"]
                std = s["accuracy_std"]
                n = s["n_runs"]
                time_s = s["time_mean"]
                print(f"  {100 * mean:5.2f}% ± {100 * std:.2f}%  ", end="")
            else:
                print(f"  {'—':>24}", end="")
        print()

    print()


def print_memory_table(aggregated: dict) -> None:
    """Print weight memory comparison table."""
    datasets = sorted(aggregated.keys())
    variants = ["v1", "v2", "v3", "v4"]

    print("### Weight Memory (parameters) & Sparsity")
    print()
    print(f"{'Dataset':<14}", end="")
    for v in variants:
        print(f"  {VARIANT_LABELS[v]:>24}", end="")
    print()

    print(f"{'':-<14}", end="")
    for _ in variants:
        print(f"  {'':-<24}", end="")
    print()

    for ds in datasets:
        print(f"{DATASET_LABELS[ds]:<14}", end="")
        for v in variants:
            if v in aggregated.get(ds, {}):
                s = aggregated[ds][v]
                n_params = s.get("n_parameters", 0)
                sparsity = s.get("weight_sparsity_pct", 0)
                print(f"  {n_params:>8,} params  ", end="")
            else:
                print(f"  {'—':>24}", end="")
        print()

    print()


def print_ternary_gap_table(aggregated: dict) -> None:
    """Print ternary gap (v2 - v1 accuracy) for each dataset."""
    print("### Ternary Gap (FP16 − Ternary STE)")
    print()
    print(f"{'Dataset':<14}  {'FP16':>8}  {'Ternary STE':>14}  {'Gap (pp)':>10}")
    print(f"{'':-<14}  {'':->8}  {'':->14}  {'':->10}")
    for ds in sorted(aggregated.keys()):
        v1 = aggregated[ds].get("v1", {})
        v2 = aggregated[ds].get("v2", {})
        if v1 and v2:
            v1_acc = v1["accuracy_mean"]
            v2_acc = v2["accuracy_mean"]
            gap = v2_acc - v1_acc
            print(
                f"{DATASET_LABELS[ds]:<14}  "
                f"{100 * v2_acc:6.2f}%  "
                f"{100 * v1_acc:6.2f}%  "
                f"{100 * gap:>+7.2f} pp"
            )
    print()


def save_csv(aggregated: dict, output_path: str) -> None:
    """Save comparison table as CSV."""
    datasets = sorted(aggregated.keys())
    variants = ["v1", "v2", "v3", "v4", "v5"]

    with open(output_path, "w") as f:
        # Header
        f.write("dataset," + ",".join(variants) + "\n")
        for ds in datasets:
            row = [DATASET_LABELS[ds]]
            for v in variants:
                if v in aggregated.get(ds, {}):
                    s = aggregated[ds][v]
                    row.append(f'{100 * s["accuracy_mean"]:.2f}')
                else:
                    row.append("")
            f.write(",".join(row) + "\n")


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L1: Aggregate experiment results and produce comparison tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="l1_results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output file path",
    )
    return parser.parse_args()


def main() -> None:
    """Run the aggregator."""
    args = parse_args()

    print("=" * 72)
    print("  L1: Ternary STE Baseline Suite — Results Summary")
    print("=" * 72)
    print()

    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in '{args.results_dir}'.")
        print("Run the experiment first:")
        print(f"  python -m ph_neuro.examples.run_l1_baseline_suite --dataset mnist --variant v1")
        return

    print(f"Loaded {len(results)} result files from '{args.results_dir}'")
    print()

    aggregated = aggregate_by_dataset_variant(results)

    # Print tables
    print_comparison_table(aggregated)
    print_ternary_gap_table(aggregated)
    print_memory_table(aggregated)

    # Save CSV
    if args.output:
        save_csv(aggregated, args.output)
        print(f"CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
