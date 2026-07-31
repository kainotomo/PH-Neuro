#!/usr/bin/env python3
"""L7 Results Aggregator — collects depth-vs-width results and produces comparison tables.

Usage:
    # Aggregate all results from the output directory
    python -m ph_neuro.examples.aggregate_l7_results --results-dir l7_results

    # Produce comparison table as CSV
    python -m ph_neuro.examples.aggregate_l7_results --results-dir l7_results --output summary.csv

Output:
    - Console: formatted comparison tables (Accuracy vs Depth, Ternary Gap vs Depth)
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

FORMAT_LABELS = {
    "ternary": "Ternary STE",
    "fp16": "FP16",
}


# ── Loading ─────────────────────────────────────────────────────────


def load_results(results_dir: str) -> dict:
    """Load all L7 JSON result files from a directory.

    Returns:
        Dict: ``{(dataset, depth, weight_format, seed): result_dict}``
    """
    results: dict = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        key = (data["dataset"], data["depth"], data["weight_format"], data["seed"])
        results[key] = data
    return results


# ── Aggregation ─────────────────────────────────────────────────────


def aggregate_by_depth_format(results: dict) -> dict:
    """Aggregate results by (dataset, depth, weight_format), computing mean and std.

    Returns:
        Nested dict: ``{dataset: {depth: {weight_format: {metric: value}}}}``
    """
    grouped: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for (dataset, depth, weight_format, _seed), data in results.items():
        grouped[dataset][depth][weight_format].append(data)

    aggregated: dict = defaultdict(dict)

    for dataset in sorted(grouped.keys()):
        for depth in sorted(grouped[dataset].keys()):
            aggregated[dataset][depth] = {}
            for weight_format in sorted(grouped[dataset][depth].keys()):
                entries = grouped[dataset][depth][weight_format]
                accs = [e["best_accuracy"] for e in entries]
                times = [e["training_time_seconds"] for e in entries]
                n_params = entries[0]["n_parameters"]

                summary = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "accuracy_max": float(np.max(accs)),
                    "time_mean": float(np.mean(times)),
                    "n_runs": len(entries),
                    "n_parameters": int(n_params),
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

                # Efficiency: accuracy per 100K params
                if n_params > 0:
                    summary["accuracy_per_100k_params"] = (
                        100.0 * summary["accuracy_mean"] * 100000 / n_params
                    )

                aggregated[dataset][depth][weight_format] = summary

    return aggregated


def compute_ternary_gaps(aggregated: dict) -> dict:
    """Compute ternary gap (FP16 accuracy - ternary accuracy) per dataset × depth.

    Returns:
        Dict: ``{dataset: {depth: {"ternary_gap": float}}}``
    """
    gaps: dict = defaultdict(dict)
    for dataset in sorted(aggregated.keys()):
        for depth in sorted(aggregated[dataset].keys()):
            depth_data = aggregated[dataset][depth]
            if "ternary" in depth_data and "fp16" in depth_data:
                ternary_acc = depth_data["ternary"]["accuracy_mean"]
                fp16_acc = depth_data["fp16"]["accuracy_mean"]
                gaps[dataset][depth] = {
                    "ternary_gap": fp16_acc - ternary_acc,
                    "ternary_acc": ternary_acc,
                    "fp16_acc": fp16_acc,
                }
    return gaps


# ── Display ─────────────────────────────────────────────────────────


def print_summary_table(aggregated: dict, gaps: dict) -> None:
    """Print formatted comparison tables."""
    for dataset in sorted(aggregated.keys()):
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        print(f"\n{'=' * 70}")
        print(f"  L7 Results: {dataset_label}")
        print(f"{'=' * 70}")

        depths = sorted(aggregated[dataset].keys())

        # ── Accuracy vs Depth table ────────────────────────────────
        print(f"\n  {'Accuracy vs Depth':^66}")
        print(f"  {'─' * 66}")
        header = (
            f"  {'Depth':>6} | {'Layers':<28} | {'Params':>8} | "
            f"{'Ternary STE':>12} | {'FP16':>12} | {'Gap':>6}"
        )
        print(header)
        print(f"  {'─' * 66}")

        for depth in depths:
            layer_sizes_str = "×".join(str(s) for s in ["784"] + ["h"] * depth + ["10"])
            n_params = list(aggregated[dataset][depth].values())[0]["n_parameters"]

            ternary_str = "     —     "
            if "ternary" in aggregated[dataset][depth]:
                t = aggregated[dataset][depth]["ternary"]
                ternary_str = f"{100 * t['accuracy_mean']:6.2f}% ± {100 * t['accuracy_std']:.2f}"

            fp16_str = "     —     "
            if "fp16" in aggregated[dataset][depth]:
                f_val = aggregated[dataset][depth]["fp16"]
                fp16_str = f"{100 * f_val['accuracy_mean']:6.2f}% ± {100 * f_val['accuracy_std']:.2f}"

            gap_str = "  —  "
            if dataset in gaps and depth in gaps[dataset]:
                gap = gaps[dataset][depth]["ternary_gap"]
                gap_str = f"{100 * gap:5.2f}pp"

            print(
                f"  D={depth:1d}     | {layer_sizes_str:<28} | "
                f"{n_params:>8,} | {ternary_str:>12} | {fp16_str:>12} | {gap_str:>6}"
            )

        # ── Weight sparsity table ──────────────────────────────────
        print(f"\n  {'Weight Sparsity (% of weights = 0)':^50}")
        print(f"  {'─' * 50}")
        header2 = f"  {'Depth':>6} | {'Ternary STE':>14} | {'FP16':>14}"
        print(header2)
        print(f"  {'─' * 50}")
        for depth in depths:
            t_sparsity = "      —      "
            if "ternary" in aggregated[dataset][depth]:
                t_sparsity = f"{aggregated[dataset][depth]['ternary'].get('weight_sparsity_pct', 0):6.2f}%"

            f_sparsity = "      —      "
            if "fp16" in aggregated[dataset][depth]:
                f_sparsity = f"{aggregated[dataset][depth]['fp16'].get('weight_sparsity_pct', 0):6.2f}%"

            print(f"  D={depth:1d}     | {t_sparsity:>14} | {f_sparsity:>14}")

        # ── Training time table ────────────────────────────────────
        print(f"\n  {'Training Time (seconds)':^40}")
        print(f"  {'─' * 40}")
        header3 = f"  {'Depth':>6} | {'Ternary STE':>12} | {'FP16':>12}"
        print(header3)
        print(f"  {'─' * 40}")
        for depth in depths:
            t_time = "     —     "
            if "ternary" in aggregated[dataset][depth]:
                t_time = f"{aggregated[dataset][depth]['ternary']['time_mean']:6.1f}s"

            f_time = "     —     "
            if "fp16" in aggregated[dataset][depth]:
                f_time = f"{aggregated[dataset][depth]['fp16']['time_mean']:6.1f}s"

            print(f"  D={depth:1d}     | {t_time:>12} | {f_time:>12}")


def to_csv(aggregated: dict, gaps: dict, output_path: str) -> None:
    """Write aggregated results to a CSV file."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "depth", "weight_format",
            "accuracy_mean", "accuracy_std", "accuracy_max",
            "time_mean_seconds", "n_parameters",
            "weight_sparsity_pct", "weight_zero_pct",
            "weight_pos_pct", "weight_neg_pct",
            "n_runs", "ternary_gap_pp",
        ])

        for dataset in sorted(aggregated.keys()):
            for depth in sorted(aggregated[dataset].keys()):
                for weight_format in sorted(aggregated[dataset][depth].keys()):
                    s = aggregated[dataset][depth][weight_format]
                    ternary_gap = ""
                    if dataset in gaps and depth in gaps[dataset]:
                        ternary_gap = f"{100 * gaps[dataset][depth]['ternary_gap']:.4f}"
                    writer.writerow([
                        dataset, depth, weight_format,
                        f"{s['accuracy_mean']:.6f}",
                        f"{s['accuracy_std']:.6f}",
                        f"{s['accuracy_max']:.6f}",
                        f"{s['time_mean']:.2f}",
                        s["n_parameters"],
                        s.get("weight_sparsity_pct", ""),
                        s.get("weight_zero_pct", ""),
                        s.get("weight_pos_pct", ""),
                        s.get("weight_neg_pct", ""),
                        s["n_runs"],
                        ternary_gap,
                    ])

    print(f"\nCSV saved: {output_path}")


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L7 Results Aggregator — Depth vs Width Scaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="l7_results",
        help="Directory containing L7 result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    """Aggregate and display L7 results."""
    args = parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}")
        return

    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result files from {args.results_dir}")
    aggregated = aggregate_by_depth_format(results)
    gaps = compute_ternary_gaps(aggregated)
    print_summary_table(aggregated, gaps)

    if args.output:
        to_csv(aggregated, gaps, args.output)


if __name__ == "__main__":
    main()
