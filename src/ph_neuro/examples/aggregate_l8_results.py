#!/usr/bin/env python3
"""L8 Results Aggregator — collects JSON results and produces comparison tables.

Usage:
    # Aggregate all results from the output directory
    python -m ph_neuro.examples.aggregate_l8_results --results-dir l8_results

    # Produce comparison table as text file
    python -m ph_neuro.examples.aggregate_l8_results \\
        --results-dir l8_results --output summary.txt

Output:
    - Console: formatted comparison tables
    - Text file: optionally written to file
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# ── Config ──────────────────────────────────────────────────────────

PROTOCOL_LABELS = {
    "split": "Split MNIST",
    "permuted": "Permuted MNIST",
}

WEIGHT_LABELS = {
    "ternary": "Ternary STE",
    "fp16": "FP16",
}


# ── Loading ─────────────────────────────────────────────────────────


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all JSON result files from a directory.

    Returns:
        Dict: ``{(protocol, weight_format, seed): result_dict}``
    """
    results: dict[str, dict] = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json") or fname == "aggregated_summary.txt":
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        key = (data["protocol"], data["weight_format"], data["seed"])
        results[key] = data
    return results


# ── Aggregation ─────────────────────────────────────────────────────


def aggregate_results(results: dict) -> dict:
    """Aggregate results by (protocol, weight_format), computing mean and std.

    Returns:
        Nested dict: ``{protocol: {weight_format: {metric: value}}}``
    """
    grouped: dict = defaultdict(lambda: defaultdict(list))

    for (protocol, weight_format, _seed), data in results.items():
        grouped[protocol][weight_format].append(data)

    aggregated: dict = defaultdict(dict)

    for protocol in sorted(grouped.keys()):
        for weight_format in sorted(grouped[protocol].keys()):
            entries = grouped[protocol][weight_format]
            forgetting = [e["metrics"]["average_forgetting"] for e in entries]
            accs = [e["metrics"]["average_accuracy"] for e in entries]
            times = [e["total_training_time_seconds"] for e in entries]

            # Per-task forgetting (average across runs)
            n_tasks = len(entries[0]["metrics"]["per_task_forgetting"])
            per_task_forget = np.zeros(n_tasks)
            for e in entries:
                per_task_forget += np.array(e["metrics"]["per_task_forgetting"])
            per_task_forget /= len(entries)

            summary = {
                "forgetting_mean": float(np.mean(forgetting)),
                "forgetting_std": float(np.std(forgetting)),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "time_mean": float(np.mean(times)),
                "n_runs": len(entries),
                "per_task_forgetting_mean": per_task_forget.tolist(),
            }

            aggregated[protocol][weight_format] = summary

    return dict(aggregated)


# ── Formatting ──────────────────────────────────────────────────────


def print_comparison_table(aggregated: dict) -> str:
    """Build a formatted comparison table as a string."""
    lines: list[str] = []
    protocols = sorted(aggregated.keys())
    weight_formats = ["ternary", "fp16"]

    lines.append("")
    lines.append("L8: Forgetting Baseline — Comparison Table")
    lines.append("=" * 80)
    lines.append("")

    # Main metrics table
    lines.append(
        f"{'Protocol':<16} {'Weight':<14} {'Avg Forgetting':>16} {'Avg Accuracy':>16} {'Runs':>6}"
    )
    lines.append("-" * 68)

    for protocol in protocols:
        for wf in weight_formats:
            if wf in aggregated.get(protocol, {}):
                s = aggregated[protocol][wf]
                forget_mean = 100 * s["forgetting_mean"]
                forget_std = 100 * s["forgetting_std"]
                acc_mean = 100 * s["accuracy_mean"]
                acc_std = 100 * s["accuracy_std"]
                n = s["n_runs"]
                lines.append(
                    f"{PROTOCOL_LABELS.get(protocol, protocol):<16} "
                    f"{WEIGHT_LABELS.get(wf, wf):<14} "
                    f"{forget_mean:5.2f}% ± {forget_std:.2f}%  "
                    f"{acc_mean:5.2f}% ± {acc_std:.2f}%  "
                    f"{n:>4}"
                )
        lines.append("")

    # Ternary-FP16 gap table
    lines.append("")
    lines.append("Forgetting Gap (FP16 − Ternary)")
    lines.append("-" * 60)
    lines.append(
        f"{'Protocol':<16} {'FP16 Forgetting':>18} {'Ternary Forgetting':>20} {'Gap (pp)':>10}"
    )
    lines.append("-" * 64)

    for protocol in protocols:
        fp16 = aggregated.get(protocol, {}).get("fp16")
        ternary = aggregated.get(protocol, {}).get("ternary")
        if fp16 and ternary:
            fp16_f = 100 * fp16["forgetting_mean"]
            ternary_f = 100 * ternary["forgetting_mean"]
            gap = fp16_f - ternary_f
            lines.append(
                f"{PROTOCOL_LABELS.get(protocol, protocol):<16} "
                f"{fp16_f:>8.2f}% ± {100 * fp16['forgetting_std']:.2f}%  "
                f"{ternary_f:>8.2f}% ± {100 * ternary['forgetting_std']:.2f}%  "
                f"{gap:>+8.2f} pp"
            )
    lines.append("")

    # Per-task forgetting detail
    lines.append("")
    lines.append("Per-Task Forgetting (averaged across seeds)")
    lines.append("-" * 80)

    for protocol in protocols:
        lines.append(f"\n--- {PROTOCOL_LABELS.get(protocol, protocol)} ---")
        # Determine max tasks
        max_tasks = 0
        for wf in weight_formats:
            if wf in aggregated.get(protocol, {}):
                pt = aggregated[protocol][wf]["per_task_forgetting_mean"]
                max_tasks = max(max_tasks, len(pt))

        lines.append(
            f"{'Task':<8} " + "  ".join(f"{WEIGHT_LABELS.get(wf, wf):>18}" for wf in weight_formats)
        )
        for task_i in range(max_tasks):
            row = f"Task {task_i + 1:<3}"
            for wf in weight_formats:
                if wf in aggregated.get(protocol, {}):
                    pt = aggregated[protocol][wf].get("per_task_forgetting_mean", [])
                    if task_i < len(pt):
                        row += f"  {100 * pt[task_i]:>8.2f}%     "
                    else:
                        row += f"  {'—':>18}"
            lines.append(row)

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def print_ternary_weight_summary(results: dict) -> str:
    """Build a summary of ternary weight statistics across tasks."""
    lines: list[str] = []
    lines.append("")
    lines.append("Ternary Weight Snapshots (first run per protocol)")
    lines.append("-" * 80)
    lines.append("")

    for (protocol, weight_format, seed), data in results.items():
        if weight_format != "ternary" or seed != 42:
            continue
        snapshots = data.get("weight_snapshots", {})
        if not snapshots:
            continue

        lines.append(f"--- {PROTOCOL_LABELS.get(protocol, protocol)} ---")
        lines.append(f"{'After':<12} {'Sparsity (% 0)':>15} {'% +1':>8} {'% -1':>8} {'Params':>10}")
        lines.append("-" * 53)

        for key in sorted(snapshots.keys(), key=lambda k: int(k) if k != "-1" else -1):
            label = "Init" if key == "-1" else f"Task {int(key) + 1}"
            s = snapshots[key]
            lines.append(
                f"{label:<12} "
                f"{s['weight_sparsity_pct']:>8.2f}%    "
                f"{s['weight_pos_pct']:>5.2f}%  "
                f"{s['weight_neg_pct']:>5.2f}%  "
                f"{int(s['n_parameters']):>10,}"
            )
        lines.append("")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L8: Aggregate forgetting baseline results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="l8_results",
        help="Directory containing L8 result JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path (writes to console if not specified)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the aggregation and print/save results."""
    args = parse_args()

    # Load results
    results = load_results(args.results_dir)
    if not results:
        print(f"No result files found in '{args.results_dir}'.")
        print("Run the L8 experiment first:")
        print("  python -m ph_neuro.examples.run_l8_forgetting_baseline ...")
        return

    n_runs = len(results)
    protocols = set(k[0] for k in results)
    weight_formats = set(k[1] for k in results)
    print(
        f"Loaded {n_runs} run(s): {len(protocols)} protocol(s), "
        f"{len(weight_formats)} weight format(s)"
    )

    # Aggregate
    aggregated = aggregate_results(results)
    table = print_comparison_table(aggregated)

    # Ternary weight stats
    weight_summary = print_ternary_weight_summary(results)

    # Compose full output
    full_output = table + "\n" + weight_summary

    # Output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"\nSummary written to: {args.output}")
    else:
        print(full_output)


if __name__ == "__main__":
    main()
