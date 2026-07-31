#!/usr/bin/env python3
"""B1 Results Aggregator — collects EWC results and compares with L8 baseline.

Usage:
    # λ sweep table (5 runs, seed=42)
    python -m ph_neuro.examples.aggregate_b1_results \\
        --results-dir b1_results --mode sweep

    # Full run comparison vs L8 baseline (best λ, 3 seeds)
    python -m ph_neuro.examples.aggregate_b1_results \\
        --results-dir b1_results --mode full --l8-dir l8_results

Output:
    - Console: formatted tables
    - Text file: optionally written to file (--output)
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

# ── Loading ─────────────────────────────────────────────────────────


def load_results(results_dir: str) -> list[dict]:
    """Load all B1 JSON result files from a directory.

    Returns:
        List of result dicts (one per run).
    """
    results: list[dict] = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or fname == "aggregated_summary.txt":
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("experiment") == "B1 EWC + Ternary STE":
            results.append(data)
    return results


def load_l8_results(l8_dir: str) -> dict[tuple[str, int], dict]:
    """Load L8 ternary baseline results keyed by ``(protocol, seed)``.

    Returns:
        Dict: ``{(protocol, seed): result_dict}`` for ternary runs only.
    """
    results: dict[tuple[str, int], dict] = {}
    for fname in sorted(os.listdir(l8_dir)):
        if not fname.endswith(".json") or fname == "aggregated_summary.txt":
            continue
        path = os.path.join(l8_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("weight_format") == "ternary":
            results[(data["protocol"], data["seed"])] = data
    return results


# ── Aggregation helpers ─────────────────────────────────────────────


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


# ── Sweep table ─────────────────────────────────────────────────────


def build_sweep_table(results: list[dict]) -> str:
    """Build a table of λ vs forgetting/accuracy for the λ sweep runs."""
    # Group by lambda
    by_lambda: dict[float, list[dict]] = defaultdict(list)
    for r in results:
        by_lambda[r["ewc_lambda"]].append(r)

    lines: list[str] = []
    lines.append("")
    lines.append("B1: EWC λ Sweep — Split MNIST (seed=42)")
    lines.append("=" * 88)
    lines.append("")
    lines.append(
        f"{'λ':<10} {'Runs':>5} {'Avg Forgetting':>16} {'Avg Accuracy':>16} {'Tasks':>6}"
    )
    lines.append("-" * 55)

    for lam in sorted(by_lambda.keys()):
        runs = by_lambda[lam]
        forget_mean, forget_std = _mean_std(
            [r["metrics"]["average_forgetting"] for r in runs]
        )
        acc_mean, acc_std = _mean_std([r["metrics"]["average_accuracy"] for r in runs])
        n_tasks = runs[0]["n_tasks"]
        lines.append(
            f"{lam:<10.4g} {len(runs):>5} "
            f"{100 * forget_mean:>8.2f}% ± {100 * forget_std:<6.2f} "
            f"{100 * acc_mean:>8.2f}% ± {100 * acc_std:<6.2f} "
            f"{n_tasks:>6}"
        )

    lines.append("")
    lines.append("L8 baseline (no EWC, ternary, seed=42): ~37.33% forgetting, "
                 "62.16% accuracy on Split MNIST")
    lines.append("")
    return "\n".join(lines)


# ── Full comparison table ───────────────────────────────────────────


def build_full_table(results: list[dict], l8_results: dict) -> str:
    """Build a B1-vs-L8 comparison table for the full run (best λ)."""
    # Group B1 results by protocol
    by_protocol: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_protocol[r["protocol"]].append(r)

    lines: list[str] = []
    lines.append("")
    lines.append("B1: EWC vs L8 Baseline — Comparison Table")
    lines.append("=" * 100)
    lines.append("")

    lines.append(f"{'Protocol':<14} {'Method':<12} {'Avg Forgetting':>18} {'Avg Accuracy':>18} {'Runs':>6}")
    lines.append("-" * 68)

    for protocol in sorted(by_protocol.keys()):
        runs = by_protocol[protocol]
        b1_forget_mean, b1_forget_std = _mean_std(
            [r["metrics"]["average_forgetting"] for r in runs]
        )
        b1_acc_mean, b1_acc_std = _mean_std(
            [r["metrics"]["average_accuracy"] for r in runs]
        )
        lam = runs[0]["ewc_lambda"]

        # L8 baseline for the same protocol (ternary runs)
        l8_forget: list[float] = []
        l8_acc: list[float] = []
        for r in runs:
            key = (protocol, r["seed"])
            if key in l8_results:
                l8_forget.append(l8_results[key]["metrics"]["average_forgetting"])
                l8_acc.append(l8_results[key]["metrics"]["average_accuracy"])
        l8_forget_mean, l8_forget_std = _mean_std(l8_forget)
        l8_acc_mean, l8_acc_std = _mean_std(l8_acc)

        lines.append(
            f"{PROTOCOL_LABELS.get(protocol, protocol):<14} {'EWC (λ=' + f'{lam:g})':<10} "
            f"{100 * b1_forget_mean:>8.2f}% ± {100 * b1_forget_std:<6.2f} "
            f"{100 * b1_acc_mean:>8.2f}% ± {100 * b1_acc_std:<6.2f} "
            f"{len(runs):>6}"
        )
        lines.append(
            f"{'':<14} {'L8 (no EWC)':<12} "
            f"{100 * l8_forget_mean:>8.2f}% ± {100 * l8_forget_std:<6.2f} "
            f"{100 * l8_acc_mean:>8.2f}% ± {100 * l8_acc_std:<6.2f} "
            f"{len(l8_forget):>6}"
        )
        if l8_forget:
            d_forget = l8_forget_mean - b1_forget_mean
            d_acc = b1_acc_mean - l8_acc_mean
            lines.append(
                f"{'':<14} {'Δ (EWC − L8)':<12} "
                f"{'':>8} {100 * d_forget:+.2f} pp "
                f"{'':>8} {100 * d_acc:+.2f} pp"
            )
        lines.append("")

    lines.append("Δ Forgetting: positive = EWC forgets LESS than baseline (good).")
    lines.append("Δ Accuracy:   positive = EWC has HIGHER final accuracy than baseline.")
    lines.append("")
    return "\n".join(lines)


# ── Per-task detail (optional) ──────────────────────────────────────


def build_per_task_table(results: list[dict]) -> str:
    """Per-task forgetting averaged across seeds, for the best-λ runs."""
    by_protocol: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_protocol[r["protocol"]].append(r)

    lines: list[str] = []
    lines.append("")
    lines.append("Per-Task Forgetting (averaged across seeds)")
    lines.append("-" * 70)

    for protocol in sorted(by_protocol.keys()):
        runs = by_protocol[protocol]
        n_tasks = runs[0]["n_tasks"]
        lines.append(f"\n--- {PROTOCOL_LABELS.get(protocol, protocol)} (λ={runs[0]['ewc_lambda']:g}) ---")
        lines.append(f"{'Task':<8} {'Forgetting':>14} {'Final Acc':>14}")
        for task_i in range(n_tasks):
            forgets = [
                r["metrics"]["per_task_forgetting"][task_i]
                for r in runs
                if task_i < len(r["metrics"]["per_task_forgetting"])
            ]
            accs = [
                r["metrics"]["per_task_accuracy"][task_i]
                for r in runs
                if task_i < len(r["metrics"]["per_task_accuracy"])
            ]
            f_mean, f_std = _mean_std(forgets)
            a_mean, a_std = _mean_std(accs)
            lines.append(
                f"Task {task_i + 1:<3} "
                f"{100 * f_mean:>6.2f}% ± {100 * f_std:<6.2f} "
                f"{100 * a_mean:>6.2f}% ± {100 * a_std:<6.2f}"
            )
    lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 EWC results aggregator")
    parser.add_argument("--results-dir", type=str, default="b1_results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "full", "per-task"],
        default="sweep",
        help="sweep: λ table | full: vs L8 | per-task: per-task forgetting",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=None,
        dest="ewc_lambda",
        help="Only consider runs at this λ (default: auto-select for full/per-task)",
    )
    parser.add_argument("--l8-dir", type=str, default="l8_results")
    parser.add_argument("--output", type=str, default=None, help="Optional text file")
    return parser.parse_args()


def _filter_by_lambda(results: list[dict], ewc_lambda: float | None) -> list[dict]:
    """Keep only runs at a given λ; auto-select the most common λ if None.

    The sweep and full runs share the same results directory, so the
    directory usually contains runs at several different λ values. When
    aggregating a full run we only want the runs at the chosen (best) λ.
    """
    if ewc_lambda is not None:
        return [r for r in results if abs(r["ewc_lambda"] - ewc_lambda) < 1e-9]
    if not results:
        return results
    counts: dict[float, int] = defaultdict(int)
    for r in results:
        counts[r["ewc_lambda"]] += 1
    best_lam = max(counts, key=counts.get)
    return [r for r in results if abs(r["ewc_lambda"] - best_lam) < 1e-9]


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        print(f"No B1 results found in {args.results_dir}")
        return

    l8_results = load_l8_results(args.l8_dir)

    if args.mode == "sweep":
        table = build_sweep_table(results)
    elif args.mode == "per-task":
        table = build_per_task_table(_filter_by_lambda(results, args.ewc_lambda))
    else:
        table = build_full_table(_filter_by_lambda(results, args.ewc_lambda), l8_results)

    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
