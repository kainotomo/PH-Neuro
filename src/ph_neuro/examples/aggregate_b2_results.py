#!/usr/bin/env python3
"""B2 Results Aggregator — collects QLoRA results and compares baselines.

Usage:
    # Rank sweep table (16 runs, seed=42)
    python -m ph_neuro.examples.aggregate_b2_results \\
        --results-dir b2_results --mode sweep

    # Full run comparison vs L8 baseline + B1 EWC (best rank, 3 seeds)
    python -m ph_neuro.examples.aggregate_b2_results \\
        --results-dir b2_results --mode full --lora-r 8

    # Per-task accuracy breakdown
    python -m ph_neuro.examples.aggregate_b2_results \\
        --results-dir b2_results --mode per-task --lora-r 8

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
PRETRAIN_LABELS = {
    "full": "full pretrain",
    "task1": "task1 (1-ep)",
}

# ── Loading ─────────────────────────────────────────────────────────


def load_results(results_dir: str) -> list[dict]:
    """Load all B2 JSON result files from a directory."""
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
        if data.get("experiment") == "B2 QLoRA + Frozen Ternary Backbone":
            results.append(data)
    return results


def load_l8_results(l8_dir: str) -> dict[tuple[str, int], dict]:
    """Load L8 ternary baseline results keyed by ``(protocol, seed)``."""
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


def load_b1_results(b1_dir: str) -> dict[tuple[str, int], dict]:
    """Load B1 EWC ternary results keyed by ``(protocol, seed)``.

    Only the best-λ runs are kept (the B1 aggregator convention is that
    the most-common λ in the directory is the chosen one).
    """
    runs: list[dict] = []
    for fname in sorted(os.listdir(b1_dir)):
        if not fname.endswith(".json") or fname == "aggregated_summary.txt":
            continue
        path = os.path.join(b1_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("experiment") == "B1 EWC + Ternary STE":
            runs.append(data)
    if not runs:
        return {}
    # Most common λ = the selected best λ.
    counts: dict[float, int] = defaultdict(int)
    for r in runs:
        counts[r["ewc_lambda"]] += 1
    best_lam = max(counts, key=counts.get)
    return {
        (r["protocol"], r["seed"]): r
        for r in runs
        if abs(r["ewc_lambda"] - best_lam) < 1e-9
    }


# ── Aggregation helpers ─────────────────────────────────────────────


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _group_key(r: dict) -> tuple[str, str]:
    return (r["protocol"], r["pretrain_protocol"])


# ── Sweep table ─────────────────────────────────────────────────────


def build_sweep_table(results: list[dict]) -> str:
    """Table of rank vs forgetting/accuracy for each protocol × pretrain."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_group[_group_key(r)].append(r)

    lines: list[str] = []
    lines.append("")
    lines.append("B2: QLoRA Rank Sweep (seed=42)")
    lines.append("=" * 90)
    lines.append("")

    for protocol in sorted(PROTOCOL_LABELS):
        for pretrain in ("full", "task1"):
            runs = by_group.get((protocol, pretrain))
            if not runs:
                continue
            lines.append(f"--- {PROTOCOL_LABELS[protocol]} / {PRETRAIN_LABELS[pretrain]} ---")
            lines.append(f"{'Rank':<6} {'Avg Forgetting':>16} {'Avg Accuracy':>16} {'Backbone Test':>16}")
            lines.append("-" * 58)
            for r in sorted(runs, key=lambda x: x["lora_rank"]):
                lines.append(
                    f"{r['lora_rank']:<6} "
                    f"{100 * r['metrics']['average_forgetting']:>8.2f}% "
                    f"{100 * r['metrics']['average_accuracy']:>8.2f}% "
                    f"{100 * r.get('backbone_test_accuracy', 0.0):>8.2f}%"
                )
            lines.append("")

    lines.append("L8 baseline (no CL mechanism, ternary):")
    lines.append("  Split MNIST: ~37.33% forgetting, 62.16% accuracy")
    lines.append("  Permuted MNIST: ~54.86% forgetting, 41.92% accuracy")
    lines.append("")
    return "\n".join(lines)


# ── Full comparison table ───────────────────────────────────────────


def build_full_table(
    results: list[dict], l8_results: dict, b1_results: dict
) -> str:
    """B2 vs L8 vs B1 comparison for the full run (best rank)."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_group[_group_key(r)].append(r)

    lines: list[str] = []
    lines.append("")
    lines.append("B2: QLoRA vs L8 (no CL) vs B1 (EWC) — Comparison Table")
    lines.append("=" * 110)
    lines.append("")

    for protocol in sorted(PROTOCOL_LABELS):
        for pretrain in ("full", "task1"):
            runs = by_group.get((protocol, pretrain))
            if not runs:
                continue
            rank = runs[0]["lora_rank"]
            lines.append(f"--- {PROTOCOL_LABELS[protocol]} / {PRETRAIN_LABELS[pretrain]} (r={rank}) ---")
            lines.append(
                f"{'Method':<24} {'Avg Forgetting':>16} {'Avg Accuracy':>16} {'Runs':>6}"
            )
            lines.append("-" * 66)

            f_mean, f_std = _mean_std([r["metrics"]["average_forgetting"] for r in runs])
            a_mean, a_std = _mean_std([r["metrics"]["average_accuracy"] for r in runs])
            lines.append(
                f"{'QLoRA (B2)':<24} "
                f"{100 * f_mean:>8.2f}% ± {100 * f_std:<6.2f} "
                f"{100 * a_mean:>8.2f}% ± {100 * a_std:<6.2f} "
                f"{len(runs):>6}"
            )

            # L8 baseline (same protocol, matched seeds)
            l8_f: list[float] = []
            l8_a: list[float] = []
            for r in runs:
                key = (protocol, r["seed"])
                if key in l8_results:
                    l8_f.append(l8_results[key]["metrics"]["average_forgetting"])
                    l8_a.append(l8_results[key]["metrics"]["average_accuracy"])
            if l8_f:
                lf_mean, lf_std = _mean_std(l8_f)
                la_mean, la_std = _mean_std(l8_a)
                lines.append(
                    f"{'L8 (no CL)':<24} "
                    f"{100 * lf_mean:>8.2f}% ± {100 * lf_std:<6.2f} "
                    f"{100 * la_mean:>8.2f}% ± {100 * la_std:<6.2f} "
                    f"{len(l8_f):>6}"
                )
                lines.append(
                    f"{'Δ (QLoRA − L8)':<24} "
                    f"{'':>8} {100 * (lf_mean - f_mean):+.2f} pp "
                    f"{'':>8} {100 * (a_mean - la_mean):+.2f} pp"
                )

            # B1 EWC (same protocol, matched seeds)
            b1_f: list[float] = []
            b1_a: list[float] = []
            for r in runs:
                key = (protocol, r["seed"])
                if key in b1_results:
                    b1_f.append(b1_results[key]["metrics"]["average_forgetting"])
                    b1_a.append(b1_results[key]["metrics"]["average_accuracy"])
            if b1_f:
                bf_mean, bf_std = _mean_std(b1_f)
                ba_mean, ba_std = _mean_std(b1_a)
                lam = b1_results[(protocol, runs[0]["seed"])]["ewc_lambda"]
                lines.append(
                    f"{f'B1 EWC (λ={lam:g})':<24} "
                    f"{100 * bf_mean:>8.2f}% ± {100 * bf_std:<6.2f} "
                    f"{100 * ba_mean:>8.2f}% ± {100 * ba_std:<6.2f} "
                    f"{len(b1_f):>6}"
                )
                lines.append(
                    f"{'Δ (QLoRA − B1)':<24} "
                    f"{'':>8} {100 * (bf_mean - f_mean):+.2f} pp "
                    f"{'':>8} {100 * (a_mean - ba_mean):+.2f} pp"
                )
            lines.append("")

    lines.append("Δ Forgetting: positive = method forgets LESS than baseline.")
    lines.append("Δ Accuracy:   positive = method has HIGHER accuracy.")
    lines.append("QLoRA forgetting is 0 by design (frozen backbone).")
    lines.append("")
    return "\n".join(lines)


# ── Per-task detail ─────────────────────────────────────────────────


def build_per_task_table(results: list[dict]) -> str:
    """Per-task accuracy for the best-rank runs, averaged across seeds."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in results:
        by_group[_group_key(r)].append(r)

    lines: list[str] = []
    lines.append("")
    lines.append("Per-Task Accuracy (averaged across seeds)")
    lines.append("-" * 70)

    for protocol in sorted(PROTOCOL_LABELS):
        for pretrain in ("full", "task1"):
            runs = by_group.get((protocol, pretrain))
            if not runs:
                continue
            n_tasks = runs[0]["n_tasks"]
            lines.append(
                f"\n--- {PROTOCOL_LABELS[protocol]} / {PRETRAIN_LABELS[pretrain]} "
                f"(r={runs[0]['lora_rank']}) ---"
            )
            lines.append(f"{'Task':<8} {'Final Acc':>14}")
            for task_i in range(n_tasks):
                accs = [
                    r["metrics"]["per_task_accuracy"][task_i]
                    for r in runs
                    if task_i < len(r["metrics"]["per_task_accuracy"])
                ]
                a_mean, a_std = _mean_std(accs)
                lines.append(
                    f"Task {task_i + 1:<3} "
                    f"{100 * a_mean:>6.2f}% ± {100 * a_std:<6.2f}"
                )
    lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B2 QLoRA results aggregator")
    parser.add_argument("--results-dir", type=str, default="b2_results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sweep", "full", "per-task"],
        default="sweep",
        help="sweep: rank table | full: vs L8/B1 | per-task: per-task accuracy",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="Only consider runs at this rank (default: auto-select for full/per-task)",
    )
    parser.add_argument("--l8-dir", type=str, default="l8_results")
    parser.add_argument("--b1-dir", type=str, default="b1_results")
    parser.add_argument("--output", type=str, default=None, help="Optional text file")
    return parser.parse_args()


def _filter_by_rank(results: list[dict], rank: int | None) -> list[dict]:
    """Keep only runs at a given rank; auto-select the most common rank."""
    if rank is not None:
        return [r for r in results if r["lora_rank"] == rank]
    if not results:
        return results
    counts: dict[int, int] = defaultdict(int)
    for r in results:
        counts[r["lora_rank"]] += 1
    best_rank = max(counts, key=counts.get)
    return [r for r in results if r["lora_rank"] == best_rank]


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    if not results:
        print(f"No B2 results found in {args.results_dir}")
        return

    if args.mode == "sweep":
        table = build_sweep_table(results)
    elif args.mode == "per-task":
        table = build_per_task_table(_filter_by_rank(results, args.lora_r))
    else:
        l8_results = load_l8_results(args.l8_dir)
        b1_results = load_b1_results(args.b1_dir)
        table = build_full_table(
            _filter_by_rank(results, args.lora_r), l8_results, b1_results
        )

    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Table written to: {args.output}")


if __name__ == "__main__":
    main()
