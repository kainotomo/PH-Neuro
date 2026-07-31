#!/usr/bin/env python3
"""B3 Precision Comparison — Results Aggregator.

Collects the four-way precision comparison (ternary vs INT8 vs INT4 vs
FP16) for Split MNIST and Permuted MNIST and prints:

- A comparison table: average forgetting + average accuracy per
  precision, with a ``Δ vs FP16`` column (positive = forgets less than
  the full-precision upper bound).
- An optional per-task accuracy/forgetting breakdown.

The ``ternary`` and ``fp16`` runs live in ``l8_results/`` (L8 control,
identical hyperparameters/seeds); the ``int8``/``int4`` runs live in
``b3_results/`` (this experiment). The aggregator merges both.

Usage::

    # Full comparison table
    python -m ph_neuro.examples.aggregate_b3_results \\
        --results-dir b3_results --l8-dir l8_results --mode comparison

    # Per-task accuracy + forgetting breakdown
    python -m ph_neuro.examples.aggregate_b3_results \\
        --results-dir b3_results --l8-dir l8_results --mode per-task

    # Write the table to a file
    python -m ph_neuro.examples.aggregate_b3_results \\
        --results-dir b3_results --l8-dir l8_results --output b3_results/aggregated_summary.txt
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
PRECISION_LABELS = {
    "ternary": "Ternary (STE)",
    "int8": "INT8 (QAT)",
    "int4": "INT4 (QAT)",
    "fp16": "FP16",
}
# Order for display, strongest quantization first.
PRECISION_ORDER = ["ternary", "int4", "int8", "fp16"]


# ── Loading ─────────────────────────────────────────────────────────


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_b3_results(results_dir: str) -> dict[tuple[str, str, int], dict]:
    """Load B3 precision runs keyed by ``(protocol, weight_format, seed)``.

    Only runs with ``experiment == "B3 Precision Comparison"`` are kept.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or fname == "aggregated_summary.txt":
            continue
        data = _read_json(os.path.join(results_dir, fname))
        if not data or data.get("experiment") != "B3 Precision Comparison":
            continue
        out[(data["protocol"], data["weight_format"], data["seed"])] = data
    return out


def load_l8_results(l8_dir: str) -> dict[tuple[str, str, int], dict]:
    """Load L8 control runs for ternary and fp16, same key schema.

    L8 and B3 share the protocol / seed conventions and hyperparameters,
    so their results are directly comparable.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for fname in sorted(os.listdir(l8_dir)):
        if not fname.endswith(".json") or fname == "aggregated_summary.txt":
            continue
        data = _read_json(os.path.join(l8_dir, fname))
        if not data or data.get("weight_format") not in ("ternary", "fp16"):
            continue
        out[(data["protocol"], data["weight_format"], data["seed"])] = data
    return out


def merge_results(
    b3: dict[tuple[str, str, int], dict],
    l8: dict[tuple[str, str, int], dict],
) -> dict[tuple[str, str, int], dict]:
    """Merge B3 and L8 runs; B3 takes precedence on key collisions."""
    merged = dict(l8)
    merged.update(b3)
    return merged


# ── Aggregation helpers ─────────────────────────────────────────────


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _group_runs(
    runs: dict[tuple[str, str, int], dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group runs by ``(protocol, weight_format)``."""
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (protocol, wf, _seed), r in runs.items():
        by_group[(protocol, wf)].append(r)
    return by_group


# ── Comparison table ────────────────────────────────────────────────


def build_comparison_table(runs: dict[tuple[str, str, int], dict]) -> str:
    """Four-way precision comparison per protocol, with Δ vs FP16."""
    by_group = _group_runs(runs)

    lines: list[str] = []
    lines.append("")
    lines.append("B3: Precision Comparison — Forgetting + Accuracy")
    lines.append("=" * 96)
    lines.append("")

    for protocol in sorted(PROTOCOL_LABELS, key=lambda p: ("permuted", "split").index(p)):
        lines.append(f"--- {PROTOCOL_LABELS[protocol]} ---")
        header = f"{'Precision':<16} {'Avg Forgetting':>18} {'Avg Accuracy':>18} {'Runs':>5}"
        lines.append(header)
        lines.append("-" * len(header))

        rows: list[tuple[str, str, float, float, float, float, int]] = []
        fp16_forgetting: float | None = None
        for precision in PRECISION_ORDER:
            group = by_group.get((protocol, precision))
            if not group:
                continue
            forget = [r["metrics"]["average_forgetting"] for r in group]
            acc = [r["metrics"]["average_accuracy"] for r in group]
            f_mean, f_std = _mean_std(forget)
            a_mean, a_std = _mean_std(acc)
            if precision == "fp16":
                fp16_forgetting = f_mean
            rows.append(
                (precision, PRECISION_LABELS[precision], f_mean, f_std, a_mean, a_std, len(group))
            )

        for _precision, label, f_mean, f_std, a_mean, a_std, n in rows:
            lines.append(
                f"{label:<16} "
                f"{100 * f_mean:>8.2f}% ± {100 * f_std:<5.2f} "
                f"{100 * a_mean:>8.2f}% ± {100 * a_std:<5.2f} "
                f"{n:>5}"
            )

        # Δ vs FP16 (positive = forgets less than FP16)
        if fp16_forgetting is not None:
            lines.append("-" * len(header))
            lines.append("Δ vs FP16 (positive = forgets LESS):")
            for precision, label, f_mean, _f_std, _a_mean, _a_std, _n in rows:
                if precision == "fp16":
                    continue
                delta = fp16_forgetting - f_mean
                lines.append(f"  {label:<14} {100 * delta:+.2f} pp")
        lines.append("")

    lines.append("Δ Forgetting: positive = precision forgets LESS than FP16.")
    lines.append("(quantization noise as implicit regularization)")
    lines.append("")
    return "\n".join(lines)


# ── Per-task table ──────────────────────────────────────────────────


def build_per_task_table(runs: dict[tuple[str, str, int], dict]) -> str:
    """Per-task accuracy + forgetting averaged across seeds."""
    by_group = _group_runs(runs)

    lines: list[str] = []
    lines.append("")
    lines.append("Per-Task Accuracy & Forgetting (averaged across seeds)")
    lines.append("=" * 96)
    lines.append("")

    for protocol in sorted(PROTOCOL_LABELS, key=lambda p: ("permuted", "split").index(p)):
        for precision in PRECISION_ORDER:
            group = by_group.get((protocol, precision))
            if not group:
                continue
            n_tasks = group[0]["n_tasks"]
            lines.append(
                f"--- {PROTOCOL_LABELS[protocol]} / {PRECISION_LABELS[precision]} ---"
            )
            lines.append(f"{'Task':<8} {'Final Acc':>14} {'Forgetting':>14}")
            for task_i in range(n_tasks):
                accs = [
                    r["metrics"]["per_task_accuracy"][task_i]
                    for r in group
                    if task_i < len(r["metrics"]["per_task_accuracy"])
                ]
                fgts = [
                    r["metrics"]["per_task_forgetting"][task_i]
                    for r in group
                    if task_i < len(r["metrics"]["per_task_forgetting"])
                ]
                a_mean, a_std = _mean_std(accs)
                f_mean, f_std = _mean_std(fgts)
                lines.append(
                    f"Task {task_i + 1:<3} "
                    f"{100 * a_mean:>6.2f}% ± {100 * a_std:<5.2f} "
                    f"{100 * f_mean:>6.2f}% ± {100 * f_std:<5.2f}"
                )
            lines.append("")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B3 precision comparison results aggregator"
    )
    parser.add_argument("--results-dir", type=str, default="b3_results")
    parser.add_argument("--l8-dir", type=str, default="l8_results")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["comparison", "per-task"],
        default="comparison",
        help="comparison: 4-way forgetting/accuracy table | "
        "per-task: per-task accuracy and forgetting",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional text file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    b3 = load_b3_results(args.results_dir)
    l8 = load_l8_results(args.l8_dir)
    runs = merge_results(b3, l8)

    if not runs:
        print(f"No B3 results found in {args.results_dir} and no L8 results in {args.l8_dir}")
        return

    if args.mode == "per-task":
        table = build_per_task_table(runs)
    else:
        table = build_comparison_table(runs)

    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"Table written to: {args.output}")


if __name__ == "__main__":
    main()
