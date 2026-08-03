#!/usr/bin/env python3
"""Aggregate MoE DQT pilot (E019) results across seeds into a summary.

Usage:
    python -m ph_neuro.examples.aggregate_moe_results [RESULTS_DIR]

Reads ``moe_results_final/results_mnist_seed*.json`` (default) and prints a
comparison table: dense vs MoE accuracy, load balancing, sparsity, time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def summarize(dir_path: str) -> None:
    files = sorted(glob.glob(os.path.join(dir_path, "results_mnist_seed*.json")))
    if not files:
        print(f"No result JSONs found in {dir_path}")
        return

    dense_best, moe_best, moe_bal, moe_sp, dense_sp = [], [], [], [], []
    for path in files:
        d = json.load(open(path))
        seed = d["seed"]
        ep = d["config"]["epochs"]
        db = d["dense"]["best_accuracy"]
        mb = d["moe"]["best_accuracy"]
        lb = d["moe"]["load_balancing"]
        br = lb["balance_ratio"]
        sel = lb["selection_fractions"]
        dense_best.append(db)
        moe_best.append(mb)
        moe_bal.append(br)
        moe_sp.append(d["moe"]["weight_stats"]["sparsity_pct"])
        dense_sp.append(d["dense"]["weight_stats"]["sparsity_pct"])
        print(f"  seed={seed} ep={ep:>2}  Dense {100*db:5.2f}%  "
              f"MoE {100*mb:5.2f}%  Δ {100*(mb-db):+6.2f}pp  "
              f"balance_ratio {br:.3f}  sel=[{', '.join(f'{s:.2f}' for s in sel)}]  "
              f"sparsity D/M {dense_sp[-1]:.1f}/{moe_sp[-1]:.1f}%")

    def mean(xs):
        return statistics.mean(xs) if xs else 0.0

    def stdev(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    print()
    print("  ── Summary (mean ± std over seeds) ──")
    print(f"  Dense DQT best: {100*mean(dense_best):.2f}% ± {100*stdev(dense_best):.2f}")
    print(f"  MoE DQT   best: {100*mean(moe_best):.2f}% ± {100*stdev(moe_best):.2f}")
    print(f"  Δ accuracy    : {100*(mean(moe_best)-mean(dense_best)):+.2f}pp")
    print(f"  balance_ratio : {mean(moe_bal):.3f} ± {stdev(moe_bal):.3f}")
    print(f"  MoE sparsity  : {mean(moe_sp):.1f}% ± {stdev(moe_sp):.1f}")
    print(f"  Dense sparsity: {mean(dense_sp):.1f}% ± {stdev(dense_sp):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MoE DQT pilot results")
    parser.add_argument("dir", nargs="?", default="moe_results_final",
                        help="Results directory (default: moe_results_final)")
    args = parser.parse_args()
    summarize(args.dir)


if __name__ == "__main__":
    main()
