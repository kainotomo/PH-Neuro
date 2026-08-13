#!/usr/bin/env python3
"""E031 — cross-seed aggregation + verdict vs the LOCKED protocol criteria.

Reads every per-cell result JSON from ``results/brain/e031/`` and computes
the across-seed statistics the protocol requires (mean ± SD, cross-seed
paired t-test on per-seed Δppl, Cohen's d). Writes:

* ``results/brain/e031/summary_e031.json`` — structured summary.
* ``results/brain/e031/summary_e031.md`` — a markdown results table ready to
  paste into ``docs/brain/05-e031-minimal-viable.md``.

Also prints the pre-registered verdict (protocol §7) at the 100K primary
test point.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from ph_neuro.brain.stats import cross_seed_summary

SUCCESS = {
    "frozen_target_ppl": None,  # set from data
    "frozen_source_ppl": None,
}


def load_results(results_dir: str) -> dict:
    by_key: dict[tuple, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            r = json.load(fh)
        key = (r["baseline"], r.get("adaptation_tokens", 0))
        by_key.setdefault(key, []).append(r)
    return by_key


def summarize_group(rs: list[dict]) -> dict:
    """Aggregate across seeds for one (baseline, budget) group."""
    if not rs:
        return {}
    seeds = sorted(r["seed"] for r in rs)
    ms = [r["metrics"] for r in rs]  # runner stores metrics under 'metrics'
    target_delta = cross_seed_summary(ms, "target_ppl_delta")
    source_delta = cross_seed_summary(ms, "source_ppl_delta")
    forgetting = cross_seed_summary(ms, "forgetting_pct")
    tgt = cross_seed_summary(ms, "target_block_cohens_d")
    return {
        "seeds": seeds,
        "n": len(rs),
        "target_ppl_delta": target_delta,
        "source_ppl_delta": source_delta,
        "forgetting_pct": forgetting,
        "target_block_cohens_d": tgt,
        "target_ppl_delta_per_seed": [
            round(r["metrics"]["target_ppl_delta"], 4) for r in sorted(rs, key=lambda r: r["seed"])
        ],
        "target_ppl_frozen": rs[0]["metrics"]["target_ppl_frozen"],
        "target_ppl_plastic": [
            round(r["metrics"]["target_ppl_plastic"], 4)
            for r in sorted(rs, key=lambda r: r["seed"])
        ],
        "source_ppl_frozen": rs[0]["metrics"]["source_ppl_frozen"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/brain/e031")
    ap.add_argument("--primary-budget", type=int, default=100_000)
    args = ap.parse_args(argv)

    by_key = load_results(args.results_dir)
    out: dict = {"groups": {}, "verdict": {}}
    summary_lines: list[str] = []

    for (baseline, budget), rs in sorted(by_key.items()):
        key = f"{baseline}" if budget == 0 else f"{baseline}_{budget // 1000}k"
        g = summarize_group(rs)
        out["groups"][key] = g
        summary_lines.append(
            f"| {key} | {g['target_ppl_frozen']:.2f} | "
            f"{g['target_ppl_plastic']} | "
            f"{g['target_ppl_delta']['mean']:+.3f} ± {g['target_ppl_delta']['sd']:.3f} "
            f"(p={g['target_ppl_delta']['p']:.3f}) | "
            f"{g['forgetting_pct']['mean']:+.3f}% | "
            f"d={g['target_block_cohens_d']['mean']:+.3f} |"
        )

    # Verdict at the primary 100K point.
    frozen = out["groups"].get("frozen") or {}
    primary_budget_tag = f"{args.primary_budget // 1000}k"

    # Recompute the across-seed paired t on Δppl for the primary point.
    primary: dict = {}
    for baseline in ("surprise", "constM", "random"):
        key = f"{baseline}_{primary_budget_tag}" if baseline in ("surprise", "constM") else baseline
        rs = sorted(by_key.get((baseline, args.primary_budget if baseline != "random" else 0), []),
                    key=lambda r: r["seed"])
        if not rs:
            continue
        primary[baseline] = {
            "per_seed_delta": [r["metrics"]["target_ppl_delta"] for r in rs],
            **cross_seed_summary([r["metrics"] for r in rs], "target_ppl_delta"),
            "forgetting_pct_mean": cross_seed_summary(
                [r["metrics"] for r in rs], "forgetting_pct"
            )["mean"],
        }

    frozen_t = frozen.get("target_ppl_frozen")
    if frozen_t is not None and primary.get("surprise"):
        s = primary["surprise"]
        c = primary.get("constM")
        rnd = primary.get("random")
        checks = {
            "delta_gt_0": s["mean"] > 0,
            "p_lt_0_05": s["p"] < 0.05,
            "delta_gt_random": rnd is not None and s["mean"] > rnd["mean"],
            "forgetting_lt_1pct": s["forgetting_pct_mean"] < 1.0,
            "surprise_ge_constM": c is not None and s["mean"] >= c["mean"],
            "delta_ge_0_5": s["mean"] >= 0.5,
        }
        passed = all(checks.values())
        out["verdict"] = {
            "primary_budget": args.primary_budget,
            "checks": checks,
            "passed": passed,
            "label": "✅ GO" if passed else "❌ NO-GO",
        }

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary_e031.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # Markdown results table (paste-ready for docs/brain/05-e031-minimal-viable.md).
    md = [
        "# E031 cross-seed summary",
        "",
        "| baseline/budget | frozen target ppl | plastic target ppl | Δppl (mean±SD) | p | forgetting | block d |",
        "|:----------------|:-----------------:|:------------------:|:---------------:|:--:|:---------:|:-------:|",
        *summary_lines,
        "",
    ]
    v = out.get("verdict", {})
    if v:
        md += [
            f"**Verdict at {v['primary_budget']} tokens:** {v['label']}",
            "",
            "| pre-registered check | result |",
            "|:---------------------|:------:|",
        ]
        for k, val in v["checks"].items():
            md.append(f"| `{k}` | {'✅' if val else '❌'} |")
        md.append("")
    md_path = os.path.join(args.results_dir, "summary_e031.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
