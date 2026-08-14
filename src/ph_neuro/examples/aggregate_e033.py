#!/usr/bin/env python3
"""E033 — cross-seed aggregation + verdict vs the pre-registered criteria.

Reads every per-cell result JSON from ``results/brain/e033/`` and computes the
across-seed statistics (mean ± SD, cross-seed paired t-test on per-seed Δppl,
Cohen's d) for every config (tag). Also computes the pre-registered verdict
at the 100K primary point:

    1. Δppl_PC ≥ 0.5  (the practical bar)
    2. Δppl_PC > 0    (sign agreement with the E032 LoRA bound — unlike the
                       E032 local Hebbian's −1.35)
    3. forgetting < 1% (source degradation)
    4. ratio Δppl_PC / Δppl_LoRA  (vs E032's ≈ −0.84)

and the pre-registered kill-criteria consequence (the LAST local-rule
experiment): if Δppl ≤ 0 (or worse than random), the local-rule scientific
question is CLOSED and the project pivots to the backprop-LoRA product path.

Writes ``results/brain/e033/summary_e033.json`` + ``summary_e033.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from ph_neuro.brain.stats import cross_seed_summary

# E032 LoRA best (rank-1, 344K, matched budget) at 100K — the comparison bound.
# Loaded from the E032 summary when present; falls back to the reported value.
LORA_BEST_DELTA = 1.520


def load_results(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def load_lora_best(e032_summary: str | None) -> float | None:
    if e032_summary and os.path.exists(e032_summary):
        with open(e032_summary) as fh:
            s = json.load(fh)
        lb = s.get("verdict", {}).get("lora_best")
        if lb and lb.get("delta"):
            return float(lb["delta"])
    return None


def summarize_group(rs: list[dict]) -> dict:
    if not rs:
        return {}
    rs = sorted(rs, key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rs]
    ms = [r["metrics"] for r in rs]
    tgt = cross_seed_summary(ms, "target_ppl_delta")
    forgetting = cross_seed_summary(ms, "forgetting_pct")
    block_d = cross_seed_summary(ms, "target_block_cohens_d")
    return {
        "seeds": seeds,
        "n": len(rs),
        "target_ppl_delta": tgt,
        "forgetting_pct": forgetting,
        "target_block_cohens_d": block_d,
        "target_ppl_delta_per_seed": [round(m["target_ppl_delta"], 4) for m in ms],
        "target_ppl_plastic": [round(m["target_ppl_plastic"], 4) for m in ms],
        "target_ppl_frozen": rs[0]["metrics"]["target_ppl_frozen"],
        "source_ppl_frozen": rs[0]["metrics"]["source_ppl_frozen"],
        "plastic_params": rs[0]["plastic_weights"]["count"],
        "inverse_params": rs[0].get("inverse_weights", {}).get("count", 0),
        "mean_abs_error": rs[0].get("inverse_weights", {}).get("mean_abs_error", None),
    }


def fmt_group(g: dict) -> str:
    t = g["target_ppl_delta"]
    f = g["forgetting_pct"]
    return (
        f"Δppl={t['mean']:+.3f}±{t['sd']:.3f} (p={t['p']:.3f}) | "
        f"forgetting={f['mean']:+.3f}% | d={g['target_block_cohens_d']['mean']:+.3f}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/brain/e033")
    ap.add_argument("--e032-summary", default="results/brain/e032/summary_e032.json")
    ap.add_argument("--primary-budget", type=int, default=100_000)
    args = ap.parse_args(argv)

    rs = load_results(args.results_dir)
    by_tag: dict[str, list[dict]] = {}
    for r in rs:
        # Only cells at the pre-registered primary budget participate in the
        # verdict group; any smoke/anneal cells (same tag, different budget)
        # are excluded so they never pollute the cross-seed stats.
        if r.get("adaptation_tokens", 0) != args.primary_budget:
            continue
        by_tag.setdefault(r["tag"], []).append(r)

    groups: dict[str, dict] = {tag: summarize_group(v) for tag, v in sorted(by_tag.items())}

    # ── primary verdict (PC at 100K, ≥3 seeds) ─────────────────────
    lora_best = load_lora_best(args.e032_summary) or LORA_BEST_DELTA
    pc_group = None
    pc_tag = None
    for tag, v in sorted(by_tag.items()):
        g = groups[tag]
        if (v[0]["method"] == "predictive_coding"
                and v[0]["adaptation_tokens"] == args.primary_budget
                and g["target_ppl_delta"]["n"] >= 3):
            pc_group, pc_tag = g, tag
            break

    verdict: dict = {
        "primary_budget": args.primary_budget,
        "lora_best_delta": lora_best,
        "pc_tag": pc_tag,
        "pc": None,
    }
    if pc_group is not None:
        t = pc_group["target_ppl_delta"]
        f = pc_group["forgetting_pct"]
        ratio = t["mean"] / lora_best if lora_best > 0 else None
        verdict["pc"] = {
            "tag": pc_tag,
            "delta_mean": t["mean"],
            "delta_sd": t["sd"],
            "p": t["p"],
            "cohens_d": t["cohens_d"],
            "forgetting_mean": f["mean"],
            "delta_ge_0_5": bool(t["mean"] >= 0.5),
            "sign_agreement_gt_0": bool(t["mean"] > 0),
            "forgetting_lt_1pct": bool(f["mean"] < 1.0),
            "lora_ratio": ratio,
            "all_seeds_positive": bool(all(x > 0 for x in pc_group["target_ppl_delta_per_seed"])),
        }
        # Pre-registered kill-criteria consequence (LAST local-rule experiment)
        pc = verdict["pc"]
        if pc["delta_mean"] > 0 and pc["p"] < 0.05:
            verdict["consequence"] = (
                "PC shows a positive, significant effect — a local rule CAN "
                "adapt a frozen LM (breakthrough direction). Ratio to LoRA "
                f"{ratio:.3f}. Additional formulations are permitted."
            )
        else:
            verdict["consequence"] = (
                "PC failed the pre-registered criteria at matched budget "
                f"(Δppl = {pc['delta_mean']:+.3f}, p = {pc['p']:.3f}). The "
                "local-rule scientific question is CLOSED: the project pivots "
                "to the backprop-LoRA product path. No second PC variant, no "
                "hyperparameter rescue, no capacity escalation."
            )
    else:
        verdict["consequence"] = "no PC group with ≥3 seeds at the primary budget"

    out = {"groups": {t: g for t, g in groups.items()}, "verdict": verdict}
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary_e033.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ── markdown table ─────────────────────────────────────────────
    md = [
        "# E033 cross-seed summary",
        "",
        "| config | frozen tgt ppl | plastic tgt ppl | Δppl (mean±SD) | p | forgetting | block d |",
        "|:-------|:--------------:|:---------------:|:---------------:|:--:|:----------:|:-------:|",
    ]
    for tag in sorted(groups):
        g = groups[tag]
        if not g.get("target_ppl_delta", {}).get("n"):
            continue
        t = g["target_ppl_delta"]
        f = g["forgetting_pct"]
        md.append(
            f"| {tag} | {g['target_ppl_frozen']:.2f} | "
            f"{g['target_ppl_plastic']} | {t['mean']:+.3f} ± {t['sd']:.3f} "
            f"(p={t['p']:.3f}) | {f['mean']:+.3f}% | "
            f"{g['target_block_cohens_d']['mean']:+.3f} |"
        )
    md += ["", "**Verdict (pre-registered, 100K primary point):**", ""]
    pc = verdict.get("pc")
    if pc:
        md += [
            f"- **Δppl_PC = {pc['delta_mean']:+.3f} ± {pc['delta_sd']:.3f} (p={pc['p']:.3f}, "
            f"per-seed {pc_group['target_ppl_delta_per_seed']})**",
            f"- **Δppl ≥ 0.5 practical bar?** {'✅' if pc['delta_ge_0_5'] else '❌'}",
            f"- **Sign agreement with LoRA (Δppl > 0)?** "
            f"{'✅' if pc['sign_agreement_gt_0'] else '❌'} "
            f"(all 3 seeds positive: {pc['all_seeds_positive']})",
            f"- **Forgetting < 1%?** {'✅' if pc['forgetting_lt_1pct'] else '❌'} "
            f"(mean {pc['forgetting_mean']:+.3f}%)",
            f"- **Ratio Δppl_PC / Δppl_LoRA = {pc['lora_ratio']:.3f}** "
            f"(LoRA bound {lora_best:+.3f})",
        ]
    md += ["", f"**Consequence (pre-registered):** {verdict['consequence']}", ""]
    md_path = os.path.join(args.results_dir, "summary_e033.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
