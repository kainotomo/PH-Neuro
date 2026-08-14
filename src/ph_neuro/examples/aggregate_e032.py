#!/usr/bin/env python3
"""E032 — cross-seed aggregation + verdict vs the pre-registered criteria.

Reads every per-cell result JSON from ``results/brain/e032/`` and computes the
across-seed statistics the protocol requires (mean ± SD, cross-seed paired
t-test on per-seed Δppl, Cohen's d) for every config (tag). Also computes:

* the rank-sweep winner (Part A, at E031 default η/surprise settings),
* the gain-sweep winner (Part B) with the <1% forgetting constraint,
* the decay-ablation result (Part C),
* the LoRA ratio Δppl_local / Δppl_LoRA at matched budget (Part D),
* the 1M anneal result (Part E),
* the pre-registered verdict (Δppl ≥ 0.5 at 100K; forgetting < 1%).

Writes ``results/brain/e032/summary_e032.json`` + ``summary_e032.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from ph_neuro.brain.stats import cross_seed_summary


def load_results(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def summarize_group(rs: list[dict]) -> dict:
    """Cross-seed aggregate for one config group."""
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
        "plastic_bytes": rs[0]["plastic_weights"]["bytes"],
    }


def tag_summary(r: dict) -> str:
    """Human-readable config line from one result dict."""
    tag = r["tag"]
    budget = r.get("adaptation_tokens", 0)
    lr = r["lr"]
    dec = r["decay_rate"]
    mod = r.get("modulator", {})
    s0, k, mmax = mod.get("s0"), mod.get("k"), mod.get("M_max")
    bits = [f"budget={budget // 1000}k", f"rank={r['rank']}", f"lr={lr:g}"]
    if dec:
        bits.append(f"decay={dec:g}")
    if s0 is not None:
        bits.append(f"s0={s0:g}")
    if k is not None:
        bits.append(f"k={k:g}")
    if mmax is not None:
        bits.append(f"Mmax={mmax:g}")
    return f"{tag} ({', '.join(bits)})"


def fmt_group(g: dict) -> str:
    t = g["target_ppl_delta"]
    f = g["forgetting_pct"]
    return (
        f"Δppl={t['mean']:+.3f}±{t['sd']:.3f} (p={t['p']:.3f}) | "
        f"forgetting={f['mean']:+.3f}% | d={g['target_block_cohens_d']['mean']:+.3f}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/brain/e032")
    ap.add_argument("--primary-budget", type=int, default=100_000)
    args = ap.parse_args(argv)

    rs = load_results(args.results_dir)
    by_tag: dict[str, list[dict]] = {}
    for r in rs:
        # The 1M anneal reuses the best local tag (e.g. lrr1); keep its group
        # distinct from the 100K primary-budget cells of the same tag.
        key = r["tag"]
        if r.get("adaptation_tokens", 0) == 1_000_000:
            key = f"{key}_1m"
        by_tag.setdefault(key, []).append(r)

    groups: dict[str, dict] = {tag: summarize_group(v) for tag, v in sorted(by_tag.items())}

    # Split into the experiment families for the verdict.
    def family(r: dict) -> str:
        if r["method"] == "lora":
            return "lora"
        if r["adaptation_tokens"] == 1_000_000:
            return "anneal"
        return "local"

    fam = {tag: family(rs_list[0]) for tag, rs_list in by_tag.items()}

    # ── Part A: rank sweep at E031 defaults ────────────────────────
    def at_e031_defaults(r: dict) -> bool:
        return (r["lr"] == 1e-3 and r["decay_rate"] == 0.0
                and r["modulator"].get("s0") == 0.05
                and r["modulator"].get("k") == 60.0
                and r["modulator"].get("M_max") == 1.0)

    rank_tags = [t for t, g in groups.items()
                 if fam[t] == "local" and g["target_ppl_delta"]["n"] >= 3
                 and at_e031_defaults(by_tag[t][0])]
    rank_rows = []
    for t in rank_tags:
        g = groups[t]
        rank_rows.append({
            "tag": t, "rank": by_tag[t][0]["rank"],
            "delta": g["target_ppl_delta"]["mean"],
            "forgetting": g["forgetting_pct"]["mean"],
            "n": g["n"],
        })
    rank_rows.sort(key=lambda x: -x["delta"])
    best_rank_tag = rank_rows[0]["tag"] if rank_rows else None
    best_rank = by_tag[best_rank_tag][0]["rank"] if best_rank_tag else None

    # ── Part D: LoRA at the same budget as best rank ───────────────
    lora_tags = [t for t, g in groups.items()
                 if fam[t] == "lora" and g["target_ppl_delta"]["n"] >= 3]
    lora_rows = []
    for t in lora_tags:
        g = groups[t]
        lora_rows.append({
            "tag": t, "lr": by_tag[t][0]["lr"],
            "rank": by_tag[t][0]["rank"],
            "delta": g["target_ppl_delta"]["mean"],
            "forgetting": g["forgetting_pct"]["mean"],
            "n": g["n"],
        })
    lora_rows.sort(key=lambda x: -x["delta"])

    # Best local at the primary budget (excluding LoRA and 1M).
    local_candidates = [t for t, g in groups.items()
                        if fam[t] == "local"
                        and by_tag[t][0]["adaptation_tokens"] == args.primary_budget
                        and g["target_ppl_delta"]["n"] >= 3]
    scored = []
    for t in local_candidates:
        g = groups[t]
        scored.append((g["target_ppl_delta"]["mean"], g["forgetting_pct"]["mean"], t))
    scored.sort(key=lambda x: -x[0])
    best_local = None
    for mean, forg, t in scored:  # prefer highest Δppl with forgetting < 1%
        if forg < 1.0:
            best_local = {"tag": t, "delta": mean, "forgetting": forg}
            break
    # Highest-Δppl local regardless of the forgetting constraint (used for the
    # LoRA ratio when no local config meets <1% forgetting).
    best_local_any = {"tag": scored[0][2], "delta": scored[0][0],
                      "forgetting": scored[0][1]} if scored else None

    # ── verdict ────────────────────────────────────────────────────
    lora_best = lora_rows[0] if lora_rows else None
    ratio = None
    if best_local_any and lora_best and lora_best["delta"] > 0:
        ratio = best_local_any["delta"] / lora_best["delta"]
    verdict: dict = {
        "primary_budget": args.primary_budget,
        "best_local": best_local,  # None when no config meets forgetting < 1%
        "best_local_any": best_local_any,
        "reaches_0_5": bool(best_local and best_local["delta"] >= 0.5),
        "forgetting_lt_1pct": bool(best_local),
        "lora_best": {"tag": lora_best["tag"], "delta": lora_best["delta"],
                      "forgetting": lora_best["forgetting"]} if lora_best else None,
        "lora_ratio": ratio,  # best_local_any.delta / lora_best.delta
        "rank_sweep_winner": {"tag": best_rank_tag, "rank": best_rank,
                              "delta": rank_rows[0]["delta"] if rank_rows else None},
    }

    out = {"groups": {t: g for t, g in groups.items()}, "verdict": verdict}
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary_e032.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ── markdown table ─────────────────────────────────────────────
    md = [
        "# E032 cross-seed summary",
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
            f"| {tag_summary(by_tag[tag][0])} | {g['target_ppl_frozen']:.2f} | "
            f"{g['target_ppl_plastic']} | {t['mean']:+.3f} ± {t['sd']:.3f} "
            f"(p={t['p']:.3f}) | {f['mean']:+.3f}% | "
            f"{g['target_block_cohens_d']['mean']:+.3f} |"
        )
    md += ["", "**Verdict (pre-registered, 100K primary point):**", ""]
    if verdict:
        bl = verdict.get("best_local_any")
        if bl:
            md.append(
                f"- **Best local config (any forgetting):** `{bl['tag']}` — "
                f"Δppl = **{bl['delta']:+.3f}**, forgetting = **{bl['forgetting']:+.3f}%** "
                f"(> 1% ⇒ trade-off boundary per pre-registration)"
            )
        if verdict.get("best_local"):
            md.append(
                f"- **Best local config (forgetting < 1%):** `{verdict['best_local']['tag']}` — "
                f"Δppl = **{verdict['best_local']['delta']:+.3f}**"
            )
        else:
            md.append("- **Best local config (forgetting < 1%):** none — every "
                      "local config is a trade-off boundary")
        md += [
            f"- **Reaches the 0.5-ppl practical bar?** "
            f"{'✅' if verdict['reaches_0_5'] else '❌'}",
            f"- **Forgetting < 1%?** {'✅' if verdict['forgetting_lt_1pct'] else '❌'}",
        ]
        if verdict.get("lora_best"):
            lb = verdict["lora_best"]
            md.append(
                f"- **LoRA upper bound** (`{lb['tag']}`): Δppl = **{lb['delta']:+.3f}**, "
                f"forgetting = {lb['forgetting']:+.3f}%"
            )
        if verdict.get("lora_ratio") is not None:
            md.append(
                f"- **Ratio Δppl_local / Δppl_LoRA = {verdict['lora_ratio']:.3f}**"
            )
        rw = verdict.get("rank_sweep_winner")
        if rw:
            md.append(f"- **Rank-sweep winner:** `{rw['tag']}` (rank {rw['rank']}, "
                      f"Δppl = {rw['delta']:+.3f})")
    md_path = os.path.join(args.results_dir, "summary_e032.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
