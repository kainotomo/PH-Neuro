#!/usr/bin/env python3
"""E035 — cross-seed aggregation + verdict vs the pre-registered criteria.

Reads every per-cell result JSON from ``results/brain/e035/`` plus the reused
E034 float gated-LoRA single-domain summary (``results/brain/e034/
summary_e034.json``) and computes:

* **Experiment 1 (single-domain, WikiText-2 → PubMed, 100K):** float gated
  LoRA (E034 base, +0.902) vs T-A-q / T-A-qft / T-B / T-C — cross-seed mean ±
  SD of Δppl, source forgetting, per-step training overhead, and storage
  (float32 1.38 MB vs 2-bit packed 86 KB → 16×).
* **Pre-registered checks (§7.1):** any variant Δppl ≥ 90% of float (+0.902 →
  ≥ 0.81); source forgetting < 1%; storage 16× on disk; identity invariant
  (reported from unit tests).
* **Selection rule (§7.0):** best ternary variant = highest single-domain mean
  Δppl (ties: lower forgetting, then lower storage bytes) — used for the
  two-domain run.
* **Experiment 2 (sequential two-domain, best variant):** backward transfer on
  PubMed ``BT = pubmed_ppl_after_p2 − pubmed_ppl_after_p1`` (< 0.1 bar),
  domain-2 adaptation (CNN Δppl > 0), source forgetting < 1%.

Writes ``results/brain/e035/summary_e035.json`` + ``summary_e035.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from ph_neuro.brain.stats import cross_seed_summary

# Pre-registered float gated-LoRA single-domain baseline (E034, 100K, 3 seeds).
FLOAT_GATED_DELTA = 0.902
FLOAT_GATED_SD = 0.182
TERNARY_BAR_90PCT = 0.9 * FLOAT_GATED_DELTA  # ≥ 0.81
BT_BAR = 0.1  # pre-registered two-domain selectivity bar


def load_results(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def load_e034_float(results_dir: str) -> dict | None:
    """Read the E034 single-domain gated mean Δppl from its summary (if present)."""
    path = os.path.join(results_dir, "summary_e034.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        s = json.load(fh)
    verdict = s.get("verdict", {}).get("single")
    if not verdict:
        return None
    return {
        "mean": verdict.get("gated_delta_mean", FLOAT_GATED_DELTA),
        "sd": verdict.get("gated_delta_sd", FLOAT_GATED_SD),
        "forgetting_mean": verdict.get("gated_forgetting_mean"),
        "per_seed": verdict.get("gated_delta_per_seed"),
    }


def summarize_group(rs: list[dict]) -> dict:
    if not rs:
        return {}
    rs = sorted(rs, key=lambda r: r["seed"])
    ms = [r["metrics"] for r in rs]
    times = [m.get("mean_step_time_s", float("nan")) for m in ms]
    storage = rs[0].get("storage", {})
    return {
        "seeds": [r["seed"] for r in rs],
        "n": len(rs),
        "adapter": rs[0]["adapter"],
        "tag": rs[0]["tag"],
        "target_ppl_delta": cross_seed_summary(ms, "target_ppl_delta"),
        "forgetting_pct": cross_seed_summary(ms, "forgetting_pct"),
        "target_block_cohens_d": cross_seed_summary(ms, "target_block_cohens_d"),
        "target_ppl_delta_per_seed": [round(m["target_ppl_delta"], 4) for m in ms],
        "target_ppl_plastic": [round(m["target_ppl_plastic"], 4) for m in ms],
        "target_ppl_frozen": rs[0]["metrics"]["target_ppl_frozen"],
        "source_ppl_frozen": rs[0]["metrics"]["source_ppl_frozen"],
        "mean_surprise_M": round(
            sum(m.get("mean_surprise_M", 0.0) or 0.0 for m in ms) / len(ms), 4),
        "mean_step_time_s": round(sum(t for t in times if t == t) / max(
            sum(1 for t in times if t == t), 1), 4),
        "storage": storage,
        "plastic_params": rs[0]["plastic_weights"]["count"],
    }


def summarize_phase2(rs: list[dict]) -> dict:
    if not rs:
        return {}
    p2 = [r["phase2_metrics"] for r in rs]
    deltas = cross_seed_summary(
        [{"target_ppl_delta": p["phase2_ppl_delta"]} for p in p2],
        "target_ppl_delta",
    )
    return {
        "seeds": [r["seed"] for r in rs],
        "n": len(rs),
        "phase2_ppl_delta": deltas,
        "phase2_ppl_delta_per_seed": [round(p["phase2_ppl_delta"], 4) for p in p2],
        "phase2_ppl_plastic": [round(p["phase2_ppl_plastic"], 4) for p in p2],
        "phase2_ppl_frozen": p2[0]["phase2_ppl_frozen"],
    }


def fmt(t: dict, unit: str = "") -> str:
    return f"{t['mean']:+.3f}±{t['sd']:.3f} (p={t['p']:.3f}){unit}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/brain/e035")
    ap.add_argument("--e034-dir", default="results/brain/e034")
    ap.add_argument("--primary-budget", type=int, default=100_000)
    args = ap.parse_args(argv)

    rs = load_results(args.results_dir)
    e034_float = load_e034_float(args.e034_dir)

    single: dict[str, list[dict]] = {}
    two: dict[str, list[dict]] = {}
    for r in rs:
        if r.get("adaptation_tokens") != args.primary_budget:
            continue
        key = f"{r['adapter']}:{r['tag']}"
        if r.get("phases") == 1:
            single.setdefault(key, []).append(r)
        elif r.get("phases") == 2:
            two.setdefault(key, []).append(r)

    single_sum = {k: summarize_group(v) for k, v in single.items()}
    two_sum = {k: summarize_group(v) for k, v in two.items()}
    two_p2 = {k: summarize_phase2(v) for k, v in two.items()}

    # ── group keys by adapter family (pick the tag with the most seeds) ──
    # Operate on the SUMMARIZED groups (single_sum/two_sum), which carry
    # "adapter"/"tag"/"n"; the raw groupings (single/two) are lists of cells.
    def best_key(mapping: dict, adapter: str) -> str | None:
        cands = [k for k in mapping if mapping[k]["adapter"] == adapter]
        if not cands:
            return None
        return max(cands, key=lambda k: mapping[k]["n"])

    float_key = best_key(single_sum, "float")
    # T-A-q vs T-A-qft are both "adapter: ta" but distinguished by tag.
    ta_q_key = max(
        (k for k in single_sum if single_sum[k]["adapter"] == "ta"
         and single_sum[k]["tag"].endswith("_q")),
        key=lambda k: single_sum[k]["n"], default=None)
    ta_qft_key = max(
        (k for k in single_sum if single_sum[k]["adapter"] == "ta"
         and single_sum[k]["tag"].endswith("_qft")),
        key=lambda k: single_sum[k]["n"], default=None)
    tb_key = best_key(single_sum, "tb")
    tc_key = best_key(single_sum, "tc")

    groups = {
        "float": single_sum.get(float_key) if float_key else None,
        "ta_q": single_sum.get(ta_q_key) if ta_q_key else None,
        "ta_qft": single_sum.get(ta_qft_key) if ta_qft_key else None,
        "tb": single_sum.get(tb_key) if tb_key else None,
        "tc": single_sum.get(tc_key) if tc_key else None,
    }

    # ── 90% bar check (any variant) ────────────────────────────────
    bar_results: dict[str, dict] = {}
    for name, g in groups.items():
        if g and g["target_ppl_delta"]["n"] >= 3:
            d = g["target_ppl_delta"]["mean"]
            f = g["forgetting_pct"]["mean"]
            bar_results[name] = {
                "delta_mean": d,
                "delta_sd": g["target_ppl_delta"]["sd"],
                "p": g["target_ppl_delta"]["p"],
                "ge_90pct_bar": bool(d >= TERNARY_BAR_90PCT),
                "forgetting_lt_1pct": bool(f < 1.0),
                "storage_16x": bool(g["storage"].get("reduction_factor", 1.0) >= 15.5),
                "reduction_factor": g["storage"].get("reduction_factor"),
                "packed_bytes": g["storage"].get("packed_bytes"),
                "disk_bytes": g["storage"].get("disk_bytes"),
                "mean_step_time_s": g["mean_step_time_s"],
            }

    # ── selection rule (§7.0): best ternary variant ───────────────
    ternary_cands = [name for name in ("ta_q", "ta_qft", "tb", "tc")
                     if bar_results.get(name)]
    best_variant: str | None = None
    if ternary_cands:
        best_variant = max(
            ternary_cands, key=lambda n: (bar_results[n]["delta_mean"], -bar_results[n]["forgetting_lt_1pct"])
        )

    # ── two-domain backward transfer (best variant) ───────────────
    bt: dict | None = None
    if best_variant and two:
        # p1 = best variant single-domain PubMed ppl (per seed)
        g_best = groups[best_variant]
        p1_by_seed = dict(zip(g_best["seeds"], g_best["target_ppl_plastic"])) if g_best else {}
        two_best_key = max(
            (k for k in two_sum if two_sum[k]["adapter"] == g_best["adapter"]),
            key=lambda k: two_sum[k]["n"], default=None)
        if two_best_key and p1_by_seed:
            per_seed = {}
            for r in two[two_best_key]:
                s = int(r["seed"])
                p2 = float(r["metrics"]["target_ppl_plastic"])
                if s in p1_by_seed:
                    per_seed[s] = p2 - float(p1_by_seed[s])
            if per_seed:
                seeds = sorted(per_seed)
                vals = [per_seed[s] for s in seeds]
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
                bt = {
                    "best_variant": best_variant,
                    "two_tag": two_best_key,
                    "seeds": seeds,
                    "per_seed": {str(s): round(v, 4) for s, v in per_seed.items()},
                    "mean": round(mean, 4),
                    "sd": round(var ** 0.5, 4),
                    "n": len(vals),
                    "lt_0_1": bool(mean < BT_BAR),
                }

    # ── verdict ───────────────────────────────────────────────────
    verdict: dict = {
        "float_gated_baseline_delta": FLOAT_GATED_DELTA,
        "ternary_bar_90pct": round(TERNARY_BAR_90PCT, 4),
        "bt_bar": BT_BAR,
        "e034_float_delta": (e034_float["mean"] if e034_float else FLOAT_GATED_DELTA),
    }
    any_pass_90 = any(v["ge_90pct_bar"] for v in bar_results.values())
    best_res = bar_results.get(best_variant) if best_variant else None
    verdict["best_variant"] = best_variant
    verdict["any_variant_ge_90pct"] = bool(any_pass_90)
    verdict["best_ge_90pct"] = bool(best_res and best_res["ge_90pct_bar"]) if best_res else False
    verdict["best_forgetting_lt_1pct"] = bool(
        best_res and best_res["forgetting_lt_1pct"]) if best_res else False
    verdict["storage_16x_confirmed"] = bool(
        any(v["storage_16x"] for v in bar_results.values()))
    verdict["selectivity_bt_lt_0_1"] = bool(bt and bt["lt_0_1"]) if bt else None

    out = {
        "single": {name: g for name, g in groups.items() if g},
        "two": {m: g for m, g in two_sum.items() if g},
        "two_phase2": {m: g for m, g in two_p2.items() if g},
        "bar_checks": bar_results,
        "best_variant": best_variant,
        "backward_transfer": bt,
        "verdict": verdict,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary_e035.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ── markdown ──────────────────────────────────────────────────
    md = ["# E035 cross-seed summary", ""]
    md += ["## Experiment 1 — single-domain (WikiText-2 → PubMed, 100K)", ""]
    md += ["| variant | Δppl (mean±SD, p) | per-seed | source forgetting | s/step | storage |",
           "|:--------|:------------------:|:--------:|:----------------:|:------:|:-------:|"]
    for key, label in [("float", "float gated (E034 base)"), ("ta_q", "T-A-q (post-train q)"),
                       ("ta_qft", "T-A-qft (+calib)"), ("tb", "T-B (DQT)"), ("tc", "T-C (STE)")]:
        g = groups.get(key)
        if not g or g["target_ppl_delta"]["n"] < 3:
            continue
        t = g["target_ppl_delta"]
        f = g["forgetting_pct"]
        st = g["storage"]
        red = st.get("reduction_factor", 1.0)
        stxt = f"{red:.0f}×" if st.get("packed") else "fp32"
        md.append(
            f"| {label} | {fmt(t)} | {g['target_ppl_delta_per_seed']} | "
            f"{f['mean']:+.3f}% | {g['mean_step_time_s']:.2f} | {stxt} |"
        )
    md += ["", "**Pre-registered bar (90% of float +0.902 = ≥ 0.81):**", ""]
    for name in ("ta_q", "ta_qft", "tb", "tc"):
        v = bar_results.get(name)
        if not v:
            md.append(f"- {name}: (no 3-seed single-domain group yet)")
            continue
        md.append(
            f"- **{name}**: Δppl = **{v['delta_mean']:+.3f} ± {v['delta_sd']:.3f}** "
            f"(p={v['p']:.3f}) → "
            f"{'✅' if v['ge_90pct_bar'] else '❌'} ≥ 0.81 | "
            f"forgetting {'✅' if v['forgetting_lt_1pct'] else '❌'}<1% | "
            f"storage {'✅' if v['storage_16x'] else '❌'} "
            f"({v['reduction_factor']:.1f}×, {v['packed_bytes']} B packed / "
            f"{v['disk_bytes']} B disk) | {v['mean_step_time_s']:.2f} s/step"
        )
    if best_variant:
        md += ["", f"**Selection rule (§7.0) → best ternary variant: `{best_variant}`**", ""]
    md += ["", "## Experiment 2 — sequential two-domain (WikiText → PubMed → CNN/DailyMail)", ""]
    if bt:
        md += [
            f"**Best variant: `{bt['best_variant']}`** — backward transfer on PubMed "
            f"BT = **{bt['mean']:+.4f} ± {bt['sd']:.4f}** (per-seed "
            f"{[bt['per_seed'][str(s)] for s in bt['seeds']]}) → "
            f"{'✅' if bt['lt_0_1'] else '❌'} < 0.1",
            "",
        ]
    else:
        md += ["(no two-domain cell for the best variant yet)", ""]

    v = verdict
    md += ["## Verdict", "",
           f"- Float gated baseline Δppl = **+{v['float_gated_baseline_delta']:.3f}** "
           f"(E034; 90% bar = **{v['ternary_bar_90pct']:.2f}**).",
           f"- **Any variant ≥ 90% bar?** {'✅' if v['any_variant_ge_90pct'] else '❌'}",
           f"- **Best variant (`{v['best_variant']}`) ≥ 90%?** "
           f"{'✅' if v['best_ge_90pct'] else '❌'}",
           f"- **Source forgetting < 1% (best)?** "
           f"{'✅' if v['best_forgetting_lt_1pct'] else '❌'}",
           f"- **Storage 16× confirmed?** {'✅' if v['storage_16x_confirmed'] else '❌'}",
           f"- **Selectivity BT < 0.1 (best variant, two-domain)?** "
           f"{'✅' if v['selectivity_bt_lt_0_1'] else ('❌' if v['selectivity_bt_lt_0_1'] is False else '⏳')}",
           ]
    md_path = os.path.join(args.results_dir, "summary_e035.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
