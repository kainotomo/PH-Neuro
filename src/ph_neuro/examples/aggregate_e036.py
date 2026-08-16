#!/usr/bin/env python3
"""E036 — cross-seed aggregation + verdict vs the pre-registered criteria.

Reads every per-cell result JSON from ``results/brain/e036/`` and computes:

* **B1 (independent per-domain adapters):** D1/D2/D3 Δppl (the fresh
  per-domain adapters), source forgetting, storage (3 × 86 KB).
* **B2 (interference floor):** D3 Δppl (continuing adapter), BT on D1/D2,
  storage (1 × 86 KB).
* **C (consolidation):** D3 Δppl for the warm-started full ST (the
  forward-transfer quantity) and for the consolidated LT (the deployed
  artifact); BT on D1/D2 (LT-based, pure later-domain interference); storage
  (LT + sparse deltas).
* **Forward transfer:** per-seed Δppl_C_D3 − Δppl_B1_D3 (+ the C/B1 ratio),
  and adaptation speed (steps to plateau from the D3 50K-token probes).
* **Pre-registered gates (§7):** C's D3 Δppl ≥ B1's D3 Δppl; C's BT on
  D1/D2 < 0.1; C's storage ≤ B1's storage (deployed and LT+deltas figures).

Writes ``results/brain/e036/summary_e036.json`` + ``summary_e036.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from ph_neuro.brain.stats import cross_seed_summary

BT_BAR = 0.1  # pre-registered backward-transfer selectivity bar


def load_results(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def _find(rs: list[dict], condition: str, b1_domain: str | None, seed: int) -> dict | None:
    for r in rs:
        if r["condition"] != condition:
            continue
        if condition == "b1" and r.get("b1_domain") != b1_domain:
            continue
        if int(r["seed"]) == seed:
            return r
    return None


def steps_to_plateau(probes: list[dict], tol: float = 0.005) -> int | None:
    """First probe step from which all later probes are within ``tol`` of final."""
    if not probes:
        return None
    pts = sorted(probes, key=lambda p: int(p["adapt_step"]))
    final = float(pts[-1]["ppl"])
    vals = [(int(p["adapt_step"]), float(p["ppl"])) for p in pts]
    for i, (s, _) in enumerate(vals):
        if all(p2 <= final * (1.0 + tol) for _, p2 in vals[i:]):
            return s
    return None


def b1_d3_plateau(rs: list[dict], seed: int) -> int | None:
    r = _find(rs, "b1", "c4", seed)
    return steps_to_plateau(r["probe_metrics"]) if r else None


def fmt(t: dict, unit: str = "") -> str:
    if not t:
        return "n/a"
    return f"{t['mean']:+.3f}±{t['sd']:.3f} (p={t['p']:.3f}){unit}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/brain/e036")
    ap.add_argument("--tag", default=None,
                    help="restrict to a tag (default: the group with most seeds)")
    ap.add_argument("--primary-budget", type=int, default=100_000)
    args = ap.parse_args(argv)

    rs = load_results(args.results_dir)
    # Only the primary-budget cells (100K) count; smoke (10K) cells are
    # excluded by the budget filter.
    rs = [r for r in rs if int(r.get("adaptation_tokens", 0)) == args.primary_budget]
    seeds = sorted({int(r["seed"]) for r in rs})

    def pick(condition: str, b1_domain: str | None):
        cands = [r for r in rs if r["condition"] == condition and
                 (r.get("b1_domain") == b1_domain if condition == "b1" else True)]
        if not cands:
            return None
        if args.tag:
            cands = [r for r in cands if r.get("tag") == args.tag]
        return cands

    b1_d1 = pick("b1", "pubmed")
    b1_d2 = pick("b1", "cnn")
    b1_d3 = pick("b1", "c4")
    b2 = pick("b2", None)
    c = pick("c", None)

    # ── per-seed forward-transfer / storage tables ─────────────────
    per_seed: dict = {}
    for s in seeds:
        row: dict = {"seed": s}
        r = _find(b1_d1 or [], "b1", "pubmed", s)
        if r:
            row["b1_d1_delta"] = r["metrics"]["domain_ppl_delta"]
            row["b1_source_forgetting"] = r["metrics"]["forgetting_pct"]
        r = _find(b1_d2 or [], "b1", "cnn", s)
        if r:
            row["b1_d2_delta"] = r["metrics"]["domain_ppl_delta"]
        r = _find(b1_d3 or [], "b1", "c4", s)
        if r:
            row["b1_d3_delta"] = r["metrics"]["domain_ppl_delta"]
            row["b1_d3_plateau"] = steps_to_plateau(r["probe_metrics"])
            row["b1_storage"] = r["storage"].get("total_packed_bytes")
        r = _find(b2 or [], "b2", None, s)
        if r:
            row["b2_d3_delta"] = r["metrics"]["d3_ppl_delta_after_d3"]
            row["b2_bt_d1"] = r["metrics"]["bt_d1"]
            row["b2_bt_d2"] = r["metrics"]["bt_d2"]
            row["b2_d1_delta"] = r["metrics"]["d1_ppl_delta_after_d3"]
            row["b2_d2_delta"] = r["metrics"]["d2_ppl_delta_after_d3"]
            row["b2_storage"] = r["storage"].get("total_packed_bytes")
        r = _find(c or [], "c", None, s)
        if r:
            row["c_d3_st_delta"] = r["metrics"]["d3_ppl_st_delta_after_d3"]
            row["c_d3_lt_delta"] = r["metrics"]["d3_ppl_lt_delta_after_d3"]
            row["c_bt_d1"] = r["metrics"]["bt_d1"]
            row["c_bt_d2"] = r["metrics"]["bt_d2"]
            row["c_d1_delta"] = r["metrics"]["d1_ppl_delta_after_d3"]
            row["c_d2_delta"] = r["metrics"]["d2_ppl_delta_after_d3"]
            row["c_plateau"] = steps_to_plateau(r["probe_metrics"])
            st = r["storage"]
            row["c_storage_deployed"] = st["lt_total_bytes"]
            row["c_storage_with_deltas"] = st["c_total_bytes"]
            row["c_storage_index_variant"] = st["c_total_index_variant_bytes"]
        if row != {"seed": s}:
            per_seed[s] = row

    # B1 total storage = 3 full adapters (from the b1 cells; use the measured
    # per-adapter packed bytes).
    b1_all = [r for r in (b1_d1 or []) + (b1_d2 or []) + (b1_d3 or [])]

    def _b1_storage(seed: int) -> int:
        tot = 0
        for d in ("pubmed", "cnn", "c4"):
            r = _find(b1_all, "b1", d, seed)
            if r:
                tot += int(r["storage"].get("total_packed_bytes",
                                            r["storage"].get("disk_bytes", 0)))
        return tot

    # ── summaries ──────────────────────────────────────────────────
    def summ(rows, key):
        d = [{"target_ppl_delta": row[key]} for row in rows if row.get(key) is not None]
        return cross_seed_summary(d, "target_ppl_delta") if d else {}

    seed_rows = list(per_seed.values())
    summary = {
        "seeds": seeds,
        "b1_d1_delta": summ(seed_rows, "b1_d1_delta"),
        "b1_d2_delta": summ(seed_rows, "b1_d2_delta"),
        "b1_d3_delta": summ(seed_rows, "b1_d3_delta"),
        "b2_d3_delta": summ(seed_rows, "b2_d3_delta"),
        "b2_bt_d1": summ(seed_rows, "b2_bt_d1"),
        "b2_bt_d2": summ(seed_rows, "b2_bt_d2"),
        "c_d3_st_delta": summ(seed_rows, "c_d3_st_delta"),
        "c_d3_lt_delta": summ(seed_rows, "c_d3_lt_delta"),
        "c_bt_d1": summ(seed_rows, "c_bt_d1"),
        "c_bt_d2": summ(seed_rows, "c_bt_d2"),
    }

    # ── forward transfer: per-seed paired Δppl_C_D3 − Δppl_B1_D3 ──
    ft = [row["c_d3_st_delta"] - row["b1_d3_delta"]
          for row in seed_rows if row.get("c_d3_st_delta") is not None
          and row.get("b1_d3_delta") is not None]
    ft_summary = cross_seed_summary(
        [{"target_ppl_delta": v} for v in ft], "target_ppl_delta") if ft else {}
    ft_ratio = None
    if summary["c_d3_st_delta"].get("mean") and summary["b1_d3_delta"].get("mean"):
        ft_ratio = (summary["c_d3_st_delta"]["mean"] / summary["b1_d3_delta"]["mean"])

    # ── adaptation speed (steps to plateau, C vs B1) ───────────────
    speed = {"c": {}, "b1": {}}
    for row in seed_rows:
        if row.get("c_plateau") is not None:
            speed["c"][row["seed"]] = row["c_plateau"]
        if row.get("b1_d3_plateau") is not None:
            speed["b1"][row["seed"]] = row["b1_d3_plateau"]

    # ── storage ────────────────────────────────────────────────────
    b1_storage_rows = {s: _b1_storage(s) for s in seeds}
    storage = {}
    for s in seeds:
        row = per_seed.get(s, {})
        b1_bytes = b1_storage_rows.get(s)
        storage[s] = {
            "b1_bytes": b1_bytes,
            "b2_bytes": row.get("b2_storage"),
            "c_deployed_bytes": row.get("c_storage_deployed"),
            "c_with_deltas_bytes": row.get("c_storage_with_deltas"),
            "c_index_variant_bytes": row.get("c_storage_index_variant"),
        }

    def ratio(a, b):
        return (a / b) if a and b else None

    storage_means = {}
    if storage:
        vals = list(storage.values())
        b1s = [v["b1_bytes"] for v in vals if v["b1_bytes"]]
        cdep = [v["c_deployed_bytes"] for v in vals if v["c_deployed_bytes"]]
        cdel = [v["c_with_deltas_bytes"] for v in vals if v["c_with_deltas_bytes"]]
        cidx = [v["c_index_variant_bytes"] for v in vals if v["c_index_variant_bytes"]]
        b2s = [v["b2_bytes"] for v in vals if v["b2_bytes"]]
        storage_means = {
            "b1_bytes": (sum(b1s) / len(b1s)) if b1s else None,
            "b2_bytes": (sum(b2s) / len(b2s)) if b2s else None,
            "c_deployed_bytes": (sum(cdep) / len(cdep)) if cdep else None,
            "c_with_deltas_bytes": (sum(cdel) / len(cdel)) if cdel else None,
            "c_index_variant_bytes": (sum(cidx) / len(cidx)) if cidx else None,
        }
        storage_means["c_over_b1_deployed"] = ratio(
            storage_means["c_deployed_bytes"], storage_means["b1_bytes"])
        storage_means["c_over_b1_with_deltas"] = ratio(
            storage_means["c_with_deltas_bytes"], storage_means["b1_bytes"])
        storage_means["c_over_b1_index_variant"] = ratio(
            storage_means["c_index_variant_bytes"], storage_means["b1_bytes"])
        storage_means["b2_over_b1"] = ratio(
            storage_means["b2_bytes"], storage_means["b1_bytes"])

    # ── pre-registered gates ───────────────────────────────────────
    c_d3 = summary["c_d3_st_delta"].get("mean")
    b1_d3 = summary["b1_d3_delta"].get("mean")
    c_bt_d1 = summary["c_bt_d1"].get("mean")
    c_bt_d2 = summary["c_bt_d2"].get("mean")
    gates = {
        "ft_delta_ge_0": bool(ft_summary.get("mean", 0) >= 0),
        "ft_delta": ft_summary,
        "ft_ratio_c_over_b1": ft_ratio,
        "c_d3_ge_b1_d3": bool(c_d3 is not None and b1_d3 is not None and c_d3 >= b1_d3),
        "c_bt_d1_lt_0_1": bool(c_bt_d1 is not None and c_bt_d1 < BT_BAR),
        "c_bt_d2_lt_0_1": bool(c_bt_d2 is not None and c_bt_d2 < BT_BAR),
        "c_storage_le_b1_deployed": bool(
            storage_means.get("c_over_b1_deployed") is not None and
            storage_means["c_over_b1_deployed"] <= 1.0),
        "c_storage_le_b1_with_deltas": bool(
            storage_means.get("c_over_b1_with_deltas") is not None and
            storage_means["c_over_b1_with_deltas"] <= 1.0),
    }

    out = {
        "per_seed": per_seed,
        "summary": summary,
        "forward_transfer": {
            "per_seed_delta": ft,
            "delta": ft_summary,
            "ratio_c_over_b1": ft_ratio,
            "speed_steps_to_plateau": speed,
        },
        "storage": {"per_seed": storage, "means": storage_means},
        "gates": gates,
        "bt_bar": BT_BAR,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary_e036.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    # ── markdown ──────────────────────────────────────────────────
    md = ["# E036 cross-seed summary", ""]
    md += ["| condition | D1 (PubMed) Δppl | D2 (CNN) Δppl | D3 (C4) Δppl | BT_D1 | BT_D2 |",
           "|:----------|:-----------------:|:-------------:|:------------:|:-----:|:-----:|"]
    for label, d1, d2, d3, bt1, bt2 in [
        ("B1 (independent)", "b1_d1_delta", "b1_d2_delta", "b1_d3_delta", None, None),
        ("B2 (continuing)", None, None, "b2_d3_delta", "b2_bt_d1", "b2_bt_d2"),
        ("C (consolidation, ST_D3)", None, None, "c_d3_st_delta", "c_bt_d1", "c_bt_d2"),
        ("C (consolidation, LT_D3)", None, None, "c_d3_lt_delta", "c_bt_d1", "c_bt_d2"),
    ]:
        md.append(
            f"| {label} | {fmt(summary.get(d1, {})) if d1 else '—'} | "
            f"{fmt(summary.get(d2, {})) if d2 else '—'} | "
            f"{fmt(summary.get(d3, {})) if d3 else '—'} | "
            f"{fmt(summary.get(bt1, {})) if bt1 else '—'} | "
            f"{fmt(summary.get(bt2, {})) if bt2 else '—'} |"
        )
    md += ["", "**Forward transfer (C ST_D3 vs B1, per-seed paired):**", ""]
    md += [f"- Δppl_C_D3 − Δppl_B1_D3 = **{fmt(ft_summary)}** "
           f"(per-seed {[round(v, 4) for v in ft]})"]
    md += [f"- ratio C/B1 on D3 Δppl = **{ft_ratio:.3f}**" if ft_ratio else "- ratio n/a"]
    md += ["", "**Adaptation speed (steps to plateau, D3 50K probes):**", ""]
    md += [f"- C: {speed['c']}", f"- B1: {speed['b1']}"]
    md += ["", "**Storage:**", ""]
    sm = storage_means
    md += [f"- B1 (3 adapters): **{sm.get('b1_bytes')} B**",
           f"- B2 (1 adapter): **{sm.get('b2_bytes')} B** "
           f"(ratio {sm.get('b2_over_b1'):.3f})" if sm.get("b2_bytes") else "",
           f"- C deployed (LT): **{sm.get('c_deployed_bytes')} B** "
           f"(ratio {sm.get('c_over_b1_deployed'):.3f})",
           f"- C LT+deltas (bitmap): **{sm.get('c_with_deltas_bytes')} B** "
           f"(ratio {sm.get('c_over_b1_with_deltas'):.3f})",
           f"- C LT+deltas (int32-index, conservative): "
           f"**{sm.get('c_index_variant_bytes')} B** "
           f"(ratio {sm.get('c_over_b1_index_variant'):.3f})"]
    md += ["", "**Pre-registered gates (§7):**", ""]
    md += [f"- C D3 ≥ B1 D3 (forward transfer): "
           f"{'✅' if gates['c_d3_ge_b1_d3'] else '❌'} "
           f"({gates['ft_delta'].get('mean', 0):+.3f} mean paired Δ)",
           f"- C BT_D1 < 0.1: {'✅' if gates['c_bt_d1_lt_0_1'] else '❌'} "
           f"(mean {c_bt_d1:+.4f})" if c_bt_d1 is not None else "",
           f"- C BT_D2 < 0.1: {'✅' if gates['c_bt_d2_lt_0_1'] else '❌'} "
           f"(mean {c_bt_d2:+.4f})" if c_bt_d2 is not None else "",
           f"- C storage ≤ B1 (deployed): "
           f"{'✅' if gates['c_storage_le_b1_deployed'] else '❌'} "
           f"(ratio {sm.get('c_over_b1_deployed'):.3f})",
           f"- C storage ≤ B1 (LT+deltas): "
           f"{'✅' if gates['c_storage_le_b1_with_deltas'] else '❌'} "
           f"(ratio {sm.get('c_over_b1_with_deltas'):.3f})"]

    with open(os.path.join(args.results_dir, "summary_e036.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n".join(md))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
