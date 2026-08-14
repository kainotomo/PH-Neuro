#!/usr/bin/env python3
"""E034 — cross-seed aggregation + verdict vs the pre-registered criteria.

Reads every per-cell result JSON from ``results/brain/e034/`` plus the reused
E032 plain-LoRA single-domain cells (``results/brain/e032/``) and computes:

* **Experiment 1 (single-domain, WikiText-2 → PubMed, 100K):** gated LoRA vs
  reused plain LoRA — cross-seed mean ± SD of Δppl, source forgetting, mean
  surprise M, the effective-mean-lr, and the pre-registered success checks:
  gated Δppl ≥ 0.5, source degradation < 1%, Δppl_gated ≤ Δppl_plain.
* **Experiment 2 (sequential two-domain, WikiText → PubMed → CNN/DailyMail):**
  plain vs gated — do both adapt to domain 2 (CNN Δppl > 0)? backward
  transfer on domain 1 (PubMed) ``BT = pubmed_ppl_after_p2 −
  pubmed_ppl_after_p1`` (the **selectivity claim**: BT_gated < BT_plain),
  domain-1 retention after the full sequence, and source degradation.
* **Optional control:** const_reduced vs gated (same total effective learning,
  only the lr's temporal shape differs).

``pubmed_ppl_after_p1`` is the single-domain result of the *same* method and
seed: plain → E032 lora_lr1e3 (identical stream/init/rule); gated → the E034
single-domain gated cell. The two-domain stream up to the phase-1 end is
bit-identical to the single-domain stream, so the pairing is exact.

Writes ``results/brain/e034/summary_e034.json`` + ``summary_e034.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from ph_neuro.brain.stats import cross_seed_summary

# E032 plain LoRA best-lr cell (rank 1, 344K, lr = 1e-3) — the single-domain
# plain baseline (reused; identical protocol, no cells re-run).
E032_PLAIN_TAG = "lora_lr1e3"
E032_PLAIN_DELTA_MEAN = 1.520  # fallback if the E032 JSON is unavailable


def load_results(results_dir: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def _ppl_after_p1_by_seed(results: list[dict], method: str, phases: int) -> dict:
    """``{seed: target_ppl_plastic}`` for single-domain cells of ``method``."""
    out: dict[int, float] = {}
    for r in results:
        if (r.get("method") == method and r.get("phases") == phases
                and r.get("adaptation_tokens") == 100_000):
            out[int(r["seed"])] = float(r["metrics"]["target_ppl_plastic"])
    return out


def load_e032_plain(e032_dir: str) -> dict[int, float]:
    """PubMed ppl after phase 1 for plain LoRA, per seed (from E032)."""
    out: dict[int, float] = {}
    for path in sorted(glob.glob(os.path.join(e032_dir, "*.json"))):
        if os.path.basename(path).startswith("summary"):
            continue
        with open(path) as fh:
            r = json.load(fh)
        if (r.get("tag") == E032_PLAIN_TAG and r.get("method") == "lora"
                and r.get("adaptation_tokens") == 100_000):
            out[int(r["seed"])] = float(r["metrics"]["target_ppl_plastic"])
    return out


def summarize_group(rs: list[dict], key: str) -> dict:
    if not rs:
        return {}
    rs = sorted(rs, key=lambda r: r["seed"])
    ms = [r["metrics"] for r in rs]
    return {
        "seeds": [r["seed"] for r in rs],
        "n": len(rs),
        "target_ppl_delta": cross_seed_summary(ms, "target_ppl_delta"),
        "forgetting_pct": cross_seed_summary(ms, "forgetting_pct"),
        "target_block_cohens_d": cross_seed_summary(ms, "target_block_cohens_d"),
        "target_ppl_delta_per_seed": [round(m["target_ppl_delta"], 4) for m in ms],
        "target_ppl_plastic": [round(m["target_ppl_plastic"], 4) for m in ms],
        "mean_surprise_M": round(
            sum(m.get("mean_surprise_M", 0.0) or 0.0 for m in ms) / len(ms), 4
        ),
        "effective_mean_lr": round(
            sum(m.get("effective_mean_lr", 0.0) or 0.0 for m in ms) / len(ms), 6
        ),
        "target_ppl_frozen": rs[0]["metrics"]["target_ppl_frozen"],
        "source_ppl_frozen": rs[0]["metrics"]["source_ppl_frozen"],
        "plastic_params": rs[0]["plastic_weights"]["count"],
    }


def summarize_phase2(rs: list[dict]) -> dict:
    """Cross-seed summary of the two-domain phase-2 (CNN) metrics."""
    if not rs:
        return {}
    p2 = [r["phase2_metrics"] for r in rs]
    seeds = [r["seed"] for r in rs]
    deltas = cross_seed_summary([{"target_ppl_delta": p["phase2_ppl_delta"]} for p in p2],
                                "target_ppl_delta")
    return {
        "seeds": seeds,
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
    ap.add_argument("--results-dir", default="results/brain/e034")
    ap.add_argument("--e032-dir", default="results/brain/e032")
    ap.add_argument("--primary-budget", type=int, default=100_000)
    args = ap.parse_args(argv)

    rs = load_results(args.results_dir)
    e032_plain = load_e032_plain(args.e032_dir)

    # ── single-domain groups (phases=1) AND two-domain groups (phases=2),
    # keyed by ``(method, tag)`` so a bonus cell (e.g. the 100K smoke_gated)
    # never pollutes a 3-seed group or double-counts a seed. ─────────────
    single: dict[str, list[dict]] = {}
    two: dict[str, list[dict]] = {}
    for r in rs:
        if r.get("adaptation_tokens") != args.primary_budget:
            continue
        key = f"{r['method']}:{r['tag']}"
        if r.get("phases") == 1:
            single.setdefault(key, []).append(r)
        elif r.get("phases") == 2:
            two.setdefault(key, []).append(r)

    def tag_method(key: str) -> str:
        return key.split(":", 1)[0]

    single_sum = {k: summarize_group(v, "target_ppl_delta") for k, v in single.items()}
    two_sum = {k: summarize_group(v, "target_ppl_delta") for k, v in two.items()}
    two_p2 = {k: summarize_phase2(v) for k, v in two.items()}

    # ── backward transfer on domain 1 (PubMed) ────────────────────
    # BT(method) = pubmed_ppl_after_p2(method, 2-dom) − pubmed_ppl_after_p1(method).
    # pubmed_ppl_after_p1: plain → E032 (reused); gated → E034 single-domain.
    p1_plain = e032_plain  # {seed: pubmed ppl after phase 1, plain}
    p1_gated = _ppl_after_p1_by_seed(rs, "lora_gated", 1)

    def backward_transfer(two_results: list[dict], p1_by_seed: dict) -> dict | None:
        if not two_results or not p1_by_seed:
            return None
        per_seed = {}
        for r in two_results:
            s = int(r["seed"])
            p2 = float(r["metrics"]["target_ppl_plastic"])
            if s in p1_by_seed:
                per_seed[s] = p2 - float(p1_by_seed[s])  # + = forgetting of PubMed
        if not per_seed:
            return None
        seeds = sorted(per_seed)
        vals = [per_seed[s] for s in seeds]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        sd = var ** 0.5
        return {
            "seeds": seeds,
            "per_seed": {str(s): round(v, 4) for s, v in per_seed.items()},
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "n": len(vals),
        }

    # Pick the tag with the most seeds per (method, phases) family.
    def best_key(mapping: dict, method: str, phases: int) -> str | None:
        cands = [k for k in mapping if tag_method(k) == method]
        if not cands:
            return None
        return max(cands, key=lambda k: len(mapping[k]))

    two_plain_key = best_key(two, "lora", 2)
    two_gated_key = best_key(two, "lora_gated", 2)
    single_plain_key = best_key(single, "lora", 1)
    single_gated_key = best_key(single, "lora_gated", 1)
    single_control_key = best_key(single, "lora_const_reduced", 1)

    bt_plain = backward_transfer(two.get(two_plain_key, []), p1_plain)
    bt_gated = backward_transfer(two.get(two_gated_key, []), p1_gated)

    # ── verdict ────────────────────────────────────────────────────
    plain_single = single_sum.get(single_plain_key) if single_plain_key else None
    gated_single = single_sum.get(single_gated_key) if single_gated_key else None
    control_single = single_sum.get(single_control_key) if single_control_key else None
    plain_two = two_sum.get(two_plain_key) if two_plain_key else None
    gated_two = two_sum.get(two_gated_key) if two_gated_key else None

    verdict: dict = {}
    if gated_single and gated_single["target_ppl_delta"]["n"] >= 3:
        g = gated_single["target_ppl_delta"]
        f = gated_single["forgetting_pct"]
        plain_delta = (plain_single["target_ppl_delta"]["mean"]
                       if plain_single and plain_single["target_ppl_delta"]["n"] >= 3
                       else E032_PLAIN_DELTA_MEAN)
        verdict["single"] = {
            "gated_delta_mean": g["mean"],
            "gated_delta_sd": g["sd"],
            "gated_p": g["p"],
            "gated_delta_per_seed": gated_single["target_ppl_delta_per_seed"],
            "gated_forgetting_mean": f["mean"],
            "gated_delta_ge_0_5": bool(g["mean"] >= 0.5),
            # "source degradation < 1%" = forgetting (ppl increase) under +1%;
            # a negative value (source *improved*) also passes.
            "gated_forgetting_lt_1pct": bool(f["mean"] < 1.0),
            "gated_delta_le_plain": bool(g["mean"] <= plain_delta + 1e-9),
            "plain_delta_mean": plain_delta,
            "mean_surprise_M": gated_single["mean_surprise_M"],
            "effective_mean_lr": gated_single["effective_mean_lr"],
        }
    if gated_two and gated_two["target_ppl_delta"]["n"] >= 3:
        verdict["two"] = {
            "plain_p2_delta_mean": (two_p2[two_plain_key]["phase2_ppl_delta"]["mean"]
                                    if two_plain_key and two_p2.get(two_plain_key) else None),
            "gated_p2_delta_mean": (two_p2[two_gated_key]["phase2_ppl_delta"]["mean"]
                                    if two_gated_key and two_p2.get(two_gated_key) else None),
            "bt_plain": bt_plain,
            "bt_gated": bt_gated,
            "selectivity_claim": None,
        }
        if bt_plain and bt_gated and bt_plain["n"] >= 3 and bt_gated["n"] >= 3:
            bt_diff = bt_gated["mean"] - bt_plain["mean"]  # < 0 → gated forgets less
            # seed agreement: gated BT < plain BT in >= 2/3 seeds
            # (per_seed dicts are keyed by string seeds; bt_plain["seeds"] is
            # the int list — compare via str keys).
            agree = sum(
                1 for s in bt_plain["seeds"]
                if str(s) in bt_gated["per_seed"]
                and bt_gated["per_seed"][str(s)] < bt_plain["per_seed"][str(s)]
            )
            verdict["two"]["selectivity_claim"] = {
                "bt_gated_minus_plain": round(bt_diff, 4),
                "bt_gated_lt_plain": bool(bt_diff < 0),
                "seed_agreement": f"{agree}/{len(bt_plain['seeds'])}",
                "seed_agreement_ge_2of3": bool(agree >= 2),
                "claim_holds": bool(bt_diff < 0 and agree >= 2),
            }

    out = {
        "single": {m: g for m, g in single_sum.items() if g},
        "two": {m: g for m, g in two_sum.items() if g},
        "two_phase2": {m: g for m, g in two_p2.items() if g},
        "backward_transfer": {"plain": bt_plain, "gated": bt_gated},
        "verdict": verdict,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "summary_e034.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ── markdown ───────────────────────────────────────────────────
    md = ["# E034 cross-seed summary", ""]
    md += ["## Experiment 1 — single-domain (WikiText-2 → PubMed, 100K)", ""]
    md += ["| method | Δppl (mean±SD, p) | per-seed | source forgetting | mean M | eff. mean lr |",
           "|:-------|:------------------:|:--------:|:----------------:|:------:|:------------:|"]
    for key, label in [(single_plain_key, "plain (E032 reuse)"),
                       (single_gated_key, "gated"),
                       (single_control_key, "const_reduced control")]:
        g = single_sum.get(key) if key else None
        if not g or g["target_ppl_delta"]["n"] < 3:
            continue
        t = g["target_ppl_delta"]
        f = g["forgetting_pct"]
        md.append(
            f"| {label} | {fmt(t)} | {g['target_ppl_delta_per_seed']} | "
            f"{f['mean']:+.3f}% | {g['mean_surprise_M']:.3f} | "
            f"{g['effective_mean_lr']:.2e} |"
        )
    v1 = verdict.get("single")
    if v1:
        md += ["", "**Exp. 1 verdict:**", "",
               f"- Gated Δppl = **{v1['gated_delta_mean']:+.3f} ± {v1['gated_delta_sd']:.3f}** "
               f"(p={v1['gated_p']:.3f}, per-seed {v1['gated_delta_per_seed']})",
               f"- **Δppl ≥ 0.5 bar?** {'✅' if v1['gated_delta_ge_0_5'] else '❌'}",
               f"- **Source degradation < 1%?** {'✅' if v1['gated_forgetting_lt_1pct'] else '❌'} "
               f"(mean {v1['gated_forgetting_mean']:+.3f}%)",
               f"- **Δppl_gated ≤ Δppl_plain** ({v1['gated_delta_mean']:+.3f} vs "
               f"{v1['plain_delta_mean']:+.3f}): "
               f"{'✅' if v1['gated_delta_le_plain'] else '❌'} (learns less by design)",
               f"- **M-trace:** mean M = {v1['mean_surprise_M']:.3f}, "
               f"effective mean lr = {v1['effective_mean_lr']:.2e}",
               ""]
    md += ["## Experiment 2 — sequential two-domain (WikiText → PubMed → CNN/DailyMail)", ""]
    md += ["| method | PubMed Δppl after seq. | CNN Δppl (adapt to d2) | source forgetting |",
           "|:-------|:-----------------------:|:----------------------:|:-----------------:|"]
    for key, label in [(two_plain_key, "plain"), (two_gated_key, "gated")]:
        g = two_sum.get(key) if key else None
        p2 = two_p2.get(key) if key else None
        if not g or not p2 or g["target_ppl_delta"]["n"] < 3:
            continue
        t = g["target_ppl_delta"]
        f = g["forgetting_pct"]
        c = p2["phase2_ppl_delta"]
        md.append(
            f"| {label} | {fmt(t)} | {fmt(c)} | {f['mean']:+.3f}% |"
        )
    md += ["", "**Backward transfer on domain 1 (PubMed):** BT = pubmed_ppl_after_p2 − "
               "pubmed_ppl_after_p1 (+ = forgetting).", ""]
    md += ["| method | BT mean | BT per seed |",
           "|:-------|:-------:|:-----------:|"]
    for key, label in [("plain", "plain"), ("gated", "gated")]:
        bt = (bt_plain if key == "plain" else bt_gated)
        if bt:
            md.append(f"| {label} | {bt['mean']:+.4f} ± {bt['sd']:.4f} | "
                      f"{[bt['per_seed'][str(s)] for s in bt['seeds']]} |")
    v2 = verdict.get("two")
    if v2 and v2.get("selectivity_claim"):
        sc = v2["selectivity_claim"]
        md += ["", "**Selectivity claim (pre-registered: BT_gated < BT_plain):**",
               f"- BT_gated − BT_plain = **{sc['bt_gated_minus_plain']:+.4f}** "
               f"→ BT_gated {'<' if sc['bt_gated_lt_plain'] else '≥'} BT_plain "
               f"→ {'✅ **CLAIM HOLDS**' if sc['claim_holds'] else '❌ claim fails'}",
               f"- Seed agreement: {sc['seed_agreement']} "
               f"({'✅' if sc['seed_agreement_ge_2of3'] else '❌'} ≥ 2/3)",
               ""]
    md_path = os.path.join(args.results_dir, "summary_e034.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
