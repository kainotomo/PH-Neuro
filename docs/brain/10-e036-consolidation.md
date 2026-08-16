# Step 2.3 — Consolidation Mechanism (E036)

> **Status:** ✅ COMPLETE — pre-registered 2026-08-15; smoke (4 cells) + full
> three-condition × three-seed run (15 cells) 2026-08-15→16, **0 failures**.
> **Bottom line:** the pre-registered **top-K=10% (by |ΔW|) sparse
> consolidation — LONG-TERM store + per-domain SHORT-TERM with warm-start —
> DOES NOT deliver forward transfer on the T-C ternary stack; it actively
> HURTS D3.** C's D3 Δppl = **+0.020 ± 0.001** vs B1's fresh **+0.178 ± 0.034**
> (paired forward transfer **−0.158 ± 0.034, p=0.015** — significantly
> negative; C/B1 ratio **0.112**), and C's D3 plateau is ~**2× slower**
> (360 vs 176 steps). C's BT ≈ 0 (< 0.1) **but vacuously** — the top-10% LT
> captures essentially **no knowledge** (D1/D2 LT effects +0.000/+0.007), so
> there is no interference but also no retention. C's storage **wins**
> (deployed 0.333×, LT+deltas 0.931×), but only by throwing away the adapted
> knowledge. **The mechanism fails because the T-C sign pattern is 100% dense:
> |ΔW| latent magnitude is a poor importance proxy for a sign-quantized
> adapter (every nonzero entry contributes ±1 regardless of magnitude), and
> warm-starting from a sparse LT sign pattern is a destructive prior.** The
> forward-transfer potential exists only with FULL retention (B2 continuing
> D3 +0.187 > B1 +0.178) — but B2 pays real interference (BT_D1 +0.265).
> **Question (one line):** With per-domain T-C ternary STE adapters + the
> surprise gate (forgetting already ≈ 0), can a **sleep-like consolidation
> mechanism** — a long-term store that accumulates the top-K% of each
> domain's short-term plastic changes — give **forward transfer** (each new
> domain warm-starts from it and adapts faster/better) **without
> reintroducing interference** (backward transfer < 0.1), at **equal-or-less
> storage** than N full per-domain adapters?
> **Base:** the proven **T-C ternary STE gated LoRA** adapter (E035): rank-1
> `o_proj`+`down_proj` = 344,064 params, 86,016 B 2-bit packed, surprise-gated
> lr `η·M_t` (α=0.99, s₀=0.05, k=60, M_max=1.0, η=1e-3), WikiText-2 warmup
> M=0. Single-domain Δppl +0.892 ± 0.206 (99% of float), two-domain BT
> −0.0118 < 0.1 — the on-device stack is "surprise-gated, ternary,
> continually-learning adapters", 86 KB per domain.

---

## 1. The scientific question — what can consolidation add?

E034/E035 established that per-domain adapters + the surprise gate already
keep forgetting at ≈ 0 (BT −0.009 float / −0.0118 ternary on a two-domain
sequence). So consolidation is **not** needed to *stop forgetting* — the
selectivity already does that. The task specifies two candidate value-adds,
and E036 tests **both** on a **three-domain** sequence (the first 3-domain
test in the product path):

1. **Forward transfer** — a long-term (LT) store that accumulates cross-domain
   knowledge; each new domain warm-starts its short-term (ST) from it and
   should adapt **faster / better** than a fresh adapter. This is the
   biologically-motivated claim: consolidated knowledge (e.g. "how scientific
   text reads") transfers to the next domain.
2. **Storage management** — one long-term adapter + small per-domain deltas
   vs N full adapters (86 KB each). If the LT captures the *important* (top-K)
   plasticity, the product stores **one shared adapter** per device instead of
   N.

With three domains (WikiText-2 → PubMed → CNN/DailyMail → **C4**), the
forward-transfer test becomes strong: D3 is the *hardest* domain (frozen ppl
13.57 vs 10.66/11.46/11.97), so any warm-start advantage shows clearly.

## 2. Why now — the last brain mechanism on the product path

Phase 2 has validated four brain mechanisms on the product adapter:
- **Surprise gate** (E034) — selectivity (BT −0.009 vs plain +1.854);
- **Ternary storage** (E035) — 2-bit STE adapter at 99% of float quality,
  16× smaller (86 KB);
- **Selectivity** (E034/E035) — the gate + per-domain adapters preserve
  earlier domains (BT < 0.1, 3/3 seeds);
- **Consolidation** (E036, this step) — sleep-like transfer of important
  plastic changes from short-term to long-term storage.

If E036 shows forward transfer without interference at ≤ B1 storage, **all
four mechanisms are tested on the product path — Phase 2 complete**, ready
for Phase 3 (multi-domain at scale, scaling laws, vision) and the paper.

## 3. Context — what the previous steps established (recap)

| Step | Method (matched 344K budget, 100K) | Δppl (mean) | Source | Verdict |
|:----:|:-----------------------------------|:-----------:|:------:|:--------|
| E031 | vector-bias surprise Hebbian (98K) | +0.034 | +0.37% | stable, sub-threshold |
| E032 | low-rank Hebbian (344K) | −1.349 | +13.1% | destructive |
| E032 | backprop LoRA (344K, lr=1e-3) | **+1.520** | −6.53% | exceeds 0.5 bar |
| E033 | predictive coding (344K) | +0.001 | −0.012% | stable, inert |
| E034 | **gated LoRA (344K, η=1e-3)** | **+0.902** | −2.66% | gate adds value; selectivity (BT −0.009) |
| E035 | **T-C ternary STE gated LoRA (344K)** | **+0.892** | −2.26% | 2-bit at 99% of float, 16× storage, two-domain BT −0.0118 |

**E036 base = E035's T-C adapter** (the selected product adapter). The E035
two-domain run (PubMed → CNN) was a *continuing* adapter — the interference
floor style. E036's **B2** reproduces that as a 3-domain continuing adapter;
**B1** is the independent per-domain baseline; **C** is the new consolidation.

## 4. Design

### 4.1 The sequence (3 domains, 100K each, + WikiText-2 warmup)

| Phase | Domain | Source | License | Frozen ppl | Shift vs Wiki |
|:-----:|:-------|:-------|:--------|:----------:|:-------------:|
| warmup | WikiText-2 (train, M=0) | `Salesforce/wikitext` | cc-by-sa-3.0 | 10.664 | — |
| **D1** | PubMed | `ccdv/pubmed-summarization` | undeclared (PMC-derived) | 11.457 | +7.4% |
| **D2** | CNN/DailyMail | `abisee/cnn_dailymail` (3.0.0) | **apache-2.0** | 11.971 | +12.3% |
| **D3** | **C4 (Common Crawl web)** | `allenai/c4` (`en`) | **ODC-BY** | **13.568** | **+27.2%** |

**D3 choice — C4 (documented, verified 2026-08-15):**
* **License ODC-BY** (Open Data Commons Attribution 1.0) — attribution-only,
  **no non-commercial / no share-alike restriction** → product-path compatible
  (consistent with the project's permissive-license rule, which rejects only
  NC/SA-restricted licenses such as pile-of-law's cc-by-nc-sa-4.0 or
  codeparrot/github-code's "other").
* **Availability verified**: loads natively (parquet/streaming) on this
  machine; deterministic 500K eval subsample of the **validation** split
  (doc-permutation seed 42 over a fixed 4000-doc head), 300K train buffer
  (seed 42 over a fixed 3000-doc head), 50K probe (seed 43). All bit-identical
  across seeds.
* **Why NOT legal**: SCOTUS (8.41), LEDGAR (8.46) and EUR-LEX (6.51) frozen
  ppl are all *lower* than WikiText-2 (10.66) — legal text is more predictable
  for SmolLM2. A D3 that is *easier* than the running EMA keeps the surprise
  gate **closed** at the D3 boundary (loss *falls* → s < 0 → M ≈ 0) → D3
  adaptation ≈ 0 for every condition → the forward-transfer test is **vacuous**.
  C4 (13.57) is *harder* than CNN (11.97), so the gate opens (loss rises past
  the EMA) and D3 adaptation is a real, discriminating quantity.
* **Domain distinctness**: general web text — a genuinely new 4th register
  (encyclopedia → scientific → news → informal web), and the **hardest** of the
  four, so the LT's accumulated knowledge has the most room to help.

### 4.2 Three conditions (same protocol, 3 seeds 42/43/44)

All use the **T-C ternary STE gated LoRA** adapter (E035) at rank 1 =
344,064 params, AdamW wd=0, surprise lr `η·M_t` (η=1e-3), 100-step WikiText
warmup (M=0), 98 adapt steps per domain (100K), EMA running continuously
across the adapt phases (no reset at boundaries — the E035 convention, so each
boundary opens a bounded plasticity window).

| ID | Condition | Mechanism | Storage |
|:--:|:----------|:----------|:--------|
| **B1** | Independent per-domain adapters, no consolidation (the current best) | Each domain trains a **fresh** T-C adapter (its own WikiText warmup → its domain). Adapters never touch other domains → BT = 0 by construction. | 3 × 86,400 B = **259,200 B** |
| **B2** | Interference floor | One **single continuing** T-C adapter across all 3 domains (EMA continuous, no resets) — the E035 two-domain style extended to 3. | 1 × 86,400 B = **86,400 B** |
| **C** | Consolidation | **LT store + per-domain ST.** After each domain: transfer the top-K% (by \|ΔW\|) of the ST's latent-score changes into LT (**add rule**); reset ST; next domain **warm-starts ST from LT**. | LT (86,400 B) + per-domain sparse deltas |

### 4.3 The consolidation mechanism (C) — pre-registered choices

All decisions documented before running:

1. **K = 10%** (the top decile of latent-score changes by |ΔW| magnitude, taken
   over the **global** 344,064-param budget). Rationale: the largest plasticity
   events carry the most transferable information; 10% is a substantial but
   conservative transfer (a 90%-sparse delta) — large enough to move knowledge,
   small enough to keep the delta far smaller than a full adapter.
2. **Transfer rule = add** (`LT ← LT + Δ_topK`, sparse top-K add). Copy would
   overwrite earlier domains' contributions; add accumulates the store.
3. **LT decay = none** (persistent store). The surprise gate already limits
   overwriting during adaptation; an LT decay would add a hyperparameter with no
   pre-registered benefit.
4. **Warm-start rule = copy LT into ST** (ST's latent scores *and* scales are
   overwritten by LT's before each new domain). ST's ΔW for the next transfer
   is measured relative to this warm-start, so each delta is that domain's own
   change. **LT keeps the canonical init scales** (A_scale = 1/sqrt(d_in),
   B_scale = 1e-2) — only latent-score **signs** transfer, so LT's injection
   magnitude stays ~0.01 (matching the float adapter) without re-tuning.
5. **ST reset**: after the transfer, ST is re-initialized to LT (the warm-start
   *is* the reset — D1's LT is the all-zero store, so D1's ST is exactly the
   fresh E035 adapter, making D1 identical for B1 and C).
6. **What transfers**: the T-C **latent scores** (A_latent, B_latent) — the
   sign pattern is the knowledge; scale magnitudes are reset per the canonical
   init.

**Storage accounting (pre-registered, on-device format):**
* **C deployed artifact** = the **LT** (one packed T-C adapter: 86,016 B +
  384 B scales = 86,400 B). This is what runs at inference — the storage-
  management claim (1 adapter instead of N).
* **C full storage (LT + deltas)** = LT (86,400 B) + 3 per-domain deltas, each
  stored as a **1-bit presence mask** (344,064 bits = 43,008 B) + **2-bit
  packed ternary signs** of the kept entries (K% × 344,064/4 B = 8,602 B at
  K=10%) + 2 fp32 per-A/B scales (8 B) = **51,618 B/delta** → C = 86,400 +
  154,854 = **241,254 B**.
* **Conservative upper bound** = the int32-index sparse variant (n_kept × 4 B
  indices + signs + scales = 146,234 B/delta → C = 525,102 B) — reported for
  transparency, not the primary accounting (a bitmap is the natural on-device
  sparse-ternary format).
* B1 = 3 × 86,400 = 259,200 B. So **C/B1 = 0.33 (deployed) or 0.93 (LT+deltas,
  bitmap)** — both ≤ 1.0.

### 4.4 Eval schedule (window 512 / stride 256, frozen caches reused)

| Condition | After D1 | After D2 | After D3 (full sequence) | D3 probes |
|:----------|:---------|:---------|:--------------------------|:----------|
| B1 | D1 adapter on PubMed | D2 adapter on CNN | D3 adapter on C4 (+ source) | every 10 steps in D3 (B1's c4 cell) |
| B2 | continuing on PubMed | continuing on PubMed + CNN | continuing on PubMed + CNN + C4 | — |
| C | **LT₁** on PubMed | **LT₂** on PubMed + CNN | **ST₃** on C4 (full D3), **LT₃** on PubMed + CNN + C4 | every 10 steps in D3 |

**Backward transfer (C):** `BT_C(D1) = ppl(LT₃ on PubMed) − ppl(LT₁ on
PubMed)`; `BT_C(D2) = ppl(LT₃ on CNN) − ppl(LT₂ on CNN)` — the pure
interference from later-domain deltas entering the shared store. B1's BT = 0
by construction; B2's BT is the interference floor.

**Forward transfer (D3):** C's D3 adaptation uses **ST₃** (the full
warm-started short-term — apples-to-apples with B1's full fresh D3 adapter).
The consolidated LT₃'s D3 Δppl is reported separately (the deployed artifact).

**Adaptation speed:** 50K-token C4 probe evals (seed-43 subsample, independent
of the 500K seed-42 eval corpus) every 10 steps in the D3 phase. Steps to
plateau = first step from which all later probe ppls are within **+0.5%** of
the final probe ppl.

## 5. Pre-Registered Success Criteria (before running)

| # | Criterion | Rationale |
|:-:|:----------|:----------|
| 1 | **Forward transfer non-negative:** `Δppl_C_D3 ≥ Δppl_B1_D3` (C's warm-started ST₃ vs B1's fresh D3 adapter on C4); ideally `>`. | The LT store must not hurt D3 adaptation; the claim is it helps (faster/better). Also report the C/B1 ratio. |
| 2 | **Backward transfer on D1 and D2: `BT_C(D1) < 0.1` AND `BT_C(D2) < 0.1`** (LT-based, §4.4). | Consolidation must not reintroduce interference — adding later domains' deltas to the shared store must not degrade earlier domains beyond a 0.1-ppl tolerance (the E035 selectivity bar). |
| 3 | **Storage ≤ B1: C's total ≤ B1's total** — both the deployed-LT figure (ratio 0.33) and the LT+deltas (bitmap) figure (ratio 0.93) must be ≤ 1.0. | The storage-management claim: one shared adapter + small deltas replaces N full adapters. |
| 4 | Report the **C/B1 ratio on D3 Δppl and on storage**; report **adaptation speed** (steps to plateau, C vs B1) — reported, not gated. | The task's explicit reporting requirements. |
| 5 | D3 adaptation must be non-vacuous for **all** conditions: the gate must open at the D3 boundary (mean D3 adapt M > 0), i.e. C4 (13.57) must be harder than the running EMA after CNN. | Sanity — verified by C4's frozen ppl in §4.1. |

**Pass:** criteria 1 + 2 + 3 all hold → consolidation gives forward transfer
without interference at equal-or-less storage → the complete Phase 2 product
path (surprise gate + ternary + selectivity + consolidation) holds → Phase 3.
**Fail (any):** report honestly which criterion fails and why; e.g. if BT ≥ 0.1
the shared store reintroduces interference; if storage > B1 the deltas cost
more than they save; if forward transfer < 0 the warm-start hurts D3.

## 6. Implementation plan

| File | Change |
|:-----|:-------|
| `src/ph_neuro/brain/lora.py` | **Extend** — E036 consolidation machinery: `tc_latent_state`, `tc_set_latent_state`, `zero_lt_state`, `latent_change_topk` (global top-K by \|ΔW\|), `add_delta_to_lt` (add rule), `warm_start_st_from_lt`, `sparse_delta_storage`. |
| `src/ph_neuro/brain/datasets.py` | **Extend** — C4 (D3) loaders (`c4_train_ids`/`c4_eval_ids`/`c4_probe_ids`) + `make_four_domain_batch_iter`. |
| `src/ph_neuro/examples/run_e036_consolidation.py` | E036 runner: `--condition b1\|b2\|c`, boundary-aware gated training, LT/ST bookkeeping, boundary + final evals, D3 probes, checkpoint/resume (LT + boundary flags + eval cache), storage report. |
| `src/ph_neuro/examples/aggregate_e036.py` | Cross-seed aggregation + verdict vs §5 (forward transfer paired by seed, BT gates, storage ratios, steps to plateau). |
| `scripts/run_e036_consolidation.sh` | Orchestrator: `smoke` → `b1` → `b2` → `c` → `agg`; skip-if-exists; GPU gate; frozen-cache reuse. |
| `tests/brain/test_e036_consolidation.py` | Unit tests: latent-state round-trip, zero-LT identity, top-K masking (global threshold, true float Δ at kept positions), add accumulation, warm-start injection, sparse-storage accounting, mini two-domain consolidation, steps-to-plateau. |

**Operational rules (unchanged):** GPU gate ≥ 6 GiB free, checkpoints every
100 steps + at boundaries + SIGINT/SIGTERM (atomic temp+rename, skip-if-exists
via result JSON), `PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`,
Triton-bmm workaround + eager attention, logs → `logs/brain/e036/`, results →
`results/brain/e036/`, venv `.venv/bin/python`.

## 7. Protocol notes

The measurement protocol is **unchanged** (metric, window/stride, budgets,
baseline reuse, statistics, thresholds). The **new** elements are (a) the
**third evaluation domain — C4** (`allenai/c4` config `en`, license **ODC-BY**;
frozen ppl 13.568 on a deterministic 500K validation subsample — harder than
all prior domains, so the surprise gate opens at the D3 boundary) and (b) the
**consolidation mechanism itself** (an adapter-management change, not a
measurement change) — recorded here and appended to the LOCKED protocol's
deviation log (§11 of `04-evaluation-protocol.md`) as a Step 2.3 entry. No
post-hoc criterion changes.

## 8. Results

E036 ran the pre-registered protocol: smoke (10K, all conditions, seed 42) +
full three-domain (B1/B2/C × 3 seeds × 100K/domain). **15 cells, 0 failures.**
Frozen baselines (seed-independent caches): WikiText-2 **10.664**, PubMed
**11.457**, CNN/DailyMail **11.971**, C4 **13.568**.

### 8.1 Cross-condition table (100K, 3 seeds)

| Condition | D1 (PubMed) Δppl | D2 (CNN) Δppl | **D3 (C4) Δppl** | BT_D1 | BT_D2 |
|:----------|:-----------------:|:-------------:|:----------------:|:-----:|:-----:|
| **B1** (independent per-domain) | **+0.892 ± 0.206** (p=0.017) | **+0.106 ± 0.044** (p=0.052) | **+0.178 ± 0.034** (p=0.012) | 0 (by constr.) | 0 (by constr.) |
| **B2** (continuing adapter) | — | — | **+0.187 ± 0.017** (p=0.003) | **+0.265 ± 0.123** | **−0.039 ± 0.010** |
| **C** (consolidation, ST_D3) | — | — | **+0.020 ± 0.001** (p=0.002) | **−0.001 ± 0.001** | **−0.008 ± 0.003** |
| **C** (consolidation, LT_D3) | — | — | **+0.020 ± 0.001** | −0.001 ± 0.001 | −0.008 ± 0.003 |

Per-seed (D3 Δppl, BT): B1 +0.141/+0.208/+0.186; B2 +0.168/+0.195/+0.199
(BT_D1 +0.124/+0.351/+0.320); C +0.020/+0.021/+0.018 (BT_D1 −0.001/−0.001/
+0.000, BT_D2 −0.011/−0.007/−0.005).

### 8.2 The consolidation condition (C) in detail (seed 42, representative)

* D1: ST trains a fresh adapter (identical to B1's — D1's LT is the zero
  store). After D1: transfer top-10% by |ΔW| into LT; **the LT's D1 effect on
  PubMed is +0.0000** (11.4572 → 11.4572) — the sparse top-10% store captures
  **none** of the D1 adapter's +0.655 gain.
* D2: ST warm-starts from LT₁, trains CNN. LT₂'s CNN effect: **+0.007**
  (11.9706 → 11.9635) — ~6% of B1's fresh CNN gain (+0.106).
* D3: ST warm-starts from LT₂, trains C4 with the gate open (mean D3 M ≈ 0.5+,
  C4's +27% frozen shift opens the window). **ST₃'s C4 Δppl = +0.020** vs B1's
  fresh +0.141 — the sparse-LT warm-start **costs 0.121 ppl of adaptation**.
* BT (LT-based): BT_D1 = −0.001, BT_D2 = −0.011 — both < 0.1, but only because
  the LT is nearly empty (no interference because no knowledge).

### 8.3 Forward transfer (the decisive quantity)

* Paired Δppl_C_D3 − Δppl_B1_D3 = **−0.158 ± 0.034 (p = 0.015)**, per-seed
  −0.121/−0.187/−0.168 — **significantly NEGATIVE** (3/3 seeds).
* **Ratio C/B1 on D3 Δppl = 0.112** (C achieves ~11% of B1's fresh D3 gain).
* **Adaptation speed**: C's D3 plateau (50K-probe, within 0.5% of final) at
  steps **357–367** vs B1's **170–180** — C is **~2× slower** to plateau; the
  warm-start from the sparse LT does not accelerate D3, it slows it.
* B2 (full-retention continuing) D3 = +0.187 vs B1 +0.178 — the only condition
  where forward transfer is positive (~5%), at the cost of BT_D1 +0.265.

### 8.4 Storage (C ≤ B1, both pre-registered figures)

| Artifact | Bytes | Ratio vs B1 |
|:---------|------:|:-----------:|
| B1 (3 × 86,400) | **259,200** | 1.000 |
| B2 (1 × 86,400) | **86,400** | 0.333 |
| C deployed (LT) | **86,400** | **0.333** |
| C LT + 3 sparse deltas (bitmap) | **241,254** | **0.931** |
| C LT + 3 sparse deltas (int32-index, conservative) | 525,102 | 2.026 |

Storage passes both pre-registered bars — but the "savings" come from a store
that retains ~1% of the adapted knowledge.

## 9. Pre-registered criteria — full check

| § | Criterion | Result |
|:-:|:----------|:-------|
| 7-1 | **C's D3 ≥ B1's D3 (forward transfer non-negative)** | ❌ **−0.158 ± 0.034 (p=0.015); C/B1 = 0.112** — significantly **negative** (3/3 seeds) |
| 7-2 | **C's BT on D1 < 0.1** | ✅ **−0.001 ± 0.001** — but **vacuous** (LT captures ~0) |
| 7-2 | **C's BT on D2 < 0.1** | ✅ **−0.008 ± 0.003** — but **vacuous** |
| 7-3 | **C's storage ≤ B1** (deployed and LT+deltas) | ✅ **0.333** and **0.931** — storage management works |
| 7-5 | D3 adaptation non-vacuous (gate opens) | ✅ mean D3 M > 0 (C4 frozen +27% shift opens the window; B1's fresh D3 +0.178 confirms D3 is learnable) |

**The primary criterion (forward transfer) FAILS decisively.** The BT and
storage criteria pass, but the BT pass is vacuous and the storage pass is
achieved by discarding the learned knowledge.

## 10. Verdict

**Does the long-term store give forward transfer without interference at
equal-or-less storage? NO — for the pre-registered top-K |ΔW| consolidation on
the T-C ternary stack.**

1. **The consolidation mechanism FAILS its primary claim.** The top-K=10% (by
   |ΔW| latent magnitude) sparse transfer + warm-start gives **significantly
   negative forward transfer** (−0.158 ± 0.034 on D3, p=0.015; 3/3 seeds), ~9×
   worse than a fresh D3 adapter, and is ~2× slower to plateau. The sparse
   LT sign pattern is a **destructive prior** for the next domain.
2. **Why (the mechanistic diagnosis, pre-registration-amendment §7):** the T-C
   adapter's sign pattern is **100% dense** (sign(A)/sign(B) nonzero fraction
   = 1.0 after training), and a sign-quantized injection weights every nonzero
   entry equally (±1). Therefore **|ΔW| latent magnitude is a poor importance
   proxy** — a rank-1 outer product loses ~(1−K)² of its injection when K% of
   entries are dropped, so a top-10% store captures ~1% of the injection (the
   LT's D1/D2 effects are +0.000/+0.007). The failure is intrinsic to
   *sparse* consolidation on a *dense-sign* representation.
3. **Where forward transfer DOES appear:** the faithful continuing adapter
   (B2) — which retains the full previous adapters — reaches D3 +0.187 vs B1
   +0.178 (a small positive transfer), but pays **real interference**:
   BT_D1 = **+0.265 ± 0.123** (the 3-domain continuing adapter forgets D1, a
   direct extension of E035's 2-domain BT −0.0118 to a longer sequence). B1's
   independent per-domain adapters remain the only condition with **zero
   interference by construction** while still adapting every domain.
4. **Storage management works but is self-defeating.** C ≤ B1 (0.333×
   deployed, 0.931× LT+deltas), yet the compact store retains ~1% of the
   adapted knowledge — it saves bytes by erasing what was learned.
5. **Honest bottom line for Phase 2:** the fourth brain mechanism
   (consolidation) is **negative on the product path as designed**. The
   on-device stack that Phase 2 has validated is *surprise gate + ternary
   storage + selectivity* (E034/E035); consolidation does not add forward
   transfer and only "saves" storage by discarding knowledge. The T-C ternary
   representation needs a different consolidation abstraction (e.g. full-sign
   or module-level transfer — which collapse toward B2's storage, or a
   non-ternary LT) before a long-term store can help.

## 11. Next-step implications

* **The product remains B1-style per-domain T-C adapters** (86 KB each,
  selective, zero interference by construction) — consolidation as tested does
  not beat it.
* **The dense-sign × sparse-transfer incompatibility is the key negative
  finding**: any future consolidation for the ternary stack must transfer at a
  granularity that preserves the sign pattern (full per-module transfer, which
  costs like B2, or a hybrid with a coarser long-term representation), not
  top-K by |ΔW| latent magnitude.
* **B2's 3-domain BT_D1 (+0.265) extends E035's selectivity caveat**: the
  surprise gate's near-zero BT held for 2 domains but degrades on a 3-domain
  continuing adapter — supporting the per-domain-adapter product over the
  continuing adapter as the sequence grows.
* **Phase 3 queue**: multi-domain at scale should use B1-style per-domain
  adapters (proven scalable, selective); the consolidation question is closed
  for the sparse top-K mechanism but could be revisited with a different
  store representation. The paper can report E036 as a clean negative:
  sparse-ternary consolidation fails on a dense-sign adapter — a mechanistic
  (not tuning) result.

## 12. Reproducibility

- Deterministic: `torch.manual_seed(seed)` per process; block-shuffled batch
  order seeded; eval corpora fixed subsamples (seed 42/43); B1's b1_pubmed
  cell reproduces E035's single-domain T-C seed 42 **bit-identically**
  (Δppl +0.655, forgetting −2.224%) — protocol reproduction confirmed.
- LT/ST bookkeeping (top-K mask, sparse add, warm-start) is pure float32
  tensor math on CPU; checkpoints save LT states + boundary flags + eval
  cache + RNG so resume continues bit-exactly.
- Everything to re-run: `bash scripts/run_e036_consolidation.sh <mode>`
  (`smoke | b1 | b2 | c | all | agg`); skip-if-exists via result JSON;
  results `results/brain/e036/`, logs `logs/brain/e036/`, frozen caches
  copied from E034/E035 + C4 computed once.
