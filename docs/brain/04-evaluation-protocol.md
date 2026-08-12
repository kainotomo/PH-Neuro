# Step 0.5 — Evaluation Protocol (LOCKED)

> **Status:** ✅ **COMPLETE — PROTOCOL LOCKED (2026-08-12)**
> **Goal:** Define exactly how we measure success, how we compute the metric, what baselines we compare against, and what constitutes success/failure — **before any experiments run**.
> **Lock rule:** Once Phase 1.1 experiments begin, NO metric, baseline, domain, or threshold in this document may change without documenting a **protocol deviation** (dated note appended to this file + ROADMAP).
> **This document is the measurement spec for Step 1.1.** The Step 1.1 chat implements exactly what is written here.

---

## 0. What We Are Measuring (one line)

**Does surprise-modulated vector-bias Hebbian plasticity on a frozen pre-trained LM produce a measurable, statistically significant perplexity improvement on a target domain, without forgetting the source domain?**

Everything below operationalizes this sentence. All numbers in brackets `[V]` were **measured on 2026-08-12** with the actual Phase 1.1 eval model (`HuggingFaceTB/SmolLM2-1.7B`, bf16, eager attention) — not assumed.

---

## 1. Primary Metric

**Domain adaptation perplexity improvement:**
$$\Delta\text{ppl} = \text{ppl}_{\text{frozen}}(\text{target}) \;-\; \text{ppl}_{\text{plastic}}(\text{target})$$

Positive Δppl = plastic weights improved target-domain performance. Reported **per seed**, then summarized as mean ± SD across ≥3 seeds.

**Measured frozen baselines (SmolLM2-1.7B, 512-token window, eager attn):**
| Domain | ppl | 95% CI | σ_block (nats) | n_blocks |
|:-------|:---:|:------:|:--------------:|:--------:|
| WikiText-2 (source) | **10.65** | 10.21–11.11 | 0.262 | 589 |
| PubMed (target, 250K subsample) | **11.67** | 11.03–12.35 | 0.320 | 122+ |

**Interpretation of the shift:** PubMed is only **+9.5% harder** than WikiText-2 for SmolLM2-1.7B. This is the *desired* regime — a moderate, measurable domain shift (scientific register, terminology) that is large enough to detect a real signal but small enough that the surprise sigmoid operates in its sensitive range (per Step 0.3, s₀=0.05, k=60: a ~10% loss elevation saturates M; a ~5% elevation sits at the midpoint). An extreme shift (e.g., prose→code) would saturate M trivially and not discriminate surprise-modulated from constant-M plasticity. **PubMed is therefore the correct target, not code.**

---

## 2. Evaluation Domains (VERIFIED — not assumed)

### Source — WikiText-2
- **HF ID:** `Salesforce/wikitext`, config **`wikitext-2-raw-v1`**
- **Split:** `test` [V] 4,358 rows, 241,211 words
- **Token counts (test, no special tokens):** [V] **301,948** (SmolLM2 tokenizer), 283,287 (GPT-2 tokenizer)
- **Availability:** ✅ loads from HF (cached locally 2026-08-12); splits train 36,718 / val 3,760 / test 4,358
- **Eval corpus for the protocol:** the **full test split** (301,948 SmolLM2 tokens). No subsampling.

### Target — PubMed (scientific/medical register)
- **HF ID:** `ccdv/pubmed-summarization` (parquet, robust loading) — PubMed Central-derived, Cohan et al. 2018 (NAACL-HLT 2018), `@inproceedings{cohan-etal-2018-discourse}`
- **Split:** `test` [V] 6,658 documents. Fields: `article` (body), `abstract`, `id`. **Document = `abstract + " " + article`.**
- **Token counts (test, full text):** [V] **~28.999M tokens** (SmolLM2 tok) — **too large to evaluate in full repeatedly.** (Avg doc ≈ 3,092 whitespace tokens ≈ 4,356 BPE tokens.)
- **License:** ⚠️ **not declared** on the dataset card. It is a widely-used academic summarization corpus built from PubMed Central open-access abstracts/articles (NIH data; see the original `armancohan/long-summarization`). For research evaluation this is standard practice, but the **protocol notes the license as "undeclared on HF card; PMC-derived, Cohan et al. 2018"** and treats PubMed text as research-use evaluation data only (not redistributed). Document this in the experiment report.
- **Locked eval corpus:** a **deterministic 500,000-token subsample** of the PubMed test split (tokenize with the eval model's tokenizer, concat abstract+article, `random.Random(42)` document permutation, accumulate until ≥500K tokens). Fixed seed → bit-identical across every seed/baseline → paired comparisons are valid. (Chosen size: see §6 — 500K gives ~977 blocks, ~30× the d=0.5 sample-size requirement.)

### Alternatives considered (rejected, with reason)
| Alternative | Verdict | Why |
|:------------|:-------:|:----|
| `codeparrot/github-code` (general → code) | ❌ | License = `other` (incompatible with project's permissive-license rule from Step 0.1); shift too extreme → M saturates, ablations collapse. |
| `scientific_papers` (armanc, config `pubmed`) | ⚠️ | Same underlying PMC data; loading script instead of parquet; `ccdv/pubmed-summarization` is more robust. Not chosen. |
| `pubmed_abstracts` (HF) | ❌ | 401 / gated on load attempt 2026-08-12. |
| Legal / news / books corpora | ⏳ | Phase 1.2+ follow-up targets, not Phase 1.1. |

---

## 3. Adaptation Data Budgets (VERIFIED against EMA dynamics)

**Locked learn() config (from Step 0.4):** `batch_size=4`, `seq_len=256` → **1,024 tokens/step**.

| Budget | Tokens | ≈pages | Steps (b4/s256) | ~Sequences (s256) | EMA regime (α=0.99, τ=100 steps ≈ 102K tok) | Role |
|:-------|:------:|:------:|:---------------:|:-----------------:|:--------------------------------------------|:-----|
| **Micro** | 1K | 2 | **1** | 3 | 0.01τ — EMA never moves | Code sanity only (catches loader/loop bugs). Not a scientific point. |
| **Small** | 10K | 20 | **10** | 39 | 0.1τ — EMA barely tracks; M ≈ **constant** | **Go/no-go gate (fast mechanism test): does vector-bias Hebbian move ppl at all?** |
| **Medium** | 100K | 200 | **98** | 390 | ≈1τ — **the surprise window** (1/e residual) | **PRIMARY surprise test point.** Surprise-modulated vs constant-M are distinguishable here. |
| **Large** | 1M | 2,000 | **977** | 3,906 | ≈9.8τ — fully annealed | Saturation + forgetting-at-scale test. |

**Critical verified finding (this is why the budgets changed role vs the draft):**
> With α=0.99 and b4/s256 (1,024 tok/step), the EMA time constant is **τ = 100 steps ≈ 102,400 tokens**. A 10K-token run is only **10 steps (0.1τ)** — the EMA has decayed only ~10% of an initial surprise, so **M stays effectively constant (=1.0) for the whole run**. Therefore:
> - **10K CANNOT test surprise modulation** — it degenerates into the constant-M condition. It is only a mechanism-viability gate.
> - **100K is the minimum budget where surprise-modulated ≠ constant-M** (98 steps ≈ 1τ; ~37% of a sustained surprise signal remains at the end → a real modulation curve).
> - **1M fully exercises the anneal** (0.005% residual → M returns to ≈0.018 → automatic learning-rate decay to near-zero).

**EMA warmup requirement (VERIFIED design implication):** Because Phase 1.1 adapts *on the target domain directly*, the EMA baseline L̂ must be established **on the source domain first**, otherwise L̂ starts at the first PubMed loss and the domain shift produces **s ≈ 0 → M ≈ 0.018 (≈ no learning)** for the entire run. Locked procedure:
1. **Warmup:** run `warmup_steps ≥ 50` steps on **WikiText-2 train** with `M=0` (EMA settles on source loss; no plastic update). [50 steps = 51.2K tokens = 0.5τ — enough to seed L̂; 100 steps safer.]
2. **Adapt:** switch input to **PubMed train**, keep EMA running, apply surprise-modulated updates. The first target steps see L ≫ L̂ → s > s₀ → M high → the intended "plasticity window."
3. The warmup tokens do **not** count toward the adaptation budget (they are the baseline reference, not the learning budget).

---

## 4. Baselines (with Step 1.1 implementation notes)

| # | Baseline | Implementation (Step 1.1) | Expected |
|:-:|:---------|:--------------------------|:---------|
| **B1** | **Frozen (zero plasticity)** | Eval with `with brain.without_plasticity():` — hooks return output unchanged → this *is* the frozen model by construction (Step 0.4 invariant I1). | The floor. Plasticity must beat this. |
| **B2** | **Random plastic weights** | Init every plastic bias to `randn(0, 0.01)` on a **fixed seed** (reuse seed 42), skip learning, eval. Controls for added capacity alone. | Plasticity training must beat random-initialized bias (which is just a fixed perturbation). |
| **B3** | **Constant M (no surprise)** | `modulator_cfg = {"mode": "constant", "M": 1.0}` — the Hebbian rule with M=1 always. **THE key ablation.** | Surprise-modulated (B4) must beat or match constant-M. This isolates whether the surprise signal matters. |
| **B4** | **Surprise-modulated (the method)** | Locked defaults: α=0.99, s₀=0.05, k=60, M_max=1.0, η=1e-3, decay λ=0.0 (Phase 1.1), warmup per §3. | The hypothesis. |
| **B5** | **LoRA (practical upper bound)** | **Rank-1 LoRA on the 24 `o_proj` modules only** (SmolLM2) / 12 `attn.c_proj` (GPT-2). Params: 24·(2048+2048)·1 = **98,304 (SmolLM2) / 18,432 (GPT-2) — EXACT match to the vector-bias plastic budget.** AdamW lr=5e-5, wd=0.0, 3 seeds, same budgets. Feasible on 8 GB (frozen bf16 base + tiny adapter states). | The realistic ceiling — same parameter budget, backprop update rule. |
| **B6** | **Full fine-tuning (true upper bound)** | **⚠️ INFEASIBLE on SmolLM2-1.7B / RTX 4060 8 GB — see §4.1.** Feasible on GPT-2 124M only (Phase 1.3). | If feasible (GPT-2): AdamW lr=5e-5, 3 seeds, same budgets. |

### 4.1 Full fine-tuning memory feasibility — VERIFIED (2026-08-12)

SmolLM2-1.7B = 1,711,376,384 params. State-only minimums (activations, CUDA context, fragmentation excluded):

| Config | Weights | Grads | Optimizer | **Total** | Fits 8 GB? |
|:-------|:-------:|:-----:|:---------:|:---------:|:----------:|
| fp32 weights + fp32 AdamW | 6.85 GB | 6.85 GB | 13.69 GB | **27.38 GB** | ❌ |
| bf16 weights + fp32 grads | 3.42 GB | 6.85 GB | 13.69 GB (fp32 AdamW) | **23.96 GB** | ❌ |
| bf16 + bf16 grads + 8-bit AdamW | 3.42 GB | 3.42 GB | 3.42 GB | **8.56 GB** | ❌ (usable ≈ 7.5 GB; Xwayland holds ~0.6 GB) |
| bf16 + fp32 grads + 8-bit AdamW | 3.42 GB | 6.85 GB | 3.42 GB | **13.69 GB** | ❌ |

**Conclusion:** even the most aggressive combination (bf16 weights + bf16 grads + 8-bit AdamW + gradient checkpointing + batch 1) needs **8.56 GB of pure state** before a single activation — already over the ~7.5 GB usable on the 8 GB card. **Full fine-tuning of SmolLM2-1.7B is INFEASIBLE on the RTX 4060.** → The SmolLM2 upper bound is **B5 (LoRA)**. Full fine-tuning is implemented only on **GPT-2 124M** in **Phase 1.3** (~2 GB state — feasible), as the cross-architecture upper bound.

---

## 5. Perplexity Computation (LOCKED)

- **Tokenizer:** the **model's own tokenizer** (SmolLM2 or GPT-2 as the eval model). No special tokens added for eval (`add_special_tokens=False`).
- **Window:** **fixed context window = 512 tokens for BOTH models.** Rationale (verified): SmolLM2's max ctx is 8192 and GPT-2's is 1024; using each model's max would make Phase 1.3 (cross-architecture) numbers **incomparable**. A fixed 512 window (well under both models' max, above the 256 adaptation length) makes every ppl comparable across models, budgets, and seeds. **This is the locked choice; full-context (8192/1024) ppl is reported only as an optional diagnostic, never as a primary comparison.**
- **Stride:** **256 tokens (50% overlap)** for both models (consistent with the draft's 50% convention; 512 window → stride 256). A 50% overlap roughly doubles effective samples at ~2× compute; the power analysis (§6) already uses the conservative non-overlap n.
- **First token handling:** the **first token of each window is skipped** (no left context). Matches HF's standard sliding-window ppl.
- **Aggregation:** primary = **unweighted per-token average** (every predicted token counts once; the standard LM ppl and the quantity the EMA/surprise is built on). Secondary = **weighted per-sequence average** (mean of per-window ppl) reported alongside for robustness.
- **Numerics:** compute NLL in float32 (logits→float32); final ppl = exp(mean NLL).
- **Sliding-window rule (HF convention):** a token at position t with only `k < 511` tokens of context in the first window is evaluated only in the window where it has the most context; overlapping windows each evaluate all their positions with their own context (tokens are not double-subtracted; each window is a full causal LM forward).

---

## 6. Statistical Protocol (LOCKED)

1. **Seeds:** ≥3 (42, 43, 44 — matching project convention). Each seed = fresh plastic-weight init + fresh data shuffle order. Frozen baseline is seed-independent (same model, same eval corpus) — measured once and reused.
2. **Report:** mean ± SD across seeds for all metrics.
3. **Confidence intervals:** **bootstrap 95% CI** on Δppl — resample **blocks** (512-token windows, preserving paired structure: same blocks for frozen & plastic) 10,000×, percentile CI. Paired bootstrap (frozen & plastic on the *same* blocks) is mandatory — it removes corpus-composition variance.
4. **Significance:** **paired t-test on Δppl across seeds** (per-seed Δppl paired to the frozen baseline on identical eval tokens). Report t, df, p. Also report the paired block-level t-test (n = blocks, paired frozen↔plastic) as supporting evidence.
5. **Effect size:** **Cohen's d** = mean(Δppl per seed) / SD(Δppl per seed); paired d also reported (block-level). Small d=0.2, medium d=0.5, large d=0.8.
6. **Sample size (VERIFIED):** to detect **d=0.5 at 80% power, two-sided α=0.05**, with per-block std σ (measured: 0.262 WikiText-2 / 0.320 PubMed):
   $$n_{\text{chunks}} = \left(\frac{z_{0.975} + z_{0.80}}{d}\right)^2 = \left(\frac{2.802}{0.5}\right)^2 \approx 31.4 \text{ blocks} \;\Rightarrow\; n_{\text{tokens}} = 31.4 \times 512 \approx \mathbf{16,074} \text{ tokens}.$$
   - **WikiText-2 test (301,948 tok)** = 589 non-overlap blocks ≈ **18.8×** the requirement.
   - **PubMed eval corpus (500,000 tok)** = 977 non-overlap blocks ≈ **31×** the requirement.
   - **Both test corpora are comfortably large enough.** The detectable floor at the planned corpora (conservative unpaired, 95% CI): WikiText-2 ≈ 0.022 nats ≈ 0.23 ppl; PubMed ≈ 0.020 nats ≈ 0.23 ppl (≈2% relative). A **0.5 ppl** effect (4.3% relative on ppl 11.67) is **~10–25× above the noise floor** → easily detectable. The paired design (same blocks) is even stronger.
   - **Practical implication:** the binding constraint on detectability is the **number of seeds / across-seed variance**, not test-set size. With 3 seeds, the across-seed t-test needs a consistent Δppl sign + low across-seed spread; the block-level paired test carries most of the power.

---

## 7. Pre-Registered Failure Criteria (LOCKED, concrete)

### Phase 1.1 success (ALL must hold at the primary test point — 100K tokens):
1. **Δppl > 0** on PubMed, **p < 0.05** (paired, across seeds) AND
2. **Δppl > random-plastic (B2) Δppl** (plasticity training beats a fixed random perturbation) AND
3. **Source degradation < 1%** relative on WikiText-2 test (frozen 10.65 → plastic ≥ 10.54 ppl) AND
4. **Surprise-modulated (B4) ≥ constant-M (B3)** (the surprise signal is not worse than always-learning) AND
5. **Δppl ≥ 0.5 ppl** (practically meaningful — **not** merely statistically significant; see §7.1).

### Partial success (valuable, publishable):
- Δppl > 0, p < 0.05, but **0 < Δppl < 0.5 ppl** → plasticity helps but trivially (report honestly).
- Δppl ≥ 0.5 but **surprise ≤ constant-M** → plasticity helps, our modulator design doesn't — the mechanism works, the signal is wrong.

### Informative failure (publishable as a negative result):
- **Δppl ≈ 0 (or < random-plastic)** at 100K for all configurations → local Hebbian plasticity cannot adapt a pre-trained transformer beyond random perturbation. Consistent with E001–E019 (Hebbian cannot train from scratch); now shown for adaptation too.

### Phase 1.1 is a hard FAIL (stop):
1. No statistically significant Δppl at 100K **and** no signal at 1M;
2. Δppl positive but **below random-plastic** (B2) → the update rule is actively worse than a random perturbation;
3. Source degradation ≥ 1% (plasticity is destroying the frozen model's source ability).

### 7.1 "Practically meaningful" vs "statistically significant but trivial"
- **Statistically detectable floor:** ≈ 0.23 ppl (2% relative) at the planned corpora. Any Δppl above ~0.2 ppl is detectable above measurement noise.
- **Practically meaningful (pre-registered bar):** **Δppl ≥ 0.5 ppl** (≈4.3% relative on PubMed ppl 11.67). Rationale: 0.5 ppl is a real, user-visible improvement on a strong model (SmolLM2-1.7B), is ~10–25× the noise floor (so it is robust, not borderline), and matches the "0.5 ppl" figure the project brief treats as meaningful.
- **Statistically significant but trivial:** 0.2–0.5 ppl. Report as such — a real effect, but not a practical win.
- **Below 0.2 ppl:** treat as "within noise" regardless of nominal significance (spurious).

---

## 8. Secondary Metrics

| Metric | Definition | Threshold |
|:-------|:-----------|:----------|
| **Forgetting (source ppl)** | Δppl_source = ppl_plastic(source) − ppl_frozen(source); negative = improved. | ≤ +1% relative (lenient) / ≤ +0.1% (strict) |
| **Forward transfer** | After learning PubMed, eval on a **third held-out domain** (e.g., `scientific_papers/arxiv` or CNN/DailyMail as a *third* probe) — improvement over frozen = positive transfer. | Reported, no gate |
| **Backward transfer** | Re-eval source (WikiText-2) after each budget to trace forgetting as a function of adaptation size. | See forgetting |
| **Plasticity efficiency** | Δppl per plastic parameter (98,304 SmolLM2). | Reported |
| **Surprise dynamics** | Per-step M trace: warmup≈0 → shift spike → anneal. Report mean M, % steps M>0.5, final M. | Sanity check (§3) |

---

## 9. Experiment Tracking & Files

- **Results dir:** `results/brain/e031/` (e031 = Phase 1.1 minimal viable; naming per draft, now pinned).
- **File:** `results/brain/e031/{model}_{target}_{budget}_{baseline}_seed{seed}.json`
  - model: `smolllm2_1p7b` | `gpt2_124m`; target: `pubmed`; budget: `1k|10k|100k|1m`; baseline: `frozen|random|constM|surprise|lora`
- **JSON schema (extend the draft):**
```json
{
  "experiment": "e031_minimal_viable",
  "model": "HuggingFaceTB/SmolLM2-1.7B",
  "plasticity": "vector_bias",
  "modulator": "surprise_ema",
  "target_domain": "pubmed",
  "adaptation_tokens": 100000,
  "warmup_steps": 100,
  "seed": 42,
  "eval": {"window": 512, "stride": 256, "aggregation": "unweighted"},
  "metrics": {
    "source_ppl_frozen": 10.65,
    "source_ppl_plastic": 10.63,
    "source_ppl_delta": -0.02,
    "target_ppl_frozen": 11.67,
    "target_ppl_plastic": 11.15,
    "target_ppl_delta": 0.52,
    "target_ppl_delta_ci95": [0.31, 0.73],
    "paired_t": 8.1, "paired_p": 0.015, "cohens_d": 1.4,
    "forgetting_pct": 0.19,
    "mean_surprise_M": 0.41,
    "pct_steps_M_gt_05": 0.22,
    "final_M": 0.02
  },
  "plastic_weights": {"count": 98304, "bytes": 393216, "mean_magnitude": 0.011, "sparsity": 0.88}
}
```
- **Baseline registry (pre-measured, reused across seeds):** frozen ppl per domain (seed-independent). Random-plastic has 3 seeds.
- **Harness scripts** (Step 1.1): `research/scripts/verify_eval_datasets.py` + `calibrate_eval_power.py` are the verified dataset/eval primitives; Step 1.1 builds the `BrainWrapper.evaluate()` on top.

---

## 10. What Step 1.1 Must Deliver (checklist)

- [ ] `BrainWrapper.evaluate(texts, window=512, stride=256)` → frozen & plastic ppl, per-block NLL
- [ ] `warmup` on WikiText-2 train (M=0) → then adapt on PubMed train
- [ ] 4 baselines × 4 budgets × 3 seeds on SmolLM2-1.7B (100K = primary point)
- [ ] LoRA rank-1 o_proj baseline (98,304 params) at 100K & 1M, 3 seeds
- [ ] Bootstrap paired CI + paired t-test + Cohen's d
- [ ] M-trace sanity check (warmup≈0 → spike → anneal)
- [ ] Result JSONs in `results/brain/e031/`

---

## 11. Protocol Deviation Log

| Date | Deviation | Reason | Approved by |
|:-----|:----------|:-------|:------------|
| — | none yet | — | — |

---

## 12. Verification Artifacts (this step)

- `research/scripts/verify_eval_datasets.py` → `research/scripts/eval_dataset_verify.json` — splits, token counts, budget→step table
- `research/scripts/calibrate_eval_power.py` → `research/scripts/eval_power_calibration.json` — SmolLM2-1.7B frozen ppl on both domains, σ_block, d=0.5 sample size
- Env note: this machine has **no C compiler** → torch 2.13 Triton fused `bmm` (RoPE) fails at JIT. Verified workaround used in calibration: `deregister_op_overrides(disable_op_symbols="bmm")` + `attn_implementation="eager"`. Step 1.1 must apply the same to avoid the Triton path.
