# Step 1.3 (re-scoped) — Error-Based Local Rules: Predictive Coding (E033)

> **Status:** ✅ COMPLETE — pre-registered 2026-08-14; smoke (10K) + primary
> (100K × 3 seeds) run 2026-08-14, 0 failures.
> **Question:** The last untested local-rule family with a credit-assignment
> story: can an **error-based, no-backprop predictive-coding rule** adapt a
> frozen LLM at the matched-budget point where E032's low-rank Hebbian was
> catastrophically unstable? E032's verdict (capacity/gain/decay are not the
> missing ingredient; **credit assignment is**) leaves exactly one candidate
> family. This experiment is the **final local-rule test**.
> **Bottom line:** PC at the matched 344K budget is **stable but inert**:
> Δppl_PC = **+0.001 ± 0.003** (p = 0.737, per-seed +0.001 / +0.004 /
> −0.003) — statistically null, ~1500× below the LoRA bound (+1.52), below
> even the 0.2-ppl "within noise" floor. The error-driven update **removes
> E032's instability** (source *improved* −0.012%, no forgetting) but the
> plastic weights barely move (mean|B| ≈ 0.0016) and the reconstruction
> error carries no usable adaptation direction. **The local-rule scientific
> question is CLOSED (pre-registered kill criterion): the project pivots to
> the backprop-LoRA product path.**
> **Spec:** [03-architecture.md](03-architecture.md) (extended for
> predictive coding) + [04-evaluation-protocol.md](04-evaluation-protocol.md)
> (LOCKED; see §9 for the explicit re-scoping amendment).
> **Pre-registered:** 2026-08-14 (success criteria §7, kill criteria §8,
> formulation choice §4, hyperparameters §5).

---

## 1. Experiment Summary

E033 keeps the E031/E032 protocol exactly (frozen `HuggingFaceTB/SmolLM2-1.7B`
bf16, WikiText-2 → PubMed, window 512/stride 256, 100K primary point, 3 seeds
42/43/44, surprise modulator locked at α=0.99 / s₀=0.05 / k=60 / M_max=1.0)
and changes only the **plastic update direction**: from the E032
surprise-modulated **Hebbian correlation** (`ΔW ∝ pre ⊗ post`) to an
**error-driven predictive-coding rule** (`ΔW ∝ ε ⊗ post` where ε is a
per-injection-site reconstruction error). The rank-1 low-rank structure, the
init, the injection sites, and the parameter budget are **identical** to the
E032 Part D LoRA baseline (344,064 params) — so the comparison isolates the
learning rule, exactly as E032's Part A vs D did.

| Item | Value |
|:-----|:------|
| Model | `HuggingFaceTB/SmolLM2-1.7B`, bf16, eager attention |
| Plasticity | **Predictive coding (PC-ERR)**: rank-1 `W_plastic = B·A` at the same 48 sites (`o_proj` + `down_proj`) |
| Update | `ΔA = η·M·mean((Bᵀ·post)⊗ε) / rms` · `ΔB = η·M·mean(post⊗(A·ε)) / rms`, `ε = pre − W_inv·post` (signed per-dim error) |
| Inverse | Per-site low-rank linear inverse `W_inv = U·V` (rank 8), learned by local recirculation (no backprop) |
| Plastic params | **344,064** (rank 1) — EXACT match to the E032 LoRA budget/structure/init |
| Inverse params | ~2.75M (auxiliary machinery, ~11 MB fp32, reported separately — like AdamW states) |
| Surprise | Locked E031 defaults; gates only the A/B injection update |
| Budgets | **10K** (mechanism gate) · **100K** (primary point), 3 seeds |
| Eval | window 512 / stride 256, unweighted per-token mean NLL, float32 (locked) |

**Baselines (for the matched-budget contrast):** frozen (Δppl = 0), E032
backprop LoRA at the same 344K budget (Δppl = **+1.52**, best lr), E032 local
low-rank Hebbian at the same budget (Δppl = **−1.35** at rank 1). The E032
LoRA and Hebbian numbers are **reused from the E032 report** — they ran on the
identical protocol, so no cells are re-run.

## 2. Why Now — the E032 verdict

E032 answered the two E031 bottleneck hypotheses **decisively in the negative**
for local plasticity and identified the binding constraint:

1. **Capacity destroys.** Every low-rank Hebbian config is negative (rank 1:
   −1.35; rank 2: −1648; rank 4: −3172 mean Δppl) with catastrophic forgetting
   (up to +99,000%). The unsupervised correlation, given matrix capacity,
   amplifies the strongest channels (**Hebbian concentration**; max |B|
   element 1.8 → 58) and — via the surprise gate's positive feedback (loss
   spike → M → larger maladaptive update) — destroys the model.
2. **Gain is the poison, not the cure.** η↑, s₀↓, k↓, M_max↑ are all
   monotonically destructive (best −1.35 → −126K at η=1e-2); damage tracks
   mean M.
3. **Decay is neutral.** λ ∈ {1e-5, 1e-4} ≈ λ=0 — the damage is a
   within-window feedback loop, not secular weight growth.
4. **Backprop is the difference.** Matched-budget LoRA **exceeds the 0.5
   practical bar at every lr** (best +1.52) with the source *improved*
   (−6.5 to −8.5%). The best local config is opposite in sign (ratio
   ≈ −0.84).

**E032's explicit next-step option #1** was *"error-based local rules
(predictive coding / target propagation) — local rules that still propagate
some error signal per layer, giving credit assignment without full backprop …
the only path that could close the 3-order-of-magnitude gap to the LoRA
bound."* E033 is that experiment, and the **final** test of the "no backprop"
thesis (§8 kill criteria).

## 3. Scope re-scoping (recorded, not silent)

The original Step 1.3 in `ROADMAP.md`/`BRAIN.md` was **Architectural
Generalization** (repeat the vector-bias experiment on GPT-2 124M). E032's
verdict (§12 of its report) explicitly queued error-based local rules as the
higher-value next step, and the Phase 1.1/1.2 story is incomplete without a
decisive test of the only remaining local-rule family. This step **re-scopes
Step 1.3 to the predictive-coding test** on the primary model (SmolLM2-1.7B,
the model where the local-rule question must be answered), keeping the
measurement protocol unchanged (metric, domains, budgets, window/stride,
baselines, statistics, thresholds). The re-scoping is recorded as a **protocol
amendment** in §9 and appended to the LOCKED protocol's deviation log (§11 of
`04-evaluation-protocol.md`). The GPT-2 replication remains a possible cheap
follow-up only if PC is positive (§8); it is **not** the primary question of
this step.

## 4. Formulation choice — the design decision

Three local, no-backprop predictive-coding formulations were on the table
(from [01-plasticity-mechanisms.md](01-plasticity-mechanisms.md) §4 and the
task spec):

| # | Formulation | ε | ΔW |
|:-:|:------------|:--|:---|
| **1** | **Per-injection-site reconstruction error** | `ε = x − W_inv·post` (small linear inverse per site predicts the block input from the output) | `ΔW = η·M·ε ⊗ post` (signed, per-dimension) |
| 2 | Difference target propagation | `ε = target − actual` (feedback connections produce a per-layer target) | `ΔW = η·M·ε ⊗ pre` |
| 3 | Feedback alignment variant | `ε_i = B_i·ε_{i+1}` (fixed random B) | `ΔW = η·M·ε ⊗ post` |

**Chosen: Formulation 1 (PC-ERR).** Justification, tied directly to E032's
failure modes:

### 4.1 Why Formulation 1 fixes E032's concentration

E032's low-rank Hebbian is `ΔB ∝ post ⊗ (A·pre)` — a **raw correlation** of
the frozen output and the (A-filtered) input. Correlations are always
positive-definite in the co-activation direction: whatever co-occurs is
strengthened, the dominant channels grow fastest, and max |B| runs away
(concentration). Formulation 1 replaces `pre` with **ε = pre − W_inv·post**,
the signed reconstruction residual. ε is:

- **Signed per dimension** — it can be positive *or* negative, so the update
  is error-driven (anti-Hebbian where the reconstruction overshoots, Hebbian
  where it undershoots), **not a raw correlation**.
- **Self-limiting** — as the inverse W_inv fits the local
  input→output mapping, ‖ε‖ shrinks and the updates shrink. There is no
  concentration: ε does not grow with B (B only appears inside the update
  through A·ε, and ε is B-independent because it uses the **pre-injection
  frozen post** — the same feedback-free amendment E032 needed for stability).
- **Bounded** — ε is the residual of a well-posed least-squares problem
  (predicting the block input from the block output), so ‖ε‖ is bounded by
  ‖x‖ + ‖x̂‖, both unit-scale after RMSNorm.

### 4.2 Why Formulation 1 fixes E032's surprise positive feedback

E032's feedback loop: *plastic perturbation → loss spike → surprise M rises →
next update is larger → perturbation worse.* The loop is destructive because
the update direction itself was maladaptive (a correlation with no error
content) — bigger M amplified a *bad* direction. In Formulation 1 the update
direction is **corrective**: it pushes toward reducing the per-layer
reconstruction error. The surprise gate is retained exactly as E031 validated
it (protective), but the direction it amplifies now has error content — M
spikes amplify a principled, error-reducing step, not a correlation that
concentrates noise.

### 4.3 Why Formulation 1 is the credit-assignment answer

E032's bottom line was "the missing ingredient is credit assignment." ε at
each site is a **per-layer, signed error signal**: "the frozen output no
longer reconstructs its input under the new domain." Unlike the global
surprise scalar (which says "learn now" but carries no per-layer credit),
ε says *where* and *in which direction* the representation drifted. This is
the local credit-assignment story PC is designed to provide (Rao & Ballard
1999; Whittington & Bogacz 2017: PC ≈ backprop with local Hebbian updates).

### 4.4 Why not Formulations 2 and 3

- **Formulation 2 (DTP)** requires trained per-layer feedback networks. The
  survey ([01-plasticity-mechanisms.md](01-plasticity-mechanisms.md) §6)
  rejected it: training the feedback nets with backprop violates the
  no-backprop constraint; training them with local rules just moves the
  problem; inverting multi-head attention is ill-posed; and DTP has never been
  demonstrated at transformer scale.
- **Formulation 3 (feedback alignment with random B)** re-propagates a
  top-down error through fixed random matrices. The project's NTH-4b result
  (dense, continuous feedback through latent scores still failed to train
  hidden layers; "the feedback signal is a correlation, not a gradient") is a
  direct, unfavorable precedent. It also needs a top-down error chain through
  the frozen backbone that is not locally available without extra machinery,
  and its errors would remain correlations (not reconstruction residuals),
  leaving E032's concentration risk partially intact.

### 4.5 The update (final, implemented)

Per injection site (48 sites), with `pre` = projection input (captured),
`post` = **pre-injection frozen** projection output (captured; feedback-free):

$$x̂ = U\,(V\,post) \qquad \varepsilon = x - x̂ \qquad (\text{low-rank inverse } W_{inv}=U\cdot V,\ \text{rank 8})$$

Inverse update — **local recirculation, no backprop, not surprise-gated** (it
tracks the local reconstruction statistics; its own small η_inv and decay keep
it stable):

$$\Delta V = \eta_{inv}\,\frac{\mathrm{mean}_{bs}\big((U^{\top}\varepsilon)\otimes post\big)}
{\mathrm{rms}(U^{\top}\varepsilon)\,\mathrm{rms}(post)}, \qquad
\Delta U = \eta_{inv}\,\frac{\mathrm{mean}_{bs}\big(\varepsilon\otimes (V\,post)\big)}
{\mathrm{rms}(\varepsilon)\,\mathrm{rms}(V\,post)}$$

Plastic (model-affecting) update — **error-driven, surprise-gated**, the
projection of `ΔW = η·M·mean(ε ⊗ post)` onto the low-rank manifold `W = B·A`:

$$\Delta A = \eta\, M\,\frac{\mathrm{mean}_{bs}\big((B^{\top}post)\otimes \varepsilon\big)}
{\mathrm{rms}(B^{\top}post)\,\mathrm{rms}(\varepsilon)},\qquad
\Delta B = \eta\, M\,\frac{\mathrm{mean}_{bs}\big(post\otimes (A\,\varepsilon)\big)}
{\mathrm{rms}(post)\,\mathrm{rms}(A\,\varepsilon)}$$

**Reading:** the only difference from the E032 Hebbian update is that the
input-side activation `pre` is replaced by the reconstruction error `ε`. The
A factor is driven by the frozen output weighted by the error it causes; the B
factor by the frozen output weighted by the A-filtered error. Both use only
`pre`, `post`, the site's own A/B, and the site's own U/V — fully local, no
backprop. `ΔW ∝ ε ⊗ post` is the task-spec form; the per-factor rms
normalisation is carried over from E032's stabilised implementation so the
step stays a bounded correlation.

## 5. Pre-registered configuration (before running)

| Param | Value | Rationale (locked) |
|:------|:------|:-------------------|
| plasticity | `predictive_coding` | Formulation 1 (PC-ERR), §4 |
| rank (A/B) | **1** | exact match to the E032 LoRA baseline structure |
| inv_rank (U/V) | 8 | small linear inverse; ~11 MB fp32 auxiliary |
| η (plastic lr) | 1e-3 | E032 best local default |
| η_inv (inverse lr) | 1e-3 | comparable adaptation rate |
| inv_decay | 1e-4 | keeps W_inv bounded (stability) |
| A/B decay λ | 0.0 | E032 found decay neutral |
| Surprise | α=0.99, s₀=0.05, k=60, M_max=1.0 | LOCKED E031 defaults |
| Warmup | 100 steps WikiText (M=0; W_inv fits the source-domain map) | LOCKED protocol §3 |
| Budget | **100K primary**, 3 seeds 42/43/44 | LOCKED protocol |
| Eval | window 512 / stride 256 / unweighted / float32 | LOCKED protocol §5 |

**Matched-budget guarantee:** the plastic parameters are `A+B` at rank 1 =
344,064 — exactly the E032 Part D LoRA budget, same init (A ~ N(0, 1/d_in),
B = 0), same injection sites, same architecture. The inverse U/V (~2.75M
params, 11 MB) are **auxiliary learning machinery** (analogous to the AdamW
m/v states of the LoRA baseline, which are likewise not counted in the 344K
budget): they do not touch the model output directly and are reported
separately (`inverse_weights`). **The model-affecting budget is matched.**

## 6. Implementation (build on E032, no rewrite)

| File | Change |
|:-----|:-------|
| `src/ph_neuro/brain/block_wrappers.py` | `InjectionPoint` gains `U`/`V`/`inv_rank` (the linear inverse `W_inv = U·V`); both wrappers accept `inv_rank=` (vector-bias/low-rank paths unchanged). |
| `src/ph_neuro/brain/brain_wrapper.py` | `plasticity="predictive_coding"` + `inv_rank`/`inv_lr`/`inv_decay`; error-driven A/B update (§4.5); local-recirculation U/V update; U/V in `state_dict`/`load_state_dict`/`to`/checkpoints; `inverse_parameter_count()` and `mean_inverse_error()`; A/B param count unchanged → budget matched. |
| `tests/brain/test_predictive_coding.py` | 16 tests: shapes, I1 identity, no-backprop, derived-update numerics (ΔA/ΔB/ΔU/ΔV), inverse decay, matched-budget-vs-low_rank, serialization. |
| `src/ph_neuro/examples/run_e033_predictive_coding.py` | Single-cell runner (clone of the E032 runner): `--plasticity predictive_coding --rank 1 --tag pc --inv-rank/--inv-lr/--inv-decay`, protocol-schema JSON with `method: predictive_coding` + `inverse_weights`. |
| `src/ph_neuro/examples/aggregate_e033.py` | Cross-seed aggregation + verdict vs §7/§8 (Δppl, p, forgetting, LoRA ratio, kill-criteria consequence). |
| `scripts/run_e033_predictive_coding.sh` | Orchestrator: `smoke` (10K gate) → `primary` (3×100K), skip-if-exists, GPU gate, frozen-cache reuse. |
| `results/brain/e033/` + `logs/brain/e033/` | Result JSONs (protocol schema, `method: predictive_coding`) + logs. Frozen evals **reuse the E031/E032 cache** (identical seed-independent baseline). |

**Operational rules (unchanged):** GPU gate ≥ 6 GiB free (exit policy),
checkpoints every 100 steps (atomic, skip-if-exists), SIGINT/SIGTERM handlers,
`PYTHONUNBUFFERED=1`, `TOKENIZERS_PARALLELISM=false`, Triton-bmm workaround +
eager attention (no C compiler), logs → `logs/brain/e033/`, results →
`results/brain/e033/`.

## 7. Pre-Registered Success Criteria (before running)

At the **100K primary point**, 3 seeds:

| # | Criterion | Pass? |
|:-:|:----------|:------|
| 1 | **Δppl_PC ≥ 0.5 ppl** on PubMed (the practical bar) | ❌ Δppl_PC = **+0.001 ± 0.003** (p = 0.737) — ~500× below the bar, below the 0.2 "within noise" floor |
| 2 | **Sign agreement with LoRA: Δppl_PC > 0** (unlike E032's −1.35; the update must be error-driven, not a raw correlation) | ❌ mean +0.001 is nominally > 0 but statistically indistinguishable from 0 (p = 0.737); seed 44 is **negative** (−0.003). Not a robust positive sign. |
| 3 | **Forgetting < 1%** on WikiText-2 (source degradation) | ✅ mean **−0.012%** — the source *improved*; PC is stable (no forgetting at all) |
| 4 | **Ratio Δppl_PC / Δppl_LoRA reported** (vs E032's ≈ −0.84) | ✅ ratio ≈ **0.000** (Δppl_PC +0.001 vs LoRA +1.520 — ~1500× gap) |

## 8. Time-Box & Kill Criteria (pre-registered consequence — CRITICAL)

This is the **LAST local-rule experiment**. Per the pre-registration:

1. **One shot per formulation.** No second PC variant, no hyperparameter
   rescue, no capacity escalation.
2. **Kill criterion:** if PC at matched budget fails — **Δppl_PC ≤ 0** or
   worse than random (Δppl < 0 with p ≥ 0.05) — the **local-rule scientific
   question is CLOSED**. The project **pivots to the backprop-LoRA product
   path** (E032's proven +1.52 bound) as the adaptation mechanism.
3. **Conditional extension only:** additional formulations (e.g. the §4
   Formulation 3 feedback-alignment variant) are permitted **only if** the
   primary shows **Δppl > 0 with p < 0.05**.

The decision is applied automatically by the aggregator (`aggregate_e033.py`)
and recorded in §12. **Applied: PC failed → CLOSED (§12).**

## 9. Protocol Amendment — explicit re-scoping (not silent)

**Recorded 2026-08-14.** The original Step 1.3 scope in `ROADMAP.md`/
`BRAIN.md` was **Architectural Generalization** (GPT-2 124M replication of the
vector-bias result). E032's verdict (§12 of `06-e032-capacity-gain.md`)
explicitly queued error-based local rules as the highest-value next test —
the only remaining local-rule family with a credit-assignment story, and the
only path that could close the ~3-order-of-magnitude gap to the LoRA bound.
This step **re-scopes Step 1.3 to the predictive-coding experiment** on the
primary model. The measurement protocol (metric, domains, budgets, baselines,
statistics, thresholds) is unchanged. This re-scoping is recorded here and
appended to the LOCKED protocol's deviation log (§11 of
`04-evaluation-protocol.md`) rather than silently changing scope.

## 10. Results

E033 ran the pre-registered protocol exactly: 10K smoke (mechanism gate, seed
42) + 100K primary × 3 seeds (42/43/44), 0 failures, matched rank-1 344K
budget, surprise-gated error-driven PC update. The verdict is a **clean null**:

| Seed | Δppl (PubMed) | 95% CI | Source forgetting | block p | block d | mean M | mean\|B\| | mean\|ε\| |
|:----:|:-------------:|:------:|:-----------------:|:-------:|:-------:|:------:|:--------:|:--------:|
| 42 | **+0.001** | [−0.000, 0.002] | −0.008% | 0.113 | +0.036 | 0.100 | 0.0011 | 0.139 |
| 43 | **+0.004** | [0.002, 0.005] | −0.025% | <1e-4 | +0.128 | 0.167 | 0.0019 | 0.137 |
| 44 | **−0.003** | [−0.004, −0.001] | −0.004% | 0.002 | −0.071 | 0.155 | 0.0018 | 0.139 |
| **mean** | **+0.001 ± 0.003** | — | **−0.012%** | p = 0.737 | +0.031 | 0.141 | 0.0016 | 0.138 |

### 10.1 What worked: the instability is gone

The single most important mechanical result: **the error-driven update is
stable.** Across all 3 seeds and the smoke:
* No NaN, no divergence, no catastrophic forgetting — **source ppl improved**
  in every seed (−0.008% / −0.025% / −0.004%). This is the direct fix of
  E032's failure modes: the signed reconstruction error ε (instead of the raw
  pre ⊗ post correlation) cannot concentrate, and the plastic weights stay
  bounded (max element 0.35–0.82, mean|B| ≈ 0.0016). E032's low-rank Hebbian
  at the same budget destroyed the model (Δppl −1.35, +13% forgetting);
  E033's PC at the same budget does nothing but breaks nothing.
* The surprise gate behaved exactly as designed (mean M 0.10–0.17, spikes to
  M > 0.9 on hard PubMed steps) yet — because the update direction is now
  corrective — even a surprise spike (seed 44's final M = 0.98) produced no
  damage. E031's protective-gate property is preserved.

### 10.2 What failed: no adaptation — the plastic weights barely move

The decisive finding is *inertia*, not instability. After 100K tokens of
adaptation the plastic weights are still at their init scale:
* **mean|A| ≈ 0.0107** — statistically identical to the init (A ~ N(0, 1/d_in)
  ⇒ mean|A| ≈ 0.011 on `down_proj` d_in = 8192). **A essentially never
  moved.**
* **mean|B| ≈ 0.0016** — grew from 0 but is ~10⁻³ of the frozen weight scale,
  and the injection B·(A·x) is therefore a ~10⁻³ relative perturbation — far
  too small to change ppl (hence Δppl ≈ 0).
* **mean|ε| ≈ 0.138, unchanged across seeds** — the reconstruction error did
  not shrink with adaptation (the inverse W_inv never closed the gap), and
  more importantly it did not carry a direction that moves the plastic
  weights toward a lower-LM-loss configuration.

Mechanistic reading: ε = pre − W_inv·post is the residual of an
**underdetermined** inverse problem (the block input is not reconstructible
from a single projection output — many x map to the same post). The residual
is dominated by this uninformative underdetermination, not by a domain-shift
signal the plastic weights can act on. The error-driven update therefore
makes the model neither better nor worse: it removes the correlation-driven
destruction (E032) but provides **no usable credit assignment** for this
task. The per-site reconstruction error is a *consistent* but *uninformative*
signal — the PC story needs a level of per-layer consistency that a single
linear inverse per projection cannot provide on a 1.7B residual transformer.

## 11. Result Tables

### 11.1 Primary (100K, matched budget) — full protocol-schema metrics

| Metric | seed 42 | seed 43 | seed 44 | mean ± SD |
|:-------|:-------:|:-------:|:-------:|:---------:|
| target ppl frozen | 11.457 | 11.457 | 11.457 | 11.457 |
| target ppl plastic | 11.456 | 11.454 | 11.460 | 11.457 ± 0.003 |
| **Δppl** | **+0.001** | **+0.004** | **−0.003** | **+0.001 ± 0.003** |
| source ppl frozen | 10.664 | 10.664 | 10.664 | 10.664 |
| source ppl plastic | 10.663 | 10.661 | 10.663 | 10.662 |
| forgetting | −0.008% | −0.025% | −0.004% | −0.012% |
| block paired p | 0.113 | <1e-4 | 0.002 | (cross-seed) p = 0.737 |
| block Cohen's d | +0.036 | +0.128 | −0.071 | +0.031 |
| mean surprise M | 0.100 | 0.167 | 0.155 | 0.141 |
| % steps M > 0.5 | 8.1% | 14.6% | 14.1% | 12.3% |
| mean\|A\| (plastic) | 0.0107 | 0.0108 | 0.0108 | 0.0108 |
| mean\|B\| (plastic) | 0.0011 | 0.0019 | 0.0018 | 0.0016 |
| max \|plastic\| | 0.593 | 0.352 | 0.816 | 0.59 |
| mean\|U\| (inverse) | 0.0088 | 0.0087 | 0.0088 | 0.0088 |
| mean\|V\| (inverse) | 0.0020 | 0.0020 | 0.0020 | 0.0020 |
| mean\|ε\| | 0.139 | 0.137 | 0.139 | 0.138 |
| plastic params | 344,064 | 344,064 | 344,064 | **344,064 (matched)** |
| inverse params | 2,752,512 | 2,752,512 | 2,752,512 | 2,752,512 (aux) |

### 11.2 The matched-budget contrast (this is the whole comparison)

| Method (rank-1, 344K, 100K) | Δppl (mean) | Source | Forgetting | Verdict |
|:----------------------------|:-----------:|:------:|:----------:|:--------|
| **Frozen** (baseline) | 0 | — | — | floor |
| **E033 predictive coding** | **+0.001** | improved | −0.012% | **stable, inert** |
| **E032 low-rank Hebbian** | **−1.349** | destroyed | +13.1% | catastrophically unstable |
| **E032 backprop LoRA** | **+1.520** | improved | −6.5% | exceeds the 0.5 bar |

At the **identical** parameter budget, architecture, init and injection sites,
only the update rule differs: raw correlation destroys (−1.35), backprop
credit assignment adapts (+1.52), and the error-driven PC signal is stable
but carries no usable direction (+0.001). The PC-to-LoRA ratio ≈ **0.000**.

### 11.3 Smoke (10K, mechanism gate)

| Metric | value |
|:-------|:------|
| Δppl (seed 42) | +0.0003 (≈0, expected at 0.1τ) |
| source forgetting | −0.008% |
| mean M | 0.028 |
| mean\|ε\| | 0.142 |

The smoke validated the mechanism end-to-end (stable learning, correct
surprise dynamics, matched budget, no NaN) — a GO for the primary.

## 12. Verdict

**Final verdict (4 cells, 0 failures): PC at the matched budget is a clean
null — and therefore the pre-registered kill criterion fires.**

* **Criterion 1 (Δppl ≥ 0.5): NOT met.** Δppl_PC = +0.001 ± 0.003 (p = 0.737),
  ~500× below the practical bar and below the 0.2-ppl "within noise" floor.
* **Criterion 2 (sign agreement with LoRA): NOT met.** The mean is nominally
  positive but statistically zero (p = 0.737), and seed 44 is negative —
  there is no robust positive sign.
* **Criterion 3 (forgetting < 1%): met.** −0.012% — the source *improved*.
  PC is the first local rule since E031's vector-bias to be fully stable at
  matrix capacity. The error-driven direction **fixed E032's instability.**
* **Criterion 4 (ratio):** Δppl_PC / Δppl_LoRA ≈ **0.000** (−1500× gap).
  E032's ratio was −0.84 (destructive); E033's is 0.000 (inert).

**What this answers, definitively:**
1. **The missing ingredient is NOT a signed/error-driven local direction per
   se** — replacing the correlation with the reconstruction error removes the
   destruction but does not create adaptation. E032's "credit assignment"
   verdict stands, but the conclusion sharpens: *local* error signals of the
   predictive-coding family do not provide the credit assignment a frozen
   transformer needs to adapt, at least not through a per-projection linear
   inverse.
2. **The local-rule research question is CLOSED.** E031 (vector-bias Hebbian:
   +0.034, stable, sub-threshold), E032 (low-rank Hebbian: destructive), E033
   (predictive coding: stable, inert) span the three axes a local rule could
   plausibly occupy — a stable-but-trivial signal, an unstable correlation,
   and an error-driven but uninformative signal. **No local no-backprop rule
   reaches the 0.5-ppl practical bar at a matched budget.**
3. **The project pivots to the backprop-LoRA product path** (the pre-registered
   consequence): E032's proven +1.52 bound at 344K params, source improved,
   is the product adaptation mechanism. LoRA is not "local" in the brain-like
   sense, but it is the only matched-budget rule that works.

**Pre-registered consequence (applied):** *PC failed the pre-registered
criteria at matched budget. The local-rule scientific question is CLOSED: the
project pivots to the backprop-LoRA product path. No second PC variant, no
hyperparameter rescue, no capacity escalation.*

### What this means for Phase 2

* **Step 2.1 (Low-Rank Plastic Matrices) is answered negative twice over** —
  E032 (Hebbian) and E033 (predictive coding) both fail at matched budget;
  the Phase 2.1 "more capacity → better adaptation" hypothesis for local
  rules is **falsified**.
* **Backprop LoRA becomes the Phase 2 scaling mechanism** (the product path):
  it is the proven +1.52 bound; Phase 2.2 (ternary plastic weights via
  DQT/hysteresis) can be re-scoped to apply the existing ternary/DQT
  infrastructure to **LoRA adapters** rather than local-rule plastic weights.
* The Brain Wrapper infrastructure (hooks, checkpoints, surprise modulator,
  protocol harness) remains the vehicle — it cleanly hosts LoRA-style
  adapters (E032 Part D already proved this).

## 13. Reproducibility

- Same operational rules as E031/E032: GPU gate ≥ 6 GiB free, checkpoints
  every 100 steps (atomic, skip-if-exists), SIGINT/SIGTERM handlers,
  `PYTHONUNBUFFERED=1`, logging to `logs/brain/e033/`.
- Deterministic: `torch.manual_seed(seed)` per process; deterministic A/U init
  (process RNG), block-shuffled batch order (seeded), PubMed eval subsample
  fixed at seed 42 (bit-identical across seeds/configs → paired stats valid).
- Everything to re-run: `bash scripts/run_e033_predictive_coding.sh primary`
  (skips completed cells).
- Frozen evals cached under `results/brain/e033/cache/` (reused from E031).
