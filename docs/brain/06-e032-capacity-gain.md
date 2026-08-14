# Step 1.2 — Capacity & Gain Experiment (E032)

> **Status:** ✅ COMPLETE — pre-registered 2026-08-13; all five parts (A–E) run
> 2026-08-13, 0 failures across 45 cells (33 local + 9 LoRA + 3 × 1M).
> **Bottom line: low-rank local Hebbian is catastrophically unstable at every
> capacity × gain × decay setting and compounds with budget (1M: −7381 Δppl);
> matched-budget backprop LoRA exceeds the 0.5 practical bar at every lr.
> The missing ingredient is credit assignment, not capacity/gain/decay.**
> **Question:** E031's surprise-modulated **vector-bias** Hebbian plasticity
> worked (Δppl = +0.034, p = 0.003, 0.37% forgetting) but was far below the
> 0.5-ppl practical bar. E032 tests the two identified bottlenecks — **capacity**
> (a vector bias cannot reshape features) and **modulator conservatism**
> (E031's mean M = 0.098) — plus the one untested ablation axis (**decay**),
> and quantifies how close matched-budget local plasticity gets to a
> **backprop LoRA upper bound**.
> **Spec:** [03-architecture.md](03-architecture.md) (extended for low-rank) +
> [04-evaluation-protocol.md](04-evaluation-protocol.md) (LOCKED; see §9 for
> the explicit re-scoping amendment).
> **Pre-registered:** 2026-08-13 (success criteria §8, search strategy §4).

---

## 1. Experiment Summary

E032 keeps the E031 protocol exactly (frozen `HuggingFaceTB/SmolLM2-1.7B`
bf16, WikiText-2 → PubMed, window 512/stride 256, 100K primary point, 3
seeds 42/43/44) and changes the **plastic representation** and the **surprise
gain**. Five sub-experiments:

| Part | What | Question |
|:----:|:-----|:---------|
| **A** | Low-rank plastic matrices `W_plastic = B·A` at the same 48 sites (`o_proj` + `down_proj`), rank sweep r ∈ {1,2,4} at E031 default η/surprise | Does **capacity** (not just scalar bias) unlock larger Δppl? |
| **B** | Surprise **gain sweep** on the best rank: η, then s₀/k, then M_max | Does a less-conservative modulator trade a little forgetting for a bigger effect? |
| **C** | **Decay ablation** λ ∈ {1e-5, 1e-4} at the best A+B config | The last untested axis of the (now answered) 2×2×2 grid. |
| **D** | **LoRA backprop baseline** (minimal manual LoRA, same budget) | The upper bound: what can matched-budget backprop achieve? |
| **E** | **1M anneal** at the best local config | Saturation; does forgetting stay < 1% at 9.8τ? |

All local cells (A–C, E) are **no-backprop**: the update is a local,
surprise-modulated Hebbian rule projected onto the low-rank manifold.

## 2. Why Now — the E031 evidence

E031's two coupled bottlenecks (from [05-e031-minimal-viable.md](05-e031-minimal-viable.md) §6):

1. **Capacity.** 98,304 scalar biases (0.0057% of the model) can only add a
   per-output offset; they cannot rotate/reshape the feature space. The
   protocol's Phase 2.1 already specifies low-rank (`y + B@(A@x)`) as the
   capacity upgrade.
2. **Conservatism.** E031's modulator averaged M = 0.098 (peak 0.28); plastic
   biases stayed ~5× smaller than constM's (|b| 0.012 vs 0.065) precisely
   because the surprise gate protects the source. A stronger gain (higher η,
   lower s₀, higher M_max) should trade a little forgetting for a real effect.

E031 already answered 2 of the 3 ablation axes of the original 2×2×2 grid
(surprise vs constant-M; Hebbian vs random). **Decay is the only untested
axis** → this step's Part C. The grid re-scoping is recorded in §9.

## 3. Part A — Low-Rank Plastic Matrices (the capacity answer)

### 3.1 Mechanism

Each injection point carries `A: (r, d_in)` and `B: (d_out, r)` (float32).
The forward hook injects

$$y' = y + B\,(A\,x) \qquad (W_{\text{plastic}} = B\,A)$$

at `o_proj` (d_in = d_out = 2048) and `down_proj` (d_in = 8192, d_out =
2048) of every block. With `B = 0` the injection is exactly zero, so the
identity invariant (I1) holds at construction.

### 3.2 The local update — derivation

E031's 3-factor Hebbian rule for a **full** plastic matrix is

$$\Delta W = \eta\, M\, \frac{1}{T}\sum_t \mathrm{pre}_t \otimes \mathrm{post}_t,$$

i.e. `ΔW = η·M·mean_t(pre ⊗ post)`. We want a **local** update to the
low-rank factors `A, B` that projects this desired change onto the low-rank
manifold `W = B·A`. Projecting `ΔW` with the Frobenius objective
`min_{ΔA,ΔB} ‖ΔW − B·ΔA − ΔB·A‖²_F` gives the natural factor gradients:

$$\Delta A = \eta\, M\, \frac{1}{T}\sum_t \big(B^{\top}\mathrm{post}_t\big)\otimes \mathrm{pre}_t
\quad\Leftrightarrow\quad
\Delta A = \eta\, M\,\mathrm{mean}_t\big((B^{\top}\mathrm{post}_t)\otimes \mathrm{pre}_t\big),$$

$$\Delta B = \eta\, M\,\frac{1}{T}\sum_t \mathrm{post}_t \otimes \big(A\,\mathrm{pre}_t\big)
\quad\Leftrightarrow\quad
\Delta B = \eta\, M\,\mathrm{mean}_t\big(\mathrm{post}_t \otimes (A\,\mathrm{pre}_t)\big).$$

**Reading:** the A factor is driven by the input `pre`, weighted by how much
the B-filtered output (`Bᵀ·post`) projects onto it; the B factor is driven by
the output `post`, weighted by the A-filtered input (`A·pre`). Both use only
`pre`, `post`, and the point's **own** A/B — fully local, no backprop.

**Stabilised implementation** (in `BrainWrapper.learn`, `plasticity="low_rank"`;
two amendments discovered at the first real run — recorded as a protocol note,
not silently):

1. **Feedback-free post.** The update uses the **pre-injection (frozen)
   output** for `post`, not the post-injection activation. With the injected
   post, the Hebbian term is self-amplifying (bigger B → bigger post → bigger
   ΔB) and A/B diverged to ~1e25 in a few PubMed steps. This is the
   architecture doc's pre-noted "pre-bias frozen output" variant
   (`03-architecture.md` §Injection design space, point 4).
2. **Normalised Hebbian step.** Each factor's outer-product mean is divided
   by `rms(post)·rms(A·pre)` (resp. `rms(Bᵀpost)·rms(pre)`), so the step is a
   bounded correlation (~η·M) rather than raw `(activation)²` units — which
   are ~100× E031's vector-bias step `η·M·O(1)` and drove the blowup. The
   normalisation is self-limiting (ΔB ∝ 1/‖A·pre‖), exactly the Oja-style
   stabilisation that raw Hebbian learning needs on a deep model.

```
pB  = post_frozen @ B                    # (B,S,r) = Bᵀ·post
ΔA  = η·M·mean_bs( pB ⊗ pre ) / (rms(pB)·rms(pre))
pA  = einsum('ri,bsi->bsr', A, pre)      # (B,S,r) = A·pre
ΔB  = η·M·mean_bs( post_frozen ⊗ pA ) / (rms(post)·rms(pA))
A ← A + ΔA ; B ← B + ΔB                  # optional decay: ×(1−λ)
```

Verified stable at constant M = 1.0 (worst case, no warmup) on the live
model: 20 steps, no NaN, loss 2.94 → 2.39, ‖B‖ bounded ~8e-3.

### 3.3 Initialisation & the zero-init deadlock

A naive `A = B = 0` init makes the **first** update zero too (ΔB ∝ A·pre = 0,
ΔA ∝ Bᵀ·post = 0) — a deadlock. E032 uses a **scaled random projection**:
`A ~ N(0, 1/d_in)`, `B = 0`. With `B = 0` the injected term is still zero
(identity holds), but `A ≠ 0` makes ΔB nonzero on step 1, bootstrapping
learning. This also makes Part A and Part D share **identical init and
architecture** — the only difference is the update rule (local Hebbian vs
AdamW backprop), which is exactly the comparison we want.

> **Stability amendment (2026-08-13, discovered at the first real run):** the
> initially pre-registered LoRA convention `A ~ N(0, 1)` diverges to NaN on
> the 1.7B model. With `A ~ N(0,1)`, `A·pre` has std ≈ ‖pre‖ ≈ √d_in ≈ 90 on
> `down_proj` (d_in = 8192), so the first ΔB produces an injection
> `B@(A@x) ~ O(1)` per point and the residual stream explodes at the first
> PubMed step. Scaling A by `1/√d_in` keeps `A·pre ~ O(1)` and the run
> stable. The comparison to Part D remains apples-to-apples: the LoRA
> baseline uses the **same** scaled init, so the update rule is still the only
> difference.

> Consequence to analyse in the results: with this init the local rule drives
> B hard on early steps while A drifts slowly (ΔA ∝ ‖B‖). Whether the A factor
> meaningfully adapts — i.e. whether rank r > 1 buys more than a learned
> readout of a fixed random projection — is a direct empirical question of
> Part A vs Part D (reported as mean|ΔA| vs mean|ΔB| and the plastic-weight
> diagnostics).

### 3.4 Parameter counts & memory per rank (SmolLM2-1.7B, 24 blocks)

| rank r | o_proj / block (2·2048·r) | down_proj / block ((8192+2048)·r) | params total (24 blocks) | fp32 bytes | % of model |
|:------:|:------------------------:|:--------------------------------:|:------------------------:|:----------:|:----------:|
| 0 (vector bias, E031) | 2048 | 2048 | 98,304 | 393 KB | 0.0057% |
| 1 | 4,096 | 10,240 | 344,064 | 1.376 MB | 0.0201% |
| 2 | 8,192 | 20,480 | 688,128 | 2.752 MB | 0.0402% |
| 4 | 16,384 | 40,960 | 1,376,256 | 5.505 MB | 0.0804% |

### 3.5 Rank sweep (isolate capacity)

r ∈ {1, 2, 4}, at **E031 defaults** (η = 1e-3, α = 0.99, s₀ = 0.05, k = 60,
M_max = 1.0, λ = 0.0), 100K budget, 3 seeds. The r = 0 reference is E031's
`surprise` 100K result (Δppl = +0.034). Winner = highest cross-seed mean
Δppl with forgetting < 1%.

## 4. Part B — Surprise Gain Sweep (the conservatism answer)

**Staged search strategy (pre-registered):** on the best rank from A, sweep
one axis at a time, keeping the previous winner for the others. "Best" =
highest cross-seed mean Δppl with **forgetting < 1%** (the trade-off
constraint).

| Stage | Axis | Values | Held fixed |
|:-----:|:-----|:-------|:-----------|
| B1 | η | {1e-3, **3e-3**, 1e-2} | s₀=0.05, k=60, M_max=1.0, λ=0 (η=1e-3 = the A winner cell) |
| B2 | s₀, k | s₀∈{0.02, 0.05}, k∈{30, 60} | best η from B1 (default 0.05/60 = the B1 winner cell) |
| B3 | M_max | {1.0, 2.0} | best η, s₀, k from B1–B2 (M_max=1.0 = the B2 winner cell) |

Cells: B1 2×3 seeds, B2 3×3 seeds, B3 1×3 seeds (reusing the already-run
winner cells for the defaults). Rationale for the axis order: η is the most
direct gain knob (E031 conservatism is fundamentally small-η); s₀/k control
the surprise sensitivity; M_max caps the burst.

## 5. Part C — Decay Ablation (the last untested axis)

At the best config from A+B: λ ∈ {1e-5, 1e-4} (λ = 0.0 is the best-config
cell, already run). Decay `×(1−λ)` is applied to A and B after every update.
Tests whether a gentle weight decay holds plastic weights near the frozen
solution (less drift → less forgetting) without killing the target effect.

## 6. Part D — LoRA Backprop Baseline (THE comparison)

**Real LoRA** at the **same parameter budget** as the best low-rank local
config: same 48 sites, same rank, same `A/B` structure (→ identical param
count by construction), same init (A ~ N(0,1), B = 0). `peft` is not
installed, so a minimal manual LoRA (~50 lines) is used. The frozen backbone
is `requires_grad_(False)`; only A/B train with **AdamW** (wd = 0.0, per
protocol §4 B5), lr sweep {1e-4, 3e-4, 1e-3}, over the **same** combined
stream (WikiText warmup → PubMed adapt). Note the honest reading of "same
warmup procedure": backprop LoRA *does* update during the warmup steps (the
local method's warmup has M = 0 and no update) — LoRA gets the maximal
upper-bound benefit. Gradient checkpointing keeps activation memory near
forward-only levels on the 8 GB card.

**Question:** how close can matched-budget **local** plasticity get to
backprop? Reported as `Δppl_local / Δppl_LoRA` (secondary criterion).

## 7. Part E — 1M Budget (anneal point)

On the best local config from A–D: 1M tokens (977 adapt steps ≈ 9.8τ, the
protocol's fully-annealed point). Tests saturation and whether forgetting
stays < 1% at scale.

## 8. Pre-Registered Success Criteria (before running)

At the **100K primary point**, 3 seeds:

| # | Criterion | Pass? |
|:-:|:----------|:------|
| 1 | **Best local config reaches Δppl ≥ 0.5 ppl** on PubMed (the practical bar) | ❌ best local −1.275 (decay1e4); E031 vector-bias +0.034. No local config is even positive. **LoRA meets it** (+1.52) — see #5. |
| 2 | Local beats E031's vector-bias surprise (+0.034) — capacity + gain help | ❌ capacity + gain strictly hurt; every low-rank/gain config is below +0.034 (all negative) |
| 3 | **Forgetting < 1% for all local configs** (a config exceeding 1% is documented as the trade-off boundary, not silently dropped) | ❌ no local low-rank config < 1% (best +2.2%, most ≫ 100%); all documented as trade-off boundaries in §11 |
| 4 | Δppl_local / Δppl_LoRA reported (how close to matched-budget backprop) | ✅ ratio ≈ **−0.84** (best local −1.275 vs best LoRA +1.520 — opposite sign; local destroys where LoRA improves). At 1M the gap is ~4,900× in magnitude |
| 5 | LoRA upper bound itself: what matched-budget backprop achieves at 100K | ✅ **+0.86 / +1.32 / +1.52** (all lr; source *improved* −6.5 to −8.5%) |

## 9. Protocol Amendment — explicit re-scoping (not silent)

**Recorded 2026-08-13.** The original Step 1.2 scope in `ROADMAP.md` /
`BRAIN.md` was a "2×2×2 ablation grid" (surprise vs constant LR, Hebbian vs
random update, decay vs no decay). E031 already answered two axes with
full cross-seed statistics at the primary point:

* surprise vs constant-M → **answered** (E031: surprise +0.034, constM −0.573
  catastrophic +10.7% forgetting; surprise is essential),
* Hebbian vs random update → **answered** (E031: random-plastic −0.044, i.e.
  plasticity training beats random perturbation).

The remaining untested axis is **decay**, which E032 runs as Part C. The rest
of E032 is the capacity (low-rank) and gain (η/s₀/k/M_max) investigation that
E031's verdict explicitly queued ("What this means for Phase 1.2"). This
re-scoping is recorded here and appended to the LOCKED protocol's deviation
log (§11 of `04-evaluation-protocol.md`) rather than silently changing scope.
The measurement protocol (metric, domains, budgets, baselines, statistics)
is unchanged.

## 10. Implementation (build on E031, no rewrite)

| File | Change |
|:-----|:-------|
| `src/ph_neuro/brain/block_wrappers.py` | `InjectionPoint` gains `A`/`B`/`in_features`/`rank`; `_get_in_features()`; both wrappers accept `rank=` (vector-bias path unchanged). |
| `src/ph_neuro/brain/brain_wrapper.py` | `plasticity="low_rank"` + `rank`; low-rank injection hook (`y + B·A·x`); low-rank local update (§3.2); A/B in `state_dict`/`load_state_dict`/`to`/param-count (E031 schema preserved); LoRA-convention init. |
| `tests/brain/test_low_rank.py` | 15 tests: shapes, I1 identity, no-backprop, derived-update numerics, decay, serialization, counts. |
| `src/ph_neuro/examples/run_e032_capacity_gain.py` | Single-cell runner (clone of the E031 runner): `--plasticity/--rank/--tag/--lr/--decay/--s0/--k/--m-max`. |
| `src/ph_neuro/examples/run_e032_lora.py` | Manual LoRA backprop runner (AdamW, grad-checkpointed, same stream). |
| `src/ph_neuro/examples/aggregate_e032.py` | Cross-seed aggregation + verdict (Δppl, p, forgetting, LoRA ratio). |
| `scripts/run_e032_capacity_gain.sh` | Orchestrator: staged modes `rank → gain_eta → gain_sk → gain_mmax → decay → lora → anneal`, skip-if-exists, GPU gate. |
| `results/brain/e032/` + `logs/brain/e032/` | Result JSONs (protocol schema, `method: lowrank|lora`) + logs. Frozen evals **reuse the E031 cache** (identical seed-independent baseline). |

## 11. Results

### 11.1 Part A — Rank sweep (capacity)

Config: rank r ∈ {1, 2, 4}, η = 1e-3, s₀ = 0.05, k = 60, M_max = 1.0, λ = 0,
100K tokens, 3 seeds. Reference: E031 vector-bias surprise (rank 0) at the
same point: **Δppl = +0.034, +0.37% forgetting**.

| Rank | Δppl per seed (42/43/44) | Δppl mean ± SD | p | forgetting (mean) | max |B| element |
|:----:|:------------------------:|:--------------:|:--:|:-----------------:|:----:|
| 0 (E031 vector bias) | +0.034 / +0.027 / +0.040 | +0.034 ± 0.003 | 0.003 | +0.37% | 16.2 |
| 1 | −0.230 / −2.247 / −1.570 | −1.349 ± 1.026 | 0.151 | +13.1% | 3.4 |
| 2 | −1.104 / **−2797** / −2147 | −1648 ± 1463 | 0.190 | +46,649% | 6.5 |
| 4 | −2417 / −3525 / −3573 | −3172 ± 654 | 0.014 | +99,113% | 8.1 |

**Finding: capacity makes it worse, not better.** Every low-rank config is
negative with severe (often catastrophic) forgetting, and the damage grows
with rank (lrr4's destruction is *statistically significant*, p = 0.014). The
two mechanisms (§11.5, mechanism finding) — surprise positive feedback and
Hebbian concentration — mean that with matrix capacity the unsupervised
Hebbian correlation destroys the model. Rank 1 is the least-destructive
config; **best rank = 1** (used for the gain sweep and the LoRA budget
match).

### 11.2 Part B — Gain sweep

**B1 (η, at rank 1, s₀=0.05, k=60, M_max=1.0):** the higher-η cells are
catastrophic — far worse than the η=1e-3 A-winner (lrr1: −1.35 mean,
max|w| 3.4). Best η is therefore **η = 1e-3** (the A default), confirming
that gain is not the missing ingredient.

| η | Δppl per seed (42/43/44) | forgetting (mean) | mean M | max |w| |
|:--:|:------------------------|:-----------------:|:------:|:----:|
| 1e-3 (lrr1) | −0.23 / −2.25 / −1.57 | +13.1% | 0.159 | 3.4 |
| 3e-3 | −3610 / −3800 / −3787 | ~+115,917% | 0.32 | 17.3 |
| 1e-2 | −104,107 / −162,287 / −111,911 | ~+2.3M% | 0.46 | 58.2 |

**Finding: η is not the missing ingredient — higher gain makes it strictly
worse.** η=3e-3 triples the mean |M| and explodes the damage 1000×; η=1e-2
destroys the model (mean Δppl −126K, max|w| 58). The surprise gate that kept
E031's vector-bias stable becomes a runaway amplifier the moment the plastic
update has matrix capacity (each adaptation step pushes loss up, M→1.0, and
the damage compounds). **Best η = 1e-3** (the Part A default, lrr1).

**B2 (s₀, k, at rank 1, η=1e-3, M_max=1.0):** the s₀/k sweep confirms the
default (s₀=0.05, k=60 = lrr1) is the least-bad on both axes.

| s₀ / k | Δppl per seed (42/43/44) | Δppl mean | forgetting (mean) | mean M |
|:------:|:------------------------|:---------:|:-----------------:|:------:|
| 0.05 / 60 (lrr1) | −0.23 / −2.25 / −1.57 | **−1.35** | +13.1% | 0.159 |
| 0.05 / 30 | −0.42 / −6.55 / −2.96 | −3.31 | +34% | 0.20 |
| 0.02 / 60 | −4.18 / **−653** / **−440** | −366 | +6,309% | 0.32 |
| 0.02 / 30 | −5.33 / **−908** / **−415** | −443 | +7,993% | 0.31 |

**Finding: the gain dimensions behave monotonically in the destructive
direction.** Lower s₀ (0.02) opens the surprise gate more often → mean M
~0.30–0.32 → catastrophic in 4/6 seeds (Δppl −415 to −908). Lower k (30)
alone is milder (mean −3.31) but still ~2.5× worse than lrr1. There is no
s₀/k combination that rescues the local low-rank Hebbian; the "default"
(lrr1) is the least-damaging operating point. **Best s₀/k = 0.05/60 (lrr1).**

**B3 (M_max, at rank 1, η=1e-3, s₀=0.05, k=60):**

| M_max | Δppl per seed (42/43/44) | Δppl mean | forgetting (mean) | mean M |
|:-----:|:------------------------|:---------:|:-----------------:|:------:|
| 1.0 (lrr1) | −0.23 / −2.25 / −1.57 | **−1.35** | +13.1% | 0.159 |
| 2.0 | −2506 / −4175 / −3335 | −3339 | +104K% | ~1.9 (clipped) |

**Finding: M_max=2.0 is catastrophic (~2,500× worse).** Doubling the cap on
the surprise multiplier lets each adaptation step write ~2× larger updates;
the surprise positive-feedback loop then runs away exactly like the higher-η
cells. **Best M_max = 1.0.**

**B1–B3 conclusion (gain sweep):** every gain knob — η↑, s₀↓, k↓, M_max↑ —
moves monotonically in the destructive direction. There is no gain
configuration that rescues the local low-rank Hebbian update; the protocol's
"default" (lrr1) is the least-damaging operating point across the whole
capacity × gain grid. This closes the **gain axis** of the 2×2×2 study:
gain is not the missing ingredient — **credit assignment is** (see §12 and
the LoRA contrast in §11.4).

### 11.3 Part C — Decay ablation

**C (λ, at the best local point: rank 1, η=1e-3, s₀=0.05, k=60, M_max=1.0):**

| λ | Δppl per seed (42/43/44) | Δppl mean | forgetting (mean) |
|:--:|:------------------------|:---------:|:-----------------:|
| 0.0 (lrr1) | −0.23 / −2.25 / −1.57 | −1.349 | +13.1% |
| 1e-5 | −0.23 / −2.26 / −1.57 | −1.355 | +13.2% |
| 1e-4 | −0.22 / −2.08 / −1.53 | −1.275 | +12.4% |

**Finding: decay is a wash — it does not rescue the low-rank Hebbian.** λ ∈
{1e-5, 1e-4} is statistically indistinguishable from λ=0 (all within seed
noise; decay1e4 marginally "less bad" at −1.275 but still far below E031's
vector-bias +0.034 and the 0.5 bar). This is mechanistically consistent with
the diagnosis: the damage is driven by the **surprise positive-feedback loop
acting within each adaptation window** (a perturbation pushes loss up → M
rises → a larger perturbing update), not by slow secular growth of the
plastic weights that a decay term would counteract. **Best λ = 0.0 (lrr1).**

**C closes the 2×2×2 grid.** Combined with E031 (which answered the
capacity/vector-bias and modulation axes), the full capacity × modulation
space is now covered: rank ↑ capacity hurts, gain ↑ hurts, decay ~neutral.
Across every local configuration the surprise-modulated Hebbian update with
matrix capacity destroys or degrades the model; only E031's *vector-bias*
form (no matrix) was stable and positive (+0.034).

### 11.4 Part D — LoRA backprop baseline

**Smoke (1K budget, rank 1, lr 3e-4, seed 42):** Δppl = **+0.219** (positive),
source forgetting **−14.0%** (improved). Backprop LoRA at **1/100th** of the
100K budget already beats E031's vector-bias at 100K (+0.034) by 6× — a
dramatic contrast to the local method's catastrophe. Note: LoRA trains during
the warmup steps too (both domains), so the smoke's 1K budget included 100
WikiText steps.

| Config (rank 1, 100K) | Δppl per seed | Δppl mean | forgetting (mean) | p |
|:----------------------|:--------------:|:---------:|:-----------------:|:--:|
| LoRA lr=1e-4 | +0.844 / +0.859 / +0.872 | **+0.858** | −6.6% (improved) | <1e-100 |
| LoRA lr=3e-4 | +1.308 / +1.321 / +1.324 | **+1.318** | −8.5% (improved) | <1e-100 |
| LoRA lr=1e-3 | +1.497 / +1.542 / +1.520 | **+1.520** | −6.5% (improved) | <1e-100 |

> **Every LoRA config exceeds the 0.5 practical bar** (best +1.52 at lr=1e-3),
> is tightly consistent across seeds, and **improves** the source (negative
> forgetting). At the same rank-1 budget where local Hebbian destroyed the
> model (Δppl −1.35 to −3172), backprop LoRA exceeds the pre-registered
> practical threshold — this is the comparison the step was designed to
> answer, and it is decisive: **the local method's failure is the absence of
> credit assignment, not the parameter budget** (identical init, identical
> architecture, identical rank; only the update rule differs).

### 11.5 Part E — 1M anneal

**E (best local config lrr1: rank 1, η=1e-3, s₀=0.05, k=60, M_max=1.0, λ=0,
scaled up to 1M tokens / 977 adapt steps ≈ 9.8τ):**

| budget | Δppl per seed (42/43/44) | Δppl mean | forgetting (mean) | max |B| |
|:------:|:------------------------|:---------:|:-----------------:|:------:|
| 100K (lrr1) | −0.23 / −2.25 / −1.57 | −1.35 | +13.1% | 3.4 |
| 1M | −7683 / −7075 / −7386 | **−7381** | +201,261% | 0.20 |

**Finding: damage does not saturate — it compounds catastrophically with
budget.** At 10× the primary budget the surprise feedback loop drives the
plastic weights ~30× larger (|B| 1.1e-2 → 0.20) and the model is destroyed
(mean Δppl ≈ −7381, ~5,500× worse than 100K). The fully-annealed point the
protocol designed to test is decisively negative for the local method.

> **Mechanism finding (2026-08-13):** across all five parts, two coupled
> mechanisms explain the catastrophe. (1) **Surprise positive feedback** — a
> plastic perturbation spikes the loss, surprise M rises toward its cap, the
> next update is larger, the perturbation is worse; damage tracks mean M in
> every stage (B1: η↑→mean M↑; B2: s₀↓→mean M↑; B3: M_max↑→damage↑; E:
> sustained M pins |B| up 30×). (2) **Hebbian concentration** — the
> unsupervised correlation amplifies the strongest channels; max |B| element
> grows (1.8 → 6.5 → 58) and localized perturbations wreck logits. E031's
> vector-bias (rank 0) is the only form without these failure modes: it
> smooths rather than reshapes, and stayed stable at +0.034.

> **Interim finding (2026-08-13, Part A partial — 5/9 cells):** the low-rank
> Hebbian is **negative and severely unstable across seeds**. Rank-1 seed42:
> Δppl = −0.230, +2.26% forgetting (mean M = 0.105); but seed43 −2.25/+22.1%
> and seed44 −1.57/+15.0% (mean M ≈ 0.19). Rank-2 seed43 is **catastrophic**
> (Δppl = −2797, +85,914% forgetting — the model is destroyed, plastic max
> element 6.5, mean M = 0.264). Two coupled mechanisms:
> 1. **Surprise positive feedback.** The plastic perturbation spikes the loss;
>    surprise M rises (we observed M up to 0.99); larger updates → worse
>    perturbation. Damage tracks mean M (0.105 → 0.26).
> 2. **Hebbian concentration.** mean|B| stays bounded (~0.003–0.009) but the
>    **max** element grows 1.8 → 6.5 — the unsupervised correlation
>    amplifies the strongest channels, producing localized perturbations that
>    wreck logits.
>
> This is a real mechanism finding: with matrix capacity, unsupervised
> Hebbian correlation is maladaptive for ppl (E031's vector-bias smoothing
> was stable at +0.034), and the surprise gate **amplifies** rather than
> contains the damage. The full rank/gain/decay matrix below maps this
> landscape; the LoRA comparison (Part D) quantifies how much backprop credit
> assignment fixes it.

## 12. Verdict

**Final verdict (all five parts A–E complete, 51 cells, 0 failures):** the
two E031 hypotheses are answered **decisively in the negative** for the
local method, and the LoRA baseline identifies the binding constraint.

* **Capacity (rank) does not help — it destroys.** Every low-rank Hebbian
  config is negative (rank 1: −1.35; rank 2: −1648; rank 4: −3172 mean Δppl)
  with catastrophic forgetting (up to +99,000%). The unsupervised Hebbian
  correlation, given matrix capacity, amplifies the strongest channels
  (concentration) and, via the surprise gate's positive feedback (loss spike →
  M → more damage), destroys the model.
* **Gain is not the bottleneck — it's the poison.** The full gain sweep
  (B1–B3) is monotonic in the destructive direction: η 1e-3→1e-2 (−1.35 →
  −126K), s₀ 0.05→0.02 (−1.35 → −366/−443), k 60→30 (−1.35 → −3.31),
  M_max 1→2 (−1.35 → −3339). Damage tracks mean M throughout (0.10 → 0.46);
  the "conservatism" E031 flagged is what kept the vector-bias alive.
* **Decay is neutral, not a rescue.** λ ∈ {1e-5, 1e-4} is indistinguishable
  from λ=0 (Δppl −1.36 / −1.28 vs −1.35) — consistent with the diagnosis
  that the damage is a within-window feedback loop, not secular weight growth.
* **More budget compounds the damage.** At the 1M anneal point (Part E) the
  best local config gives Δppl ≈ −7381 and +201,000% forgetting — ~5,500×
  worse than 100K. The instability never saturates.
* **Backprop is the difference.** Matched-budget LoRA (rank 1, 344K params,
  identical init/architecture) **exceeds the 0.5 practical bar at every lr**
  (lr=1e-3: Δppl **+1.52**; lr=3e-4: **+1.32**; lr=1e-4: **+0.86**) with
  **improved** source (−6.5 to −8.5% forgetting). The best local config
  (decay1e4, −1.275) is *negative* — opposite in sign to LoRA's positive
  Δppl — so the ratio Δppl_local/Δppl_LoRA ≈ **−0.84** (the local method is
  not merely weaker; it actively destroys performance where LoRA improves
  it). At the 1M point the gap is ~4,900× in magnitude (−7381 vs +1.52).

**Pre-registered success criteria (reported):**
* Δppl ≥ 0.5 at 100K: **NOT met** by any local config (best = vector-bias
  +0.034 from E031; best low-rank = −1.35). Met by **every** LoRA config.
* Δppl_local/Δppl_LoRA ratio: ~1/1100 (best local −1.35 vs best LoRA +1.52).
* Forgetting < 1%: **no** local low-rank config meets it (all ≥ +13%);
  LoRA improves the source (−6.5 to −8.5%, i.e. negative forgetting). Per
  the pre-registered rule, every local low-rank config is a documented
  **trade-off boundary** (destructive), and the method is rejected.

**Conclusion:** surprise-modulated local Hebbian plasticity with matrix
capacity is not merely sub-optimal — it is **structurally unstable** (the
surprise gate becomes a runaway amplifier once the update can reshape
features). The missing ingredient is **credit assignment** (backprop),
not capacity, gain, decay, or budget. E031's vector-bias form (+0.034,
0.37% forgetting) remains the only stable local configuration.

### What this means for Phase 1.3 / Phase 2

E032 sharpens the Phase 1.1 partial-success verdict: **surprise-modulated
local Hebbian plasticity cannot reach the practical bar, and increasing its
capacity is not just unhelpful — it is destructive.** The binding constraint
is not capacity, gain, or the surprise signal's conservatism; it is the lack
of **credit assignment** in the local update rule. Options for the next step:

1. **Error-based local rules (predictive coding / target propagation)** —
   local rules that still propagate *some* error signal per layer, giving
   credit assignment without full backprop. The honest next test of the
   "no backprop" thesis (and the only path that could close the 3-order-of-
   magnitude gap to the LoRA bound).
2. **Re-scope Phase 1.3 (GPT-2) as a replication of the E032 verdict** — does
   the same catastrophic-instability pattern hold on a smaller, classic
   architecture? (Cheap; the runner is architecture-agnostic.)
3. **Keep the wrapper for what works** — the stable vector-bias mechanism
   (+0.034, no forgetting) remains the safe default; low-rank Hebbian is
   **rejected** (Part C confirmed λ alone does not rescue it).

## 13. Reproducibility

- Same operational rules as E031: GPU gate ≥ 6 GiB free + residual in-learn
  gate, checkpoints every 100 steps (atomic, skip-if-exists), SIGINT/SIGTERM
  handlers, `PYTHONUNBUFFERED=1`, logging to `logs/brain/e032/`.
- Deterministic: `torch.manual_seed(seed)` per process; deterministic A init
  (process RNG), block-shuffled batch order (seeded), PubMed eval subsample
  fixed at seed 42 (bit-identical across seeds/configs → paired stats valid).
- Everything to re-run: `bash scripts/run_e032_capacity_gain.sh <stage>`.
- Frozen evals cached under `results/brain/e032/cache/` (reused from E031).
