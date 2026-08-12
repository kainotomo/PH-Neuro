# Step 0.3 — Surprise & Neuromodulation Signal Design

> **Status:** ✅ Complete (2026-08-12)
> **Goal:** Define the "surprise" signal that modulates plasticity. Determine what information tells the brain "learn now" and how to compute it efficiently.
> **Decision:** Sequence-level relative prediction error (normalized loss deviation from EMA) → sigmoid modulation → one global float32 scalar M per update, broadcast to all layers.
> **Constrained by (Step 0.2):** Δb = η · M · mean(post) at `down_proj` + `o_proj` per block; M is a global scalar; no backprop through frozen layers; compute M in float32 (bf16 underflows M ≈ 10⁻³ → 0).

---

## The Problem

In 3-factor Hebbian learning, ΔW = η · M · pre ⊗ post, the modulator M determines whether and how strongly learning occurs. A constant M (M=1 always) would be equivalent to plain Hebbian learning — which we already know is not task-discriminative.

We need M to reflect something task-relevant: when the model encounters unexpected, surprising, or important information, M should be high (learn more). When the model encounters predictable, routine information, M should be low (don't waste plasticity capacity).

---

## Context Recap (from Steps 0.1 & 0.2)

| Item | Value |
|:-----|:------|
| Primary model | SmolLM2-1.7B — 24 LLaMA-modern blocks, d_model 2048, SwiGLU, RoPE, RMSNorm, vocab 49,152 |
| Gen-test model | GPT-2 124M — 12 classic pre-norm blocks, d_model 768, GELU, LayerNorm, vocab 50,257 |
| Selected mechanism | 3-factor Hebbian with global surprise modulator: Δb = η · M · mean(post), vector bias injected at `down_proj` + `o_proj` per block |
| Hard constraints | (1) M is a **global scalar** broadcast to all layers; (2) M computable from the model's output with **no backprop** through frozen layers; (3) M computed in **float32** (bf16 underflows M ≈ 10⁻³ → 0) |

The design task is therefore narrow: **given only the model's token-level output distribution, produce one float32 scalar M per update step that is high when the model is surprised and low when it is not** — and do it robustly on two models with very different absolute loss scales.

---

## Candidate Surprise Formulations

Six families were surveyed. Each is evaluated against the specific constraint set: (a) computable from the output distribution alone, (b) no backprop, (c) usable on **both** SmolLM2-1.7B and GPT-2 124M (whose absolute loss scales differ), and (d) survives the *mild* domain-shift case (SmolLM2 on PubMed — trained on diverse web/educational data, so raw loss elevation is modest).

| # | Formulation | Output-only? | No-backprop? | Model-agnostic? | Phase 1.1 verdict |
|:-:|:------------|:------------:|:------------:|:---------------:|:------------------|
| 1 | Prediction error (loss dev. from EMA) | ✅ | ✅ | ✅ (if normalized) | **PRIMARY** |
| 2 | Bayesian surprise (KL over θ) | ❌ | ❌ | — | Rejected |
| 3 | Information content (−log P per token) | ✅ | ✅ | ✅ | Raw source (aggregates into #1), not the modulator |
| 4 | Novelty (distance from memory) | ⚠️ needs memory | ✅ | ✅ | Phase 2 per-layer variant |
| 5 | Uncertainty (output entropy) | ✅ | ✅ | ✅ | Phase 1.2 ablation variant |
| 6 | Gradient-norm (‖∇L‖) | ❌ needs backward | ❌ | — | Rejected |

### 1. Prediction Error (Loss Deviation from EMA)

M = max(0, L_current — L_expected)

Where L_expected is an exponential moving average of recent loss values:
L_expected ← α · L_expected + (1—α) · L_current, α ≈ 0.99

**Interpretation:** "I'm doing worse than usual right now — something unexpected is happening. Pay attention and learn."

**Biological analog:** Norepinephrine (noradrenaline) — signals unexpected uncertainty, increases plasticity and attention.

**Pros:** Dead simple to compute. Already available (we compute loss anyway). Clear biological grounding.

**Cons:** One global scalar for the entire model. Different layers might experience different levels of "surprise." A sentence with unfamiliar vocabulary might surprise early layers (unusual token patterns) but not late layers (the sentence structure is normal).

**Variants:**
- Per-layer EMA of layer-specific loss or reconstruction error
- Token-level surprise: M = -log P(token | context) for each token position
- Sequence-level surprise: M = mean(-log P) over the sequence

**Evaluation for our models (why this is the primary):**
- **Token → global aggregation is natural**: both models expose per-token −log P at the logits; the sequence mean of that quantity IS the cross-entropy loss L. Token-level surprise therefore aggregates into exactly the scalar we need, with no extra machinery.
- **In-domain behavior**: L is stationary, so L − L̂ fluctuates around zero; ReLU-style surprise is ≈ 0 → minimal plasticity, protecting source performance.
- **Out-of-domain behavior**: L jumps above L̂ and stays there for roughly τ = 1/(1−α) steps (α=0.99 → τ≈100 steps ≈ 25k tokens at seq 256), giving a **sustained** M ≫ 0 — a plasticity window, not a spike. As L̂ catches up, M anneals: automatic learning-rate decay tied to progress.
- **EMA stabilizes fast enough**: the EMA converges within a few τ; the persistent-surprise window during a domain shift is exactly the temporal profile we want (aggressive early, gentle later).
- **Critical refinement for our two models**: GPT-2 124M (ppl≈29 → loss≈3.4) and SmolLM2-1.7B (much lower loss) have different absolute loss scales. The raw deviation L − L̂ is not comparable across them → the surprise must be **normalized** (relative to L̂, or z-scored) so one formulation transfers to both. See Modulation Function Design.

---

### 2. Bayesian Surprise

M = KL( P(θ | D_new) || P(θ | D_old) )

The KL divergence between the posterior over parameters after seeing new data and the prior (based on old data).

**Interpretation:** "How much does this new data change my beliefs?"

**Biological analog:** Acetylcholine — signals expected uncertainty, enhances attention to informative stimuli.

**Pros:** Principled information-theoretic measure. Distinguishes surprising-but-irrelevant from surprising-and-informative.

**Cons:** Requires maintaining a distribution over parameters — computationally expensive. Approximation needed for practical use.

**Evaluation for our models:** requires a posterior over parameters P(θ|D); maintaining/updating it needs gradients (or per-layer inverses) through the backbone — violating the no-backprop constraint. Even cheap approximations (Laplace / Fisher-diagonal) need Hessian information we refuse to compute. **Rejected for Phase 1.** It remains the principled upper bound for what a "belief-change" signal would look like.

---

### 3. Information Content (Negative Log Probability)

M = -log P(token | context)

The model's own estimate of how unlikely the observed token is.

**Interpretation:** "I didn't see that coming." Rare or unusual tokens get high surprise.

**Biological analog:** Mismatch negativity (MMN) — an EEG component elicited by unexpected stimuli.

**Pros:** Already computed by the model during inference (it's part of the loss). Token-level granularity.

**Cons:** Might be noisy for individual tokens. High surprise on a single rare word might not warrant widespread plasticity.

**Variants:**
- Smoothed over a window of tokens
- Thresholded: only trigger learning if surprise exceeds some percentile

**Evaluation for our models:** both tokenizers (SmolLM2 vocab 49,152 / GPT-2 50,257, both BPE) make per-token −log P available for free during the forward pass. But raw per-token surprise is **noisy and rarity-dominated**: a single novel biomedical subword spikes −log P even on in-domain text, and hard passages raise it across the board. Per-token values are the *source signal*; the meaningful aggregate for a global scalar is the sequence mean (= the loss, feeding Formulation #1). Kept as the raw building block, not as the modulator itself.

---

### 4. Novelty (Distance from Training Distribution)

M = distance( h_current, h_nearest_neighbor_in_memory )

Where h is the hidden representation and memory stores recent examples.

**Interpretation:** "I've never seen anything like this before."

**Biological analog:** Dopamine — novelty detection in the hippocampus and ventral tegmental area.

**Pros:** Distinguishes unfamiliar content from familiar-but-surprising content. Good for domain adaptation (new domain = high novelty → high plasticity).

**Cons:** Requires maintaining a memory of recent examples. Computational cost of nearest-neighbor search. Memory grows over time.

**Evaluation for our models:** the one no-backprop route to a **per-layer** surprise: capture each layer's output on a reference (in-domain) corpus once, then measure drift to the target stream's layer outputs. This is the natural upgrade if the global signal proves insufficient (see Granularity Decision). Cost: an extra reference forward pass + a persistent signature store; at vector-bias granularity the distance computation is trivial. **Deferred to Phase 2** — Phase 1.1 keeps the loss-based global signal to stay minimal.

---

### 5. Uncertainty (Entropy of Output Distribution)

M = H( P(token | context) ) = -Σ P(token) · log P(token)

**Interpretation:** "I'm not sure what comes next." High uncertainty = model is confused = should learn.

**Biological analog:** Acetylcholine — signals uncertainty, modulates learning rate.

**Pros:** Already available from the output distribution. Captures model confusion, not just input statistics.

**Cons:** High uncertainty might indicate inherently unpredictable content (not something learnable). Distinguishing epistemic uncertainty (model doesn't know) from aleatoric uncertainty (task is inherently random) is hard.

**Evaluation for our models:** entropy H = −Σ P·log P is free from the same logits. It captures "model is confused" rather than "input differs from training" — complementary to #1. Caveat: inherently high-entropy positions (punctuation, enumeration, near-uniform next-token distributions) raise H regardless of domain, mixing aleatoric with epistemic uncertainty; on a strong LM the signal tends to be flat and weak. Kept as a **Phase 1.2 ablation variant** to test whether "confusion" adds anything over "deviation from expected loss."

---

### 6. Gradient-Norm Surprise

M = ||∇L||  (magnitude of the gradient of the loss with respect to plastic weights)

**Interpretation:** "How much would the model change if it COULD backprop?" This uses the gradient as a proxy for learnability.

**Biological analog:** Not directly biological, but related to "how much learning potential does this example have?"

**Pros:** Directly measures learnability. High gradient norm = high potential for improvement.

**Cons:** Requires computing gradients through the plastic weights (but NOT through frozen layers). This is a local computation (only plastic weights, not the whole model). Actually, this IS a form of backprop, just restricted to plastic weights. Might be too close to standard training.

**Evaluation for our models:** even "restricted to plastic weights," ∂L/∂W for a hidden projection is not locally available — it requires backprop through the frozen backbone to reach the layer. **Rejected on constraint grounds** (violates the no-backprop rule of Step 0.2), not on principle.

---

## Modulation Function Design

Given a raw surprise value s, how do we map it to a learning-rate multiplier M(s)?

### Options:

1. **Linear:** M = s (clip to [0, max_M])
   - Simplest. But might cause instability for extreme surprise values.

2. **Sigmoid:** M = max_M / (1 + exp(-k · (s — s_0)))
   - Smooth threshold. Parameters k and s_0 control the transition steepness and midpoint.
   - Biologically plausible (neural activation functions are sigmoidal).

3. **Step:** M = max_M if s > threshold, else 0
   - Simplest biologically: learn only when surprise exceeds threshold.
   - But: loses gradient of information.

4. **Power law:** M = s^p (p < 1 for sub-linear, p > 1 for super-linear)
   - Sub-linear: diminishing returns for extreme surprise (biologically plausible — extreme stress impairs learning).

5. **Normalized:** M = (s — μ_s) / σ_s (z-score)
   - Always relative to recent experience. Adapts to changing baseline.

### Evaluation against our two models

| Option | Bounded? | Mild-surprise sensitivity (SmolLM2 on PubMed)? | Transfers across loss scales? | Verdict |
|:-------|:--------:|:-----------------------------------------------:|:-----------------------------:|:--------|
| Linear + clip | ⚠️ needs clip | Low (small s → tiny M) | ❌ raw s not comparable | Rejected as primary |
| **Sigmoid** | ✅ [0, M_max] | ✅ tunable midpoint s₀ | ✅ if s is normalized | **PRIMARY** |
| Step | ✅ | ❌ kills signal below threshold | ⚠️ | Rejected |
| Power law | ⚠️ unbounded | ⚠️ | ❌ raw s not comparable | Rejected as primary |
| z-score + sigmoid | ✅ | ✅ self-calibrating | ✅ | Phase 1.2 variant |

### Recommendation: two stages — normalize, then saturate

**Stage 1 (normalize, float32):** s_t = (L_t − L̂_t) / L̂_t — the *relative* deviation of the current sequence loss from its EMA. Dimensionless and comparable across SmolLM2-1.7B and GPT-2 124M despite different absolute loss scales. (Ablation variant: z-score s_t = (L_t − L̂_t)/σ̂_t — self-calibrating to the noise floor, at the cost of an EMA-of-variance state.)

**Stage 2 (modulate):** M_t = M_max / (1 + exp(−k·(s_t − s₀))).

Justification tied to the models:
- **Bounded [0, M_max]** — SmolLM2-1.7B occasionally hits wildly-off passages (OOV-heavy, malformed); a bounded map stops one bad sequence from triggering an outsized plasticity burst. M_max = 1.0 caps plasticity at the nominal Hebbian rate.
- **Soft threshold at s₀** — hard thresholding (option 3) is precisely wrong for the mild-shift case (SmolLM2 on PubMed): surprise there is modest, and a threshold would zero it out. The sigmoid keeps small-but-nonzero M in the mild regime while saturating on strong shifts (e.g., prose→code).
- **Model-transferable** — with normalized s, one (k, s₀) serves both models; the shape depends on the *relative* size of the domain shift, not the absolute loss.
- **Biologically plausible** — neuromodulator release is a saturating, sigmoidal function of deviation from baseline (norepinephrine / NMDA-gated plasticity), as the existing doc notes.

**Starting defaults (Phase 1.1, to tune):** M_max = 1.0, s₀ = 0.05, k = 60. s=0 → M ≈ 0.018 (≈ no learning); s=s₀ → M=0.5; s=2·s₀ → M≈0.95 (saturated). In-domain fluctuation of ±5% around L̂ therefore sits at/below the midpoint and yields near-zero M — protecting source performance; a ≥10% deviation (domain shift) saturates M.

**Precision (from Step 0.2):** compute L, L̂, s, and M in **float32** end-to-end; bf16 would quantize M ≈ 10⁻³ to 0 (the flagged underflow). Only the broadcast M scalar crosses into the update; plastic weights and Hebbian accumulation are float32 in Phase 1.1.

---

## Granularity Decision: Global vs Per-Layer vs Per-Token

**Decision: one global float32 scalar M per update step (per sequence/batch), broadcast to all 24 layers.** This is what Step 0.2 specified; the analysis below confirms it is also the right scientific choice, and explains why per-layer behavior emerges for free.

### Why not per-token
Per-token surprise is dominated by **token rarity**, not domain shift — a single novel subword spikes −log P on in-domain text too. A per-token M would trigger one-token plasticity bursts that are mostly noise, and it fights the update rule: Δb = η·M·mean(post) already averages post over positions, so per-token M would need a per-position multiply (mean_t(M_t·post_t)) for no clear gain. The thing we are adapting to (a domain, a register, a corpus) is a sequence/block-level phenomenon.

### Why not per-layer
1. **No-backprop constraint**: a per-layer M derived from the output requires per-layer credit assignment (which layer is responsible for the surprise?) — exactly the machinery that failed in NTH-4b. The one no-backprop route (per-layer reference-signature distance, Formulation #4) adds a reference forward pass + signature store and is a Phase 2 refinement, not a Phase 1.1 primitive.
2. **The frozen-backbone hypothesis says global is enough**: Step 0.2's core claim is that M only needs to say "learn now"; the credit ("which features") is already encoded in the structured activations. A global M is the cleanest test of that hypothesis; per-layer M would smuggle credit assignment back in and confound the result.
3. **Activation-weighting approximates per-layer behavior for free**: Δb = η·M·mean(post) scales with each layer's own post-activation magnitude. Late layers (task-specific content) have large active post-activations in the new domain → larger updates; early layers (domain-invariant syntax) → smaller updates — even though M is identical. Effective plasticity is already layer-weighted by the data.

### When to revisit
If Phase 1.1 shows too much plasticity churn in early layers, or insufficient adaptation in late layers, the per-layer variant (layer-signature distance, Formulation #4) becomes the Phase 2 knob. Pre-registered here so the ablation path is explicit.

---

## Biological Mapping

| Signal | Brain Analog | Function | Effect on Plasticity |
|:-------|:------------|:---------|:---------------------|
| High prediction error | Norepinephrine | Unexpected uncertainty | **Increase plasticity** (attend to surprising events) |
| High uncertainty | Acetylcholine | Expected uncertainty | **Increase plasticity** (learn when uncertain) |
| Reward better than expected | Dopamine (phasic) | Reward prediction error | **Strengthen** recent patterns |
| Reward worse than expected | Dopamine (dip) | Negative prediction error | **Weaken** recent patterns |
| Extreme stress | Cortisol | Threat | **Impair plasticity** (freeze, don't learn from trauma) |
| Familiar / predictable | Low norepinephrine | Routine | **Low plasticity** (conserve energy, don't overwrite) |
| Sequence loss dev. from EMA (our M) | Norepinephrine | Unexpected uncertainty | **Saturating increase of plasticity** (sigmoid) |
| Sleep / consolidation | Acetylcholine (low) + replay | Memory transfer | **Structural consolidation** |

---

## Final Recommendation

**Signal:** sequence-mean cross-entropy loss deviation from its own EMA, normalized as a **relative deviation** — the aggregate of per-token −log P (inherits both models' natural per-token surprise, smoothed over the sequence).

**Granularity:** one global float32 scalar M per update step, broadcast to all 24 layers. Per-layer weighting emerges from activation magnitudes in Δb = η·M·mean(post); no per-layer signal is computed (would need credit assignment).

**Modulation:** sigmoid on normalized surprise.

**Design sketch (not implementation):**

    L_t  = mean over sequence of −log P(token_t | context)      # from logits, float32
    L̂_t  ← α·L̂_{t−1} + (1−α)·L_t                                # EMA, α≈0.99, float32
    s_t  = (L_t − L̂_t) / L̂_t                                    # relative deviation
    M_t  = M_max / (1 + exp(−k·(s_t − s₀)))                     # sigmoid, float32
    Δb_l = η · M_t · mean_t(post_l)                             # per layer l ∈ {down_proj, o_proj}

**Why this supersedes the preliminary ReLU + linear choice:** the preliminary used raw absolute deviation with a linear clip — fine for one model on a strong shift, but (1) absolute deviation is not comparable between GPT-2 124M (loss≈3.4) and SmolLM2-1.7B (much lower), so it would need per-model re-tuning; (2) linear mapping under-weights the mild-shift regime that is the actual SmolLM2-on-PubMed case; (3) no natural bound. Normalization + sigmoid fixes all three while staying a single output-only scalar.

**Expected temporal behavior (to verify in Phase 1.1):**
- **In-domain:** s ≈ 0 (fluctuations around the midpoint) → M ≈ 0 → minimal plasticity → source performance protected.
- **Mild OOD (PubMed on SmolLM2):** s small but positive and sustained for ~τ steps → small but usable M → gentle adaptation; M anneals as L̂ catches up, preventing over-learning.
- **Strong OOD:** s large → M saturates → aggressive adaptation, then automatic anneal.
- **The signal is RELATIVE:** even if SmolLM2 handles PubMed well (low absolute loss), the deviation still marks the domain boundary. A small surprise means a small — but present — adaptation signal; that is the harder, more interesting regime for the modulator's sensitivity, and precisely why the sigmoid (not a threshold) was chosen.

**Variants to test in ablation (Phase 1.2):**
- Uncertainty-based M (output entropy, Formulation #5)
- z-score normalization instead of relative deviation
- Sigmoid shape sweep (k, s₀) and EMA time constant α
- Per-layer reference-signature distance M (Formulation #4) — Phase 2

---

## Next Steps (Phase 1.1 hooks — this step's investigation is complete)

- [x] Surprise formulation: normalized loss-deviation-from-EMA ✅
- [x] Granularity: global scalar, float32 ✅
- [x] Modulation function: sigmoid on relative deviation ✅
- [ ] Implement loss EMA tracking (float32) + surprise → M in the Brain Wrapper forward loop
- [ ] Visualize M on in-domain vs. out-of-domain text to confirm expected behavior (in-domain ≈ 0, OOD > s₀) before training anything
- [ ] Tune (α, s₀, k, M_max): start α=0.99, s₀=0.05, k=60, M_max=1.0
- [ ] Design the full modulator API (Brain Wrapper, Step 0.4)
