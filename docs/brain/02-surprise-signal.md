# Step 0.3 — Surprise & Neuromodulation Signal Design

> **Status:** ⬜ Not Started
> **Goal:** Define the "surprise" signal that modulates plasticity. Determine what information tells the brain "learn now" and how to compute it efficiently.

---

## The Problem

In 3-factor Hebbian learning, ΔW = η · M · pre ⊗ post, the modulator M determines whether and how strongly learning occurs. A constant M (M=1 always) would be equivalent to plain Hebbian learning — which we already know is not task-discriminative.

We need M to reflect something task-relevant: when the model encounters unexpected, surprising, or important information, M should be high (learn more). When the model encounters predictable, routine information, M should be low (don't waste plasticity capacity).

---

## Candidate Surprise Formulations

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

---

### 2. Bayesian Surprise

M = KL( P(θ | D_new) || P(θ | D_old) )

The KL divergence between the posterior over parameters after seeing new data and the prior (based on old data).

**Interpretation:** "How much does this new data change my beliefs?"

**Biological analog:** Acetylcholine — signals expected uncertainty, enhances attention to informative stimuli.

**Pros:** Principled information-theoretic measure. Distinguishes surprising-but-irrelevant from surprising-and-informative.

**Cons:** Requires maintaining a distribution over parameters — computationally expensive. Approximation needed for practical use.

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

---

### 4. Novelty (Distance from Training Distribution)

M = distance( h_current, h_nearest_neighbor_in_memory )

Where h is the hidden representation and memory stores recent examples.

**Interpretation:** "I've never seen anything like this before."

**Biological analog:** Dopamine — novelty detection in the hippocampus and ventral tegmental area.

**Pros:** Distinguishes unfamiliar content from familiar-but-surprising content. Good for domain adaptation (new domain = high novelty → high plasticity).

**Cons:** Requires maintaining a memory of recent examples. Computational cost of nearest-neighbor search. Memory grows over time.

---

### 5. Uncertainty (Entropy of Output Distribution)

M = H( P(token | context) ) = -Σ P(token) · log P(token)

**Interpretation:** "I'm not sure what comes next." High uncertainty = model is confused = should learn.

**Biological analog:** Acetylcholine — signals uncertainty, modulates learning rate.

**Pros:** Already available from the output distribution. Captures model confusion, not just input statistics.

**Cons:** High uncertainty might indicate inherently unpredictable content (not something learnable). Distinguishing epistemic uncertainty (model doesn't know) from aleatoric uncertainty (task is inherently random) is hard.

---

### 6. Gradient-Norm Surprise

M = ||∇L||  (magnitude of the gradient of the loss with respect to plastic weights)

**Interpretation:** "How much would the model change if it COULD backprop?" This uses the gradient as a proxy for learnability.

**Biological analog:** Not directly biological, but related to "how much learning potential does this example have?"

**Pros:** Directly measures learnability. High gradient norm = high potential for improvement.

**Cons:** Requires computing gradients through the plastic weights (but NOT through frozen layers). This is a local computation (only plastic weights, not the whole model). Actually, this IS a form of backprop, just restricted to plastic weights. Might be too close to standard training.

---

## Modulation Function Design

Given a raw surprise value s, how do we map it to a learning rate multiplier M(s)?

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
| Sleep / consolidation | Acetylcholine (low) + replay | Memory transfer | **Structural consolidation** |

---

## Recommendation (Preliminary)

**Primary (for Phase 1.1):** Prediction error (loss deviation from EMA), scalar, global. M = ReLU(L — L_ema). Linear modulation with clipping.

This is the simplest defensible choice: it requires only the output loss (already computed), has a clear biological analog (norepinephrine), and provides a scalar that should correlate with domain shift (unfamiliar domain → higher loss → higher surprise → more plasticity).

**Variants to test in ablation (Phase 1.2):**
- Token-level vs. sequence-level surprise
- Per-layer surprise (layer-specific EMA of reconstruction error)
- Uncertainty-based (entropy of output distribution)
- Sigmoid vs. linear modulation function

---

## Next Steps

- [ ] Implement loss EMA tracking
- [ ] Visualize surprise signal on in-domain vs. out-of-domain text to confirm expected behavior
- [ ] Determine modulation hyperparameters (EMA α, max learning rate, clipping)
- [ ] Design the full modulator API
