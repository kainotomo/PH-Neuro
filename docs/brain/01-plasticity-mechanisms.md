# Step 0.2 — Plasticity Mechanism Survey

> **Status:** ⬜ Not Started
> **Goal:** Identify the most promising local learning rule for frozen-backbone adaptation. Understand why each candidate might work or fail.

---

## The Constraint

The learning rule must operate with **only locally available information** at each layer:

1. **Pre-activation** — the input to this layer (from the previous layer or embedding)
2. **Post-activation** — the output of this layer
3. **A global signal** — some scalar or vector that modulates learning (surprise, reward, error)

No gradient flow through frozen layers. No backpropagation. No computation of ∂L/∂W via chain rule.

---

## Candidate Mechanisms

### 1. Plain Hebbian: ΔW = η · pre ⊗ post

**What it is:** Strengthen connections where pre- and post-synaptic neurons fire together. The simplest possible learning rule.

**History in PH-Neuro:** Tested extensively (E001–E003). Single-layer supervised WTA achieves 88.4% MNIST. Hidden layers fail — produces PCA-like features, not class-discriminative. But that was training FROM SCRATCH.

**On frozen backbone:** The frozen backbone already computes useful features. Hebbian updates on the plastic weights might reinforce patterns that are useful for the current domain. The question is whether "fire together, wire together" on top of structured representations produces adaptation or just noise.

**Computational cost:** One outer product per layer per batch. O(d²) where d is the hidden dimension. For a typical small transformer (d=768), this is 768² ≈ 590K operations — negligible.

**Key risk:** Without an error signal, Hebbian updates are purely correlational. They strengthen co-occurring patterns regardless of whether those patterns help the task. This is exactly why hidden-layer Hebbian failed before.

---

### 2. Oja's Rule: ΔW = η · (pre ⊗ post — α · post² · W)

**What it is:** Hebbian learning with weight decay normalization. Converges to the first principal component of the input.

**History in PH-Neuro:** Not directly tested. Similar in spirit to Hebbian + weight decay.

**On frozen backbone:** Oja's rule extracts dominant statistical patterns. On a frozen backbone, it would reinforce the principal directions of variation in the new domain. This might help if the domain shift is primarily a statistical shift (different word frequencies, different topics) but not if it requires new semantic knowledge.

**Key risk:** Same fundamental issue as plain Hebbian — no task-specific error signal.

---

### 3. Three-Factor (Neuromodulated) Hebbian: ΔW = η · M · pre ⊗ post

**What it is:** A modulator M (scalar or vector) gates whether the pre×post correlation is strengthened (M>0), weakened (M<0), or ignored (M=0).

**History in PH-Neuro:** Tested as NTH-1 through NTH-4b (E005, E007). Label-modulated NTH works for single-layer classification (88.15% MNIST — equivalent to WTA). For hidden layers, all feedback pathways failed: weight feedback (92% sparse ternary — signal lost), latent score feedback (dense but still no learning), random feedback (moves weights but non-discriminative).

**On frozen backbone:** The situation may be different with a frozen backbone. The pre-trained features might provide enough structure that a simple scalar modulator (surprise, reward) can drive useful adaptation. Rather than propagating M through sparse ternary weights (which failed before), we compute M globally (from the output loss or prediction error) and broadcast it to all layers.

**Key advantage:** This is biologically well-grounded (dopamine = reward prediction error, acetylcholine = expected uncertainty, norepinephrine = unexpected uncertainty). The modulator represents a global brain state that affects plasticity everywhere.

**Key risk:** Global scalar modulator may be too coarse — different layers may need different modulation. A layer-specific modulator (derived from local prediction error at each layer) might be needed.

**Computational cost:** Same as Hebbian plus one scalar multiply. Negligible.

---

### 4. Predictive Coding: ΔW = η · ε · pre, where ε = post — prediction

**What it is:** Each layer tries to predict the activity of the layer below. The prediction error ε drives weight updates. This is Rao & Ballard (1999), Friston's free energy principle.

**History in PH-Neuro:** Not directly tested. Related to Equilibrium Propagation (E008) which failed for ternary weights.

**On frozen backbone:** The frozen backbone already generates predictions (it's a language model — every token is a prediction). The prediction error is naturally available at the output (loss). For hidden layers, we need a local prediction: what does this layer "expect" the next layer to do? This requires a separate prediction network per layer, which adds parameters and complexity.

**Key advantage:** Biologically well-grounded (the brain as a prediction machine). Prediction errors naturally provide layer-local learning signals.

**Key risk:** Adds complexity (prediction networks). Might be overkill for a first experiment. The local prediction error might be too noisy to drive useful learning.

---

### 5. Forward-Forward: ΔW = η · (goodness(pos) — goodness(neg))

**What it is:** Two forward passes: one with real data (increase goodness), one with corrupted data (decrease goodness). Goodness = some layer-local function (popcount, sum of squares).

**History in PH-Neuro:** Tested as TFF-1 and TFF-2 (E004, E006). Single-layer works (87.9% MNIST). Two-layer fails (86.81%) — goodness function (popcount) carries no class-specific information.

**On frozen backbone:** The frozen backbone might produce different goodness patterns for in-domain vs. out-of-domain data. But the fundamental issue remains: goodness is a scalar per layer — it cannot tell which specific weights to change or in which direction.

**Key risk:** Same fundamental failure mode as before — goodness is not class/discriminative. Unlikely to work better on frozen backbone.

---

### 6. Target Propagation: Layer N computes a target for Layer N-1

**What it is:** Each layer has a "target activation" — what it should output. The layer above computes a target for the layer below. Weights are updated to move actual outputs toward targets.

**History in PH-Neuro:** Not tested. Requires training an inverse mapping (how to go from layer N's desired output to layer N-1's desired output).

**On frozen backbone:** The frozen layers already provide a forward mapping. The inverse mapping could be learned or approximated with random feedback. Difference Target Propagation (DTP, Lee et al. 2015) is more practical.

**Key risk:** Adds significant complexity (inverse networks per layer). High computational cost. Not suitable for a first experiment.

---

### 7. STDP (Spike-Timing-Dependent Plasticity)

**What it is:** Weight change depends on the relative timing of pre- and post-synaptic spikes. Pre-before-post → LTP (strengthen). Post-before-pre → LTD (weaken).

**On frozen backbone:** Requires converting the model to a spiking network, which is a major architectural change. Not suitable for wrapping arbitrary pre-trained models.

**Key risk:** Spiking conversion is its own research problem. Orthogonal to our goal.

---

## Ranking (for first experiment)

| Rank | Mechanism | Why | Viability | Complexity |
|:----:|:----------|:----|:---------:|:----------:|
| **1** | **3-factor Hebbian** (surprise-modulated) | Simplest, biologically grounded, already have NTH code, surprise provides task-relevant signal | 🟢 High | Low |
| 2 | Predictive Coding | Biologically grounded, local error signals, but requires prediction networks | 🟡 Medium | High |
| 3 | Plain Hebbian + decay | Simplest possible. Baseline for ablation. Expected to produce some adaptation (statistical), not task-specific | 🟡 Medium | Minimal |
| 4 | Oja's Rule | Hebbian with built-in stability. Good baseline | 🟡 Medium | Minimal |
| 5 | Forward-Forward | Already proven to fail for ternary weights. Unlikely to benefit from frozen backbone | 🔴 Low | Medium |
| 6 | Target Propagation | Too complex for first experiment. Maybe later | 🔴 Low | High |

---

## Recommendation (Preliminary)

**Primary:** Three-factor Hebbian with surprise as modulator. Start with scalar global surprise (from output loss deviation). Test layer-specific surprise in ablation.

**Baselines:** Plain Hebbian (no modulator) and random updates (no Hebbian) — to confirm that both the Hebbian direction AND the surprise modulation are necessary.

**Later exploration:** Predictive coding if 3-factor Hebbian shows promise but insufficient adaptation magnitude.

---

## Next Steps

- [ ] Formalize the 3-factor Hebbian update equation for transformer blocks
- [ ] Determine how the modulator M interacts with multi-head attention outputs
- [ ] Review NTH-1 and NTH-4 code in `src/ph_neuro/training/neuromodulated.py` for reusable components
- [ ] Design the local update that doesn't require backprop through frozen weights
