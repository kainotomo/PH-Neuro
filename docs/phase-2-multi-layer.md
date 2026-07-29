# Phase 2 — Forward Signals & Three-Factor Learning

> **Goal:** Solve the hidden-layer problem. Give every layer a local error signal without backprop.  
> **Duration:** ~3-4 weeks  
> **Hardware:** RTX 4060 8 GB — easy  
> **Success:** Ternary Forward-Forward >95% MNIST, neuromodulated Hebbian >85% MNIST

---

## Why This Phase Exists

Phase 1 conclusively showed that **unsupervised Hebbian cannot build useful deep networks.** The root cause is fundamental:

$$\Delta W_{\text{Hebbian}} = \eta \cdot \text{post}^T \cdot \text{pre} \quad \text{optimizes for} \quad \max \text{corr(pre, post)}$$

This captures **statistical structure** (PCA), not **discriminative structure**. Each hidden layer compounds the problem — correlations of correlations diverge from class-relevant features. The result: 2-layer MLP (87.9%) = 1-layer (88.4%), CNN conv layers (32.6%) = random (33.0%).

**Phase 2 implements the solution:** give every layer its own local objective, using two complementary approaches validated by the literature:

| Approach | Source | Key Idea | Ternary Fit |
|----------|--------|----------|:-----------:|
| **Forward-Forward** | Hinton (2022) | Each layer maximizes "goodness" for real data, minimizes for negative | ✅ Popcount goodness |
| **Three-Factor Hebbian** | Frémaux & Gerstner (2016) | ΔW = η · M · pre · post with modulator M | ✅ M ∈ {-1, 0, +1} |

Both are **local, backprop-free, and ternary-compatible.**

---

## Literature Foundations

### 1. The Forward-Forward Algorithm (Hinton, 2022)

> Hinton, G. "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345, 2022.

**Key insight:** Replace the forward+backward passes of backprop with **two forward passes:**
- **Positive pass:** Real data → each layer increases its "goodness"
- **Negative pass:** Generated/junk data → each layer decreases its "goodness"

**Results:** 1.37% error on MNIST (98.6% accuracy) — competitive with backprop, entirely without it.

**Goodness function:** Sum of squared activations in a layer. For ternary, this maps naturally to **popcount** (number of active neurons).

### 2. Three-Factor Learning Rules (Frémaux & Gerstner, 2016)

> Frémaux, N. & Gerstner, W. "Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules." Frontiers in Neural Circuits, 2016. (659 citations)

**Key insight:** Biological learning is not pure Hebbian — it's **neuromodulated:**

$$\Delta W = \eta \cdot M \cdot \text{pre} \cdot \text{post}$$

Where $M$ is a **third factor** (neuromodulator) that says "this correlation is good/bad/irrelevant":
- $M = +1$: Dopamine-like — reward, strengthen this association
- $M = 0$: No modulation — ignore this correlation
- $M = -1$: Punishment — weaken this association

### 3. Predictive Coding (Whittington & Bogacz, 2017)

> Whittington, J.C.R. & Bogacz, R. "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity." Neural Computation, 2017. (494 citations)

**Key insight:** Local Hebbian updates in a predictive coding framework can **approximate backpropagation** without computing gradients. Each layer predicts its input; prediction error drives local weight updates. This will be the foundation for Phase 3 (Language).

### 4. Neuro-Modulated Hebbian Learning (Tang et al., CVPR 2023)

> Tang, Y. et al. "Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation." CVPR 2023.

**Key insight:** Unsupervised Hebbian + learned neuro-modulator works for test-time adaptation. **Caveat:** Their modulator is trained with backprop — we need a backprop-free version.

---

## Staged Experiment Plan

**Strategy:** Start with the fastest, lowest-risk experiments on MNIST. Only proceed to harder benchmarks (CIFAR-10) after the mechanism is validated. Each stage has a clear go/no-go decision.

### Stage 1: MNIST Sanity Checks (Week 1)

**Goal:** Prove Forward-Forward and neuromodulated Hebbian work at all with ternary weights.

| Experiment | Architecture | Result | Time | Go/No-Go |
|-----------|-------------|:------:|:----:|:--------:|
| TFF-1: Single-layer FF | 784→10 (FF inspired WTA) | ✅ **87.9%** (matches 88.4% WTA) | ~50 s | ✅ Pass — approach validated |
| NTH-1: Label modulator | 784→10 (NTH) | ✅ **88.15%** (peak 88.74%) | ~61 s | ✅ Pass — matches WTA baseline |

**Result:** TFF-1 achieves 87.9% MNIST accuracy, matching the WTA baseline (88.4%) within expected variance. The Forward-Forward-inspired training loop is validated — zero `.backward()` calls, all weights ternary, flip rates <0.05%/step. The junk-suppression negative pass does not help the single output layer (which already has direct class access via WTA correction) but is retained as a critical component for hidden layers in Stage 2.

**Result (NTH-1):** NTH-1 achieves **88.15% MNIST accuracy** (peak 88.74% at epoch 9), matching both the WTA baseline (88.4%) and TFF-1 (87.9%) within expected variance. The label-modulator approach is validated: the unified update `Δ = lr × Mᵀ @ pre` produces identical latent-score deltas to WTA on wrong predictions, confirmed empirically with `torch.allclose(..., atol=1e-6)`. Zero `.backward()` calls, all weights ternary, flip rates 0.04%/step. Training time ~61s on RTX 4060. **Key design insight:** the modulator must only fire on wrong predictions — strengthening correct predictions net-weakens the correct class on MNIST because ~85% of pixels are dark (−1 after ternary quantization). Full documentation in `docs/experiments/E005-nth-mnist-label-modulator.md`.

**Key implementation details:**
- **Approach:** FF-inspired WTA — forward pass on real data → WTA correction (strengthen correct, weaken wrong prediction) + optional junk anti-Hebbian suppression
- **Best hyperparameters:** `lr_pos=0.01`, `lr_neg=0.0`, `θ_u=1.0`, `θ_l=0.3`, `ε=0.1`, 10 epochs
- **Negative pass finding:** Junk suppression (anti-Hebbian on corrupted data) hurts accuracy for the output layer. Recommended: `lr_neg=0.0` for single-layer, `lr_neg>0` for multi-layer hidden layers
- **Training time:** ~50 seconds on RTX 4060 (well under 2-minute target)
- **Documentation:** `docs/experiments/E004-forward-forward-mnist.md`

**Decision:** TFF-1 passes ✅. Proceed to Stage 2 (TFF-2: multi-layer FF). When hidden layers are introduced, the junk-suppression negative pass becomes the key mechanism for creating useful representations without labels.

### Stage 2: MNIST Multi-Layer (Week 2) — THE CRITICAL TEST

**Goal:** Answer the question that Phase 1 couldn't: does depth help when hidden layers have a local error signal?

| Experiment | Architecture | Target | Time | Meaning |
|-----------|-------------|--------|------|---------|
| TFF-2: 2-layer FF | 784→512→10 | **86.81%** — 🔴 Fail | ~1 hr | 🔴 Fail: FF+ternary incompatible for hidden layers |
| | | ≈ Phase 1.1 (87.9%) | | FF negative pass adds no benefit |
| | | No improvement from depth | | Pivot to NTH-4 |
| TFF-3: 3-layer FF | 784→512→256→10 | >96% MNIST | ~2 hrs | Cancelled — mechanism fails |
| NTH-4: Multi-layer NTH | 784→512→10 | **86.68%** — 🔴 Fail | ~1 hr | 🔴 Fail: NTH cannot propagate to hidden layers, 4 approaches all <88% |
| NTH-4b: Latent score feedback | 784→512→10 | **86.68%** — 🔴 Fail | ~1 min | 🔴 Fail: even dense continuous feedback can't train hidden layers |
| TFF-4: FF vs WTA | Same arch | — | — | Omnibus comparison table |

**Decision:** TFF-2 → **🔴 Fail (86.81%).** No improvement over 1-layer (87.9%). FF+ternary is incompatible for hidden layers — the FF contrastive objective (popcount/goodness) trivially saturates without competition, and with top-1 competition the FF negative pass adds no benefit beyond random bootstrapped prototypes. **Proceed to NTH-4 (multi-layer NTH) as the remaining pathway.** TFF-3 (3-layer) and TFF-5 (CNN CIFAR-10) are cancelled.

Documented in `docs/experiments/E006-forward-forward-multilayer-mnist.md`.

### Stage 3: CIFAR-10 (Week 3) — **CANCELLED** (Stage 2 failed)

The CIFAR-10 experiments (TFF-5, TFF-6) are cancelled because the fundamental mechanism (FF hidden layers) does not work on MNIST. There is no reason to expect it to work on the harder CIFAR-10 benchmark.

### NTH-4: Multi-layer NTH on MNIST

**Result: 🔴 Fail (86.68%).** All four modulator propagation approaches fail to improve over the single-layer 88% bound.

| Approach | Accuracy | Hidden Learns? | Failure Mode |
|:--------:|:--------:|:--------------:|-------------|
| A: Label broadcast | **9.80%** | ❌ | Global anti-Hebbian kills all hidden representations |
| B: Weight feedback | **85.79%** | ❌ (~0% flips) | W_out is 92% zero — feedback passes through sparse near-zero weights |
| C: Random feedback | **85.02%** | ✅ (52% dense) | Random projection drives non-discriminative changes |
| D: Latent score (NTH-4b) | **86.68%** | ❌ (~0% flips) | Dense continuous feedback through S_out fails too — sparsity not the bottleneck |

**Root cause:** The hidden-to-output weights are ternary and mostly zero (92% after training in the best case). The feedback signal `M_hidden = M_output @ W_out` passes through sparse near-zero weights, producing negligible hidden update. **NTH-4b (latent score feedback)** bypasses this by using dense continuous latent scores $S_{\text{out}}$ instead of ternary weights $W_{\text{out}}$, but the hidden layer still does not learn (0.000% flips). This proves **sparsity was not the bottleneck** — the Hebbian correlation-based update fundamentally cannot create discriminative features even with a perfect feedback pathway.

**Decision:** NTH-4 **🔴 Fail.** No remaining approach for ternary Hebbian hidden layers. Phase 2 is complete — all approaches exhausted including NTH-4b.

**Phase 3 (Language/Predictive Coding) assessment:** Unlikely to succeed. The same fundamental limitation applies: ternary Hebbian updates cannot propagate discriminative signals through hidden layers, regardless of the training objective. Predictive coding requires continuous error signals at each layer, which face the same hysteresis threshold problem.

Documented in `docs/experiments/E007-nth-multilayer-mnist.md`.

### Why This Staging Matters

```
Week 1: MNIST 1-layer    → 5 min experiments, validate the mechanism
Week 2: MNIST 2-3 layers → 1-2 hr experiments, the make-or-break test
Week 3: CIFAR-10 CNN     → 2 hr experiments, only if MNIST multi-layer succeeded
```

This prevents us from spending weeks on CIFAR-10 if the fundamental mechanism doesn't work on MNIST. Every experiment at each stage reuses existing infrastructure from Phase 0-1 — we're not building from scratch.

---

## 2a. Ternary Forward-Forward (TFF)

### Core Idea

Each layer has a **goodness function** $G(h)$ that measures how "good" the layer's output is:

$$G(h) = \text{popcount}(h) = \sum_i \mathbb{1}[h_i \neq 0]$$

For ternary activations $h \in \{-1, 0, +1\}^d$, goodness is simply the **number of active neurons.** Real data should produce high goodness; negative data should produce low goodness.

### Training Procedure

```python
def forward_forward_step(layer, x_pos, x_neg, lr):
    """One FF step for a single layer.
    
    x_pos: real data batch
    x_neg: negative data batch (generated)
    """
    # Positive pass: increase goodness for real data
    h_pos = ternary_sign(layer(x_pos))
    goodness_pos = h_pos.abs().sum(dim=1)  # popcount
    
    # Hebbian update: strengthen connections that contribute to high goodness
    layer.hebbian_update(x_pos, h_pos, lr=+lr)
    
    # Negative pass: decrease goodness for negative data
    h_neg = ternary_sign(layer(x_neg))
    goodness_neg = h_neg.abs().sum(dim=1)
    
    # Anti-Hebbian update: weaken connections that contribute to negative data
    layer.hebbian_update(x_neg, h_neg, lr=-lr)
    
    layer.refresh_weights()
    
    return goodness_pos.mean(), goodness_neg.mean()
```

### Negative Data Generation

The Forward-Forward algorithm requires negative data. Several strategies:

| Strategy | Description | Ternary-compatible? |
|----------|-------------|:-------------------:|
| **Hybrid images** | Mix two real images (e.g., top half of 3 + bottom half of 7) | ✅ |
| **Pixel shuffle** | Randomly permute pixels | ✅ |
| **Mask + predict** | Mask part of input, predict it | ✅ |
| **Mask + corrupt** | Mask part, fill with random ternary noise | ✅ |
| **Label-conditioned** | Same input, wrong label as context | ✅ |

**Recommended starting point:** Mask 50% of pixels and fill with random {-1, 0, +1} noise. Simple, fast, and clearly distinguishable from real data.

### Layer-wise Training

```python
# Greedy layer-wise Forward-Forward
# Train Layer 1
for epoch in range(epochs):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        x_neg = generate_negative(x_ternary, strategy='mask_corrupt')
        
        g_pos, g_neg = forward_forward_step(layer1, x_ternary, x_neg, lr)

# Freeze Layer 1, train Layer 2 on Layer 1's output
layer1.requires_hebbian_(False)
for epoch in range(epochs):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        with torch.no_grad():
            h1 = ternary_sign(layer1(x_ternary))
        h1_neg = generate_negative(h1, strategy='mask_corrupt')
        
        g_pos, g_neg = forward_forward_step(layer2, h1, h1_neg, lr)

# Output layer: supervised WTA (unchanged from Phase 0)
```

### Experiment Plan

| Experiment | Architecture | Target | Notes |
|-----------|-------------|--------|-------|
| TFF-1: Single layer | 784→10 (FF only) | >85% MNIST | Sanity check: does FF work at all? |
| TFF-2: 2-layer FF | 784→512→10 | >95% MNIST | Compare to Phase 1.1 (87.9%) |
| TFF-3: 3-layer FF | 784→512→256→10 | >96% MNIST | Does depth help with FF? |
| TFF-4: FF vs WTA | Same arch, both methods | — | Direct comparison |
| TFF-5: CNN FF | Conv FF on CIFAR-10 | >45% | Compare to Phase 1.2 (32.6%) |

### Ternary Goodness Variants

The standard FF goodness is $\sum h_i^2$ (sum of squared activations). For ternary $h \in \{-1, 0, +1\}^d$:

| Goodness Function | Formula | Ternary Value | Notes |
|-------------------|---------|:-------------:|-------|
| Squared sum | $\sum h_i^2$ | = popcount(h) | +1²=1, (-1)²=1, 0²=0 |
| Sum | $\sum h_i$ | = pos_count − neg_count | Can be negative! |
| Abs sum | $\sum |h_i|$ | = popcount(h) | Same as squared |
| Thresholded | $\sum \mathbb{1}[h_i \neq 0] > \tau$ | = popcount > τ | Binary good/bad |

For ternary, squared sum = abs sum = popcount. This is **elegant** — goodness is just "how many neurons fired."

---

## 2b. Neuromodulated Ternary Hebbian (NTH)

### Core Idea

Add a third factor $M$ to the Hebbian rule:

$$\Delta W = \eta \cdot M \cdot \text{pre} \cdot \text{post}$$

Where $M \in \{-1, 0, +1\}$ (or a continuous value) modulates whether the correlation pre×post should be strengthened, ignored, or weakened.

### Modulator Sources

| Source | Description | Granularity | Example |
|--------|-------------|:-----------:|---------|
| **Global reward** | Single scalar for the whole network | Per-sample | M = +1 if correct prediction, -1 if wrong |
| **Layer-local error** | Per-layer error signal | Per-layer | M = target_activity − actual_activity |
| **Prediction error** | Difference between predicted and actual next state | Per-neuron | M_i = predicted_i − actual_i |
| **Novelty** | How surprising is this input? | Per-layer | M = 1 − reconstruction_quality |
| **Label** | Ground truth label as modulator | Per-class | M_c = +1 for correct class, 0 otherwise |

### Implementation

```python
class NeuromodulatedHebbianLinear(TernaryHebbianLinear):
    """Ternary Hebbian layer with neuromodulation."""
    
    def neuromodulated_update(self, pre, post, modulator, lr):
        """
        pre: (batch, in_features) — ternary input
        post: (batch, out_features) — ternary output  
        modulator: (batch, out_features) or (batch,) or scalar
                   M ∈ {-1, 0, +1} or continuous
        lr: learning rate
        
        Δscore = lr · modulator · postᵀ @ pre
        """
        if modulator.dim() == 0 or modulator.dim() == 1:
            # Per-sample or global modulator: broadcast to all neurons
            modulated_post = post * modulator.view(-1, 1)
        else:
            # Per-neuron modulator
            modulated_post = post * modulator
        
        delta = lr * (modulated_post.T @ pre)
        self._latent_scores.scores += delta
```

### Experiment Plan

| Experiment | Modulator Type | Target | Notes |
|-----------|:-------------:|--------|-------|
| NTH-1: Label modulator | M_c = +1 for correct class | >85% MNIST | ✅ **88.15%** | Equivalent to WTA — confirmed |
| NTH-2: Error modulator | M = target − output | >85% MNIST | ⬜ Not run | Deprioritized — NTH-4 validates multi-layer |
| NTH-3: Ternary modulator | M ∈ {-1, 0, +1} | >85% MNIST | ⬜ Not run | Deprioritized — NTH-4 covers this |
| NTH-4: Multi-layer NTH | Label→propagated modulator | >90% MNIST | 🔴 **86.68%** | Fail: all 4 approaches <88% single-layer bound (NTH-4b best) |
| NTH-5: CIFAR-10 | NTH CNN | >45% | ⬜ Cancelled | Mechanism fails on MNIST — no reason to proceed |

---

## 2c. Comparison & Integration

### Three-Way Shootout

| Method | Hidden Layer Signal | Backprop-Free? | Ternary-Native? | Expected MNIST |
|--------|:------------------:|:-------------:|:---------------:|:-------------:|
| **WTA (Phase 0-1)** | None (unsupervised Hebbian) | ✅ | ✅ | 88.4% (1-layer) |
| **Forward-Forward** | Local goodness objective | ✅ | ✅ (popcount) | >95% (target) |
| **Neuromodulated Hebbian** | Modulator signal | ✅ | ✅ (M ∈ {-1,0,1}) | >90% (target) |
| Backprop (reference) | Global gradient | ❌ | ❌ (needs STE) | ~98% |

### Integration: FF + NTH

The two approaches can be **combined:**

```python
# Forward-Forward with neuromodulation
# Modulator M = +1 during positive pass, -1 during negative pass
def ff_neuromodulated_step(layer, x_pos, x_neg, lr):
    # Positive pass: M = +1
    h_pos = ternary_sign(layer(x_pos))
    layer.neuromodulated_update(x_pos, h_pos, modulator=+1.0, lr=lr)
    
    # Negative pass: M = -1
    h_neg = ternary_sign(layer(x_neg))
    layer.neuromodulated_update(x_neg, h_neg, modulator=-1.0, lr=lr)
    
    layer.refresh_weights()
```

This is essentially Forward-Forward expressed as neuromodulated Hebbian.

### Decision Matrix

| Criterion | WTA | Forward-Forward | NTH | FF+NTH |
|-----------|:---:|:---------------:|:---:|:------:|
| Simple to implement | ✅ | ⚠️ (needs neg data) | ✅ | ⚠️ |
| Works for hidden layers | ❌ | ✅ (theory) | ⚠️ (untested) | ✅ (theory) |
| Biological plausibility | ⚠️ | ⚠️ | ✅ | ✅ |
| Ternary-compatible | ✅ | ✅ | ✅ | ✅ |
| No negative data needed | ✅ | ❌ | ✅ | ❌ |
| Continual learning potential | ⚠️ (single-head fails) | ✅ (local objectives) | ✅ (modulation per task) | ✅ |

---

## Success Criteria

### Stage 1 Gates (must pass to continue)

| Milestone | Target | Verification |
|-----------|--------|-------------|
| FF single-layer MNIST | >88% (match WTA baseline) | ✅ **87.9%** — `docs/experiments/E004-forward-forward-mnist.md` |
| NTH single-layer MNIST | >85% | ⬜ Not yet run |

### Stage 2 Gates (the critical test) — DEFINITIVELY CLOSED

| Milestone | Target | Result | Verification |
|-----------|--------|:------:|-------------|
| FF 2-layer MNIST | >95% (beat 1-layer by >6pp) | 🔴 86.81% | `docs/experiments/E006-forward-forward-multilayer-mnist.md` |
| FF 3-layer MNIST | >96% | ❌ Cancelled | Mechanism fails on 2-layer |
| NTH multi-layer MNIST | >90% | 🔴 85.79-86.68% | `docs/experiments/E007-nth-multilayer-mnist.md` |
| EP multi-layer MNIST (TEP-1) | >92% | 🔴 **80-84%** | `docs/experiments/E008-equilibrium-propagation-mnist.md` |
| Continual learning with FF | <10% forgetting single-head | ❌ Cancelled | All hidden-layer mechanisms fail |

### Stage 3 (only if Stage 2 passes)

| Milestone | Target | Verification |
|-----------|--------|-------------|
| FF CNN CIFAR-10 | >45% (10+pp over random baseline) | Experiment log |
| FF+NTH CNN CIFAR-10 | >45% | Integration experiment |

---

## Deliverables

- [ ] `TernaryForwardForwardLayer` implementation
- [ ] Negative data generators (mask-corrupt, hybrid, shuffle)
- [ ] `NeuromodulatedHebbianLinear` implementation
- [ ] FF experiment script + results
- [ ] NTH experiment script + results
- [ ] Comparison table: WTA vs FF vs NTH vs FF+NTH
- [ ] Updated tests (50+ new tests)

---

## What's Next

After Phase 2 → **Phase 3: Language & Predictive Coding.** Once we've validated that local error signals (FF goodness / neuromodulator) can drive useful learning in hidden layers, we apply the same principle to sequential data using predictive coding.

---

## 2d. Equilibrium Propagation (TEP-1)

**Status:** 🔴 Failed — Worst accuracy of any 2-layer method (82.57%)

### Core Idea

Equilibrium Propagation (Scellier & Bengio, 2017) contrasts two network states — free (unperturbed forward pass) and nudged (output weakly clamped toward target) — and updates weights by the difference of their Hebbian correlations:

$$\Delta W_{ij} = \frac{\eta}{\beta} \cdot (h_i^{\text{nudged}} \cdot h_j^{\text{nudged}} - h_i^{\text{free}} \cdot h_j^{\text{free}})$$

### Ternary Simplification

Without recurrent dynamics, EP was approximated with 2 forward passes:

1. **Free phase**: Standard forward pass → `h_free`, `y_free`
2. **Nudged phase**: Compute `h_target = ternary_sign(S_out^T @ y_onehot)` — what the hidden layer SHOULD produce for the correct class, derived from output layer **latent scores** (dense fp16, not sparse ternary weights)
3. **Hidden update**: `ΔS_hidden = η × (h_target^T @ x - h_free^T @ x)`
4. **Output update**: Standard WTA (Phase 0)

### Three Variants Tested

| Variant | Accuracy | Hidden Flips | Key Finding |
|:--------|:--------:|:------------:|-------------|
| **Joint EP** (both layers) | **80.79%** | **0.006%** | Moving target: hidden chases evolving S_out |
| **Greedy EP** (frozen output) | **82.57%** | **0.005%** | Stale target: S_out doesn't adapt to new hidden reps |
| **Random prototypes** (fixed) | **81.11%** | **0.068%** | Random targets = random flips |

### Why EP Failed

EP successfully moves hidden weights (0.005%/step — **first non-backprop method to do so**). The h_target correlation reaches 0.78, confirming hidden-to-target alignment. However, this alignment DOES NOT improve classification accuracy because:

1. **Moving target (joint):** h_target = sign(S_out^T @ y_onehot) depends on S_out, which changes as the hidden layer changes. Circular dependency creates unstable dynamics.

2. **Stale target (greedy):** A frozen S_out encodes class structure for the ORIGINAL hidden features. When EP changes hidden weights, the new representations no longer match the old S_out — the frozen output layer loses accuracy.

3. **No backward propagation:** EP requires recurrent dynamics with symmetric weights to propagate the nudge backward. Feedforward ternary networks lack both. The S_out^T projection is a local approximation that provides a noisy, directionally-inconsistent signal.

### Documentation

Full experiment report: `docs/experiments/E008-equilibrium-propagation-mnist.md`

---

## Definitive Phase 2 Conclusion

After **9 experiments** across **4 fundamentally different approaches**, Phase 2 is **definitively closed**:

| Approach | Experiments | Best 2-Layer Accuracy | Verdict |
|:---------|:-----------:|:--------------------:|:--------|
| Unsupervised Hebbian | Phase 1.1, 1.2 | 87.9% | ❌ PCA, not classes |
| Forward-Forward | TFF-1, TFF-2 | 86.81% | ❌ Popcount is trivial |
| Three-factor Hebbian | NTH-1, NTH-4 (A/B/C/D) | 86.68% | ❌ Sparse/diffuse feedback |
| Equilibrium Propagation | **TEP-1** | **82.57%** | ❌ Noisy, unstable targets |

**No method trains ternary Hebbian hidden layers without backpropagation.** The fundamental limitation is the Hebbian update rule itself: correlation maximization (pre × post) cannot create class-discriminative features, regardless of how the signal is modulated or contrasted.
