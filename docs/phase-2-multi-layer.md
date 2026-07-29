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
| NTH-1: Label modulator | 784→10 (NTH) | ⬜ Not yet run | ~5 min | Must ≥ WTA baseline |

**Result:** TFF-1 achieves 87.9% MNIST accuracy, matching the WTA baseline (88.4%) within expected variance. The Forward-Forward-inspired training loop is validated — zero `.backward()` calls, all weights ternary, flip rates <0.05%/step. The junk-suppression negative pass does not help the single output layer (which already has direct class access via WTA correction) but is retained as a critical component for hidden layers in Stage 2.

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
| TFF-2: 2-layer FF | 784→512→10 | **>95% MNIST** | ~1 hr | 🟢 Great: FF+ternary works in depth |
| | | ~90-95% | | 🟡 OK: depth helps but ternary limits |
| | | ~88-90% | | 🔴 Fail: no improvement over 1-layer |
| TFF-3: 3-layer FF | 784→512→256→10 | >96% MNIST | ~2 hrs | Confirms depth scaling |
| NTH-4: Multi-layer NTH | 784→512→10 | >90% MNIST | ~1 hr | NTH alternative validation |
| TFF-4: FF vs WTA | Same arch | — | — | Direct comparison table |

**Decision:** If TFF-2 >95% → **major success**, proceed to CIFAR-10. If TFF-2 90-95% → moderate success, investigate ternary bottleneck. If TFF-2 <90% → FF+ternary may be incompatible, pivot to NTH-only or rethink.

### Stage 3: CIFAR-10 (Week 3) — ONLY IF STAGE 2 SUCCEEDS

**Goal:** Test whether Forward-Forward generalizes beyond MNIST to real vision tasks.

| Experiment | Architecture | Target | Time | Meaning |
|-----------|-------------|--------|------|---------|
| TFF-5: CNN FF | Conv(3→64,3×3)→Conv(64→128,3×3)→FC(128→10) | **>45% CIFAR-10** | ~2 hrs | 🟢 10+pp over random/Phase 1.2 (32.6%) |
| TFF-6: CNN FF+NTH | Same arch, combined | >45% | ~2 hrs | Integration test |

**Note on expectations:** Hinton's original FF paper does NOT achieve strong standalone CIFAR-10 results with pure FF — it was used primarily as unsupervised pre-training. With ternary weights, even a 10pp improvement over the Phase 1.2 random baseline (32.6%) would be a significant result. Target is deliberately modest. If TFF-5 fails (<35%), accept that CIFAR-10 is beyond ternary FF's current capabilities and proceed to Phase 3 (Language) where the natural prediction error signal is stronger.

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
| NTH-1: Label modulator | M_c = +1 for correct class | >85% MNIST | Equivalent to WTA? |
| NTH-2: Error modulator | M = target − output | >85% MNIST | Continuous M |
| NTH-3: Ternary modulator | M ∈ {-1, 0, +1} | >85% MNIST | Simplest form |
| NTH-4: Multi-layer NTH | Label→propagated modulator | >90% MNIST | Can M propagate through layers? |
| NTH-5: NTH vs WTA vs FF | Same arch | — | Three-way comparison |

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

### Stage 2 Gates (the critical test)

| Milestone | Target | Verification |
|-----------|--------|-------------|
| FF 2-layer MNIST | >95% (beat 1-layer by >6pp) | Experiment log |
| FF 3-layer MNIST | >96% (depth scaling confirmed) | Experiment log |
| NTH multi-layer MNIST | >90% | Experiment log |
| Continual learning with FF | <10% forgetting single-head | Split MNIST |

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
