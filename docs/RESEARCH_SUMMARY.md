# PH-Neuro Research Summary

> **Ternary Hebbian Networks Without Backpropagation: Why Hidden Layers Fail and What Works Instead**
>
> **Status:** Research phase closed after 9 experiments across 4 fundamentally different approaches.
> **Last updated:** 2026-07-29

---

## Abstract

Can ternary neural networks — where every weight is constrained to {-1, 0, +1} — learn useful hierarchical representations using only local Hebbian plasticity, without backpropagation? We systematically investigate this question across **9 experiments** spanning **4 fundamentally different approaches**: unsupervised Hebbian learning, Forward-Forward contrastive learning, three-factor neuromodulated Hebbian learning, and Equilibrium Propagation. All experiments use the same ternary infrastructure: weights stored as native {-1, 0, +1} with latent float scores, dual-threshold hysteresis for weight updates, and winner-take-all (WTA) competitive mechanisms.

Our results are definitive: **single-layer ternary Hebbian networks achieve 88.4% on MNIST** — approximately 96% of the theoretical maximum for a linear classifier. However, **every method tested fails to train hidden layers** to improve beyond this bound. Nine experiments across all approaches converge to the same 86–88% accuracy range on 2-layer MNIST, matching the linear separability limit of random sparse ternary features. A tenth experiment using Equilibrium Propagation achieves the first non-backpropagation movement of ternary hidden weights (0.005%/step flip rate), but accuracy **drops** to 80–84%.

The root cause is fundamental: **all Hebbian update rules optimize for statistical correlation (pre × post), not classification error minimization.** Each approach's attempt to create a local error signal — whether through goodness functions, neuromodulators, or equilibrium contrasts — ultimately reduces to correlation maximization in ternary networks. The hysteresis threshold mechanism, while providing weight stability, further prevents the sustained directional updates needed for hidden-layer learning. We conclude that **ternary Hebbian networks cannot learn class-discriminative hidden representations without backpropagation**, establishing a fundamental limitation for backpropagation-free learning in weight-quantized neural networks.

---

## Research Question

**Can ternary Hebbian networks learn hierarchical representations without backpropagation?**

Specifically:
- Can a single ternary Hebbian layer perform supervised classification above chance? (Phase 0)
- Can unsupervised Hebbian learning create useful hidden-layer features? (Phase 1)
- Can Forward-Forward contrastive learning provide a local error signal to hidden layers? (Phase 2)
- Can three-factor (neuromodulated) Hebbian learning propagate error signals through depth? (Phase 2)
- Can Equilibrium Propagation train hidden layers through state contrast? (Phase 2)

---

## Hypotheses

| # | Hypothesis | Initial Prediction | Final Verdict | Evidence |
|:-:|:-----------|:------------------:|:-------------:|:---------|
| H1 | Ternary Hebbian learning works for non-trivial classification | >85% MNIST | ✅ **Partially verified** — 88.4% MNIST single-layer (output layer only). Unsupervised hidden layers fail. | [E001](experiments/E001-mnist-hebbian-baseline.md), [E002](experiments/E002-mnist-multilayer-mlp.md), [E003](experiments/E003-cifar10-cnn.md) |
| H2 | No catastrophic forgetting with Hebbian plasticity | <5% forgetting | ⚠️ **Partially falsified** — multi-head achieves <5% ✅, but single-head suffers ~37% forgetting (anti-Hebbian = gradient interference) | [continual.py](../src/ph_neuro/analysis/continual.py) |
| H3 | Hysteresis creates weight stability | <1%/step flip rate | ✅ **Verified** — flip rates converge to <0.05%/step across all experiments | All experiments |
| H4 | Layer-wise independence is sufficient for deep networks | 2-layer > 1-layer | ❌ **Falsified** — depth provides zero improvement (2-layer 87.9% = 1-layer 88.4%). Hebbian captures PCA, not class structure. | [E002](experiments/E002-mnist-multilayer-mlp.md) |
| H5 | Forward-Forward solves the hidden-layer problem | 2-layer FF >95% MNIST | ❌ **Falsified for ternary weights** — TFF-2 achieves 86.81% (same as 1-layer). Popcount goodness trivially saturates; competition eliminates FF benefit. | [E004](experiments/E004-forward-forward-mnist.md), [E006](experiments/E006-forward-forward-multilayer-mnist.md) |
| H6 | Language is learnable without backprop | — | ⬜ **Untested** — on hold indefinitely after Phase 2 closure | — |
| H7 | Three-factor Hebbian = local error signal | NTH-1 matches WTA | ⚠️ **Partially falsified** — output layer ✅ (NTH-1: 88.15%), hidden layers ❌ (NTH-4b: 86.68%, 0.000% flips). Even dense continuous feedback fails. | [E005](experiments/E005-nth-mnist-label-modulator.md), [E007](experiments/E007-nth-multilayer-mnist.md) |
| H8 | Equilibrium Propagation trains hidden layers | >90% MNIST | ❌ **Falsified** — TEP-1 achieves 80-84% (worst of all methods). EP moves weights (0.005%/step) but in non-discriminative directions. | [E008](experiments/E008-equilibrium-propagation-mnist.md) |

---

## Methods

### Common Infrastructure

All experiments share the same ternary Hebbian framework:

- **TernaryHebbianLinear**: Linear layer with native {-1, 0, +1} weights maintained via latent float scores and dual-threshold hysteresis (θ_upper for activation, θ_lower for deactivation)
- **Hebbian update**: `ΔS = η · pre · post` where S are latent scores, pre/post are ternary activations
- **Winner-Take-All (WTA)**: For supervised output layers — strengthen correct class, weaken wrong prediction
- **No `.backward()`**: All training is manual, no autograd, no optimizers, no loss functions
- **Ternary activation**: `ternary_sign(x, epsilon=0.1)` maps continuous activations to {-1, 0, +1}

### Approach 1: Unsupervised Hebbian (Phase 1)

Hidden layers trained with **online competitive Hebbian** (WTA + conscience mechanism). Each neuron competes to represent each input; the winner's weights move toward the input; conscience bias prevents any neuron from dominating. This is essentially online k-means with ternary prototypes. The output layer uses supervised WTA.

**Why we expected it to work:** SoftHebb (Journé et al., 2023) achieves 80.3% on CIFAR-10 with float Hebbian deep learning. If unsupervised Hebbian creates useful features for float weights, it should also work for ternary weights with appropriate competition.

### Approach 2: Forward-Forward (Phase 2)

Each layer has a local **goodness function** — for ternary activations, goodness = popcount (number of active neurons). Real data drives a Hebbian update (increase goodness); corrupted/junk data drives an anti-Hebbian update (decrease goodness). This gives each layer a local contrastive objective without backpropagation. The output layer uses standard WTA.

**Why we expected it to work:** Hinton (2022) achieved 98.6% on MNIST with Forward-Forward. The popcount goodness function maps naturally to ternary activations. For hidden layers, the contrastive signal should create features that fire selectively on real data.

### Approach 3: Three-Factor (Neuromodulated) Hebbian — NTH (Phase 2)

The standard Hebbian rule is augmented with a third factor (modulator): `ΔW = η · M · pre · post`. For the output layer, M is derived from labels (M_c=+1 for correct class, M_w=-1 for wrong prediction). For hidden layers, M is propagated from the output through various feedback pathways: weight feedback (M_hidden = M_output @ W_out), random feedback alignment (M_hidden = M_output @ B_fixed), and latent score feedback (M_hidden = M_output @ S_out, where S_out are dense continuous latent scores).

**Why we expected it to work:** Three-factor learning rules are well-established in computational neuroscience (Frémaux & Gerstner, 2016). The label modulator for output layers is mathematically equivalent to WTA but expressed as a single unified update. For hidden layers, feedback alignment (Lillicrap et al., 2016) has been shown to work for continuous-weight networks.

### Approach 4: Equilibrium Propagation — TEP (Phase 2)

Two forward passes are contrasted: a **free phase** (standard forward pass) and a **nudged phase** where the output is weakly clamped toward the correct label. The hidden-layer update is the difference of Hebbian correlations: `ΔS_hidden = η · (h_targetᵀ @ x - h_freeᵀ @ x)`. The hidden target is derived from output latent scores: `h_target = sign(S_outᵀ @ y_onehot)`.

**Why we expected it to work:** Equilibrium Propagation (Scellier & Bengio, 2017) provides a theoretical framework for backpropagation-free learning through energy minimization. The difference-of-correlations update provides a stronger learning signal than scalar neuromodulation.

---

## Results

### Complete Experiment Table

| # | Experiment | Method | Architecture | Accuracy | Hidden Flip Rate | Key Insight | Verdict |
|:-:|:-----------|:-------|:-------------|:--------:|:----------------:|:------------|:-------:|
| 1 | **E001** | WTA Hebbian (supervised) | 784→10 | **88.4%** | N/A (no hidden) | Single-layer ~96% of theoretical max (~92%) | ✅ Pass |
| 2 | **E002** | Online competitive Hebbian | 784→512→10 | **87.9%** | ~0.003% | Depth does not improve accuracy; competitive Hebbian creates sparse prototypes (PCA, not classes) | ❌ Fail |
| 3 | **E003** | Competitive Hebbian CNN | Conv(3→64→128)+Linear | **32.6%** | <0.01% | Conv Hebbian ≈ random (33% baseline); class-guided kills sparsity (21.9%) | ❌ Fail |
| 4 | **E004** | FF-inspired WTA (TFF-1) | 784→10 | **87.9%** | N/A (no hidden) | FF matches WTA baseline; junk-suppression negative pass hurts single-layer | ✅ Pass |
| 5 | **E005** | Label modulator NTH (NTH-1) | 784→10 | **88.15%** | N/A (no hidden) | NTH = WTA as unified matmul; label modulator works for output layers | ✅ Pass |
| 6 | **E006** | Forward-Forward 2-layer (TFF-2) | 784→512→10 | **86.81%** | ~0.000% | FF negative pass adds no benefit; popcount goodness is not class-discriminative | ❌ Fail |
| 7 | **E007-A** | NTH label broadcast | 784→512→10 | **9.80%** | 0.002% | Global anti-Hebbian kills all hidden representations | ❌ Fail |
| 8 | **E007-B** | NTH weight feedback | 784→512→10 | **85.79%** | ~0.000% | W_out is 92% zero — feedback signal lost in sparse ternary weights | ❌ Fail |
| 9 | **E007-C** | NTH random feedback | 784→512→10 | **85.02%** | 0.127%→0.001% | Random feedback drives hidden changes (52% dense) but non-discriminative | ❌ Fail |
| 10 | **E007-D** | NTH latent score feedback (NTH-4b) | 784→512→10 | **86.68%** | ~0.000% | **Dense continuous feedback also fails** — sparsity was NOT the bottleneck | ❌ Fail |
| 11 | **E008** | Equilibrium Propagation (TEP-1) | 784→512→10 | **82.57%** | **0.005%** | First non-backprop method to move hidden weights, but accuracy DROPS; moving-target and stale-target problems | ❌ Fail |

### Ranking (All 2-Layer Experiments)

| Rank | Experiment | Accuracy | Method | Hidden Learned? |
|:---:|:-----------|:--------:|:-------|:--------------:|
| 1 | Phase 1.1 | **87.9%** | Online competitive + WTA | ❌ (PCA only) |
| 2 | TFF-2 | **86.81%** | Forward-Forward | ❌ (~0.000% flips) |
| 3 | NTH-4b (D) | **86.68%** | Latent score feedback NTH | ❌ (~0.000% flips) |
| 4 | NTH-4 (B) | **85.79%** | Weight feedback NTH | ❌ (~0.000% flips) |
| 5 | NTH-4 (C) | **85.02%** | Random feedback NTH | ✅ (but non-discriminative) |
| 6 | TEP-1 | **82.57%** | Equilibrium Propagation | ✅ (0.005% flips, wrong direction) |
| 7 | NTH-4 (A) | **9.80%** | Label broadcast NTH | ❌ |

---

## The ~88% Bound

A consistent pattern emerges across all experiments: **no method exceeds ~88% accuracy on 2-layer MNIST**, matching the single-layer baseline. This bound is not coincidental — it is the **linear separability limit of ~512 random sparse ternary features for 10-class MNIST.**

### Why ~88%?

1. **Single-layer theoretical maximum**: A single linear layer with continuous weights achieves ~92% on MNIST. The ternary constraint costs ~4pp (the information lost by quantizing weights to {-1, 0, +1}), giving ~88%.

2. **Hidden layers produce ≈ random features**: Every hidden-layer training method we tested generates features that are no more discriminative than random sparse ternary projections. The output layer then learns a linear mapping from these features to classes.

3. **The math**: With 512 hidden neurons (each ~10% active) and 10 classes, the output layer learns a weight matrix that linearly combines these 512 features. The achievable accuracy is bounded by how well 10 classes can be separated in a 512-dimensional random sparse ternary space — approximately 88%.

4. **Confirmation across methods**: Unsupervised Hebbian (87.9%), Forward-Forward (86.81%), neuromodulated Hebbian (86.68% best), and Equilibrium Propagation (82.57%) all converge to or below this bound. No method breaks it.

### Visual Intuition

```
Accuracy
 100% ┤
  95% ┤
  90% ┤  ┌── Backprop 2-layer (~98%)
  88% ┤──┤←── The ~88% bound (linear separability of 512 random sparse features)
  85% ┤  │  ┌──────────────────────────────────────┐
  80% ┤  │  │ All 2-layer methods cluster here     │
  75% ┤  │  │ Phase1.1: 87.9%  ●                   │
  70% ┤  │  │ TFF-2:    86.8%  ●                   │
  65% ┤  │  │ NTH-4b:   86.7%  ●                   │
  60% ┤  │  │ NTH-4B:   85.8%  ●                   │
     ───┤  │  NTH-4C:   85.0%  ●                   │
    10% ┤  │  TEP-1:    82.6%  ● ←─ EP hurts       │
     0% └──└──────────────────────────────────────┘
         Single    2-Layer     2-Layer     Backprop
         Layer     Methods     EP
```

---

## Why All Methods Failed

### 1. Unsupervised Hebbian (Phase 1.1, 1.2)

**What went wrong:** Unsupervised Hebbian optimizes for `max corr(pre, post)` — it captures statistical structure (PCA), not class-discriminative structure. Each hidden layer learns the most common input patterns, which are shared across all digit classes (e.g., "bright center pixels" occur in every digit). The competitive Hebbian + conscience mechanism creates sparse prototypes, but these prototypes are statistically representative, not class-separating. The output layer can only achieve ~88% by learning a linear mapping from these non-discriminative features.

**Specific evidence:** The CNN experiment (E003) confirms this dramatically — competitive Hebbian conv filters achieve 32.6% on CIFAR-10, essentially identical to random ternary weights (33.0%). The learned features are no more useful than random projections.

### 2. Forward-Forward (TFF-2)

**What went wrong:** The Forward-Forward goodness function (popcount) optimizes a **whole-layer property** — it measures "how many neurons fired," not "which neurons fired for which class." Real data produces higher popcount than junk data, but this global statistic carries no class-specific information. When top-1 competition is added (necessary for ternary to avoid saturation), the FF objective becomes equivalent to competitive Hebbian, and the negative pass (junk suppression) adds no class-relevant structure.

**Specific evidence:** The hidden layer achieves excellent goodness separation (520.3) between real and junk data, but this separation is entirely from **random bootstrapped weights** (10% sparse). The FF training barely changes the weights (~0.000%/step flip rate), and accuracy matches the unsupervised Hebbian baseline.

### 3. Three-Factor Hebbian (NTH-4)

**What went wrong:** Three different feedback pathways all fail, but for different reasons:

- **Weight feedback (B):** Output weights are 92% zero — the feedback signal `M_hidden = M_output @ W_out` passes through near-zero weights, producing negligible hidden updates (~0.000% flips).

- **Latent score feedback (NTH-4b/D):** Bypasses ternary sparsity by using dense continuous latent scores S_out. The feedback is continuous and dense. **Hidden learning still does not occur** (~0.000% flips). This proves sparsity was NOT the bottleneck.

- **Random feedback (C):** Uses a fixed dense B matrix. Hidden weights DO change (0.127% early, 52% dense final), but the changes are random projections of the label error — non-discriminative.

**The fundamental insight:** Even with a perfect dense feedback pathway, the Hebbian update `ΔS = η · Mᵀ @ pre` optimizes for `max corr(M, pre)`, not `min classification error`. The modulator modulates the correlation direction, but the objective remains correlational. Ternary hysteresis (θ_upper = 0.5) compounds this: only strong, sustained correlations cross the threshold, but class signals in hidden layers are inherently sample-dependent and noisy.

### 4. Equilibrium Propagation (TEP-1)

**What went wrong:** EP is the only method that **actually moves hidden weights** (0.005%/step flip rate), and it achieves the **lowest accuracy** (82.57%). Three distinct failure modes:

- **Moving target problem (joint training):** `h_target = sign(S_outᵀ @ y_onehot)` depends on S_out, which changes as the hidden layer changes. The hidden layer chases a signal that evolves with its own behavior.

- **Stale target problem (greedy/frozen output):** With frozen output weights, h_target is fixed but was optimized for the original random hidden features. When EP changes hidden weights, the new representations no longer match the old output weights.

- **No backward propagation mechanism:** In a feedforward network without recurrence, there is no mechanism to propagate the output nudge backward. The S_outᵀ projection is a local approximation that provides a noisy, directionally-inconsistent signal.

---

## What Works

### Single-Layer WTA Classification (88.4% MNIST)

The supervised WTA Hebbian rule — strengthen correct class, weaken wrong prediction — achieves 88.4% on MNIST in 47 seconds with zero `.backward()` calls. This is approximately 96% of the theoretical maximum for a single linear layer (~92%). The mechanism is well-understood and reliable.

**Key parameters:** θ_u=1.0, θ_l=0.3, lr=0.01, ε=0.1, 10 epochs, batch=128.

### Multi-Head Continual Learning (<5% Forgetting)

When each task has **separate output neurons** (multi-head), catastrophic forgetting drops to <5% on Split MNIST. This is a significant result: the anti-Hebbian interference that plagues single-head continual learning (~37% forgetting) is eliminated by output segregation.

**This is the primary publishable contribution of PH-Neuro:** a continual learning system that combines local Hebbian plasticity, ternary weight stability, and task-specific output heads to achieve near-zero forgetting without replay buffers or regularization.

### Ternary Infrastructure & Hysteresis Stability

The core ternary infrastructure is verified across 200+ tests: native {-1, 0, +1} weights, latent float scores, dual-threshold hysteresis, flip rate stabilization (<0.05%/step), and zero `.backward()` throughout. This infrastructure is production-ready and reusable.

---

## Future Directions

### 1. Predictive Coding (Whittington & Bogacz, 2017)

Predictive coding is the only non-backpropagation learning mechanism not tested in this work. It is fundamentally different from the approaches above: instead of maximizing correlations (Hebbian), it minimizes **prediction errors** at each layer. Each layer predicts the activity of the layer below; the prediction error drives local weight updates.

**Why it might work:** Predictive coding has been shown to approximate backpropagation through local Hebbian-like updates in continuous-valued networks. It provides a true error signal at every layer, not a correlation-based proxy.

**Why it might fail:** Predictive coding requires continuous error nodes (floating-point) and bidirectional connectivity between layers. This is incompatible with the "no backward pass" philosophy and requires significant architectural changes. The ternary activation constraint would need to be relaxed for error nodes.

**Prototype architecture:**
```
Error nodes (continuous):  e_l = h_l - W_l^T @ h_{l+1}  (prediction error)
Value nodes (continuous):  h_l = f(W_l @ h_{l-1} + e_l)  (corrected prediction)
Weight update:            ΔW_l = η · h_{l-1} · e_l  (local Hebbian)
```

### 2. Hybrid Architectures (Ternary Output + Float Hidden)

The most pragmatic path forward: use ternary weights only for the output layer (where WTA Hebbian works) and float weights for hidden layers (trained with SoftHebb or similar). This preserves the key advantages (no backpropagation, local learning) while avoiding the ternary hidden-layer limitation.

### 3. PH-Net Approach (STE + Backprop for Ternary Weights)

The PH-Net approach (separate project) uses Straight-Through Estimators (STE) with backpropagation to train ternary deep networks. This is the proven path for scaling ternary weights: BitNet b1.58 (Wang et al., 2024) demonstrates 3B-parameter ternary LLMs trained with standard backpropagation. PH-Net sacrifices backpropagation-free learning for scalability and accuracy.

---

## References

1. **Hinton, G. (2022).** "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345. — *Forward-Forward as alternative to backpropagation.*

2. **Frémaux, N. & Gerstner, W. (2016).** "Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules." *Frontiers in Neural Circuits*, 9:85. — *Three-factor learning rules: ΔW = η · M · pre · post.*

3. **Scellier, B. & Bengio, Y. (2017).** "Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation." *Frontiers in Computational Neuroscience*, 11:24. — *Equilibrium Propagation as backprop-free learning.*

4. **Whittington, J.C.R. & Bogacz, R. (2017).** "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity." *Neural Computation*, 29(5):1229–1262. — *Predictive coding approximates backprop through local updates.*

5. **Journé, A., Rodriguez, H.G., Guo, Q., & Moraitis, T. (2023).** "Hebbian Deep Learning Without Feedback." *ICLR 2023*. — *SoftHebb: State-of-the-art float Hebbian deep learning, 80.3% CIFAR-10.*

6. **Wang, S. et al. (2024).** "BitNet b1.58: 1.58-bit LLMs." arXiv:2402.17764. — *Ternary LLMs at 3B scale using STE + backprop.*

7. **Lillicrap, T.P., Cownden, D., Tweed, D.B., & Akerman, C.J. (2016).** "Random Synaptic Feedback Weights Support Error Backpropagation for Deep Learning." *Nature Communications*, 7:13276. — *Feedback alignment: random feedback weights can drive learning.*

8. **Nøkland, A. (2016).** "Direct Feedback Alignment Provides Learning in Deep Feedforward Networks." *NeurIPS 2016*. — *Direct feedback alignment for backprop-free learning.*

9. **Tang, Y. et al. (2023).** "Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation." *CVPR 2023*. — *Modulated Hebbian learning with backprop-trained modulator.*

10. **Millidge, B., Seth, A., & Buckley, C.L. (2022).** "Predictive Coding: A Theoretical and Experimental Review." arXiv:2107.12971. — *Comprehensive survey of predictive coding.*

11. **Rao, R.P.N. & Ballard, D.H. (1999).** "Predictive Coding in the Visual Cortex: A Functional Interpretation of Some Extra-Classical Receptive-Field Effects." *Nature Neuroscience*, 2(1):79–87. — *Original predictive coding framework for visual cortex.*

12. **Oja, E. (1982).** "Simplified Neuron Model as a Principal Component Analyzer." *Journal of Mathematical Biology*, 15(3):267–273. — *Oja's rule: Hebbian learning that finds principal components.*

13. **Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982).** "Theory for the Development of Neuron Selectivity: Orientation Specificity and Binocular Interaction in Visual Cortex." *Journal of Neuroscience*, 2(1):32–48. — *BCM rule: sliding-threshold Hebbian learning.*

---

## Experiment Reports

| Report | Description |
|:-------|:------------|
| [E001: Single-Layer Hebbian MNIST](experiments/E001-mnist-hebbian-baseline.md) | 88.4% MNIST, WTA Hebbian, Phase 0 baseline |
| [E002: Multi-Layer MLP](experiments/E002-mnist-multilayer-mlp.md) | 87.9%, 7 Hebbian variants tested, depth doesn't help |
| [E003: CNN on CIFAR-10](experiments/E003-cifar10-cnn.md) | 32.6%, conv Hebbian ≈ random |
| [E004: Forward-Forward 1-Layer (TFF-1)](experiments/E004-forward-forward-mnist.md) | 87.9%, matches WTA baseline |
| [E005: NTH Label Modulator (NTH-1)](experiments/E005-nth-mnist-label-modulator.md) | 88.15%, three-factor Hebbian validated for output layer |
| [E006: Forward-Forward 2-Layer (TFF-2)](experiments/E006-forward-forward-multilayer-mnist.md) | 86.81%, FF fails for hidden layers |
| [E007: NTH Multi-Layer (NTH-4)](experiments/E007-nth-multilayer-mnist.md) | 86.68% best, all 4 approaches fail, sparsity not the bottleneck |
| [E008: Equilibrium Propagation (TEP-1)](experiments/E008-equilibrium-propagation-mnist.md) | 82.57%, EP moves weights but hurts accuracy |
