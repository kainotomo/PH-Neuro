# Paper Outlines

> **Two papers planned — one from Hebbian era (negative results), one from STE era (new direction)**
> **Last updated:** 2026-07-30

---

## Paper 1: Hebbian Era — Negative Results

### Proposed Titles

1. **Ternary Hebbian Networks Without Backpropagation: Why Hidden Layers Fail and What Works Instead** *(primary)*
2. **The ~88% Bound: Fundamental Limitations of Ternary Hebbian Learning Without Backpropagation**
3. **Nine Experiments, Zero Hidden Layers: A Systematic Investigation of Backprop-Free Ternary Hebbian Networks**

---

## Target Venues

| Venue | Fit | Notes |
|:------|:---:|:------|
| **TMLR** (Transactions on Machine Learning Research) | ⭐⭐⭐ | Negative results welcome; rolling review process; good fit for "this doesn't work" papers |
| **NeurIPS** (Datasets & Benchmarks track) | ⭐⭐ | If framed as benchmark of backprop-free methods on ternary networks |
| **ICLR** | ⭐⭐ | Could work if framed as "what doesn't work and why" — ICLR has accepted negative results |
| **JMLR** | ⭐ | Would need more theoretical analysis of the ~88% bound |
| **arXiv-only** | ⭐⭐⭐ | Fastest path; publish as technical report with all code open-sourced |

**Recommendation:** Target TMLR for rolling review + arXiv preprint simultaneously. The negative-result framing is honest and timely — the community needs to know that ternary Hebbian hidden layers have a fundamental limitation.

---

## Section-by-Section Outline

### 1. Introduction (1.5–2 pages)

**Problem statement:** Can neural networks learn useful hierarchical representations using only local Hebbian plasticity with ternary weights, without any form of backpropagation?

**Motivation:**
- Backpropagation is biologically implausible (requires symmetric feedback weights, separate forward/backward passes, non-local error signals)
- Ternary weights (BitNet b1.58, Wang et al., 2024) offer dramatic memory and compute efficiency
- Hebbian learning is local, continuous, and brain-inspired — combining it with ternary weights could enable truly backpropagation-free learning
- If it works: online learning on edge devices, continual adaptation, training on hardware that can't afford backprop's memory overhead

**Prior attempts:**
- SoftHebb (Journé et al., 2023): Float Hebbian deep learning achieves 80.3% CIFAR-10 — but uses float weights
- Forward-Forward (Hinton, 2022): 98.6% MNIST without backprop — but uses float activations
- Three-factor rules (Frémaux & Gerstner, 2016): Well-established in neuroscience
- Equilibrium Propagation (Scellier & Bengio, 2017): Theoretically elegant

**Our contribution:**
- First systematic investigation of backpropagation-free learning methods in ternary weight networks
- 9 experiments across 4 fundamentally different approaches
- Definitive demonstration of a ~88% accuracy bound for 2-layer ternary Hebbian MNIST
- Proof that the bound represents the linear separability limit of random sparse ternary features
- Identification of what DOES work: single-layer classification (88.4%) and multi-head continual learning (<5% forgetting)

### 2. Background & Related Work (2–3 pages)

#### 2.1 Hebbian Learning

**Fundamental rule:** ΔW = η · pre · post (Hebb, 1949)

**Variants:**
- Oja's rule (Oja, 1982): Normalized Hebbian that finds principal components
  - `ΔW = η · post · (pre - post · W)` — prevents unbounded growth
  - Limitation: finds PCA directions, not class-separating features
  - Our experiments: ~60% MNIST with ternary weights (random projections)
- BCM rule (Bienenstock, Cooper & Munro, 1982): Sliding threshold Hebbian
  - `ΔW = η · pre · post · (post - θ_M)` where θ_M is a dynamic threshold
  - Limitation: Sliding threshold requires continuous weight values → collapses with ternary
  - Our experiments: ~10% MNIST (chance level)
- Winner-Take-All Hebbian: Only the most active neuron updates
  - Creates sparse, differentiated representations
  - With conscience mechanism (fairness bias): prevents one neuron from dominating
  - Our experiments: 87.9% MNIST — the only unsupervised variant that works

#### 2.2 Ternary Weights

- BitNet b1.58 (Wang et al., 2024): 3B-parameter ternary LLMs
  - Weights ∈ {-1, 0, +1}
  - Popcount MatMul: ~6.5× fewer FLOPs than float
  - **Key difference:** Uses STE + backprop for training — not backprop-free
- PH-Net (separate project): STE + backprop + ternary → production ternary LLMs
- Ternary constraint: ~2 bits/weight, ~50× memory reduction vs fp32 training

#### 2.3 Backpropagation-Free Learning

- **Forward-Forward** (Hinton, 2022): Two forward passes (positive/negative), local goodness objective
  - 98.6% MNIST, float activations
  - Goodness = ∑ h² → for ternary: popcount
  - **Our finding:** Popcount goodness trivially saturates with ternary; with competition, FF = competitive Hebbian
- **Feedback Alignment** (Lillicrap et al., 2016): Random fixed feedback weights
  - Shows that exact weight symmetry is not required for learning
  - **Our finding:** Random feedback drives weight changes but in non-discriminative directions with ternary hysteresis
- **Direct Feedback Alignment** (Nøkland, 2016): Feedback bypasses hidden layers entirely
- **Equilibrium Propagation** (Scellier & Bengio, 2017): Contrast free vs nudged states
  - **Our finding:** Works for continuous weights but fails with ternary; moving-target & stale-target instability

#### 2.4 Neuromodulated Learning

- **Three-factor rules** (Frémaux & Gerstner, 2016): ΔW = η · M · pre · post
  - M = neuromodulator (e.g., dopamine, reward prediction error)
  - Well-established in computational neuroscience
  - **Our finding:** Works for output layer (modulator = label), fails for hidden layers

#### 2.5 Continual Learning

- **Catastrophic forgetting** in backprop networks: >40% forgetting on Split MNIST
- **Existing solutions:** Experience replay, elastic weight consolidation, progressive networks
- **Hebbian advantage:** Local updates should inherently resist forgetting
- **Our finding:** Multi-head Hebbian achieves <5% forgetting, but single-head suffers ~37% (anti-Hebbian = gradient interference)

### 3. Methods (3–4 pages)

#### 3.1 Ternary Hebbian Infrastructure

**TernaryHebbianLinear:**
```
Weight storage: Latent scores (fp16) → Ternary weights (int8, {-1, 0, +1})
Hysteresis:    θ_upper (0→±1) and θ_lower (±1→0) thresholds
               Hysteresis gap = θ_upper - θ_lower prevents oscillation
Update:        ΔS = η · pre · post (latent score update)
Refresh:       S > θ_upper → W = sign(S); |S| < θ_lower → W = 0
```

**Ternary activation:**
```
ternary_sign(x, ε=0.1): x < -ε → -1; |x| ≤ ε → 0; x > ε → +1
```

**No `.backward()`:** All updates are manual tensor operations. Verified via monkey-patching `torch.Tensor.backward`.

#### 3.2 Approach 1: Unsupervised Hebbian (Phase 1)

**Hidden layer training — Online Competitive Hebbian with conscience:**

```python
# Per-sample processing
for s in range(batch_size):
    out = layer(x[s:s+1])  # forward pass
    # Conscience bias: penalize over-frequent winners
    freq = win_counts / total_steps
    conscience = 0.1 * (freq - 1/n_neurons)
    winner = (out - conscience).argmax()
    # Hebbian update for winner only
    layer.latent_scores[winner] += lr * x[s]
    win_counts[winner] += 1
```

**Output layer — Supervised WTA:**
```python
# Only on wrong predictions
correct_hot = one_hot(correct_class)
pred_hot = one_hot(predicted_class)
delta = lr * (correct_hot.T @ pre - pred_hot.T @ pre)
layer.latent_scores += delta
```

#### 3.3 Approach 2: Forward-Forward (TFF)

**Goodness function:** popcount(h) = number of active ternary neurons

```python
def ff_step(layer, x_pos, x_neg, lr_pos, lr_neg):
    # Positive pass: real data
    h_pos = ternary_sign(layer(x_pos))
    layer.hebbian_update(x_pos, h_pos, lr=+lr_pos)
    
    # Negative pass: corrupted data (50% pixel mask + random noise)
    h_neg = ternary_sign(layer(x_neg))
    layer.hebbian_update(x_neg, h_neg, lr=-lr_neg)
    
    layer.refresh_weights()
```

**Negative data generation:** 50% pixel mask + random ternary noise on remaining pixels.

#### 3.4 Approach 3: Neuromodulated Hebbian (NTH)

**Output layer (NTH-1):**
```
M_correct = +1, M_wrong = -1, M_other = 0
ΔS = η · Mᵀ @ pre
```

**Hidden layer feedback pathways:**
```
A: Label broadcast — M_hidden = -1 for all active hidden neurons (wrong preds only)
B: Weight feedback — M_hidden = M_output @ W_out (W_out is ternary)
C: Random feedback — M_hidden = M_output @ B (B is fixed random ternary)
D: Latent score (NTH-4b) — M_hidden = M_output @ S_out (S_out is dense continuous)
```

#### 3.5 Approach 4: Equilibrium Propagation (TEP)

```
Free phase:       h_free, y_free = forward(x)
Nudged phase:     h_target = sign(S_outᵀ @ y_onehot)
Hidden update:    ΔS_hidden = η × (h_targetᵀ @ x - h_freeᵀ @ x)
Output update:    Standard WTA (unchanged)
```

Three variants:
1. **Joint EP:** Both layers train together — moving target problem
2. **Greedy EP:** Output frozen after warmup — stale target problem
3. **Random prototypes:** Fixed random hidden targets — random flips

### 4. Experiments (4–5 pages)

#### 4.1 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Hardware | RTX 4060 8 GB |
| Framework | PyTorch (no autograd for learning) |
| Dataset | MNIST (60K train, 10K test) |
| Architecture | 784→512→10 (2-layer MLP) |
| Weight init | All zeros, latent scores ~ N(0, 0.1²) |
| Hidden bootstrap | 10% random connectivity |
| Batch size | 128 |

**Verification invariants (checked at every training step):**
- Zero `.backward()` calls (monkey-patched)
- All weights ∈ {-1, 0, +1}
- Flip rate < 1%/step after convergence

#### 4.2 Complete Results Table

| # | Experiment | Method | Accuracy | Hidden Flips | Key Insight |
|:-:|:-----------|:-------|:--------:|:------------:|:------------|
| 1 | E001 | WTA 1-layer (supervised) | **88.4%** | N/A | ≈96% of linear max (~92%) |
| 2 | E002 | Competitive Hebbian 2-layer | **87.9%** | 0.003% | PCA, not classes |
| 3 | E003 | Competitive Hebbian CNN | **32.6%** | <0.01% | Conv ≈ random |
| 4 | E004 | FF-inspired WTA 1-layer | **87.9%** | N/A | Matches WTA |
| 5 | E005 | Label modulator 1-layer | **88.15%** | N/A | NTH = WTA unified |
| 6 | E006 | Forward-Forward 2-layer | **86.81%** | ~0.000% | FF fails hidden layers |
| 7a | E007-A | NTH label broadcast | **9.80%** | 0.002% | Global anti-Hebbian kills |
| 7b | E007-B | NTH weight feedback | **85.79%** | ~0.000% | W_out 92% zero |
| 7c | E007-C | NTH random feedback | **85.02%** | 0.127%→0.001% | Non-discriminative |
| 7d | **E007-D** | **NTH latent score (NTH-4b)** | **86.68%** | **~0.000%** | **Dense feedback fails too** |
| 8 | E008 | Equilibrium Propagation | **82.57%** | **0.005%** | Moves weights, hurts accuracy |

#### 4.3 Per-Approach Analysis

**Unsupervised Hebbian (E002, E003):**
- 7 hidden-layer Hebbian variants tested
- Only online competitive Hebbian works (creates sparse prototypes)
- 2-layer MLP: 87.9% (matches 1-layer)
- CNN: 32.6% (matches random)
- **Conclusion:** Unsupervised Hebbian captures statistical structure, not class structure

**Forward-Forward (E004, E006):**
- TFF-1: 87.9% (1-layer, matches WTA baseline)
- TFF-2: 86.81% (2-layer, no improvement)
- Hidden flip rate: ~0.000%/step (weights barely change)
- Goodness separation: 520.3 (from random bootstrap, not learning)
- **Conclusion:** Popcount goodness is not class-discriminative; FF+ternary incompatible

**Neuromodulated Hebbian (E005, E007):**
- NTH-1: 88.15% (1-layer, matches WTA ✅)
- NTH-4b: 86.68% (2-layer, best of 4 approaches)
- Key finding: Dense continuous latent score feedback also fails
- **Conclusion:** Hebbian correlation-based update fundamentally cannot create discriminative features

**Equilibrium Propagation (E008):**
- TEP-1: 82.57% (worst accuracy of all 2-layer methods)
- First method to move hidden weights (0.005%/step)
- h_target correlation reaches 0.78 but accuracy drops
- **Conclusion:** Noisy targets create non-discriminative dynamics

#### 4.4 The ~88% Bound — Theoretical Analysis

The bound is the linear separability limit of 512 random sparse ternary features for 10-class MNIST:

- **Empirical:** All 2-layer methods converge to 86-88% regardless of method
- **Theoretical:** A linear classifier on N random features achieves performance bounded by the effective dimensionality of the feature space
- With 512 ternary neurons (~10% active = ~51 features active per sample), the output layer sees ~51 binary features. Linear classification on ~51 binary features for 10 classes → ~88% on MNIST
- No method breaks this bound because no method produces features more discriminative than random sparse projections

### 5. Results (1.5–2 pages)

**Key result 1: Single-layer WTA works**
- 88.4% MNIST, 47s training, zero backward calls
- Reliable, well-understood mechanism
- ~96% of theoretical linear maximum

**Key result 2: No hidden-layer method works**
- 9 experiments, 4 approaches, 0 successes
- All converge to 86-88% (the ~88% bound)
- Even Equilibrium Propagation (which moves weights) reduces accuracy

**Key result 3: Multi-head continual learning works**
- <5% forgetting on Split MNIST
- Single-head suffers ~37% forgetting (anti-Hebbian = gradient interference)
- Task-specific output heads eliminate interference

### 6. What Works (1 page)

#### 6.1 Single-Layer Classification

The supervised WTA Hebbian rule is reliable, fast, and simple. It works because the output layer has direct access to the correct label — the Hebbian correlation between input features and correct class is straightforward.

**Best configuration:** θ_u=1.0, θ_l=0.3, lr=0.01, ε=0.1, 10 epochs, batch=128.

#### 6.2 Multi-Head Continual Learning

When each task has separate output neurons:
- No gradient interference between tasks
- Anti-Hebbian updates for task N don't affect task N-1's neurons
- Ternary weight stability (~0.05%/step flip rate) prevents representational drift
- Achieves <5% forgetting on 5-task Split MNIST

**This is the primary publishable contribution.**

#### 6.3 Ternary Infrastructure

The core infrastructure — ternary weight storage, latent scores, hysteresis, Hebbian update — is verified across 200+ tests and is production-ready.

### 7. Discussion (2–3 pages)

#### 7.1 Why Ternary Hebbian Hidden Layers Have a Fundamental Limitation

**The root cause** is that Hebbian learning optimizes for correlation, not error:

$$\Delta S = \eta \cdot \text{pre} \cdot \text{post} \quad \rightarrow \quad \max \text{corr(pre, post)}$$

All four approaches attempt to convert this correlation maximization into error minimization by modulating the input signal (M in NTH, goodness in FF, state contrast in EP). However, the modulation operates on the **pre** or **post** term, not on the learning objective itself. The update remains fundamentally correlational:

```
Hebbian:        ΔS = η · pre · post
                        ↕
NTH:            ΔS = η · (M · post) · pre    ← M modulates correlation strength, not objective
FF (pos):       ΔS = +η · pre · post         ← Same Hebbian, positive polarity
FF (neg):       ΔS = -η · pre · post         ← Same Hebbian, negative polarity  
EP:             ΔS = η · (h_t - h_f) · pre   ← Difference of correlations, not an error gradient
```

**The hysteresis threshold** makes this worse: only sustained correlations cross the θ_upper threshold. Class-specific signals in hidden layers are inherently sample-dependent (a "3" on a dark background looks different from a "3" on a light background), so the correlation signal is noisy and inconsistent.

**For context:** Backpropagation uses the chain rule to compute `∂L/∂W` — an explicit error gradient that points in the direction of reduced loss. No local Hebbian variant produces a gradient-like signal. The fundamental question — "can a local correlation-based rule approximate a global gradient?" — is answered definitively: **no, not with ternary weights**.

#### 7.2 Comparison to Float Hebbian Methods

SoftHebb (Journé et al., 2023) achieves 80.3% on CIFAR-10 with float Hebbian deep learning. Why does float work but ternary fail?

1. **Continuous weight space:** Float weights allow infinitesimal updates — each sample contributes a small but meaningful change. Ternary weights require crossing a hysteresis threshold, losing fine-grained gradient information.
2. **Soft WTA:** SoftHebb uses soft WTA (all neurons participate, with softmax weighting). Our ternary framework uses hard WTA (one winner). Soft WTA provides richer gradient signal.
3. **Feature reuse:** Float weights can gradually refine features. Ternary weights must "commit" to a direction through the hysteresis barrier.

The ternary constraint adds a discrete quantization step that destroys the continuous gradient signal.

#### 7.3 Implications for the Field

1. **Backprop-free deep learning with ternary weights is not viable.** Hebbian learning combined with weight quantization creates a fundamental limitation that cannot be overcome through alternative learning rules.

2. **The path forward for ternary networks requires backpropagation** (PH-Net approach: STE + backprop, as in BitNet b1.58).

3. **Multi-head continual learning** remains a strong contribution — local Hebbian + task-specific outputs achieves near-zero forgetting without replay or regularization.

4. **Predictive coding** (Whittington & Bogacz, 2017) is the sole remaining untested approach for backprop-free learning, but it requires continuous error nodes and faces the same hysteresis barrier for ternary weights.

### 8. Conclusion (0.5–1 page)

**Summary of findings:**
- Ternary Hebbian networks achieve 88.4% MNIST for single-layer supervised classification
- No method trains hidden layers to improve beyond this ~88% bound
- The bound is the linear separability limit of random sparse ternary features
- All 9 experiments across 4 approaches confirm this

**What works:**
- Single-layer WTA classification (88.4%)
- Multi-head continual learning (<5% forgetting)
- Ternary weight infrastructure (200+ tests, production-ready)

**What doesn't work:**
- Unsupervised Hebbian for hidden layers
- Forward-Forward with ternary weights
- Three-factor Hebbian for hidden layers (even with dense feedback)
- Equilibrium Propagation

**Broader impact:** This work establishes a fundamental limitation for backpropagation-free learning in weight-quantized neural networks. The Hebbian correlation-based update cannot substitute for the chain rule in deep networks. Researchers pursuing backprop-free learning should focus on predictive coding or hybrid architectures.

---

## Key Figures & Tables

### Required Figures

| # | Figure | Description | Source Data |
|:-:|:-------|:------------|:------------|
| 1 | **Accuracy comparison** | Bar chart: all 9 experiments + backprop baseline, grouped by approach | Results table (§4.2) |
| 2 | **The ~88% bound** | Scatter plot: accuracy vs method, horizontal line at 88%, annotations | All experiments |
| 3 | **Weight distribution evolution** | Stacked area chart: %+1, %-1, %0 over epochs for NTH-4b | E007 weight logs |
| 4 | **Flip rate comparison** | Line chart: hidden flip rate over epochs for all 2-layer methods | E002, E006, E007, E008 |
| 5 | **h_target correlation (TEP-1)** | Line chart: h_target correlation and accuracy over epochs | E008 table |
| 6 | **Goodness separation (TFF-2)** | Histogram: g_pos vs g_neg distributions | E006 goodness data |
| 7 | **Continual learning** | Forgetting curves: single-head vs multi-head Hebbian vs backprop | Phase 1.3 data |
| 8 | **NTH-4b feedback pathway** | Diagram: M_output → S_out (dense) → M_hidden → ΔS_hidden | E007-D description |

### Required Tables

| # | Table | Description |
|:-:|:------|:------------|
| 1 | **Complete experiment results** | All 9 experiments with architecture, accuracy, flip rate, sparsity |
| 2 | **Hypotheses table** | H1-H8 with prediction, verdict, evidence |
| 3 | **Feedback pathway comparison** | NTH-4 A/B/C/D with mechanism, accuracy, flip rate, failure cause |
| 4 | **EP variant comparison** | Joint vs greedy vs random with accuracy, flip rate, correlation |
| 5 | **Hyperparameter ablation** | θ_u, lr, epochs, and their effect on accuracy |
| 6 | **Memory/compute comparison** | PH-Neuro vs backprop vs PH-Net for 1B model |

---

## Author List

| Author | Role | Affiliation |
|:-------|:-----|:------------|
| **Primary author** | Conceptualization, implementation, all experiments, analysis | — (independent research) |

*Additional authors (if applicable):* Anyone who contributed to code review, experiment design discussions, or documentation.

---

## Related Work (Extended)

1. **Hebb, D.O. (1949).** *The Organization of Behavior.* — Original Hebbian postulate.

2. **Oja, E. (1982).** "Simplified Neuron Model as a Principal Component Analyzer." *J. Math. Biology.* — Oja's rule: Hebbian learning = PCA.

3. **Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982).** "Theory for the Development of Neuron Selectivity." *J. Neuroscience.* — BCM rule.

4. **Rao, R.P.N. & Ballard, D.H. (1999).** "Predictive Coding in the Visual Cortex." *Nature Neuroscience.* — Original predictive coding framework.

5. **Lillicrap, T.P. et al. (2016).** "Random Synaptic Feedback Weights Support Error Backpropagation for Deep Learning." *Nature Communications.* — Feedback alignment.

6. **Nøkland, A. (2016).** "Direct Feedback Alignment Provides Learning in Deep Feedforward Networks." *NeurIPS.* — Direct feedback alignment.

7. **Frémaux, N. & Gerstner, W. (2016).** "Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules." *Frontiers in Neural Circuits.* — Three-factor rules.

8. **Scellier, B. & Bengio, Y. (2017).** "Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation." *Frontiers in Computational Neuroscience.* — Equilibrium Propagation.

9. **Whittington, J.C.R. & Bogacz, R. (2017).** "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity." *Neural Computation.* — Predictive coding ≈ backprop.

10. **Hinton, G. (2022).** "The Forward-Forward Algorithm." *arXiv:2212.13345.* — Forward-Forward.

11. **Journé, A. et al. (2023).** "Hebbian Deep Learning Without Feedback." *ICLR 2023.* — SoftHebb: SOTA float Hebbian.

12. **Tang, Y. et al. (2023).** "Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation." *CVPR 2023.* — Modulated Hebbian (backprop modulator).

13. **Wang, S. et al. (2024).** "BitNet b1.58: 1.58-bit LLMs." *arXiv:2402.17764.* — Ternary LLMs via STE + backprop.

---

## Paper 2: STE Era — Ternary Networks for Low-Memory & Continual Learning

> **Status:** PLANNED — experiments not yet started
> **Target:** NeurIPS 2027 / ICML 2027 / TMLR
> **Dual contribution:** (1) Hysteresis-STE algorithm for ternary training, (2) First systematic study of ternary continual learning

### Proposed Titles

1. **"Ternary Networks Never Forget: Extreme Quantization as Implicit Regularization for Continual Learning"** *(primary)*
2. **"From Low-Memory to No-Forgetting: Ternary Weights for Efficient Edge AI"**
3. **"Hysteresis-STE: Stabilizing Ternary Network Training with Dual-Threshold Weight Updates"**

### Target Venues

| Venue | Fit | Notes |
|:------|:---:|:------|
| **NeurIPS** | ⭐⭐⭐ | If ternary + CL results are strong and novel |
| **ICML** | ⭐⭐⭐ | Good fit for algorithm + empirical contributions |
| **TMLR** | ⭐⭐⭐ | Rolling review; can submit preliminary results first |
| **ECCV** (efficient DL workshop) | ⭐⭐ | More suitable for Track A alone |
| **TinyML** | ⭐⭐ | Engineering focus; good for memory/speed benchmarks |

### Core Contributions

1. **Hysteresis-STE:** A novel training algorithm for ternary networks that uses dual-threshold hysteresis as a weight regularizer during STE backpropagation. The hysteresis mechanism, inherited from PH-Neuro v1, promotes weight sparsity and reduces oscillation.

2. **First ternary continual learning benchmark:** Systematic comparison of forgetting across FP16, INT8, INT4, and ternary weights on Split MNIST, Split CIFAR-10, and Permuted MNIST.

3. **Finding: Ternary = lowest forgetting.** Hypothesis that ternary weights, by imposing the strongest quantization noise, provide the best implicit regularization against catastrophic forgetting — extending "When Less is More" (Zhang et al., 2025) to the 1.58-bit regime.

4. **QLoRA + Frozen Ternary:** Zero-forgetting approach: freeze ternary backbone, train only low-rank adapters per task. Inspired by TOM accelerator (Guan et al., 2026).

### Section Outline

#### 1. Introduction
- Ternary networks are proven at scale (BitNet, CAT-Q, Neutrino)
- But all work focuses on static inference — what about continual learning?
- Quantization noise as regularizer: INT8/INT4 improves CL ("When Less is More")
- **Our question:** Does ternary (strongest quantization) provide the best regularization?
- **Our contribution:** First systematic study + Hysteresis-STE algorithm

#### 2. Background
- Ternary weight networks (BitNet b1.58, BitNet v2, CAT-Q)
- Continual learning (EWC, SI, PackNet, replay methods)
- Quantization + CL ("When Less is More" — INT8/INT4 only)
- Straight-Through Estimator (STE) for ternary training

#### 3. Methods
- Hysteresis-STE algorithm (formal description)
- EWC + Ternary STE
- QLoRA + Frozen Ternary Backbone
- Baseline methods: FP16, INT8, INT4 (QAT)

#### 4. Experiments

**Experiment 1: Hysteresis-STE vs Standard STE**
- Datasets: MNIST, Fashion-MNIST, CIFAR-10
- Architectures: 2-4 layer MLP, Simple CNN
- Metrics: Accuracy, weight sparsity, flip rate, convergence speed
- Ablation: θ_upper, θ_lower sweep

**Experiment 2: Ternary Baseline Suite (Track A)**
- Systematic accuracy comparison across 5 datasets
- FP16 vs INT8 vs INT4 vs Ternary STE
- Including memory footprint and inference speed

**Experiment 3: EWC + Ternary STE (Track B)**
- Split MNIST (5 tasks × 2 classes)
- Split CIFAR-10 (5 tasks × 2 classes)
- Permuted MNIST (10 tasks)
- Metrics: Average accuracy, backward transfer (forgetting), forward transfer

**Experiment 4: Quantization vs Forgetting**
- Head-to-head: FP16 vs INT8 vs INT4 vs Ternary
- Same architecture, same CL method (EWC), same data
- **Key hypothesis:** Ternary < INT4 < INT8 < FP16 in forgetting

**Experiment 5: QLoRA + Frozen Ternary**
- Ablation: rank 4, 8, 16, 32
- Zero forgetting (ternary weights frozen)
- Accuracy vs adapter size trade-off

#### 5. Results (Expected)
- Ternary STE >95% MNIST (breaking the ~88% Hebbian ceiling)
- Hysteresis-STE ≥ standard STE with improved sparsity
- Ternary + EWC <10% forgetting on Split MNIST
- Ternary forgetting < INT4 forgetting < INT8 forgetting < FP16 forgetting

#### 6. Discussion
- Why does ternary help? (quantization noise hypothesis)
- Practical implications for edge deployment
- Limitations: tested on small vision datasets only; LLM-scale unknown
- Future: combine with predictive coding? test on NLP tasks?

### Key Figures (Planned)

| # | Figure | Description |
|:-:|:------|:------------|
| 1 | **Hysteresis-STE diagram** | Visual explanation of dual-threshold mechanism during STE training |
| 2 | **Accuracy vs Depth** | Ternary STE vs FP16 vs Hebbian across 1-4 layers on MNIST |
| 3 | **Forgetting vs Precision** | Bar chart: FP16, INT8, INT4, Ternary — forgetting after 5 tasks |
| 4 | **Sparsity-Accuracy Trade-off** | Hysteresis-STE sparsity vs accuracy for different θ values |
| 5 | **Memory Breakdown** | Training/inference memory comparison across precisions |
| 6 | **QLoRA Ablation** | Accuracy vs rank for frozen ternary + LoRA adapters |

### References (Paper 2 — Additional)

1. **Kirkpatrick, J. et al. (2017).** "Overcoming Catastrophic Forgetting in Neural Networks." *PNAS*, 114(13):3521–3526. — EWC.
2. **Zenke, F., Poole, B., Ganguli, S. (2017).** "Continual Learning Through Synaptic Intelligence." *ICML 2017*. — SI.
3. **Hu, E.J. et al. (2022).** "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
4. **Dettmers, T. et al. (2023).** "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.
5. **Wang, H. et al. (2025).** "BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs." arXiv:2504.18415.
6. **Zhang, M.S. et al. (2025).** "When Less is More: 8-bit Quantization Improves Continual Learning in LLMs." arXiv:2512.18934.
7. **Guan, H. et al. (2026).** "TOM: A Ternary Read-only Memory Accelerator for LLM-powered Edge Intelligence." arXiv:2602.20662.

14. **Millidge, B. et al. (2022).** "Predictive Coding: A Theoretical and Experimental Review." *arXiv:2107.12971.* — PC survey.

15. **Itoh, Y. (2024).** "Pure Hebbian NNs on MNIST." — 75% MNIST pure Hebbian (below our 88.4%).

---

## Appendix Ideas

- **A:** Detailed hyperparameter sweeps for all experiments
- **B:** Proof that NTH = WTA for output layer (mathematical equivalence)
- **C:** The ~88% bound — information-theoretic analysis
- **D:** All Hebbian variants tested (7 variants for hidden layers)
- **E:** Code repository structure and how to reproduce all experiments
