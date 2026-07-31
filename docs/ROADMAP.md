# PH-Neuro Roadmap

> **Status:** Strategic pivot — transitioning from Hebbian to STE-based ternary learning  
> **Last updated:** 2026-07-31 (L7 Depth vs Width ✅)

---

## Research Synthesis Documents

> **⚠️ Phase 0-2 (Hebbian) research phase closed after 9 experiments across 4 approaches.**
>
> For the definitive summary of all Phase 0-2 findings, see:
> - **[`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md)** — Complete research document: abstract, hypotheses, methods, results, analysis, and references
> - **[`PAPER_OUTLINE.md`](PAPER_OUTLINE.md)** — Paper outlines for both Phase 0-2 (negative results) and Phase 3+ (new direction)
>
> All experiment reports are in [`docs/experiments/`](experiments/).

---

## Strategic Pivot (2026-07-30)

### What Changed

A systematic literature scan (July 28-30, 2026) revealed that the ternary network landscape has transformed dramatically since early 2025:

| Development | Date | Significance |
|:-----------|:-----|:-------------|
| **BitNet b1.58** (Microsoft) | Feb 2024 | Ternary LLMs trained from scratch with STE backprop — proved ternary CAN scale |
| **BitNet b1.58 2B4T** (Microsoft) | Apr 2025 | First open-source ternary LLM (2B params, 4T tokens), MIT license |
| **BitNet v2** (Microsoft) | Apr 2025 | Native 4-bit activations with Hadamard transformation — W1.58A4 |
| **CAT-Q** (Intel, ICML 2026 Oral) | Jun 2026 | Post-training ternary quantization with only 512 calibration samples — 100,000× fewer tokens than BitNet |
| **Neutrino-8B** (Fermion Research) | Jul 2026 | 8B ternary model, 3.88 GB on disk, MMLU-Redux 67.84, Apache 2.0 |
| **TOM Accelerator** | Feb 2026 | Ternary ROM-SRAM hardware with QLoRA-based on-device tunability |
| **"When Less is More"** | Dec 2025 | INT8/INT4 quantization IMPROVES continual learning (quantization noise = implicit regularization) |
| **VibeVoice-ASR-BitNet** (Microsoft) | Jul 2026 | First ternary ASR model — 1.6-2.3× faster than whisper.cpp on edge CPUs |

### The Key Insight

**Ternary networks CAN learn deep representations — but ONLY with backpropagation (STE).** PH-Neuro's Phase 0-2 conclusively proved that Hebbian/FF/EP methods hit a fundamental ~88% ceiling on MNIST regardless of depth. Meanwhile, BitNet, CAT-Q, and Neutrino prove that STE backprop enables ternary networks at billion-parameter scale.

**The new question:** Given that ternary weights + STE backprop works, what else can we do with it that nobody has tried?

### The Research Gap: Ternary + Continual Learning

Two fields that have NEVER been combined:
1. **Ternary/1.58-bit networks** — thriving (BitNet, CAT-Q, Neutrino) but focused exclusively on static LLM inference
2. **Continual/lifelong learning** — "When Less is More" showed quantization helps (INT8/INT4), but **ternary remains untested**

**PH-Neuro is uniquely positioned** with its ternary infrastructure (packed tensors, hysteresis, flip rate tracking, 200+ tests) to bridge this gap.

### New Vision

**PH-Neuro v2: Dual-track research into ternary networks for efficient edge AI**

| | Track A: Low-Memory Supervised | Track B: Continual Learning |
|:--|:------------------------------|:---------------------------|
| **Goal** | Establish ternary STE baselines for small vision models; develop novel training techniques (Hysteresis-STE) | Prove that ternary weights + continual learning methods can achieve <10% forgetting while beating the ~88% Hebbian ceiling |
| **Method** | STE backprop + ternary weights + Hysteresis regularizer | EWC / SI / QLoRA + frozen ternary backbone |
| **Datasets** | MNIST, Fashion-MNIST, KMNIST, CIFAR-10, CIFAR-100 | Split MNIST, Split CIFAR-10, Permuted MNIST |
| **Novelty** | First systematic ternary vision benchmark since 2020; Hysteresis-STE algorithm | First combination of ternary weights with continual learning; quantization noise as forgetting regularizer |
| **Risk** | Low | Medium |
| **Timeline** | 1-2 weeks | 2-4 weeks |
| **Target Venue** | TinyML / ECCV Efficient DL Workshop | NeurIPS / ICML / TMLR |

### Why This Matters (Updated)

| Property | Backprop (FP16) | PH-Neuro v1 (Hebbian) | PH-Neuro v2 (Ternary STE) |
|----------|----------------|----------------------|--------------------------|
| Deep learning | ✅ Multiple layers work | ❌ ~88% ceiling regardless of depth | ✅ Multi-layer proven (BitNet) |
| Weight memory | 16 bits/weight | 2 bits/weight (packed) | 2 bits/weight (packed) |
| Inference compute | Float MatMul | Popcount MatMul | Popcount MatMul |
| Training memory | 4-8× model size | ~1× model size | ~2× model size (STE needs backward pass) |
| Catastrophic forgetting | Severe | ⚠️ Multi-head only | 🎯 **UNEXPLORED** — hypothesis: ternary acts as regularizer |
| Backward pass | Required | None | Required (STE) |
| Edge deployment | Heavy | Lightweight | Lightweight + deep-capable |

### The Core Hypotheses

1. **H1 — Ternary Hebbian works at all**: Ternary weights {-1, 0, +1} combined with Hebbian learning can solve non-trivial classification tasks. **Partially verified:** 88.4% MNIST single-layer (✅ >85%), but CIFAR-10 CNN failed (32.6%, ❌ <55%). Hebbian learning works for direct supervised classification but NOT for unsupervised feature learning in hidden layers.

2. **H2 — No catastrophic forgetting**: ⚠️ **PARTIALLY FALSIFIED (single-head).** Single-head WTA Hebbian suffers ~37% forgetting on Split MNIST. The anti-Hebbian update that weakens wrongly-predicted classes is **functionally identical to gradient interference** in backprop. Multi-head output (separate neurons per task) achieves <5% forgetting ✅ — but this is task-incremental only.

3. **H3 — Hysteresis creates stability**: ✅ Dual-threshold mechanism prevents oscillatory flipping and creates stable representations. Flip rates converge to <0.05%/step.

4. **H4 — Layer-wise independence is sufficient**: ❌ **FALSIFIED.** Unsupervised Hebbian captures statistical structure (PCA), not class-discriminative structure. Depth provides zero improvement (2-layer 87.9% = 1-layer 88.4%). Some form of error signal is required for hidden layers.

5. **H5 — Forward-Forward solves the hidden-layer problem**: ❌ **FALSIFIED for ternary weights.** TFF-2 (2-layer FF on MNIST) achieves 86.81% — essentially identical to Phase 0 (88.4%), Phase 1.1 (87.9%), and TFF-1 (87.9%). The FF contrastive objective (popcount goodness) trivially saturates without competition. With top-1 competition added, the FF negative pass provides no benefit beyond random bootstrapped prototypes. The ~88% bound represents the linear separability limit of 512 random sparse features for 10-class MNIST — neither unsupervised Hebbian nor Forward-Forward breaks it. Documented in `docs/experiments/E006-forward-forward-multilayer-mnist.md`. **Pivot to NTH-only for hidden layers.**

6. **H6 — Language is learnable without backprop**: ⬜ **Untested.** Predictive coding + ternary Hebbian can capture sequential structure. Moved to Phase 3 (after Forward-Forward is validated).

7. **H7 — Three-factor Hebbian = local error signal**: ⚠️ **PARTIALLY FALSIFIED (multi-layer case).** NTH-1 achieves 88.15% MNIST with label modulator (M_c=+1, M_w=-1) for the single-layer output case ✅. However, NTH-4 conclusively shows that the modulator CANNOT propagate through hidden layers with ternary weights — all three modulator propagation approaches fail to beat the ~88% single-layer bound. The three-factor framework provides a local error signal for the output layer but not for hidden layers in ternary networks. Documented in `docs/experiments/E007-nth-multilayer-mnist.md`.

8. **H8 — Equilibrium Propagation solves the hidden-layer problem**: ❌ **FALSIFIED.** TEP-1 (2-layer EP on MNIST) achieves 80-84% — WORSE than the single-layer ~88% bound. EP does move hidden weights (0.005%/step — first non-backprop method to do so), but the EP signal pushes hidden representations in non-discriminative directions. Joint training exhibits a moving-target problem; greedy training exhibits a stale-target problem. All three variants (joint EP, frozen output greedy EP, random prototype EP) fail to improve classification accuracy. Documented in `docs/experiments/E008-equilibrium-propagation-mnist.md`.

---

## Phase Status

### Hebbian Era (COMPLETED — July 2026)

| Phase | Title | Status | Key Result |
|:------|:------|:------:|:-----------|
| 0 | Core Mechanism | ✅ | 88.4% MNIST, single-layer WTA Hebbian |
| 1.1 | Multi-layer MLP | ✅ | 87.9% — depth doesn't help |
| 1.2 | CNN on CIFAR-10 | ✅ | 32.6% — conv Hebbian ≈ random |
| 1.3 | Continual Learning | ✅ | <5% multi-head ✅, single-head ❌ |
| 2 | **Forward Signals & Three-Factor** | 🔴 **RESEARCH PHASE CLOSED** | **TFF-1 ✅ 87.9%, NTH-1 ✅ 88.15%, TFF-2 ❌ 86.81%, NTH-4 ❌ 85.79%, NTH-4b ❌ 86.68%, TEP-1 ❌ 82.57% — ALL 9 approaches exhausted.** |

**Definitive conclusion:** Ternary Hebbian hidden layers CANNOT be trained without backpropagation. The ~88% MNIST bound represents the linear separability limit of random sparse ternary features.

### STE Era (NEW — July 2026 onward)

| Phase | Title | Status | Key Result |
|:------|:------|:------:|:-----------|
| **3A** | **Track A: Low-Memory Supervised Baselines** | 🟡 **L1 DONE, L2 CODE DONE** | L1: 25/25 runs. L2: layers, runner, 35 tests complete. M3+M4+M5(code) achieved |
| 3A.1 | STE TernaryLinear Implementation | ✅ **COMPLETED** | `TernarySTELinear`, `TernarySTEConv2d` + `_STESign` autograd. 22/22 tests pass |
| 3A.2 | Baseline Suite: MNIST/Fashion-MNIST/KMNIST/CIFAR-10/CIFAR-100 | ✅ **COMPLETED** | All 5 variants × 5 datasets = 25 runs done. See [`E009`](experiments/E009-ste-baseline-suite.md) |
| 3A.3 | Hysteresis-STE Algorithm | ✅ **CODE COMPLETE** | Layers, runner, sweep script, aggregator, 35 tests — awaiting full run |
| 3A.4 | Forgetting Baseline (no CL mechanism) | ✅ **COMPLETED** | 12/12 runs done. Ternary ≈ FP16 forgetting (gap <1 pp). See [`E010`](experiments/E010-l8-forgetting-baseline.md) |
| 3A.5 | Memory & Speed Benchmarks | ⬜ | Packed ternary inference speed vs FP16/INT8; training memory footprint |
| **3B** | **Track B: Continual Learning with Ternary STE** | 🟡 **B1 + B2 DONE** | EWC + ternary STE (B1) ✅, QLoRA + frozen ternary (B2) ✅ |
| 3B.1 | EWC + Ternary STE on Split MNIST | ✅ **COMPLETED** | EWC (λ=10000) reduces Split-MNIST forgetting 37.33%→**32.78%** (−4.55 pp) and raises accuracy 62.16%→**66.65%** (+4.48 pp). Permuted unchanged. See [`E013`](experiments/E013-b1-ewc-ternary-ste.md) |
| 3B.2 | QLoRA + Frozen Ternary Backbone | ✅ **COMPLETED** | **Zero forgetting (0.00% ± 0.00) in all 30 runs.** r=64 beats L8/B1 by 32-53 pp: Split 99.43% (vs L8 62.16%), Permuted 86.84-92.55% (vs L8 41.92%). Weak `task1` backbone beats `full` on Permuted. See [`E014`](experiments/E014-b2-qlora-frozen-ternary.md) |
| 3B.3 | Multi-Head Ternary EWC (5 tasks) | ⬜ | Combine multi-head architecture with EWC for maximal protection |
| 3B.4 | Comparison: Ternary vs INT8 vs INT4 vs FP16 CL | ⬜ | Replicate "When Less is More" but extend to ternary |
| **4** | **Advanced Experiments** | 🟡 **L5 + L7 DONE** | Ternary distillation (L4), BN fusion (L5) ✅, depth scaling with fixed budget (L7) ✅ |
| **5** | **Papers & Publication** | ⬜ | Paper 1: Low-Memory Vision (TinyML/ECCV), Paper 2: Continual Learning (NeurIPS/ICML) |

## Phase 3 — Track A: Low-Memory Supervised Experiments

### L1: Ternary STE Baseline Suite
**Goal:** Establish modern ternary STE baselines for small vision models (no one has done this systematically since ~2020).

**Experiment document:** [`docs/experiments/E009-ste-baseline-suite.md`](experiments/E009-ste-baseline-suite.md)

**5 variants tested per dataset** for systematic comparison:

| Variant ID | Weight Format | Training Method | Purpose |
|:-----------|:-------------|:----------------|:--------|
| **V1 — Ternary STE** | {-1, 0, +1} (2-bit packed) | STE backprop + AdamW on latent fp16 scores | **Our method** — primary benchmark |
| **V2 — FP16** | float16 | Standard backprop + AdamW | Upper bound (best possible accuracy) |
| **V3 — INT8 QAT** | int8 | Quantization-Aware Training + STE | Established quantized baseline |
| **V4 — INT4 QAT** | int4 | Quantization-Aware Training + STE | Aggressive quantization baseline |
| **V5 — Hebbian v1** | {-1, 0, +1} (2-bit packed) | WTA Hebbian (no backprop) | Legacy baseline — ~88% MNIST ceiling |

**Expected results (pre-experiment):**

| Dataset | Architecture | V1: Ternary STE | V2: FP16 | V3: INT8 QAT | V4: INT4 QAT | V5: Hebbian v1 | Ternary Gap |
|:--------|:------------|:---------------:|:--------:|:------------:|:------------:|:--------------:|:----------:|
| MNIST | 784→512→256→10 | 96-98% | ~98.5% | ~97% | ~95% | 88.4% | ~1-2pp |
| Fashion-MNIST | 784→512→256→10 | 88-91% | ~92% | ~90% | ~87% | — | ~2-3pp |
| KMNIST | 784→512→256→10 | 88-91% | ~93% | ~91% | ~88% | — | ~2-3pp |
| CIFAR-10 | Conv(32→64→128)→FC | 75-85% | ~90% | ~88% | ~82% | 32.6% | ~5-10pp |
| CIFAR-100 | Conv(64→128→256)→FC | 55-65% | ~72% | ~68% | ~60% | — | ~7-12pp |

**Key comparisons at a glance:**

| Analysis | What we learn |
|:---------|:--------------|
| V1 vs V2 | Ternary gap — how much accuracy does 2-bit cost? |
| V1 vs V3/V4 | Is ternary more efficient than INT quantization (accuracy per bit)? |
| V1 vs V5 | Does STE definitively beat the ~88% Hebbian ceiling? |
| V2 vs V3 vs V4 vs V1 | Memory-vs-accuracy Pareto frontier for vision models |

**Preliminary results (seed=42, CIFAR-10/100 still running):**

| Dataset | V1: Ternary STE | V2: FP16 | V3: INT8 QAT | V4: INT4 QAT | V5: Hebbian v1 | Ternary Gap |
|:--------|:---------------:|:--------:|:------------:|:------------:|:--------------:|:----------:|
| MNIST | **98.17%** | 98.73% | 97.58% | 98.53% | 89.02% | **0.56 pp** |
| Fashion-MNIST | **89.13%** | 90.19% | 90.14% | 89.76% | 79.70% | **1.06 pp** |
| KMNIST | **91.26%** | 93.58% | 93.41% | — | 63.23% | **2.32 pp** |
| CIFAR-10 | **72.75%** | **86.33%** | TBD | TBD | 32.6% | **13.58 pp** |
| CIFAR-100 | **39.00%** | **57.50%** | **57.13%** | **55.33%** | **6.13%** | **18.50 pp** |

**✅ Milestone M4 achieved:** Ternary STE 98.17% on MNIST — beats the 88% Hebbian ceiling by **+9.15 percentage points**.

### L2: Hysteresis-STE — Novel Training Algorithm
**Core idea:** Apply PH-Neuro's dual-threshold hysteresis during STE training as a weight regularizer.

```
Standard STE:
  forward:  W_tern = sign(W_latent)
  backward: ∂L/∂W_latent = ∂L/∂W_tern  (STE)

Hysteresis-STE:
  forward:  W_tern = tern_hyst(W_latent, θ_upper, θ_lower)
            → |W_latent| < θ_lower: W_tern = 0
            → |W_latent| > θ_upper: W_tern = sign(W_latent)
            → otherwise: unchanged from previous step
  backward: ∂L/∂W_latent = ∂L/∂W_tern  (STE)
```

**Hypothesis:** Hysteresis acts as a sparsity-promoting regularizer. Small latent weights stay at 0, only strong signals cross the threshold. This may improve generalization and reduce weight oscillation.

**Ablation:** Compare standard STE vs Hysteresis-STE across θ_upper ∈ {0.3, 0.5, 1.0, 2.0} and θ_lower ∈ {0.1, 0.15, 0.3}.

### L7: Depth vs Width Scaling
**Goal:** Determine the optimal depth-to-width ratio for ternary STE networks at a fixed parameter budget (~530K, matching L1).

**Experiment document:** [`docs/experiments/E012-l7-depth-vs-width.md`](experiments/E012-l7-depth-vs-width.md)

**Design:** 5 equal-width depth configs (D=1..5) × Ternary STE vs FP16 × 3 seeds = **30 runs** on MNIST.

**Results (30/30 runs completed):**

| Depth | Ternary STE | FP16 | Gap |
|:-----:|:-----------:|:----:|:---:|
| D=1 | 97.86% | 98.53% | 0.67 pp |
| D=2 | 98.15% | 98.56% | 0.41 pp |
| D=3 | **98.27%** | 98.68% | 0.41 pp |
| D=4 | 98.26% | 98.68% | 0.42 pp |
| D=5 | 98.24% | 98.69% | 0.45 pp |

**Key findings:**
1. **Depth scaling works for ternary** — D=1→D=3 gives +0.41 pp for ternary vs +0.15 pp for FP16. The hypothesis that repeated STE sign ops cause gradient degradation is **FALSIFIED**.
2. **Ternary gap is flat** (~0.4-0.7 pp) across all depths — no ternary depth penalty.
3. **Optimal config**: D=3 `[784, 353, 353, 353, 10]` at 98.27%.
4. **0% ternary sparsity** at all depths (standard STE + AdamW → all weights ±1).

### L8: Forgetting Baseline (Standard SGD)
**Goal:** Measure how much standard ternary STE training forgets WITHOUT any continual learning mechanism. This is the control experiment for Track B.

| Setup | Metric |
|:------|:------|
| Split MNIST, sequential SGD, no EWC, no replay | Forgetting after 5 tasks |
| Permuted MNIST, 10 tasks | Average accuracy, forgetting |
| Comparison: FP16 vs Ternary STE | Does ternary naturally forget less? |

**Status:** ✅ **COMPLETED** — 12/12 runs (2 protocols × 2 weight formats × 3 seeds).
**Key result:** Forgetting gap (FP16 − Ternary) = **+0.22 pp** (Split) and **+0.66 pp** (Permuted) — **essentially identical**. See [`E010`](experiments/E010-l8-forgetting-baseline.md).

**Implication:** Ternary weights do NOT provide natural forgetting resistance with standard STE. Hysteresis-STE (L2) or explicit CL methods (EWC, QLoRA) are needed for Track B.

---

## Phase 3 — Track B: Continual Learning Experiments

### B1: EWC + Ternary STE
**Goal:** Test whether Elastic Weight Consolidation works with ternary weights.

**Hypothesis:** Ternary weights provide a natural "stiffness" — small latent score changes don't flip ternary values. EWC further protects important weights. Combined effect: <10% forgetting while beating the ~88% Hebbian ceiling.

| Variant | Description |
|:--------|:------------|
| EWC-ternary | Standard EWC with Fisher computed after each task on ternary model |
| EWC-hysteresis | EWC + Hysteresis-STE — double protection against forgetting |
| Online EWC | Update Fisher incrementally (no separate post-task phase) |

### B2: QLoRA + Frozen Ternary Backbone
**Goal:** Zero-forgetting approach: freeze ternary weights, train only low-rank adapters.

Inspired by TOM accelerator (arXiv:2602.20662) which implements QLoRA-based on-device tunability for ternary weights.

| Variant | Description |
|:--------|:------------|
| QLoRA-rank2/4/8/16/32/64 | LoRA rank sweep on all linear layers, frozen ternary backbone (best: r=64) |
| Pretrain-full | Backbone pre-trained on full MNIST (10 epochs) |
| Pretrain-task1 | Backbone pre-trained on full MNIST (1 epoch, limited-data sim) |

**Advantage:** Zero forgetting (ternary weights never change — 0% by design) **and** higher accuracy than L8/B1 baselines.
**Disadvantage:** needs float adapters per task (r=4 → ~1% of model; r=64 → ~28%).

**Results (2026-07-31, 30 runs):** Forgetting **0.00% ± 0.00** everywhere. r=64 (3 seeds): Split 99.43%/99.22% (full/task1), Permuted 86.84%/92.55% — **+32 to +53 pp** over L8 and B1. Rank scaling monotonic; Split saturates at r≈8, Permuted still climbing at r=64. Weak `task1` backbone beats `full` on Permuted (92.55% vs 86.84%) — strong features committed to canonical layout resist permutation adaptation. See [`E014`](experiments/E014-b2-qlora-frozen-ternary.md).

### B3: Comparison — Ternary vs INT8 vs INT4 vs FP16
**Goal:** Replicate "When Less is More" (arXiv:2512.18934) findings but extend to ternary.

| Precision | Expected Forgetting | Expected Accuracy |
|:----------|:------------------:|:-----------------:|
| FP16 | Highest (~36%) | Highest (~98%) |
| INT8 | Medium (~25%) | High (~97%) |
| INT4 | Low (~15%) | Medium (~95%) |
| **Ternary** | **Lowest? (~8%)** | **Good? (~94%)** |

**Hypothesis:** Ternary provides the strongest quantization noise → best implicit regularization → lowest forgetting. This would be a novel finding.

---

## Core Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Weight format | Ternary {-1, 0, +1} | Biological analogy (excitatory/inhibitory/absent); BitNet b1.58 proven at scale; 0 = natural sparsity; popcount MatMul |
| Activation format | Ternary {-1, 0, +1} | Enables pure integer/popcount MatMul; Hebbian updates become trivial (agree → strengthen, disagree → weaken) |
| Plasticity mechanism | Latent float score + dual-threshold hysteresis | θ_upper to activate (0 → ±1), θ_lower to deactivate (±1 → 0); hysteresis gap prevents oscillation; homeostatic decay for unused synapses |
| Hebbian rule | ΔW_latent = lr × pre_activation × post_activation | Simplest Hebbian variant; both operands are ternary so update is ±lr or 0; no normalization, no competition (initially) |
| Multi-layer learning | Greedy layer-wise (like SoftHebb) | Each layer trained independently bottom-up; output of layer N is input to layer N+1; no backward signal between layers |
| Output layer | Hebbian + anti-Hebbian with label supervision | Correct class neuron: Hebbian strengthen. Wrong class neurons: anti-Hebbian weaken. Unified ternary mechanism throughout. |
| Framework | PyTorch (no autograd for learning) | PyTorch for tensor ops, data loading, infrastructure; Hebbian updates are manual (no `.backward()`); future: pure C++/WASM |
| Inference compute | Popcount (XOR + bitcount) | Ternary MatMul reduces to bitwise operations; ~6.5× fewer FLOPs than float MatMul |
| Training compute | Popcount forward + O(n) Hebbian update | No backward pass; no gradient computation; no optimizer step; training FLOPs ≈ inference FLOPs + trivial update |
| Weight storage | Naive int8 (dev) → Packed 2-bit (prod) | PyTorch has no native int2 — use 1 byte/weight for debugging in Phases 0-2, then pack 4 weights/byte in Phases 3-4 for 4× memory reduction |
| Training memory | ~1× model size (naive) / ~0.25× (packed) | No optimizer states, no gradient buffers, no activation checkpointing; even with naive int8, 1B model fits in 8 GB VRAM |
| Architecture | Agnostic (MLP, CNN, Transformer, MoE) | Not locked to any single architecture; Hebbian rule is architecture-agnostic |
| Evaluation philosophy | Accuracy vs backprop + forgetting vs backprop | Two-dimensional evaluation: static accuracy (expected loss) and continual learning (expected win) |

---

## Hardware Constraints

**Current machine:** RTX 4060 8 GB VRAM, i7-14700K (20 cores), 16 GB RAM, 1 TB SSD.

Unlike PH-Net where 8 GB VRAM is a severe bottleneck (AdamW states alone are 2× model size), PH-Neuro's memory footprint is dramatically smaller.

**Weight storage strategy — two phases:**
- **Phases 0-2 (development)**: Naive int8 — 1 byte per ternary weight. Simple, debuggable, fast.
- **Phases 3-4 (production)**: Packed 2-bit — 4 weights per int8 byte. 4× memory reduction when scaling matters.

Both strategies use the same tensor operations; packing/unpacking is transparent to the Hebbian update logic.

| Phase | Model size | Ternary weights (int8) | Ternary weights (packed 2-bit) | Latent scores (fp16) | Activations | Total (int8) | Total (packed) | Fits 8 GB? |
|-------|-----------|----------------------|-------------------------------|---------------------|-------------|-------------|---------------|-------------|
| 1 | <100K params | <0.1 MB | <0.03 MB | <0.2 MB | <10 MB | <50 MB | <50 MB | ✅ Trivial |
| 2 | <1M params | <1 MB | <0.25 MB | <2 MB | <50 MB | <100 MB | <100 MB | ✅ Trivial |
| 3 | ~100M params | ~100 MB | ~25 MB | ~200 MB | ~200 MB | ~500 MB | ~425 MB | ✅ Easy |
| 4-A | ~1B params | ~1 GB | ~250 MB | ~2 GB | ~1 GB | ~4 GB | ~3.25 GB | ✅ Fits |
| 4-B | ~7B params | ~7 GB | ~1.75 GB | ~14 GB | ~5 GB | ~26 GB | ~21 GB | ❌ Needs cloud |

> **Key insight:** PH-Neuro training on RTX 4060 can handle models ~4× larger than PH-Net because there's no optimizer, no gradient buffers, and no activation checkpointing. A 1B ternary model trains comfortably where PH-Net struggles.

### Memory Breakdown Comparison (1B model)

| Component | PH-Net (STE + AdamW) | PH-Neuro naive int8 | PH-Neuro packed 2-bit | Savings vs PH-Net (packed) |
|-----------|---------------------|--------------------|--------------------|----------------------------|
| Weights (fp32 / ternary) | 4 GB | 1 GB | 0.25 GB | 16× |
| Latent scores (fp16) | — | 2 GB | 2 GB | — |
| Optimizer states (AdamW) | 8 GB | 0 | 0 | ∞ |
| Gradients | 4 GB | 0 | 0 | ∞ |
| Activations (for backward) | ~2 GB | ~1 GB | ~1 GB | 2× |
| **Total** | **~18 GB** | **~4 GB** | **~3.25 GB** | **~5.5×** |

> **Weight packing**: 4 ternary weights per int8 byte (2 bits each: 00=0, 01=+1, 10=-1). Packing is transparent — `unpack()` converts to {-1,0,+1} int8 for MatMul, `pack()` stores compactly. Use naive int8 during development (Phases 0-2), switch to packed for scale (Phases 3-4).

> A 7B ternary model with packed weights (~1.75 GB weights + ~14 GB latent scores + ~5 GB activations ≈ ~21 GB) fits on a cloud RTX 4090 24 GB. With naive int8 it would need ~26 GB — only possible on A100.

---

## Success Criteria (Updated)

| Milestone | Target | Means of verification | Status |
|-----------|--------|----------------------|--------|
| **M0: Hebbian Core** | Ternary Hebbian MLP >85% MNIST | E001 | ✅ 88.4% |
| **M1: Multi-Head CL** | <5% forgetting on Split MNIST | continual.py | ✅ <5% (multi-head) |
| **M2: Hebbian Phase 2** | FF/NTH 2-layer >88% MNIST | E004-E008 | ❌ All <88% |
| **M3: STE TernaryLinear** | STE backward pass works, ternary invariant holds | Unit tests + E009 (V1) | ✅ 22/22 tests pass |
| **M4: Track A Baseline** | Ternary STE >95% MNIST, systematic comparison of 5 variants × 5 datasets | [`E009`](experiments/E009-ste-baseline-suite.md) | ✅ **98.17%** — beats Hebbian ceiling by 9.15 pp |
| **M5: Hysteresis-STE** | Hysteresis-STE ≥ standard STE accuracy + improved sparsity | Ablation study | 🟡 **CODE DONE** — layers, runner, tests all ✅. Full sweep pending |
| **M6: Track B EWC** | Ternary STE + EWC <10% forgetting on Split MNIST, >90% avg accuracy | Experiment log | ⬜ |
| **M7: Track B QLoRA** | Frozen ternary + LoRA achieves <1% forgetting with >85% accuracy | Experiment log | ⬜ |
| **M8: Paper 1** | Low-Memory Ternary Vision submitted | arXiv + workshop | ⬜ |
| **M9: Paper 2** | Ternary Continual Learning submitted | arXiv + conference | ⬜ |

## New References (2026 Landscape Scan)

1. **Ma, S. et al. (2024).** "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764.
2. **Wang, H., Ma, S., Wei, F. (2025).** "BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs." arXiv:2504.18415.
3. **Microsoft (2025).** "BitNet b1.58 2B4T Technical Report." arXiv:2504.12285. Model: huggingface.co/microsoft/bitnet-b1.58-2B-4T
4. **Wang, S. et al. (2026).** "CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs." arXiv:2606.26650. ICML 2026 Oral. Code: github.com/IntelChina-AI/BitTern
5. **Fermion Research (2026).** "Neutrino-8B." huggingface.co/FermionResearch/Neutrino-8B
6. **Zhang, M.S. et al. (2025).** "When Less is More: 8-bit Quantization Improves Continual Learning in Large Language Models." arXiv:2512.18934.
7. **Guan, H. et al. (2026).** "TOM: A Ternary Read-only Memory Accelerator for LLM-powered Edge Intelligence." arXiv:2602.20662.
8. **Xu, S. et al. (2026).** "VibeVoice-ASR-BitNet Technical Report." arXiv:2607.21075.
9. **Wang, J. et al. (2025).** "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs." arXiv:2502.11880. Code: github.com/microsoft/BitNet
10. **Wang, H., Ma, S., Wei, F. (2024).** "BitNet a4.8: 4-bit Activations for 1-bit LLMs." arXiv:2411.04965.

### Existing Key References (Hebbian Era)

11. **Hinton, G. (2022).** "The Forward-Forward Algorithm." arXiv:2212.13345.
12. **Frémaux, N. & Gerstner, W. (2016).** "Neuromodulated Spike-Timing-Dependent Plasticity." Frontiers in Neural Circuits, 9:85.
13. **Scellier, B. & Bengio, Y. (2017).** "Equilibrium Propagation." Frontiers in Computational Neuroscience, 11:24.
14. **Whittington, J.C.R. & Bogacz, R. (2017).** "An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network." Neural Computation, 29(5):1229–1262.
15. **Journé, A. et al. (2023).** "Hebbian Deep Learning Without Feedback." ICLR 2023.
16. **Lillicrap, T.P. et al. (2016).** "Random Synaptic Feedback Weights Support Error Backpropagation for Deep Learning." Nature Communications, 7:13276.
| **M3a: Sequence Learning** | Hebbian network learns n-gram transitions, Reber grammar, toy language (100 words, 5 rules) | Experiment log |
| **M3b: Predictive Hebbian** | Prediction error as teaching signal outperforms basic Hebbian on sequential tasks | Experiment log |
| **M3c: First LM** | 100M ternary Hebbian Transformer generates coherent paragraphs on TinyStories | Perplexity + human eval |
| **M4: Scale** | 1B ternary Hebbian model, competitive-ish with 1B backprop at 1/5th training cost | LM eval harness |
| **M5: Package** | `pip install ph-neuro` — Hebbian training, ternary inference, one command | PyPI + docs |

### Success is Relative — The Two-Dimensional Scorecard

For every benchmark, we report TWO numbers:

| Benchmark | Accuracy | Forgetting |
|-----------|----------|------------|
| Backprop baseline | 93% | 62% |
| PH-Neuro target | >65% | <5% |

**Success = we trade accuracy for unforgetfulness.** A paper that shows 70% accuracy but 2% forgetting is more interesting than 90% accuracy with 50% forgetting. The latter is just a worse backprop network.

---

## Phases

```
M0           M1-M1b          M2              M3a→M3b→M3c      M4              M5
[Core]   →   [Vision]    →   [Deep]      →   [Language]    →   [Scale]     →   [Ship]
 Mechanism    CNN proof       Multi-layer     Sequence →Pred  1B+ models      Package
 MNIST        CIFAR-10        Hierarchical    → TinyStories   Competitive     pip install
 + continual  + baselines     representations  Coherent text   + benchmarks     + docs
 ~1 week      ~2-3 weeks      ~3-4 weeks      ~6-8 weeks      ~2-3 months      ~1 week

All phases run on RTX 4060 8 GB except Phase 4-B (7B → cloud).
```

---

### Phase 0 — Core Hebbian-Ternary Mechanism

**Goal:** Build the fundamental building block: a `TernaryHebbianLinear` layer that stores ternary weights, maintains latent float scores, and updates via Hebbian rule with hysteresis. Prove it on MNIST in <1 hour.

**Duration:** ~1 week

#### 0.1 Ternary Weight Representation

- [x] `TernaryTensor`: storage of {-1, 0, +1} weights
  - **Phase 0-2 (dev)**: Naive int8 — 1 byte per weight, simple and debuggable
  - **Phase 3+ (prod)**: Packed 2-bit — 4 weights per int8 byte (00=0, 01=+1, 10=-1); pack/unpack transparent to Hebbian logic
- [x] `LatentScoreTensor`: fp16 scores paired with each weight, tracking cumulative Hebbian evidence
- [x] Conversion functions: `latent_to_ternary(scores, theta_upper, theta_lower)` with hysteresis
- [x] Weight initialization: all weights start at 0, latent scores at small random values near 0

#### 0.2 Hebbian Update Rule

- [x] Core rule: `Δlatent_score = lr × pre_activation × post_activation`
- [x] Since pre/post are ternary {-1, 0, +1}, the update is:
  - `+lr` when pre and post have same sign (both +1 or both -1) → "fire together, wire together"
  - `-lr` when pre and post have opposite signs → anti-correlation
  - `0` when either is 0 → no update (silent neuron)
- [x] Homeostatic decay: `Δlatent_score -= decay_rate × latent_score` (slow drift toward 0 for unused synapses)
- [x] Anti-Hebbian variant for output layer: `Δlatent_score = -lr × pre_activation × post_activation` for wrong-class neurons

#### 0.3 Hysteresis Threshold Mechanism

- [x] `θ_upper`: activation threshold (e.g., 5.0) — latent score must exceed this to flip 0 → ±1
- [x] `θ_lower`: deactivation threshold (e.g., 1.0) — latent score must fall below this to flip ±1 → 0
- [x] Hysteresis gap (θ_upper - θ_lower = 4.0) prevents oscillation
- [x] Once activated, a synapse is "sticky" — needs significant counter-evidence to deactivate
- [x] Configurable per-layer thresholds

#### 0.4 Forward Pass

- [x] `TernaryHebbianLinear.forward(x)`: MatMul using ternary weights + ternary activations
- [x] Implemented via popcount: `output = popcount(x AND w_pos) - popcount(x AND w_neg)`
- [x] PyTorch reference implementation first (using float MatMul for correctness), popcount optimization later
- [x] Activation function: `sign()` or `ternary_sign()` — maps to {-1, 0, +1}

#### 0.5 MNIST Sanity Check

- [x] Single `TernaryHebbianLinear` layer (784 → 10), no hidden layers (**88.4% accuracy**)
- [x] Train with Hebbian rule on output layer (WTA: strengthen correct, weaken wrong prediction)
- [x] Target: >85% accuracy in ≤10 epochs (achieved: 88.4% at epoch 10)
- [x] Verify: no `.backward()` called anywhere in training loop
- [x] Verify: weight distribution stays ternary throughout training
- [x] Verify: latent scores evolve smoothly, ternary weights flip occasionally at thresholds

> **Note**: The original target was >90%, but empirical testing showed a single ternary layer plateaus at ~88% — ~96% of the theoretical maximum (~92%) for any single linear layer on MNIST. Multi-layer networks (Phase 1) are expected to push past 90%.

#### 0.6 Unit Tests

- [x] `test_ternary_representation`: packing/unpacking correctness
- [x] `test_hebbian_update`: manual computation vs implementation
- [x] `test_hysteresis`: verify thresholds work, no oscillation for constant input
- [x] `test_no_backward`: verify `torch.autograd` is never engaged
- [x] `test_mnist_minimal`: end-to-end test, >85% accuracy

📄 Experiment report: [`E001-mnist-hebbian-baseline.md`](experiments/E001-mnist-hebbian-baseline.md)

---

### Phase 1 — Vision Proof-of-Concept

**Goal:** Show ternary Hebbian learning works on real vision tasks. Compare against float Hebbian (SoftHebb) and backprop baselines. **Demonstrate continual learning as the core differentiator.**

**Duration:** ~2-3 weeks

#### 1.1 MLP on MNIST ✅ Complete

- [x] 2-3 hidden layers, `TernaryHebbianLinear` throughout
- [x] Greedy layer-wise training: train layer 1, freeze, train layer 2, freeze, etc.
- [x] Output layer: WTA Hebbian with label supervision (from Phase 0)
- [x] **Result: 87.9%** (784→512→10, online competitive Hebbian + WTA)
- [x] 7 Hebbian variants tested for hidden layers — only competitive Hebbian works
- [x] **Key finding: Depth does not yet provide meaningful improvement** over single-layer (88.4%→87.9%). The ~88-89% range appears to be the practical limit for ternary Hebbian MLPs on MNIST.
- [x] Target adjusted: >85% (original >95% was unrealistic without global error signal)

📄 Experiment report: [`E002-mnist-multilayer-mlp.md`](experiments/E002-mnist-multilayer-mlp.md)

#### 1.2 CNN on CIFAR-10

- [ ] `TernaryHebbianConv2d`: Hebbian rule applied per filter (local receptive field × local activation)
- [ ] Architecture: 2-3 conv layers + 1-2 linear layers
- [ ] Hebbian rule for convolutions: `ΔW[h,w] = lr × input_patch[h,w] × output_neuron`
- [ ] Target: >55% accuracy (adjusted from >60% — MLP findings suggest ternary Hebbian operates ~10-15pp below backprop)
- [ ] Baseline: SoftHebb float at 80.3%, backprop at ~88%, random ternary ~10%
- [ ] CNN has natural advantages over MLP for Hebbian: locality matches receptive fields, translation invariance is built-in

#### 1.3 Continual Learning — THE Key Experiment

- [ ] **Split MNIST**: 5 binary classification tasks (0vs1, 2vs3, 4vs5, 6vs7, 8vs9) presented sequentially
  - Train on task 1, test on task 1. Train on task 2, test on tasks 1 AND 2. etc.
  - Target: <5% average forgetting (drop in task 1 accuracy after learning task 5)
  - Backprop baseline: >40% forgetting (typical for MLP without replay)
- [ ] **Permuted MNIST**: Same digits, but pixels are randomly permuted for each task
  - 5-10 sequential tasks, each with different permutation
  - Tests whether the network can learn multiple unrelated mappings
- [ ] **Why this matters**: This is the experiment that makes PH-Neuro publishable. Nobody has shown continual learning with ternary Hebbian networks.

#### 1.4 Baselines & Comparisons

- [ ] **PH-Neuro (ternary Hebbian)**: Our method
- [ ] **Float Hebbian**: Same architecture, same Hebbian rule, but float weights (no ternary constraint)
- [ ] **SoftHebb reproduction**: Implement SoftHebb's soft-WTA + float Hebbian for direct comparison
- [ ] **Backprop (PH-Net style)**: Same architecture with STE + AdamW
- [ ] **Random ternary baseline**: Random ternary weights, no learning (lower bound)

#### 1.5 Analysis Tools

- [ ] Weight distribution histograms over time (ternary: % +1, % 0, % -1)
- [ ] Latent score trajectories for individual synapses
- [ ] Activation sparsity (% of neurons outputting 0)
- [ ] Forgetting curves for continual learning experiments
- [ ] Confusion matrices per task in continual learning

📄 See: [`phase-1-vision-poc.md`](phase-1-vision-poc.md)

---

### Phase 2 — Multi-Layer & Hierarchical Representations

**Goal:** Show that greedy layer-wise Hebbian learning builds useful hierarchical features. Explore what happens when we stack many layers.

**Duration:** ~3-4 weeks

#### 2.1 Deep(er) Networks

- [ ] 5-10 layer Hebbian CNNs on CIFAR-10
- [ ] Does accuracy improve with depth? Or does Hebbian learning saturate?
- [ ] Compare: 1-layer vs 3-layer vs 5-layer vs 10-layer
- [ ] Hypothesis: Hebbian learning benefits less from depth than backprop (no global error signal to coordinate layers)

#### 2.2 Feature Visualization

- [ ] Visualize what each layer learns (filter visualization, activation maximization)
- [ ] Are the learned features interpretable? (Gabor-like edges in early layers?)
- [ ] Compare Hebbian-learned filters vs backprop-learned filters
- [ ] Hypothesis: Hebbian features are more "generic" and less task-specific

#### 2.3 Alternative Multi-Layer Strategies

- [ ] **Greedy layer-wise** (baseline, from Phase 1): train layer N, freeze, train N+1
- [ ] **Simultaneous Hebbian**: all layers update simultaneously (no freezing)
  - Hidden layers use their own output as "post" (self-organizing)
  - Danger: layers may all learn the same thing
- [ ] **Contrastive Hebbian** (Forward-Forward inspired):
  - Present positive example → Hebbian update
  - Present negative example → anti-Hebbian update
  - Layer-wise goodness function determines positive vs negative
- [ ] **Difference Target Propagation** (lightweight):
  - Instead of gradients, propagate target activations backward
  - Each layer tries to match the target from the layer above
  - Still local, but provides a "teaching signal" to hidden layers

#### 2.4 Continual Learning at Depth

- [ ] Does depth help or hurt continual learning?
- [ ] Split CIFAR-10 (5 tasks × 2 classes)
- [ ] Compare forgetting rates: 1-layer vs 5-layer vs 10-layer
- [ ] Hypothesis: Deeper networks may suffer MORE forgetting because early layers drift, affecting all subsequent layers

#### 2.5 Weight Dynamics Deep Dive

- [ ] Track "synapse lifetime": how long does a -1, 0, or +1 weight persist?
- [ ] Track "critical periods": do early training steps have disproportionate impact?
- [ ] Measure "weight entropy": how diverse are the learned weight patterns?
- [ ] Correlation between weight stability and task performance

📄 See: [`phase-2-multi-layer.md`](phase-2-multi-layer.md)

---

### Phase 3 — Language: The Brain-Inspired Approach

**Goal:** Show ternary Hebbian networks can learn sequential structure and language — not by mimicking Transformers, but by mimicking the brain's architecture for language.

**Duration:** ~6-8 weeks (3a: 1-2 weeks, 3b: 2-3 weeks, 3c: 3-4 weeks)

> **Key insight:** The brain doesn't use a monolithic Transformer. It has specialized modules (Broca, Wernicke, hippocampus), hierarchical timescales (phoneme → syllable → word → phrase), and working memory (echo state). Phase 3 is NOT "put a Transformer with Hebbian" — it's "build a brain-inspired language architecture."

---

#### Phase 3a — Sequence Learning (SANITY CHECK)

**Goal:** Before attempting natural language, prove the Hebbian mechanism can learn sequential structure at all.

- [ ] **n-gram prediction**: Train on synthetic sequences where P(next|context) follows a known distribution
  - Can the network learn "after A comes B with 80% probability"?
  - Baseline: count-based n-gram model
- [ ] **Reber grammar**: Artificial grammar with recursive rules (finite-state automaton)
  - Generate valid/invalid strings from the grammar
  - Can Hebbian learning discover the underlying rules?
  - This tests whether Hebbian captures abstract structure, not just surface correlations
- [ ] **Toy language**: 100 words, 5 grammar rules (SVO order, adjective-noun agreement, etc.)
  - Generate sentences from a small formal grammar
  - Train on sequences, evaluate on held-out grammatical sentences
  - Perfect debugging environment — small enough to trace every weight

**Success criteria:**
- n-gram: matches count-based baseline within 10%
- Reber grammar: >90% accuracy distinguishing valid from invalid strings
- Toy language: generates grammatically correct sentences >80% of the time

---

#### Phase 3b — Predictive Hebbian (THE MECHANISM)

**Goal:** Implement predictive coding as the learning mechanism. This is the brain's actual algorithm — not "fire together wire together" but "minimize prediction error."

**Why this matters:** The brain constantly predicts its next input. When prediction matches reality → strengthen (LTP). When prediction fails → correct (LTD via prediction error). This is fundamentally different from basic Hebbian and is essential for sequential learning.

```python
class PredictiveHebbianLayer(nn.Module):
    """
    Brain-inspired predictive learning:
    1. Receive current input
    2. PREDICT next input (forward prediction)
    3. Compare prediction with actual next input
    4. Hebbian update based on PREDICTION ERROR
    """
    def forward(self, x_current):
        return self.predict(x_current)  # What comes next?
    
    def learn(self, x_current, x_next, lr):
        predicted = self.predict(x_current)
        error = x_next - predicted  # Prediction error
        
        # Hebbian: "these inputs led to this error → fix the weights"
        # Positive error → strengthen (under-predicted)
        # Negative error → weaken (over-predicted)
        self.hebbian_update(x_current, error, lr)
```

**Components to build:**

- [ ] **Working Memory (Echo State)**: Leaky integration of past inputs
  - `state[t] = decay × state[t-1] + (1-decay) × input[t]`
  - Provides temporal context without backprop-through-time
  - Different decay rates for different temporal scales
- [ ] **Predictive Hebbian Update**: `Δw = lr × pre_current × (post_actual - post_predicted)`
  - NOT just correlation — it's correlation CONDITIONAL on prediction error
  - This is Rao & Ballard's predictive coding, adapted for ternary weights
- [ ] **Hierarchical Timescales**: Each layer has its own temporal integration window
  - Layer 1 (fast): decay=0.5 — phoneme-level patterns (~50ms)
  - Layer 2 (medium): decay=0.9 — word-level patterns (~500ms)
  - Layer 3 (slow): decay=0.99 — phrase-level patterns (~2s)
  - Higher layers "see" longer context
- [ ] **Layer-wise Prediction**: Each layer predicts the NEXT layer's output (not the next token directly)
  - Layer L predicts what Layer L+1 will output
  - This creates a hierarchy of predictions at increasing abstraction levels

**Experiment: Compare basic Hebbian vs predictive Hebbian on sequence tasks (from 3a)**

| Method | n-gram accuracy | Reber grammar | Toy language grammar |
|--------|----------------|---------------|---------------------|
| Basic Hebbian | ? | ? | ? |
| Predictive Hebbian | ? (expected: better) | ? (expected: better) | ? (expected: much better) |
| Baseline (n-gram / FSA) | 100% | 100% | 100% |

---

#### Phase 3c — TinyStories (NATURAL LANGUAGE)

**Goal:** Apply the predictive Hebbian architecture to real natural language. Generate coherent paragraphs.

**Only proceed after 3a and 3b succeed.**

- [ ] **Brain-inspired architecture** (not a monolithic Transformer):

| Module | Brain Region | Function | Implementation |
|--------|-------------|----------|----------------|
| Encoder | Wernicke's area | Input → meaning | Predictive Hebbian layers (fast timescale) |
| Latent Memory | Hippocampus | Episodic context | Echo state with slow decay (decay=0.99) |
| Decoder | Broca's area | Meaning → output | Predictive Hebbian layers (medium timescale) |
| Attention | Prefrontal cortex | Task focus | Ternary Hebbian attention (learned relevance) |

- [ ] **Training strategy**: Predictive Hebbian at every layer
  - Input: current token
  - Each layer predicts the next layer's output
  - Final layer predicts the next token
  - Prediction error drives Hebbian updates at ALL layers (local, not backpropagated)
- [ ] 100M total parameters across all modules
- [ ] Train on TinyStories (~2M children's stories, ~500M tokens)
- [ ] **Evaluation**:
  - Perplexity (expect higher than backprop, but << random — the brain doesn't minimize perplexity, it minimizes prediction error)
  - Generation quality: human evaluation of 100 generated paragraphs
  - Coherence: do the stories have a beginning, middle, end?
  - Grammar: does the model learn basic English syntax?
  - Diversity: do different prompts produce different stories?
- [ ] **Analysis**:
  - What do ternary embeddings look like? (t-SNE)
  - Do different layers learn different linguistic features?
  - Does the working memory actually capture long-range dependencies?
  - Weight sparsity patterns across brain-inspired modules

**Success criteria:**
- Generates paragraphs a human can read without cringing (quality ≥3/5)
- Demonstrates basic grammar (subject-verb agreement, word order)
- Different modules show functional specialization (encoder vs decoder vs memory)

📄 See: [`phase-3-language-model.md`](phase-3-language-model.md)

---

### Phase 4 — Scale & Advanced Features

---

### Phase 4 — Scale & Advanced Features

**Goal:** Push to 1B+ parameters. Explore architectural innovations specific to Hebbian learning. Measure scaling properties.

**Duration:** ~2-3 months

#### 4.1 Scale to 1B Parameters

- [ ] 1B ternary Hebbian Transformer
- [ ] Training on SlimPajama or FineWeb Edu subset
- [ ] Memory: ~3.4 GB — fits easily on RTX 4060 8 GB
- [ ] Training speed: popcount MatMul should be fast, but Hebbian update is O(n) per weight
- [ ] Optimize Hebbian update: vectorized, fused kernel if needed
- [ ] Measure: tokens/second training throughput

#### 4.2 Mixture of Experts (MoE)

- [ ] Ternary Hebbian MoE: experts are ternary Hebbian networks
- [ ] Router: can it be ternary Hebbian too? Or does routing need float precision?
- [ ] Hebbian expert selection: experts that "fire" get updated, inactive experts don't
- [ ] Natural load balancing: Hebbian learning self-organizes expert specialization?

#### 4.3 Advanced Hebbian Rules

- [ ] **BCM (Bienenstock-Cooper-Munro)**: sliding threshold for LTP/LTD based on average activity
  - `Δw = pre × post × (post - θ_M)` where θ_M adapts to average activity
  - More biologically plausible, may create better feature selectivity
- [ ] **Oja's rule**: normalized Hebbian (prevents weight explosion)
  - `Δw = lr × (pre × post - α × w × post²)`
- [ ] **Spike-Timing-Dependent Plasticity (STDP)**:
  - Not just correlation but causal timing: pre-before-post → LTP, post-before-pre → LTD
  - Challenging for rate-based networks, but could be simulated
- [ ] Ablation: which Hebbian variant works best with ternary weights?

#### 4.4 Continual Learning at Scale

- [ ] Task-incremental language learning: train on English → add French → add code
- [ ] Does the model retain English after learning French?
- [ ] Domain-incremental: Wikipedia → GitHub → PubMed
- [ ] This is where PH-Neuro should DESTROY backprop
- [ ] Measure: perplexity on old tasks after learning new tasks

#### 4.5 Interpretability & Safety

- [ ] Ternary weights are inherently interpretable: each connection is +1, -1, or 0
- [ ] Can we trace "circuits" in a ternary Hebbian network?
- [ ] Weight surgery: manually set weights to 0 to "forget" specific patterns
- [ ] This could be a unique safety advantage: auditable, editable weights

📄 See: [`phase-4-scale.md`](phase-4-scale.md)

---

### Phase 5 — Package & Publish

**Goal:** Make PH-Neuro usable by others. `pip install ph-neuro`. Write the paper.

**Duration:** ~2-3 weeks

#### 5.1 Package

- [ ] `pip install ph-neuro` with clean API
- [ ] `TernaryHebbianLinear`, `TernaryHebbianConv2d`, `TernaryHebbianTransformer`
- [ ] Simple training loop: `model.fit_hebbian(dataloader, epochs)`
- [ ] Pre-trained model zoo (MNIST, CIFAR-10, TinyStories checkpoints)
- [ ] Documentation: quickstart, API reference, examples

#### 5.2 Paper

- [ ] Core contribution: **First ternary Hebbian deep learning framework**
- [ ] Key results:
  - Vision: ternary Hebbian achieves X% of float Hebbian accuracy with 50× less memory
  - Continual learning: <5% forgetting vs >40% for backprop
  - Language: first Hebbian language model, coherent generation at 100M scale
  - Efficiency: 6.5× fewer FLOPs, 50× less training memory than backprop
- [ ] Target venue: NeurIPS, ICLR, or ICML (depending on results)
- [ ] ArXiv preprint once Phase 3 results are solid

#### 5.3 Future Horizons

- [ ] **Neuromorphic hardware deployment**: Intel Loihi, SpiNNaker
- [ ] **WebAssembly demo**: pure JS inference in browser
- [ ] **Federated Hebbian learning**: edge devices learn locally, merge ternary weights
- [ ] **Hebbian fine-tuning**: take a pre-trained model, continue Hebbian learning on new data
- [ ] **Liquid PH-Neuro**: Hebbian learning with continuous weight dynamics (no discrete steps)

📄 See: [`phase-5-package-publish.md`](phase-5-package-publish.md)

---

## Experiment Tracking

Each training run gets a log file in `docs/experiments/`:

```
docs/experiments/
├── 001-core-mlp-mnist.md
├── 002-core-cnn-mnist.md
├── 003-split-mnist-continual.md
├── 004-permuted-mnist-continual.md
├── 005-cnn-cifar10-baseline.md
├── 006-cnn-cifar10-multilayer.md
├── 007-float-hebbian-baseline.md
├── 008-softhebb-reproduction.md
├── 009-ste-baseline-suite.md  ✅ L1 COMPLETE
├── 010-ngram-hebbian-prediction.md
├── 011-reber-grammar-hebbian.md
├── 012-toy-language-hebbian.md
├── 013-predictive-vs-basic-hebbian.md
├── 014-echo-state-memory-ablation.md
├── 015-100m-lm-tinystories.md
├── 016-1b-lm-pretrain.md
└── ...
```

### Experiment Log Format

```markdown
# Experiment NNN: [Title]

- **Date:** YYYY-MM-DD
- **Git commit:** `abc1234`
- **Status:** [running | completed | failed | abandoned]

## Configuration
| Parameter | Value |
|-----------|-------|
| Architecture | e.g., "3-layer CNN, 64-128-256 filters" |
| Weight init | e.g., "All zeros, latent scores ~ Uniform(-0.1, 0.1)" |
| θ_upper / θ_lower | e.g., "5.0 / 1.0" |
| Learning rate | e.g., "0.001" |
| Decay rate | e.g., "1e-5" |
| Hebbian variant | e.g., "Basic (pre × post)" |
| Batch size | e.g., "128" |
| Epochs | e.g., "50" |
| Dataset | e.g., "CIFAR-10 (50K train, 10K test)" |
| Hardware | e.g., "RTX 4060 8 GB" |
| Training time | e.g., "12 min" |

## Results
| Metric | Value | Baseline (backprop) | Baseline (float Hebbian) |
|--------|-------|---------------------|--------------------------|
| Accuracy | XX.X% | XX.X% | XX.X% |
| Forgetting (if continual) | X.X% | XX.X% | — |
| Weight sparsity (% 0) | XX.X% | — | — |

## Observations
- What worked well?
- What failed?
- Surprising findings?

## Bugs / Issues
- [ ] Bug description and resolution

## Next Steps
- What to try next based on these results?
```

---

## Open Questions

### Core Mechanism
1. **Optimal θ_upper and θ_lower**: What hysteresis gap works best? Too wide → no learning. Too narrow → oscillation.
2. **Learning rate for ternary Hebbian**: The update is ±lr (discrete). How does lr interact with the threshold gap? Is there a "natural" lr = (θ_upper - θ_lower) / expected_steps_to_activate?
3. **Homeostatic decay rate**: How fast should unused synapses decay? Too fast → forgets useful connections. Too slow → dead weights accumulate.
4. **Weight initialization**: All zeros? Small random ternary? Random latent scores near zero? Does initialization matter given the hysteresis mechanism?

### Multi-Layer
5. **Depth scaling for Hebbian**: Does adding more layers help beyond a certain point? Is there a "Hebbian depth limit"?
6. **Layer-wise information bottleneck**: Each layer can only pass ternary activations to the next. How much information is lost per layer?
7. **Coordinated learning**: Can hidden layers receive a useful signal without backprop? Is greedy layer-wise inherently limited?

### Language & Sequence Learning
8. **Predictive coding vs basic Hebbian**: Does prediction error as a teaching signal outperform simple co-occurrence Hebbian for sequential tasks? By how much?
9. **Working memory architecture**: What decay rate(s) work best? Single rate vs hierarchical? Does echo state memory capture enough context for language?
10. **Hierarchical timescales**: Do different layers naturally learn different temporal patterns when given different decay rates? Does this create a meaningful hierarchy?
11. **Modular vs monolithic**: Does a brain-inspired modular architecture (encoder + memory + decoder) outperform a monolithic Hebbian Transformer? Is functional specialization emergent or designed?
12. **Softmax with ternary activations**: Attention scores are integer-valued (ternary Q · ternary K). Is softmax appropriate? Would a hard winner-take-all work better?
13. **Next-token prediction without cross-entropy**: Predictive Hebbian minimizes prediction error, not perplexity. Is next-token prediction fundamentally compatible with Hebbian learning?
14. **Sequence-level Hebbian**: Should Hebbian updates happen per-token or per-sequence? Per-token is more local but noisier.
15. **Toy language transfer**: Do insights from the toy language (100 words, 5 rules) transfer to natural language? Or is there a qualitative gap?

### Continual Learning
12. **Capacity limits**: How many tasks can a ternary Hebbian network learn before saturating? Is there a theoretical limit based on number of synapses?
13. **Task interference**: Even with ternary weights, can learning task B "use up" synapses needed for task A? This is different from forgetting — it's resource competition.
14. **Forward transfer**: Does learning task A help learn related task B faster? This would be a positive Hebbian prior.

### Scaling
15. **Hebbian scaling laws**: Does Hebbian learning obey the same compute-optimal scaling as backprop? Or does it have different scaling properties?
16. **Sparsity at scale**: Does weight sparsity (fraction of zeros) increase or decrease with model size?
17. **MoE and Hebbian**: Does the Hebbian rule naturally produce expert specialization? Or does it all collapse to one general expert?

### Theory
18. **What is the objective function?**: Hebbian learning doesn't minimize a loss — what does it optimize? Mutual information? Correlation? Something else?
19. **Convergence guarantees**: Under what conditions does ternary Hebbian learning converge? To what?
20. **Capacity of ternary Hebbian networks**: What's the VC dimension or expressiveness of a ternary Hebbian network compared to a float backprop network of the same architecture?

---

## Key References

### Ternary / Binary Networks
| Paper / Project | Relevance |
|-----------------|-----------|
| **BitNet b1.58** (Wang et al., 2024) | Ternary weights + activations at 3B scale — proves ternary is viable |
| **BitNet** (Wang et al., 2023) | Original binary BitNet |
| **The Era of 1-bit LLMs** (Ma et al., 2024) | Survey of 1-bit LLM approaches |
| **`microsoft/BitNet`** (bitnet.cpp) | CPU/GPU inference engine for ternary models — MIT licensed |
| **XNOR-Net** (Rastegari et al., 2016) | Binary weight networks with scaling factors |
| **TWN** (Li et al., 2016) | Ternary weight networks (ternary weights, float activations) |
| **TBN** (Wan et al., 2018) | Ternary-Binary Networks — both weights and activations quantized |
| **ProxQuant** (Bai et al., 2019) | Proximal quantization for ternary networks |

### Hebbian Learning
| Paper / Project | Relevance |
|-----------------|-----------|
| **SoftHebb** (Journé et al., ICLR 2023) | **Primary reference.** State-of-the-art Hebbian deep learning. Float weights, soft-WTA, layer-wise. 80.3% CIFAR-10, 27.3% ImageNet (top-25%). Proves Hebbian can work at scale. |
| **The Hebb Rule for Synaptic Plasticity** (Hebb, 1949) | Original Hebbian postulate: "cells that fire together wire together" |
| **Theory of Hebbian Learning** (Gerstner & Kistler, 2002) | Mathematical framework for spike-based Hebbian learning |
| **BCM Theory** (Bienenstock, Cooper, Munro, 1982) | Sliding threshold plasticity — more biologically realistic |
| **Oja's Rule** (Oja, 1982) | Normalized Hebbian rule, extracts first principal component |
| **STDP** (Bi & Poo, 1998; Markram et al., 1997) | Spike-timing-dependent plasticity — temporal Hebbian learning |

### Hebbian + Deep Learning (Alternative to Backprop)
| Paper / Project | Relevance |
|-----------------|-----------|
| **Forward-Forward** (Hinton, 2022) | Layer-wise contrastive learning without backprop. Goodness function per layer. |
| **PEPITA** (Dellaferrera et al., 2022) | Forward-only learning with random feedback — competitive with backprop on small tasks |
| **Feedback Alignment** (Lillicrap et al., 2016) | Random feedback weights work for learning — challenges backprop's necessity |
| **Direct Feedback Alignment** (Nøkland, 2016) | Error signal goes directly to each layer, bypassing chain rule |
| **Difference Target Propagation** (Lee et al., 2015) | Targets instead of gradients propagated backward |
| **Equilibrium Propagation** (Scellier & Bengio, 2017) | Energy-based learning with two phases (free and clamped) |
| **Local Representation Alignment** (Ororbia & Mali, 2023) | Local learning rules that approximate backprop |
| **Signal Propagation** (Kohan et al., 2023) | Forward-only learning via signal propagation, not gradients |

### Predictive Coding & Brain-Inspired Architecture
| Paper / Project | Relevance |
|-----------------|-----------|
| **Predictive Coding in the Visual Cortex** (Rao & Ballard, 1999) | Original predictive coding formulation — cortex minimizes prediction error hierarchically |
| **The Free-Energy Principle** (Friston, 2010) | Unified brain theory — perception and learning as prediction error minimization |
| **Predictive Coding for Deep Learning** (Whittington & Bogacz, 2017) | Shows predictive coding approximates backprop on MNIST/CIFAR |
| **Predictive Coding Networks for Video** (Lotter et al., 2017) | Predictive coding for temporal sequences — video prediction without backprop |
| **Echo State Networks** (Jaeger, 2001) | Reservoir computing — fixed recurrent network with learned readout, temporal memory without BPTT |
| **Liquid State Machines** (Maass et al., 2002) | Continuous-time reservoir computing — biologically realistic temporal processing |
| **Reservoir Computing Survey** (Lukoševičius & Jaeger, 2009) | Comprehensive survey of echo state networks for temporal pattern recognition |
| **Synfire Chains** (Abeles, 1991) | Sequences of synchronously firing neuron groups — biological mechanism for sequence memory |
| **The Brain's Language System** (Friederici, 2011) | Broca, Wernicke, arcuate fasciculus — modular language architecture in the brain |
| **Hierarchical Processing in Cortex** (Felleman & Van Essen, 1991) | Distributed hierarchical processing — inspiration for layered Hebbian learning |
| **Theta-Gamma Neural Code** (Lisman & Jensen, 2013) | Cross-frequency coupling for sequence encoding — biological timing mechanism for language |

### Continual Learning
| Paper / Project | Relevance |
|-----------------|-----------|
| **Catastrophic Forgetting** (McCloskey & Cohen, 1989; Ratcliff, 1990) | The problem PH-Neuro aims to solve |
| **EWC** (Kirkpatrick et al., 2017) | Elastic Weight Consolidation — backprop solution to forgetting |
| **SI** (Zenke et al., 2017) | Synaptic Intelligence — importance-weighted regularization |
| **Continual Learning with Hebbian Plasticity** (Various) | Sparse literature on Hebbian approaches to continual learning |
| **GEM** (Lopez-Paz & Ranzato, 2017) | Gradient Episodic Memory — replay-based approach |

### Neurobiology of Plasticity
| Paper / Project | Relevance |
|-----------------|-----------|
| **Synaptic Plasticity** (Citri & Malenka, 2008) | Review of LTP/LTD mechanisms |
| **Homeostatic Plasticity** (Turrigiano, 2011) | Mechanisms that stabilize neural activity |
| **Dendritic Computation** (London & Häusser, 2005) | Local computation in dendrites — inspiration for local learning |
| **Synaptic Scaling** (Turrigiano et al., 1998) | Global homeostatic regulation of synaptic strength |

### Sparsity & Topology
| Paper / Project | Relevance |
|-----------------|-----------|
| **SET** (Mocanu et al., 2018) | Sparse evolutionary training — connections grow and prune |
| **RigL** (Evci et al., 2020) | Gradient-based growth for sparse training |
| **Lottery Ticket Hypothesis** (Frankle & Carbin, 2019) | Sparse subnetworks can train from scratch |
| **The Sparse Manifold Transform** (Chen et al., 2022) | Sparse coding meets deep learning |

### Sibling Project
| Project | Relevance |
|---------|-----------|
| **PH-Net** (this org) | Ternary weights with STE + backprop. Trains ternary LLMs for bitnet.cpp inference. PH-Neuro explores the backprop-free path. |

---

## Relationship to PH-Net

```
                    ┌──────────────────────────────────┐
                    │     Ternary Weights {-1, 0, +1}  │
                    │     (Shared Philosophy)           │
                    └────────────┬─────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                      │
    ┌─────────▼──────────┐              ┌────────────▼───────────┐
    │      PH-Net        │              │       PH-Neuro         │
    │  STE + Backprop    │              │  Hebbian (no backprop) │
    │  Float latent w    │              │  Ternary native w      │
    │  AdamW optimizer   │              │  No optimizer          │
    │  bitnet.cpp export │              │  Pure popcount inf.    │
    │  LLM focus         │              │  Brain-inspired        │
    │  Production-ready   │              │  Research/experimental │
    └────────────────────┘              └────────────────────────┘
              │                                      │
              ▼                                      ▼
    Train ternary LLMs                     Explore alternative
    with proven methods                    learning paradigms
    (backprop + STE)                       (Hebbian + ternary)
```

**PH-Net** asks: "Can we train ternary LLMs efficiently using known methods?" → Yes, STE works.

**PH-Neuro** asks: "Can we abandon backprop entirely? What do we gain and lose?" → This is the research question.

The two projects share:
- Ternary weight representation {-1, 0, +1}
- Popcount-based inference
- The philosophy that ternary is the right abstraction for efficient neural computation
- Exploration of the ternary weight space

They differ fundamentally in their approach to learning — PH-Net uses the proven path (gradient descent), PH-Neuro explores the radical alternative (Hebbian plasticity).

---

## Appendix: Quick Start

```python
import torch
from ph_neuro import TernaryHebbianLinear, HebbianTrainer

# Define model
model = torch.nn.Sequential(
    TernaryHebbianLinear(784, 256, theta_upper=5.0, theta_lower=1.0),
    torch.nn.Sign(),  # ternary activation
    TernaryHebbianLinear(256, 10, theta_upper=5.0, theta_lower=1.0),
)

# Train without backprop
trainer = HebbianTrainer(model, lr=0.001, decay=1e-5)
trainer.fit(train_loader, epochs=10)

# No .backward() was called. No optimizer. No loss function.
```

---

> _"The brain does not compute gradients. It learns by association. PH-Neuro explores whether machines can too."_
