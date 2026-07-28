# PH-Neuro Roadmap

> **Status:** Pre-alpha research — unexplored territory  
> **Last updated:** 2026-07-28

---

## Vision

Build a **deep learning framework that learns without backpropagation** — using ternary weights {-1, 0, +1} and Hebbian plasticity. The weights are biologically inspired: excitatory (+1), inhibitory (-1), or absent (0). Learning is **local, continuous, and brain-like**: each synapse updates based solely on the activity of the two neurons it connects.

**PH-Neuro learns. PH-Net trains.** PH-Net uses STE + backprop to produce ternary LLMs. PH-Neuro explores the radical question: _can we abandon backpropagation entirely and still build useful neural networks?_

### Why This Matters

| Property | Backprop (PH-Net, GPT, etc.) | PH-Neuro (Hebbian) |
|----------|------------------------------|---------------------|
| Learning signal | Global (loss gradient from output) | Local (pre × post activity) |
| Backward pass | Required (≈2× forward FLOPs) | None |
| Optimizer states | AdamW: 2× model size in memory | None (no optimizer) |
| Activation storage | Full graph for backward pass | Not needed |
| Catastrophic forgetting | Severe (requires replay/memory) | Inherently resistant |
| Online/continuous learning | Difficult | Natural |
| Weight drift | Float weights drift continuously | Ternary weights are stable |
| Compute | Float MatMul + gradient computation | Ternary MatMul (popcount) |
| Memory (training) | 4-8× model size | ~1× model size |

### The Core Hypotheses

1. **H1 — Ternary Hebbian works at all**: Ternary weights {-1, 0, +1} combined with Hebbian learning can solve non-trivial classification tasks (MNIST >95%, CIFAR-10 competitive with float Hebbian).

2. **H2 — No catastrophic forgetting**: Because weights are discrete and Hebbian updates are local, learning new tasks does not overwrite old knowledge. Target: <5% forgetting across 10 sequential tasks, vs >60% for backprop.

3. **H3 — Hysteresis creates stability**: A dual-threshold mechanism (high threshold to activate a synapse, low threshold to deactivate) prevents oscillatory "flipping" and creates stable representations.

4. **H4 — Layer-wise independence is sufficient**: Greedy layer-wise Hebbian learning (each layer trained independently, bottom-up) can build useful hierarchical representations without any backward signal.

5. **H5 — Language is learnable without backprop**: Hebbian learning can capture the statistical regularities in text sufficient for coherent generation, even if perplexity is higher than backprop-trained models.

### What This Is NOT

- **NOT a replacement for backprop** in high-accuracy regimes. We expect lower raw accuracy.
- **NOT trying to beat GPT-4**. This is fundamental research into alternative learning paradigms.
- **NOT a quantization technique**. The weights are natively ternary, not quantized from floats.
- **NOT biologically detailed**. We borrow the Hebbian principle, not detailed synaptic dynamics (no STDP, no calcium models, no dopamine modulation — yet).

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
| Training memory | ~1× model size | No optimizer states, no gradient buffers, no activation checkpointing needed; a 7B ternary model fits in 8 GB VRAM for training |
| Architecture | Agnostic (MLP, CNN, Transformer, MoE) | Not locked to any single architecture; Hebbian rule is architecture-agnostic |
| Evaluation philosophy | Accuracy vs backprop + forgetting vs backprop | Two-dimensional evaluation: static accuracy (expected loss) and continual learning (expected win) |

---

## Hardware Constraints

**Current machine:** RTX 4060 8 GB VRAM, i7-14700K (20 cores), 16 GB RAM, 1 TB SSD.

Unlike PH-Net where 8 GB VRAM is a severe bottleneck (AdamW states alone are 2× model size), PH-Neuro's memory footprint is dramatically smaller:

| Phase | Model size | Ternary weights | Latent scores (fp16) | Activations | Total (training) | Fits 8 GB? |
|-------|-----------|----------------|---------------------|-------------|------------------|-------------|
| 1 | <100K params | <0.4 MB | <0.2 MB | <10 MB | <50 MB | ✅ Trivial |
| 2 | <1M params | <4 MB | <2 MB | <50 MB | <200 MB | ✅ Trivial |
| 3 | ~100M params | ~400 MB | ~200 MB | ~200 MB | ~1 GB | ✅ Easy |
| 4-A | ~1B params | ~4 GB | ~2 GB | ~1 GB | ~7 GB | ✅ Fits |
| 4-B | ~7B params | ~28 GB | ~14 GB | ~5 GB | ~47 GB | ❌ Needs cloud |

> **Key insight:** PH-Neuro training on RTX 4060 can handle models ~4× larger than PH-Net because there's no optimizer, no gradient buffers, and no activation checkpointing. A 1B ternary model trains comfortably where PH-Net struggles.

### Memory Breakdown Comparison (1B model)

| Component | PH-Net (STE + AdamW) | PH-Neuro (Hebbian) | Savings |
|-----------|---------------------|-------------------|---------|
| Weights (fp32 latent / ternary) | 4 GB | ~0.4 GB (ternary) | 10× |
| Latent scores (fp16) | — | 2 GB | — |
| Optimizer states (AdamW) | 8 GB | 0 | ∞ |
| Gradients | 4 GB | 0 | ∞ |
| Activations (stored for backward) | ~2 GB | ~1 GB (forward only) | 2× |
| **Total** | **~18 GB** | **~3.4 GB** | **~5×** |

> A 7B ternary model (~28 GB weights + ~14 GB latent scores = ~42 GB total) requires cloud GPU. But a 3B model (~12 GB + ~6 GB = ~18 GB) could fit with CPU offload of latent scores.

---

## Success Criteria

| Milestone | Target | Means of verification |
|-----------|--------|----------------------|
| **M0: Core Mechanism** | Ternary Hebbian MLP >95% MNIST (5 epochs) | `tests/test_core.py` |
| **M1: CNN Vision** | Ternary Hebbian CNN >60% CIFAR-10 | Experiment log |
| **M1b: Continual Learning** | <5% forgetting on split MNIST (5 tasks), backprop baseline >40% forgetting | Experiment log + comparison table |
| **M2: Multi-Layer** | 3-layer Hebbian CNN >65% CIFAR-10 (improvement over 1-layer) | Experiment log |
| **M3: First LM** | 100M ternary Hebbian Transformer generates coherent paragraphs on TinyStories | Perplexity + human eval |
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
M0           M1-M1b          M2              M3              M4              M5
[Core]   →   [Vision]    →   [Deep]      →   [Language]  →   [Scale]     →   [Ship]
 Mechanism    CNN proof       Multi-layer     TinyStories      1B+ models      Package
 MNIST        CIFAR-10        Hierarchical    Coherent text    Competitive     pip install
 + continual  + baselines     representations  + perplexity     + benchmarks     + docs
 ~1 week      ~2-3 weeks      ~3-4 weeks      ~4-6 weeks       ~2-3 months      ~1 week

All phases run on RTX 4060 8 GB except Phase 4-B (7B → cloud).
```

---

### Phase 0 — Core Hebbian-Ternary Mechanism

**Goal:** Build the fundamental building block: a `TernaryHebbianLinear` layer that stores ternary weights, maintains latent float scores, and updates via Hebbian rule with hysteresis. Prove it on MNIST in <1 hour.

**Duration:** ~1 week

#### 0.1 Ternary Weight Representation

- [ ] `TernaryTensor`: efficient storage of {-1, 0, +1} using 2-bit packing (int8 → 4 weights per byte)
- [ ] `LatentScoreTensor`: fp16 scores paired with each weight, tracking cumulative Hebbian evidence
- [ ] Conversion functions: `latent_to_ternary(scores, theta_upper, theta_lower)` with hysteresis
- [ ] Weight initialization: all weights start at 0, latent scores at small random values near 0

#### 0.2 Hebbian Update Rule

- [ ] Core rule: `Δlatent_score = lr × pre_activation × post_activation`
- [ ] Since pre/post are ternary {-1, 0, +1}, the update is:
  - `+lr` when pre and post have same sign (both +1 or both -1) → "fire together, wire together"
  - `-lr` when pre and post have opposite signs → anti-correlation
  - `0` when either is 0 → no update (silent neuron)
- [ ] Homeostatic decay: `Δlatent_score -= decay_rate × latent_score` (slow drift toward 0 for unused synapses)
- [ ] Anti-Hebbian variant for output layer: `Δlatent_score = -lr × pre_activation × post_activation` for wrong-class neurons

#### 0.3 Hysteresis Threshold Mechanism

- [ ] `θ_upper`: activation threshold (e.g., 5.0) — latent score must exceed this to flip 0 → ±1
- [ ] `θ_lower`: deactivation threshold (e.g., 1.0) — latent score must fall below this to flip ±1 → 0
- [ ] Hysteresis gap (θ_upper - θ_lower = 4.0) prevents oscillation
- [ ] Once activated, a synapse is "sticky" — needs significant counter-evidence to deactivate
- [ ] Configurable per-layer thresholds

#### 0.4 Forward Pass

- [ ] `TernaryHebbianLinear.forward(x)`: MatMul using ternary weights + ternary activations
- [ ] Implemented via popcount: `output = popcount(x AND w_pos) - popcount(x AND w_neg)`
- [ ] PyTorch reference implementation first (using float MatMul for correctness), popcount optimization later
- [ ] Activation function: `sign()` or `ternary_sign()` — maps to {-1, 0, +1}

#### 0.5 MNIST Sanity Check

- [ ] Single `TernaryHebbianLinear` layer (784 → 10), no hidden layers
- [ ] Train with Hebbian rule on output layer (supervised: post = one_hot(label))
- [ ] Target: >90% accuracy in <5 epochs (should be easy — this is essentially a Hebbian linear classifier)
- [ ] Verify: no `.backward()` called anywhere in training loop
- [ ] Verify: weight distribution stays ternary throughout training
- [ ] Verify: latent scores evolve smoothly, ternary weights flip occasionally at thresholds

#### 0.6 Unit Tests

- [ ] `test_ternary_representation`: packing/unpacking correctness
- [ ] `test_hebbian_update`: manual computation vs implementation
- [ ] `test_hysteresis`: verify thresholds work, no oscillation for constant input
- [ ] `test_no_backward`: verify `torch.autograd` is never engaged
- [ ] `test_mnist_minimal`: end-to-end test, >90% accuracy

📄 See: [`phase-0-core-mechanism.md`](phase-0-core-mechanism.md)

---

### Phase 1 — Vision Proof-of-Concept

**Goal:** Show ternary Hebbian learning works on real vision tasks. Compare against float Hebbian (SoftHebb) and backprop baselines. **Demonstrate continual learning as the core differentiator.**

**Duration:** ~2-3 weeks

#### 1.1 MLP on MNIST

- [ ] 2-3 hidden layers, `TernaryHebbianLinear` throughout
- [ ] Greedy layer-wise training: train layer 1, freeze, train layer 2, freeze, etc.
- [ ] Output layer: Hebbian + anti-Hebbian with label supervision
- [ ] Target: >95% accuracy
- [ ] Ablation: compare single-layer vs multi-layer
- [ ] Ablation: compare with/without hysteresis
- [ ] Ablation: compare with/without homeostatic decay

#### 1.2 CNN on CIFAR-10

- [ ] `TernaryHebbianConv2d`: Hebbian rule applied per filter (local receptive field × local activation)
- [ ] Architecture: 2-3 conv layers + 1-2 linear layers
- [ ] Hebbian rule for convolutions: `ΔW[h,w] = lr × input_patch[h,w] × output_neuron`
- [ ] Target: >60% accuracy (baseline: SoftHebb float at 80.3%, backprop at ~93%)
- [ ] This is ~75% of SoftHebb's performance — proving ternary doesn't kill Hebbian

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

### Phase 3 — First Language Model

**Goal:** Show ternary Hebbian Transformers can learn statistical regularities in text and generate coherent output. This is genuinely unexplored — no Hebbian network has been trained on language before.

**Duration:** ~4-6 weeks

#### 3.1 Hebbian Transformer Components

- [ ] `TernaryHebbianLinear` for all linear projections (Q, K, V, O, FFN)
- [ ] `TernaryHebbianEmbedding`: ternary embedding table (tokens → {-1, 0, +1}^d)
- [ ] Attention mechanism: standard scaled dot-product attention
  - Q, K, V are ternary → attention scores are integers in [-d, +d]
  - Softmax still needed (or explore alternatives like sparsemax, entmax)
- [ ] Hebbian rule for attention: `ΔW_Q = lr × input × query_activation`, etc.
  - Each projection matrix learns independently
- [ ] RoPE or ALiBi for position encoding (ternary compatible)

#### 3.2 Training Strategy for Language

- [ ] **Challenge**: Hebbian learning has no loss function. How do you train a language model?
- [ ] **Approach A — Next-token Hebbian**:
  - Input: sequence of tokens → ternary embeddings
  - Output: the model's ternary activation at each position
  - "Teacher" signal: one-hot vector of the actual next token
  - Hebbian update on output layer: strengthen connections that predict correct token
  - Anti-Hebbian on wrong tokens
  - Hidden layers: self-organizing (their own output is the "post" signal)
- [ ] **Approach B — Contrastive Hebbian for sequences**:
  - Real next token → Hebbian update
  - Random token → anti-Hebbian update
  - Layer-wise: each layer gets its own contrastive signal
- [ ] **Approach C — Masked Hebbian** (BERT-style):
  - Mask some tokens, predict them
  - Hebbian update based on correct vs incorrect predictions

#### 3.3 TinyStories Experiments

- [ ] 100M-param ternary Hebbian Transformer
- [ ] Train on TinyStories (~2M children's stories, ~500M tokens)
- [ ] Evaluate:
  - Perplexity (expect higher than backprop, but should be << random)
  - Generation quality: human evaluation of 100 generated paragraphs
  - Coherence: do the stories have a beginning, middle, end?
  - Grammar: does the model learn basic English syntax?
- [ ] Success: generates paragraphs a human can read without cringing
- [ ] Stretch goal: perplexity within 2× of a same-size backprop model

#### 3.4 Analysis

- [ ] What do the ternary embeddings look like? (t-SNE of token embeddings)
- [ ] Which attention heads learn interpretable patterns?
- [ ] Do different layers learn different linguistic features? (syntax early, semantics late?)
- [ ] Weight sparsity: what % of attention weights are 0?

📄 See: [`phase-3-language-model.md`](phase-3-language-model.md)

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
├── 009-100m-lm-tinystories.md
├── 010-1b-lm-pretrain.md
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

### Language
8. **Hebbian attention**: Does the Hebbian rule make sense for attention? Q, K, V projections learn based on co-occurrence — does this produce useful attention patterns?
9. **Softmax with ternary activations**: Attention scores are integer-valued (ternary Q · ternary K). Is softmax appropriate? Would a hard winner-take-all work better?
10. **Next-token prediction without loss**: Hebbian learning maximizes correlation, not minimizes perplexity. Is next-token prediction fundamentally a supervised task that needs a loss?
11. **Sequence-level Hebbian**: Should Hebbian updates happen per-token or per-sequence? Per-token is more local but noisier.

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
| **Predictive Coding** (Whittington & Bogacz, 2017) | Learning by minimizing prediction errors layer by layer |
| **Feedback Alignment** (Lillicrap et al., 2016) | Random feedback weights work for learning — challenges backprop's necessity |
| **Direct Feedback Alignment** (Nøkland, 2016) | Error signal goes directly to each layer, bypassing chain rule |
| **Difference Target Propagation** (Lee et al., 2015) | Targets instead of gradients propagated backward |
| **Equilibrium Propagation** (Scellier & Bengio, 2017) | Energy-based learning with two phases (free and clamped) |
| **Local Representation Alignment** (Ororbia & Mali, 2023) | Local learning rules that approximate backprop |
| **Signal Propagation** (Kohan et al., 2023) | Forward-only learning via signal propagation, not gradients |

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
