# Step 0.2 — Plasticity Mechanism Survey

> **Status:** ✅ Complete (2026-08-12)
> **Goal:** Identify the most promising local learning rule for frozen-backbone adaptation. Understand why each candidate might work or fail — specifically for the chosen models: SmolLM2-1.7B (primary, LLaMA-modern) and GPT-2 124M (gen-test, classic pre-norm).

---

## The Constraint

The learning rule must operate with **only locally available information** at each layer:

1. **Pre-activation** — the input to this plastic weight's projection (captured via forward hook before the linear layer)
2. **Post-activation** — the output of this plastic weight's projection (captured via forward hook after the linear layer)
3. **A global signal** — some scalar or vector that modulates learning (surprise, reward, error). Computed at the output (logits/loss) and broadcast to all layers.

No gradient flow through frozen layers. No backpropagation. No computation of ∂L/∂W via chain rule.

---

## Chosen Models (from Step 0.1)

### Primary: SmolLM2-1.7B — LLaMA-modern

| Property | Value |
|:---------|:------|
| Parameters | 1,711M (1.71B) |
| Layers | 24 transformer blocks |
| d_model | 2048 |
| d_ff (SwiGLU) | 8192 |
| Attention heads | 32 Q, 32 KV (full MHA) |
| Position encoding | RoPE (applied to Q, K) |
| Normalization | RMSNorm (pre-norm, before attn & MLP) |
| MLP activation | SwiGLU: gate_proj (SiLU) ⊙ up_proj, then down_proj |
| Vocabulary | 49,152 (BPE) |
| Max context | 8,192 tokens |
| Block path | `model.model.layers[i]` |
| Precision | bf16 (frozen backbone) |

**Key architectural properties for plasticity injection:**

- **Pre-norm RMSNorm**: Activations are normalized to unit RMS before each sublayer. This constrains the dynamic range of pre- and post-activations — Hebbian outer products will have bounded magnitudes, which is good for stability but may reduce the signal-to-noise ratio of the update.
- **SwiGLU with gate**: The MLP has three projections: `gate_proj` (2048→8192, SiLU activation), `up_proj` (2048→8192, no activation), and `down_proj` (8192→2048). The gate can zero out the up-projection's contribution — if the gate closes, the entire MLP output for that token dimension is near zero, and plasticity on `down_proj` sees no post-activation signal.
- **RoPE on Q/K only**: Position information is injected via rotation of Q and K before attention. Plastic weights on Q/K projections operate on pre-RoPE activations — they can learn position-dependent patterns because the RoPE encoding provides positional context to the post-activation.
- **Full MHA (not GQA)**: All 32 KV heads match 32 Q heads. Plasticity per head is symmetric — no need to handle the Q/KV head count mismatch that GQA introduces in smaller SmolLM2 variants (360M: 15 Q / 5 KV; 135M: 9 Q / 3 KV).
- **24 layers**: Deep network means plastic updates at early layers compound through 23 subsequent frozen layers. Small perturbations at layer 0 can have large effects at the output — both an opportunity (more leverage) and a risk (instability).

### Generalization-test: GPT-2 124M — Classic Pre-Norm

| Property | Value |
|:---------|:------|
| Parameters | 124.4M |
| Layers | 12 transformer blocks |
| d_model | 768 |
| d_ff (GELU) | 3072 |
| Attention heads | 12 (full MHA, no GQA) |
| Position encoding | Learned absolute positional embeddings (added at input) |
| Normalization | LayerNorm (pre-norm, before attn & MLP) |
| MLP activation | GELU (ungated, standard FFN: fc → GELU → proj) |
| Vocabulary | 50,257 (BPE) |
| Max context | 1,024 tokens |
| Block path | `model.transformer.h[i]` |
| Precision | bf16 (frozen backbone) |

**Key architectural differences from SmolLM2-1.7B:**

- **LayerNorm vs RMSNorm**: LayerNorm subtracts mean AND normalizes variance, while RMSNorm only normalizes scale. LayerNorm provides stronger normalization — plastic bias injection before LayerNorm will have its mean component removed. Injection AFTER the norm is safer.
- **No RoPE**: Position is encoded once at the embedding level (learned absolute positions). Plastic weights at every layer see position-agnostic activations — they cannot learn position-specific patterns without positional information leaking through the residual stream.
- **GELU (ungated) vs SwiGLU**: The classic FFN has no gate — all activations pass through GELU and contribute to the output. No risk of gate-induced signal zeroing. But also no opportunity for gate-level plasticity modulation.
- **Combined QKV projection**: GPT-2 uses a single `c_attn` (768→2304) that projects to Q, K, V concatenated, then splits. Plastic weights on this combined projection affect all three attention components simultaneously — less granular than SmolLM2's separate Q, K, V projections.
- **12 layers vs 24**: Half the depth means less compounding of plastic perturbations, which is safer but provides less adaptation leverage.

---

## The Central Scientific Question

> **"Why might local plasticity rules succeed on a frozen pre-trained backbone when they failed (or were never tested) for training deep networks from scratch?"**

### The Hebbian Failure Mode (Recap)

Our 19 experiments (E001–E019) proved definitively: **local Hebbian rules cannot train deep ternary networks from scratch.** Every method — plain Hebbian, Forward-Forward, 3-factor Hebbian with feedback, Equilibrium Propagation — hit the ~88% MNIST ceiling. Root causes:

1. **Hebbian optimizes correlation, not error**: ΔW ∝ pre ⊗ post maximizes statistical co-occurrence, not classification accuracy. Hidden layers learn PCA-like features (dominant variance directions), not class-discriminative representations.
2. **Credit assignment is impossible locally**: Without backprop, no layer knows which specific weight changes would reduce the final error. The signal that reaches hidden layers (through ternary feedback weights, random projections, or latent scores — NTH-4b) is too noisy to drive discriminative learning.
3. **Compounding noise**: Each layer's Hebbian update adds statistical noise. After N layers, the signal-to-noise ratio for class-relevant information decays exponentially.

### Why Frozen Backbones Change the Equation

A frozen pre-trained backbone provides something training from scratch never had: **structured, semantically meaningful representations at every layer.**

| Property | Training from Scratch | Frozen Pre-trained Backbone |
|:---------|:----------------------|:----------------------------|
| Layer 0 pre-activations | Raw pixels / token embeddings | Contextualized token embeddings from a pre-trained embedder |
| Hidden layer features | Random → PCA → plateau at ~88% | Semantic features: syntax (early layers), semantics (mid), task-specific (late) |
| pre ⊗ post content | Noise dominated by input statistics | Structured co-occurrence of meaningful features |
| What "correlation" means | "These pixels fire together" | "These semantic concepts co-occur in this domain" |
| Modulator meaning | Must carry credit assignment across layers | Only needs to say "this is surprising" — features are already structured |

**The key insight**: Local rules don't need to BUILD structure from scratch on a frozen backbone — they only need to MODULATE existing structure. This is a categorically easier problem:

- **Hebbian (pre ⊗ post)** on structured features reinforces genuine semantic co-occurrences in the new domain, not just pixel-level correlations.
- **A global modulator (M)** doesn't need to solve credit assignment — it just needs to say "learn this" vs "ignore this." The features being modulated are already meaningful.
- **Compounding noise is reduced** because the frozen backbone's representations are stable and high-SNR. Plastic perturbations are small relative to the frozen signal.

This is not just speculation — it mirrors the biological reality. The brain's neocortex is born with structured connectivity (evolution = pre-training). Synaptic plasticity (local Hebbian rules) doesn't build the architecture; it adapts an already-functional system to experience.

---

## Candidate Mechanisms — Detailed Evaluation

For each mechanism, we evaluate: (1) architectural compatibility with both models, (2) injection points and their implications, (3) precision requirements, (4) why it might work on frozen backbones, and (5) specific failure risks.

---

### 1. Plain Hebbian: ΔW = η · pre ⊗ post

**What it is:** Strengthen connections where pre- and post-synaptic neurons fire together. The simplest possible learning rule. No modulator, no error signal — pure correlation.

**History in PH-Neuro:** Tested extensively (E001–E003). Single-layer supervised WTA achieves 88.4% MNIST. Hidden layers fail — produces PCA-like features, not class-discriminative. All hidden-layer experiments (unsupervised, label-as-post, reward-modulated) failed to produce class-discriminative features.

#### Architectural Compatibility

**SmolLM2-1.7B (LLaMA):**

| Injection Point | pre shape | post shape | W_plastic shape | Notes |
|:----------------|:----------|:-----------|:----------------|:------|
| Q projection | (B, S, 2048) | (B, S, 2048) | (2048, 2048) | pre-RoPE; post-activation includes position via RoPE |
| K projection | (B, S, 2048) | (B, S, 2048) | (2048, 2048) | pre-RoPE; symmetric to Q |
| V projection | (B, S, 2048) | (B, S, 2048) | (2048, 2048) | No RoPE on V — purely content-based |
| O projection | (B, S, 2048) | (B, S, 2048) | (2048, 2048) | Post-attention mixed context |
| gate_proj | (B, S, 2048) | (B, S, 8192) | (2048, 8192) | **35M plastic weights per layer** — very large |
| up_proj | (B, S, 2048) | (B, S, 8192) | (2048, 8192) | Same size as gate |
| down_proj | (B, S, 8192) | (B, S, 2048) | (8192, 2048) | 16.8M plastic weights per layer |

**Total if all projections have plastic weights**: 24 layers × (4 × 2048² + 2 × 2048×8192 + 8192×2048) ≈ 24 × (16.8M + 33.6M + 33.6M + 16.8M) = 24 × 100.8M ≈ **2.4B plastic parameters** — completely impractical. **We must be selective about injection points.** A vector bias (d_model elements per projection, not full matrix) is the practical starting point: 24 layers × 7 projections × 2048 = **344K plastic parameters** (0.02% of frozen model).

**GPT-2 124M (Classic):**

| Injection Point | pre shape | post shape | W_plastic shape | Notes |
|:----------------|:----------|:-----------|:----------------|:------|
| c_attn (QKV) | (B, S, 768) | (B, S, 2304) | (768, 2304) | Combined Q,K,V — 1.8M plastic weights |
| c_proj (attn out) | (B, S, 768) | (B, S, 768) | (768, 768) | 590K plastic weights |
| c_fc (MLP in) | (B, S, 768) | (B, S, 3072) | (768, 3072) | 2.4M plastic weights |
| c_proj (MLP out) | (B, S, 3072) | (B, S, 768) | (3072, 768) | 2.4M plastic weights |

Total per layer: ~7.1M. 12 layers × 7.1M = **85M full-matrix plastic weights** (68% of frozen model). Vector bias: 12 layers × 4 projections × 768 = **37K plastic parameters** (0.03% of frozen model).

#### Why Might This Work on Frozen Backbones?

- The frozen backbone's pre-activations are semantically structured. A "pre ⊗ post" correlation on these features captures genuine co-occurrence of concepts in the target domain, not raw input statistics.
- Example: if the target domain (e.g., PubMed abstracts) frequently co-activates "insulin" and "glucose" representations, Hebbian plasticity on the MLP projections would strengthen connections between those semantic directions.
- This is domain adaptation via statistical reinforcement — the model learns "these concepts go together more often in this domain."

#### Why It Probably Won't (Risks)

- **No error signal**: Hebbian cannot distinguish "useful co-occurrence" from "spurious co-occurrence." If the target domain has a stylistic quirk (e.g., all sentences start with "We"), Hebbian reinforces that pattern even if it's irrelevant to the content.
- **RMSNorm attenuation**: In SmolLM2, RMSNorm normalizes pre-activations to unit RMS. The outer product of unit-norm vectors has entries bounded in [-1, 1], making the Hebbian update very small. Over many steps, this might still accumulate, but the per-step signal is weak.
- **Catastrophic reinforcement of dominant patterns**: Without decay or competition, Hebbian will continuously strengthen the most common patterns, eventually saturating the plastic weights.

**Verdict:** Useful as a **baseline** — proves whether ANY local adaptation occurs. Not expected to produce task-specific improvement. Needed for ablation: "does the modulator (M) matter, or is raw Hebbian enough?"

---

### 2. Oja's Rule: ΔW = η · (pre ⊗ post − α · post² · W)

**What it is:** Hebbian learning with automatic weight normalization. Converges to the first principal component of the input distribution. The subtraction term (−α · post² · W) prevents unbounded weight growth.

**History in PH-Neuro:** Not directly tested on its own. Related to Hebbian + weight decay. Our repo notes show Oja's rule produced "balanced 50/50 weights but features are random projections, not class-discriminative" in multi-layer MLP tests.

#### Architectural Compatibility

Same injection points as plain Hebbian (see table above). The additional term requires storing W_plastic as a full matrix (not just a vector), which increases memory but not computation (the −α · post² · W term is an element-wise scaling of existing weights).

#### Why Might This Work on Frozen Backbones?

- Oja's rule provides **built-in stability** — weights don't grow unboundedly. This is valuable for a 24-layer network where Hebbian updates at early layers compound.
- The normalization to principal components means Oja's rule extracts the **most statistically reliable** directions in the new domain. On structured pre-trained features, these principal directions correspond to the dominant semantic shifts between domains.
- Unlike plain Hebbian which eventually saturates, Oja's rule maintains a stable weight distribution — the system can continue adapting indefinitely.

#### Why It Probably Won't (Risks)

- **Same fundamental limitation as Hebbian**: No task-specific error signal. The principal component of pre-trained features on PubMed might be "scientific register" (formal tone, passive voice), not "medical knowledge."
- **Single principal component**: Oja's rule extracts only the first PC. Domain adaptation likely requires multiple directions. Extensions (Sanger's rule, subspace methods) add complexity.
- **post² scaling in bf16**: post is in bf16 (~3 significant digits). post² loses precision, and α · post² · W may underflow for small post values.

**Verdict:** Useful as a **stability baseline** — shows whether weight normalization helps or hurts adaptation. Not a primary candidate.

---

### 3. Three-Factor (Neuromodulated) Hebbian: ΔW = η · M · pre ⊗ post

**What it is:** A modulator M (scalar or vector per layer) gates whether the pre×post correlation is strengthened (M > 0), weakened (M < 0), or ignored (M ≈ 0).

**History in PH-Neuro:** Tested as NTH-1 through NTH-4b (E005, E007).
- **NTH-1 (single-layer, label modulator):** ✅ 88.15% MNIST — matches WTA baseline. Proves M can carry task-relevant information when directly available.
- **NTH-4 (multi-layer, weight feedback):** ❌ 85.79% — feedback through 92% sparse ternary weights loses the modulator signal.
- **NTH-4b (multi-layer, dense latent score feedback):** ❌ 86.68% — even dense continuous feedback fails. Hidden layer flip rate = 0.000%/step.
- **NTH-4b's key finding:** The failure is deeper than ternary sparsity. Even with dense fp16 feedback, the hidden layer doesn't learn because the feedback signal (M at layer N−1 derived from M at layer N) is not a valid credit assignment signal — it's a correlation, not a gradient.

#### Architectural Compatibility

**SmolLM2-1.7B (LLaMA) — Critical Analysis:**

*SwiGLU gate interaction — the most important architectural consideration:*

The SwiGLU MLP computes:
```
gate = SiLU(W_gate · x)     # SiLU(x) = x · σ(x)
up   = W_up · x
out  = W_down · (gate ⊙ up)
```

Plastic weights can be injected at three points:
1. **gate_proj**: ΔW_gate = η · M · pre ⊗ (SiLU(W_gate·pre)). The modulator M controls how the gate opens/closes. Since SiLU can be negative (unlike ReLU), the gate can both amplify and suppress. M > 0 → gate opens wider for co-active pre/post patterns; M < 0 → gate closes.
2. **up_proj**: ΔW_up = η · M · pre ⊗ (W_up·pre). Modulates the value stream. But if the gate is closed (SiLU ≈ 0), the up projection's contribution is blocked at the output — the update to up_proj has no effect on the model's behavior for that token.
3. **down_proj**: ΔW_down = η · M · (gate⊙up) ⊗ (W_down·(gate⊙up)). Here pre = gate⊙up (the SwiGLU hidden state) and post = down_proj output. **If the gate is closed, pre ≈ 0, and no Hebbian update occurs.** This is the SwiGLU gate-zeroing risk.

**Key insight for SwiGLU**: The gate projection is the most *powerful* and most *dangerous* injection point. It controls information flow — a small plastic change to gate_proj can completely reroute the MLP's computation. But it's also where a bad modulator can do the most damage.

*RMSNorm interaction:*

SmolLM2 uses pre-norm: each sublayer's input passes through RMSNorm before the linear projection. If plastic weights are injected as a bias ADDED TO the linear projection's output (not modifying the projection weights themselves), then:
- **Good**: Plastic bias after the linear layer is NOT normalized away (RMSNorm is before, not after).
- **Bad**: The pre-activation (input to RMSNorm) includes the residual stream from all previous layers. The plastic bias from layer i is added to the residual stream and will be normalized by RMSNorm at layer i+1. So plastic changes at early layers are attenuated through subsequent RMSNorm operations.

*RoPE interaction:*

RoPE is applied to Q and K after projection but before attention. Plastic weights on Q/K see pre-RoPE representations as post-activation. This means:
- The Hebbian update on Q/K captures patterns in the pre-rotation space.
- Position information from RoPE is present in the attention output (which feeds into O projection and subsequent layers) but not in the Q/K post-activations used for plasticity.
- This is actually fine — plasticity on Q/K learns content-based patterns, and the frozen RoPE provides position separately.

*GQA interaction (relevant for SmolLM2-360M and 135M scaling variants):*

In GQA, n_kv_heads < n_q_heads. The KV heads are shared across groups of Q heads. Plastic weights on K and V projections affect all Q heads in their group. This makes K/V plasticity more "coarse" than Q plasticity. For the 1.7B primary (full MHA, 32/32), this is not an issue. But it matters for scaling experiments with 360M (15/5) and 135M (9/3).

**GPT-2 124M (Classic) — Critical Analysis:**

- **LayerNorm vs RMSNorm**: LayerNorm subtracts the mean. If plastic bias is injected BEFORE LayerNorm, the mean component is removed. Injection AFTER the linear projection (as a bias to the output) avoids this.
- **No RoPE**: All position information is in the learned embedding. Plastic weights at layer N see the same representation regardless of token position — they cannot learn position-specific patterns. This is actually an advantage for domain adaptation (position-agnostic patterns transfer better).
- **GELU (ungated)**: No gate-zeroing risk. All activations contribute. Simpler dynamics — what you update is what affects the output.
- **Combined QKV projection**: Plasticity on c_attn affects Q, K, V simultaneously. Less granular than SmolLM2's separate projections, but simpler to implement.

#### Why Might This Work on Frozen Backbones When NTH-4b Failed from Scratch?

**This is the central question.** NTH-4b proved that even dense, continuous feedback fails to train hidden layers from scratch. The frozen backbone hypothesis says:

1. **M doesn't need to carry credit assignment anymore.** In NTH-4b, M at layer N−1 was computed by projecting M at layer N through latent scores: M_{N-1} = M_N @ S_out. This projection needed to approximate ∂L/∂h_{N-1} — a credit assignment signal. It failed because S_out (the output layer's latent scores) is not a valid inverse of the forward transformation.

   On a frozen backbone, M is GLOBAL — computed at the output (surprise = |loss − expected_loss|) and broadcast identically to all layers. M means "the model was surprised by this input → strengthen recently active patterns." It does NOT mean "change weight w_ij by exactly δ to reduce loss by ε." The credit assignment is handled implicitly by the frozen backbone: features that contributed to the surprise are naturally co-active in the layers where they're represented.

2. **pre ⊗ post is meaningful.** In NTH-4b, the pre and post activations of hidden layers were random ternary patterns (from random initialization) — the outer product captured noise. On a frozen backbone, pre and post are semantically meaningful. The outer product of "concept A is active" ⊗ "concept B is active" captures a genuine semantic relationship that the model can reinforce or suppress.

3. **The frozen backbone provides a stable reference.** Plastic perturbations are small relative to frozen weights. Even if the Hebbian update is slightly misdirected, the frozen backbone's computation dominates — the model doesn't collapse. In NTH-4b, the hidden layer WAS the computation — misdirected updates directly degraded performance.

**Analogy**: NTH-4b tried to build a house with Hebbian rules (impossible without a blueprint). The frozen backbone provides a fully-built house. Plasticity just rearranges the furniture.

#### Key Risks Specific to SmolLM2-1.7B

1. **SwiGLU gate zeroing**: If M < 0 (surprise is low, model should "unlearn"), the anti-Hebbian update on gate_proj could close the gate for important features. The SiLU activation has range (−0.278..., ∞), so the gate can go negative. A negative gate FLIPS THE SIGN of the up-projection contribution — worse than zeroing it.
2. **RMSNorm attenuation of early-layer plasticity**: Plastic bias added at layer 0 passes through 23 subsequent RMSNorm operations. Each RMSNorm scales its input to unit RMS, effectively normalizing away small perturbations. The effective learning rate at layer 0 may be orders of magnitude smaller than at layer 23.
3. **bf16 precision for M**: The modulator M is computed from the loss (a bf16 scalar). For small surprises (M ≈ 10⁻³), the bf16 representation has ~3 significant digits — M may be zero or dominated by quantization noise. Consider computing M in float32 even if the backbone is bf16.

#### Key Risks Specific to GPT-2 124M

1. **Combined QKV plasticity is blunt**: Changing c_attn affects Q, K, and V simultaneously. You cannot, for example, strengthen Q without also strengthening K and V. This reduces the expressiveness of plasticity.
2. **Shorter context (1024 tokens)**: Long-range dependencies are harder to capture. Plasticity may overfit to local statistics.
3. **Smaller model, smaller effect**: With only 124M frozen parameters, the model's baseline representations are less rich. Plasticity has less structure to work with — the gap between "correlation" and "task-relevant" is wider.

#### Precision Considerations

| Plastic Weight Format | Memory (vector bias, SmolLM2-1.7B) | Update Precision Needed | Feasibility |
|:----------------------|:------------------------------------|:------------------------|:------------|
| float32 | 1.38 MB (344K × 4B) | float32 recommended for accumulation | ✅ Default |
| bf16 | 0.69 MB (344K × 2B) | bf16 OK for M > 10⁻³; risk of underflow | ✅ Phase 2 |
| ternary {−1,0,+1} | 0.09 MB (344K × 2-bit) | Requires hysteresis + stochastic rounding | ⏳ Phase 2.2 |

**Recommendation**: Start with float32 plastic weights for Phase 1.1 (debuggable, no precision issues). Move to bf16 in Phase 2 once the mechanism is validated. Ternary plastic weights are the Phase 2.2 goal — use existing DQT/hysteresis infrastructure from Phase 0.

---

### 4. Predictive Coding: ΔW = η · ε · pre, where ε = post − prediction

**What it is:** Each layer maintains a prediction of the layer below's activity. The prediction error ε = actual − predicted drives weight updates. This is Rao & Ballard (1999), Friston's free energy principle. Biologically, the cortex is organized in predictive hierarchies — each region predicts the activity of the region below, and prediction errors flow upward.

**History in PH-Neuro:** Not directly tested. Related to Equilibrium Propagation (TEP-1, E008) which failed for ternary weights (82.57% — worse than single-layer 88.4%). TEP-1 was the only method that moved hidden weights (0.005%/step flip rate), but the movement was in non-discriminative directions.

#### Architectural Compatibility

Predictive coding requires **separate prediction networks** per layer. For SmolLM2-1.7B:
- Each of the 24 blocks needs a predictor that maps the block's output back to the block's input: `prediction_i = P_i(block_output_i)`, and `ε_i = block_input_i − P_i(block_output_i)`.
- The predictor P_i must be trained, adding parameters and a separate training loop.
- For a vector-bias plastic injection (not full matrix), the prediction is a single vector: the block predicts what its input SHOULD have been, compares to actual input, and the error drives plastic bias updates.

**Simplified variant (more practical):**
Instead of full predictive coding with trained inverse models, use the **surprise signal** from Step 0.3 as a proxy for prediction error. The language model's own loss is a prediction error (predicting the next token). The global surprise M = |loss − E[loss]| can be used without per-layer prediction networks — this collapses back to 3-factor Hebbian but with a theoretically grounded modulator.

#### Why Might This Work on Frozen Backbones?

- Language models ARE prediction machines — the prediction error is naturally available at the output. The theoretical fit is excellent: "the model predicted token X with confidence p, but the actual next token was Y → surprise."
- Prediction errors automatically provide **layer-relevant** signals if computed locally. A block that consistently mispredicts its input in the new domain gets a strong error signal; a block whose predictions remain accurate gets weak or zero signal.
- This provides **automatic learning rate modulation per layer** — layers that need to adapt get larger updates; layers whose representations already work for the new domain are left alone.

#### Why It Probably Won't (Risks)

- **Complexity**: Full predictive coding requires training N prediction networks (24 for SmolLM2-1.7B, 12 for GPT-2 124M). Each prediction network must be at least as expressive as the forward transformation it's inverting — this could double the parameter count.
- **Prediction network training**: The predictors need to be trained. If trained with backprop, we've violated the "no backprop through frozen layers" constraint. If trained with local rules, we've just moved the problem — how do the predictors learn?
- **Local prediction error may be uninformative**: A frozen block's input and output are highly correlated (residual connection: output ≈ input + Δ). The prediction error ε = input − P(output) may be dominated by the residual Δ, which is a small perturbation. The SNR of ε may be too low for useful learning.
- **TEP-1 precedent is unfavorable**: Equilibrium Propagation (a related energy-based method) failed on ternary weights. While the frozen backbone context is different, the failure mode — weight updates in non-discriminative directions — could recur.

**Verdict:** Theoretically elegant, practically too complex for Phase 1. Revisit in Phase 2+ if 3-factor Hebbian shows promise but insufficient adaptation magnitude. The simplified variant (global surprise as prediction error proxy) is already covered by 3-factor Hebbian.

---

### 5. Forward-Forward: ΔW = η · (goodness(pos) − goodness(neg))

**What it is:** Two forward passes per batch: one with real data (increase "goodness"), one with corrupted/negative data (decrease "goodness"). Goodness is a layer-local scalar function — sum of squared activations, or popcount for ternary.

**History in PH-Neuro:** Tested as TFF-1 and TFF-2 (E004, E006).
- **TFF-1 (single-layer):** ✅ 87.9% MNIST — works at the output layer.
- **TFF-2 (2-layer):** ❌ 86.81% — no improvement over 1 layer. Hidden layer learns "is this a real image?" not "is this a 3?" Goodness function (popcount) carries zero class-specific information. Negative pass adds nothing.

#### Architectural Compatibility

Forward-Forward requires a **goodness function** G(h) → scalar for each layer. For transformers with continuous (bf16) activations, G(h) = ||h||² (sum of squared activations) is the natural choice. The update rule:

```
ΔW_pos = η · ∂G(h_pos)/∂W    (increase goodness on real data)
ΔW_neg = −η · ∂G(h_neg)/∂W   (decrease goodness on corrupted data)
```

But computing ∂G/∂W requires... backprop through the layer! This violates the local-only constraint unless we approximate ∂G/∂W with a Hebbian-style update: ΔW ≈ η · pre ⊗ (sign of contribution to G). This approximation is exactly what failed in TFF-2.

#### Why It Probably Won't Work (Even on Frozen Backbones)

- **The fundamental problem is orthogonal to frozen vs. scratch**: Goodness is a whole-layer scalar. "Increase goodness on real data" tells the layer to increase its overall activity — it doesn't specify WHICH neurons or WHICH directions. The weight update is unconstrained: any change that increases ||h||² works.
- On a frozen backbone, the layer already has high goodness for in-domain data (the model was trained on it). The goodness contrast between in-domain and out-of-domain may be small or nonexistent.
- **TFF-2's failure mode transfers directly**: The hidden layer would learn "is this from the target domain?" not "what specific content distinguishes this domain?" This is useful for domain detection but not for domain adaptation.

**Verdict:** ❌ **Not viable.** The TFF-2 failure is fundamental to the goodness function approach, not specific to training from scratch. Do not pursue for Phase 1. Removed from ranking.

---

### 6. Target Propagation: Layer N computes a target for Layer N-1

**What it is:** Each layer has a "target activation" — what it should output to reduce the final error. The layer above computes a target for the layer below via an inverse mapping. Weights are updated to move actual outputs toward targets: ΔW = η · (target − actual) · pre^T.

**History in PH-Neuro:** Not tested. Requires training inverse mappings per layer.

#### Architectural Compatibility

For SmolLM2-1.7B with 24 layers, target propagation requires 24 inverse mappings (one per block). Each inverse mapping must be at least as expressive as the forward block it's inverting — this could triple the parameter count.

Difference Target Propagation (DTP, Lee et al. 2015) is more practical: it uses learned feedback networks trained with a separate reconstruction loss. But this requires backprop to train the feedback networks, and it still hasn't been shown to work at transformer scale.

#### Why It Probably Won't Work

- **Scale**: 24 inverse mappings for SmolLM2-1.7B is a massive engineering undertaking before any science can be done.
- **Inverse of attention**: Inverting a multi-head self-attention operation is non-trivial. The attention pattern is a function of Q·K^T — inverting this requires solving for Q and K given the attention output, which is underdetermined.
- **Never demonstrated at transformer scale**: Target propagation has been shown on MLPs and small CNNs (MNIST, CIFAR-10). No published result on transformers.

**Verdict:** ❌ **Not viable for Phase 1.** Too complex. Revisit only if all simpler mechanisms fail and there's strong theoretical reason to believe target propagation would succeed where others didn't.

---

### 7. STDP (Spike-Timing-Dependent Plasticity)

**What it is:** Weight change depends on the relative timing of pre- and post-synaptic spikes. Pre-before-post → LTP (strengthen). Post-before-pre → LTD (weaken).

**Why it's not applicable:** STDP requires the model to operate in a spiking regime — continuous-time spike events, not discrete transformer blocks. Converting SmolLM2-1.7B or GPT-2 to a spiking network is a separate research project. The temporal dynamics of transformer inference (one forward pass per token) don't map cleanly to spike timing.

**Verdict:** ❌ **Not applicable.** Orthogonal research direction. Removed from ranking.

---

## Finalized Ranking

| Rank | Mechanism | Viability | Complexity | Model-Specific Risk | Verdict |
|:----:|:----------|:---------:|:----------:|:--------------------|:--------|
| **1** | **3-factor Hebbian** (surprise-modulated) | 🟢 High | Low | SwiGLU gate zeroing; RMSNorm attenuation; bf16 precision for M | **PRIMARY** |
| 2 | Plain Hebbian + decay | 🟡 Medium | Minimal | Same as above minus M precision risk; no task signal | **Baseline #1** |
| 3 | Oja's Rule | 🟡 Medium | Minimal | post² precision in bf16; extracts only 1st PC | **Baseline #2** |
| 4 | Predictive Coding (simplified) | 🟡 Medium | High | Requires prediction networks; TEP-1 precedent unfavorable | **Phase 2 fallback** |
| — | Forward-Forward | 🔴 Low | — | Goodness function fundamentally non-discriminative (TFF-2 proven) | **Rejected** |
| — | Target Propagation | 🔴 Low | — | Too complex; inverse of attention unsolved; never shown at transformer scale | **Rejected** |
| — | STDP | 🔴 N/A | — | Requires spiking conversion | **Not applicable** |

### Ranking Methodology

Each mechanism scored on 5 axes (1–5, higher = better for Phase 1 experiment):

| Mechanism | Scientific Promise | Implementation Simplicity | Arch Compatibility (SmolLM2) | Arch Compatibility (GPT-2) | PH-Neuro Code Reuse | **Total** |
|:----------|:------------------:|:-------------------------:|:----------------------------:|:--------------------------:|:-------------------:|:---------:|
| 3-factor Hebbian | 5 | 4 | 4 | 5 | 5 (NTH code) | **23** |
| Plain Hebbian | 2 | 5 | 4 | 5 | 4 | **20** |
| Oja's Rule | 3 | 4 | 4 | 5 | 2 | **18** |
| Predictive Coding | 4 | 1 | 3 | 3 | 1 | **12** |

---

## Selected Mechanism: 3-Factor Hebbian with Global Surprise Modulator

### Why #1 — Detailed Justification

**1. Scientific promise — the frozen backbone hypothesis is most naturally tested with 3-factor Hebbian.**

The core question of PH-Neuro Brain is: *"Can a global modulator (surprise, reward, prediction error) drive useful local plasticity when the features being modulated are already semantically structured?"*

3-factor Hebbian is the minimal mechanism that tests this question. It has exactly one knob beyond plain Hebbian: the modulator M. If M adds nothing over M=1 (plain Hebbian), the hypothesis is falsified with the simplest possible experiment. If M helps, we have a clear path to improving M (layer-specific, vector-valued, different sources).

Other mechanisms either lack the task-signal component entirely (plain Hebbian, Oja) or add too many confounding variables (predictive coding adds prediction networks; target propagation adds inverse mappings).

**2. Implementation simplicity — we already have working NTH code.**

`src/ph_neuro/training/neuromodulated.py` contains `NeuromodulatedHebbianClassifier` with:
- The `neuromodulated_update(pre, post, modulator, lr)` function — ΔW = η · M · pre ⊗ post
- Verified equivalence to WTA (Δ identical to machine precision)
- 46 passing tests (`tests/integration/test_phase2_nth*.py`)
- Zero `.backward()` calls (verified via monkey-patching)

The adaptation for transformers is mechanical:
- Replace the single-layer classifier update with per-projection updates on the frozen backbone
- Replace the label modulator with a global surprise scalar
- That's it. The core update rule is unchanged.

**3. Architectural compatibility — works on both models with minimal adaptation.**

| Concern | SmolLM2-1.7B | GPT-2 124M | Mitigation |
|:--------|:-------------|:-----------|:-----------|
| SwiGLU gate zeroing | ⚠️ Risk | ✅ N/A (GELU) | Inject plastic bias AFTER gate⊙up (on down_proj), not on gate_proj itself |
| RMSNorm attenuation | ⚠️ Risk | ⚠️ Lower risk | Inject plastic bias AFTER the linear projection (as output bias), not before RMSNorm |
| bf16 precision for M | ⚠️ Risk | ⚠️ Risk | Compute M in float32; clip minimum |M| > 10⁻⁶ |
| Combined QKV in GPT-2 | ✅ N/A (separate QKV) | ⚠️ Less granular | Accept — for Phase 1, coarse plasticity is acceptable |
| GQA in scaling variants | ⚠️ For 360M/135M | ✅ N/A | For Phase 1 (1.7B, full MHA), no issue. Handle when scaling |

**4. Biological grounding — strongest theoretical foundation.**

3-factor Hebbian maps directly to established neuroscience:
- **Dopamine** → reward prediction error → M > 0 ("this was better than expected")
- **Acetylcholine** → expected uncertainty → modulates learning rate
- **Norepinephrine** → unexpected uncertainty → M ≫ 0 ("something surprising happened — learn this")
- **Serotonin** → patience/persistence → modulates decay rate

This is not just analogy — it provides a principled framework for designing M. The surprise signal from Step 0.3 will be mapped to these neuromodulatory functions.

**5. Clear ablation path.**

The 2×2×2 ablation grid (Step 1.2) is natural:
- M = surprise vs M = 1 (constant) → isolates the modulator's contribution
- Hebbian update vs random update → isolates whether the Hebbian direction matters
- Decay vs no decay → isolates whether forgetting is beneficial

### Runner-Up: Plain Hebbian + Decay

Plain Hebbian is the essential baseline. If 3-factor Hebbian with surprise modulation does NOT outperform plain Hebbian with constant M=1 and weight decay, then the modulator adds no value — the adaptation is purely statistical correlation, not task-driven. This is a critical falsification check.

---

## Architecture-Specific Risk Analysis

### SmolLM2-1.7B: SwiGLU Gate Dynamics

The SwiGLU activation function `SiLU(x) = x · σ(x)` has a minimum of approximately −0.278 at x ≈ −1.28. This means:

- **Positive gate values** (x > 0): The gate opens, and the up-projection passes through. SiLU(x) ≈ x for large x.
- **Near-zero gate values** (x ≈ 0): The gate is almost closed. SiLU(0) = 0. The up-projection's contribution is blocked.
- **Negative gate values** (x < 0): The gate is NEGATIVE. SiLU(x) < 0, which FLIPS THE SIGN of the up-projection contribution. This is a nonlinear sign inversion — qualitatively different from ReLU-based gating.

**Risk scenario**: If M < 0 (model should "unlearn") and the Hebbian update on gate_proj pushes some gate neurons negative, the SwiGLU output for those dimensions becomes negative. This inverts the contribution of those feature dimensions to the residual stream — a potentially catastrophic change from a small weight update.

**Mitigation**: For Phase 1, do NOT inject plastic weights on gate_proj. Start with plastic bias on down_proj only (post-gate, operates on the already-gated hidden representation). This avoids the gate dynamics entirely. If results are promising, add gate_proj plasticity in Phase 2 with gradient clipping on the gate bias.

### SmolLM2-1.7B: RMSNorm Attenuation Chain

With 24 layers of pre-norm RMSNorm, a plastic bias b added at layer i's output experiences:

```
b_effective_at_layer_i+1 = RMSNorm(residual_i+1 + b)
                         ≈ RMSNorm(residual_i+1)  [if ||b|| ≪ ||residual||]
```

Since RMSNorm scales to unit RMS, a small bias b is effectively normalized away. The attenuation factor per layer is approximately (||residual|| / ||residual + b||). For b ≪ residual, this is ≈ 1 − (b·residual)/||residual||².

Over 23 subsequent layers, the effective learning rate at layer 0 could be orders of magnitude smaller than at layer 23. **Layer-wise learning rate scaling may be necessary**: η_i = η_base · (1 + i/24) or similar.

### GPT-2 124M: LayerNorm Mean Subtraction

LayerNorm normalizes both mean and variance: `LN(x) = γ · (x − μ)/σ + β`. If plastic bias is injected before LayerNorm, the mean component μ is subtracted. To preserve the bias, it must be injected AFTER LayerNorm (as a bias added to the projection output).

### Both Models: bf16 Precision Cascade

The frozen backbone operates in bf16. Plastic weights and updates have three precision options:

| Stage | float32 | bf16 | Risk if bf16 |
|:------|:--------|:-----|:-------------|
| Plastic weight storage | 4B/param | 2B/param | Acceptable (weights are accumulations) |
| Hebbian update (pre ⊗ post) | 4B/op | 2B/op | **Risky** — outer product of bf16 vectors has ~3 digits; small correlations lost |
| Modulator M | 4B | 2B/op | **Risky** — M ≈ 10⁻³ may underflow to 0 |
| Weight decay | 4B | 2B/op | Acceptable (subtraction is exact in bf16 for similar magnitudes) |

**Recommendation**: Compute the Hebbian update (pre ⊗ post) and the modulator M in float32, then convert the final ΔW to the plastic weight's storage format. This adds minimal overhead (the update is a tiny fraction of the forward pass) and eliminates precision as a confounding variable.

---

## Plastic Weight Injection Architecture

### Vector Bias Approach (Phase 1)

The simplest plastic parameterization: for each trainable projection in the frozen backbone, maintain a **bias vector** b of shape (d_out,) that is added to the projection output:

```
y_frozen = W_frozen @ x          # Frozen forward pass
y_plastic = b                    # Plastic bias (learned via Hebbian)
y = y_frozen + y_plastic         # Combined output
```

The Hebbian update uses the combined pre- and post-activations:
```
pre = x                                  # Input to the projection
post = y                                 # Output AFTER adding plastic bias
Δb = η · M · mean_over_batch(post)       # Vector update: average post-activation, scaled by M
```

Wait — for a vector bias, the outer product pre ⊗ post is a matrix (d_in × d_out). How do we get a vector update?

**Correction**: For a vector bias b (shape d_out,), the Hebbian update should be:
```
Δb = η · M · mean_over_batch_and_sequence(post)
```
i.e., the average post-activation direction. This is a "1D Hebbian" — strengthen the bias in whichever direction the layer tends to activate when M > 0.

But this loses the pre ⊗ post structure — it doesn't use pre at all! A better approach for vector bias:

```
Δb = η · M · mean_over_batch_and_sequence(post − W_frozen @ x)
```
i.e., the average deviation from the frozen projection. This captures "what direction does the plastic bias need to push to make the output match what the domain requires?"

**Even better — scalar-modulated direction**:
```
Δb = η · M · mean_over_batch_and_sequence(post)
```
Where `post` is the combined (frozen + plastic) output. This is equivalent to: "when surprised, reinforce the direction the layer is currently activating." It's the simplest possible vector Hebbian update.

This will be formalized in Step 0.4 (Architecture Design). The key point for this survey: **vector bias Hebbian is O(d) per layer, not O(d²) like full-matrix Hebbian.**

### Full Matrix Approach (Phase 2)

For richer adaptation, replace the vector bias with a low-rank matrix BA^T (LoRA-style) or a full matrix W_plastic, both updated via Hebbian rules. This is deferred to Phase 2.1 (Low-Rank Plastic Matrices).

---

## Recommendation (Final — 2026-08-12)

### Primary Mechanism: 3-Factor Hebbian with Global Surprise Modulator

**Update equation (per plastic-injected projection):**
```
M = f_surprise(loss, expected_loss)     # Scalar, computed globally
Δb = η · M · mean(post)                 # Vector Hebbian update
b ← b + Δb − λ · b                      # Update with weight decay
```

Where:
- `post` = frozen_projection(x) + b (the combined output, including plastic bias)
- `f_surprise` is designed in Step 0.3
- `η` is layer-specific (compensates for RMSNorm attenuation)
- `λ` is the decay rate (forgetting is a feature)

**Injection points (Phase 1.1 — minimal viable experiment):**
1. **down_proj** (MLP output projection) in every SmolLM2 block — post-SwiGLU, no gate-zeroing risk
2. **o_proj** (attention output projection) in every block — captures attention-level adaptation

This gives 24 layers × 2 projections = 48 plastic bias vectors. For SmolLM2-1.7B: 48 × 2048 = **98,304 plastic parameters** (0.006% of frozen model). For GPT-2 124M: 12 layers × 2 projections × 768 = **18,432 plastic parameters** (0.015% of frozen model).

### Baselines

1. **Plain Hebbian (M = 1 constant)**: Same update but M is always 1. Tests whether the surprise modulator adds value.
2. **Random updates**: Δb = η · ε where ε ~ N(0, I). Tests whether ANY weight change (regardless of direction) produces measurable adaptation.
3. **Frozen only**: b = 0 always. The lower bound — confirms that plasticity is necessary for any adaptation.

### Later Exploration

- **Predictive coding (simplified)**: If 3-factor Hebbian shows promise but insufficient magnitude, add layer-specific modulators derived from local prediction error. This is the bridge between 3-factor Hebbian and full predictive coding.
- **Full-matrix / low-rank plasticity**: If vector bias works but saturates, increase capacity via LoRA-style BA^T matrices updated with Hebbian rules (Phase 2.1).
- **Ternary plastic weights**: Once the mechanism is validated in float32, compress to 2-bit using DQT/hysteresis (Phase 2.2).

---

## Next Steps

- [x] Survey 7 plasticity mechanisms — ✅ Complete
- [x] Evaluate each against SmolLM2-1.7B and GPT-2 124M specifically — ✅ Complete
- [x] Address frozen-backbone vs from-scratch question — ✅ Complete
- [x] Finalize ranking with model-specific justification — ✅ Complete
- [x] Document architecture-specific risks (SwiGLU, RMSNorm, bf16) — ✅ Complete
- [ ] **→ Step 0.3**: Design the surprise signal f_surprise(loss, expected_loss)
- [ ] **→ Step 0.4**: Formalize the plastic weight injection architecture (hooks, API)
- [ ] **→ Step 0.5**: Define the evaluation protocol (metrics, baselines, success criteria)
