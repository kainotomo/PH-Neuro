# Phase 4 — Scale & Advanced Features

> **Goal:** Push to 1B+ parameters. Explore architectural innovations. Measure scaling properties.  
> **Duration:** ~2-3 months  
> **Hardware:** RTX 4060 8 GB for 1B model; cloud GPU for 7B model  
> **Success:** 1B ternary Hebbian model competitive with small backprop models at fraction of training cost

---

## Overview

Phase 4 asks the scaling question: does Hebbian learning get better with scale? Backprop shows clear scaling laws — bigger models, more data → better performance. Does Hebbian learning follow the same trajectory, or does it saturate?

We also explore architectural innovations that are uniquely suited to Hebbian learning: MoE with self-organizing experts, continual language learning, and interpretable weight surgery.

---

## 4.1 Scale to 1B Parameters

### Model Config

```yaml
model:
  vocab_size: 32000
  d_model: 2048
  n_heads: 16
  n_layers: 24
  d_ff: 8192
  max_seq_len: 2048
  total_params: ~1B

memory:
  ternary_weights: ~400 MB (1B × 2 bits / 8)
  latent_scores: ~2 GB (1B × fp16)
  activations: ~1 GB (batch × seq × d_model)
  total: ~3.4 GB — FITS on 8 GB VRAM
```

### Training

```yaml
dataset: SlimPajama or FineWeb Edu (100B+ tokens)
batch_size: 8 (with gradient accumulation to effective 64)
seq_len: 2048
training_steps: 100,000
expected_wall_time: ~2-4 weeks on RTX 4060
```

### Optimization for Speed

Without backprop, the bottleneck becomes:
1. Forward pass (MatMul with ternary weights)
2. Hebbian update (outer product per layer per step)

```python
# Naive Hebbian update: O(batch × in_dim × out_dim) per layer
delta = lr * (pre_activation.T @ post_activation) / batch_size

# Optimized: fused CUDA kernel for ternary MatMul + Hebbian update
# This is a single kernel that:
#   1. Computes ternary MatMul (popcount)
#   2. Computes outer product for Hebbian update
#   3. Accumulates latent scores
#   4. Checks thresholds, flips weights
```

For Phase 4, implement a basic fused kernel. The Hebbian update is embarrassingly parallel (each synapse independent) — ideal for GPU.

### Scaling Law Experiment

Train models at 4 sizes: 10M, 100M, 300M, 1B. Same data, same steps.

| Params | Perplexity | Tokens/sec (training) | Memory |
|--------|-----------|----------------------|--------|
| 10M | ? | ? | <1 GB |
| 100M | ? | ? | <1 GB |
| 300M | ? | ? | ~1.5 GB |
| 1B | ? | ? | ~3.4 GB |

Plot perplexity vs params, perplexity vs FLOPs. Does it follow a power law?

Compare to Chinchilla-optimal scaling for backprop. Hypothesis: Hebbian scaling is less efficient (more params needed for same perplexity) but the compute-per-param is much lower, so the crossover point may be interesting.

---

## 4.2 Mixture of Experts (MoE)

### Why MoE for Hebbian?

MoE is a natural fit for Hebbian learning:
- Expert selection is "which neurons fire" — inherently Hebbian
- Inactive experts don't get updated — natural specialization
- Load balancing emerges from competition, not loss functions

### Ternary Hebbian MoE Architecture

```python
class TernaryHebbianMoE(nn.Module):
    def __init__(self, d_model, n_experts, expert_size, top_k=2):
        self.router = TernaryHebbianLinear(d_model, n_experts)  # Select experts
        self.experts = nn.ModuleList([
            TernaryHebbianFFN(d_model, expert_size) 
            for _ in range(n_experts)
        ])
        self.top_k = top_k
    
    def forward(self, x):
        # Router: which experts fire?
        router_logits = self.router(x)  # (batch, n_experts), float from ternary matmul
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute selected expert outputs
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = top_k_weights[mask][top_k_indices[mask] == i]
                output[mask] += weight.unsqueeze(-1) * expert_out
        
        return ternary_sign(output)
```

### Hebbian Expert Selection

The router is also ternary Hebbian. It learns which experts fire for which inputs:

```python
# Router Hebbian update:
# "This input pattern → this expert should fire"
# pre = input tokens, post = which experts were selected (one-hot-ish)
router.hebbian_update(x_ternary, expert_selection_mask, lr)
```

### Natural Load Balancing

Without an explicit load balancing loss, will experts self-organize?
- Some experts may handle common patterns (frequent tokens)
- Others may specialize in rare patterns
- "Dead" experts (never selected) naturally have their connections decay
- This is how the brain does it — no global load balancing

### Experiment

Compare:
- Dense 1B model vs MoE 1B model (4×256M experts, top-2 active → ~512M active params)
- Does MoE improve perplexity? Throughput?
- Do experts actually specialize? Measure expert activation patterns.

---

## 4.3 Advanced Hebbian Rules

### BCM (Bienenstock-Cooper-Munro)

```python
def bcm_update(pre, post, theta_M, lr):
    """
    Δw = lr × pre × post × (post - θ_M)
    
    θ_M is a sliding threshold: average of post^2 over recent history.
    If post > θ_M → LTP (strengthen)
    If post < θ_M → LTD (weaken)
    
    This creates competition: only strongly active neurons get strengthened.
    """
    delta = lr * pre * post * (post - theta_M)
    return delta

def update_theta_M(theta_M, post, tau=1000):
    """Moving average: θ_M = (1 - 1/τ) × θ_M + (1/τ) × post²"""
    return (1 - 1/tau) * theta_M + (1/tau) * (post ** 2).mean()
```

BCM is more biologically realistic and may produce:
- Better feature selectivity (neurons compete)
- Natural sparsity (only "winning" neurons get strengthened)
- Homeostatic regulation (threshold adapts to activity level)

### Oja's Rule

```python
def oja_update(pre, post, weight, lr):
    """
    Δw = lr × (pre × post - α × weight × post²)
    
    Normalized Hebbian. Prevents weight explosion.
    Converges to first principal component of input.
    """
    delta = lr * (pre * post - alpha * weight * post * post)
    return delta
```

For ternary weights, normalization is less critical (weights are bounded to {-1, 0, +1}), but Oja's rule may produce more diverse features.

### Ablation: Which Rule for Which Layer?

| Rule | Early Layers | Late Layers | Output Layer |
|------|-------------|-------------|--------------|
| Basic Hebbian | Good (feature detection) | OK | OK |
| BCM | Better (competitive features) | Good | Not needed (supervised) |
| Oja | Better (prevents collapse) | OK | Not needed |
| Anti-Hebbian | — | — | Required (wrong class suppression) |

Experiment: grid search over rule assignments per layer.

---

## 4.4 Continual Language Learning

This is where PH-Neuro should truly shine.

### Experiment: Language-Incremental Learning

```
Phase 1: Train on English Wikipedia (100M tokens)
Phase 2: Add French Wikipedia (100M tokens)  
Phase 3: Add GitHub code (100M tokens)
Phase 4: Add PubMed abstracts (100M tokens)
```

After each phase, measure:
- Perplexity on English (should stay stable — no forgetting)
- Perplexity on French (new knowledge acquired)
- Perplexity on code (new domain learned)
- Perplexity on all previous domains (cumulative knowledge)

### Expected Results

| Phase | English PPL | French PPL | Code PPL | PubMed PPL |
|-------|------------|------------|----------|------------|
| After English | 30 | — | — | — |
| After +French | 31 (+3%) | 45 | — | — |
| After +Code | 32 (+7%) | 47 (+4%) | 55 | — |
| After +PubMed | 33 (+10%) | 48 (+7%) | 57 (+4%) | 50 |

Backprop would show catastrophic forgetting: English PPL might jump to 80+ after learning French.

### Domain-Incremental Learning

Same as above but with domain shift within the same language:
- News → Fiction → Academic → Social Media → Legal

Does PH-Neuro adapt to each domain while retaining previous ones?

---

## 4.5 Interpretability & Weight Surgery

Ternary weights enable a unique capability: **manual weight editing**.

### Tracing Circuits

```python
def find_path(input_neuron, output_neuron, model):
    """Find all paths of non-zero weights from input to output."""
    # BFS through ternary weight matrices
    # Each step: which neurons in next layer does this neuron connect to?
    # Since weights are {-1, 0, +1}, connections are explicit
    ...

def explain_prediction(model, x):
    """Which synapses contributed to this prediction?"""
    # For each activated neuron, trace back through non-zero weights
    # Highlight the "voting chain" that led to the output
    ...
```

### Weight Surgery

```python
def forget_concept(model, concept_neurons):
    """Manually set weights to 0 to remove a concept."""
    for neuron_id in concept_neurons:
        # Set all outgoing connections from this neuron to 0
        model.layers[neuron_id.layer].weights[neuron_id.idx, :] = 0
        # Set all incoming connections to this neuron to 0
        model.layers[neuron_id.layer-1].weights[:, neuron_id.idx] = 0

def strengthen_concept(model, concept_neurons):
    """Manually set weights to +1 to reinforce a concept."""
    for conn in concept_connections:
        model.layers[conn.layer].weights[conn.from_idx, conn.to_idx] = 1
```

This is impossible with float networks (weights are entangled, changing one affects everything). With ternary weights, each connection is independent and semantically meaningful.

### Safety Implications

- **Auditability**: Every weight is inspectable. A +1 connection from token "kill" to token "yourself" is visible and can be flagged.
- **Editability**: Harmful connections can be surgically removed without retraining.
- **Explainability**: Predictions can be traced through explicit connection paths.

---

## 4.6 The Path to 7B (Cloud)

If 1B results are promising, scale to 7B on cloud GPU:

| GPU | VRAM | Estimated Cost | Duration |
|-----|------|---------------|----------|
| RTX 4090 24 GB | 24 GB | ~€1-2/hr | 2-4 weeks (~€300-600) |
| A100 80 GB | 80 GB | ~€2-4/hr | 1-2 weeks (~€500-1000) |

7B ternary model training memory:
- Ternary weights: ~3 GB (7B × 2 bits)
- Latent scores: ~14 GB (7B × fp16)
- Activations: ~5 GB
- **Total: ~22 GB** — fits on RTX 4090 24 GB!

This is the killer app: training a 7B model on a consumer GPU (RTX 4090) that would need an A100 80GB for backprop training.

---

## Deliverables

- [ ] 1B ternary Hebbian Transformer trained on 100B+ tokens
- [ ] Scaling law plot (params vs perplexity)
- [ ] MoE architecture with ternary Hebbian routing
- [ ] Comparison of Hebbian variants (Basic vs BCM vs Oja) at scale
- [ ] Continual language learning results (English → French → Code)
- [ ] Interpretability demo (circuit tracing, weight surgery)
- [ ] Fused CUDA kernel for Hebbian MatMul + update (if needed for speed)
- [ ] (Stretch) 7B model on cloud GPU

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 1B model doesn't improve over 100M | Medium | Hebbian learning may not scale with params — accept and document |
| Training too slow (popcount not optimized) | Medium | Implement fused kernel; PyTorch 2.0 compile may help |
| Continual learning shows some forgetting | Medium | Tune hysteresis parameters; may need per-task learning rate schedules |
| MoE experts don't specialize | Low-Medium | Add soft regularization (entropy bonus for diverse routing) |

---

## What's Next

After Phase 4 → Phase 5: Package as `pip install ph-neuro`, write paper, release pre-trained models.
