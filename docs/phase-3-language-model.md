# Phase 3 — First Language Model

> **Goal:** Show ternary Hebbian Transformers can learn statistical regularities in text and generate coherent output.  
> **Duration:** ~4-6 weeks  
> **Hardware:** RTX 4060 8 GB — 100M model fits easily (~400 MB ternary weights + 200 MB latent scores)  
> **Success:** 100M ternary Hebbian Transformer generates coherent paragraphs on TinyStories

---

## Overview

This is genuinely unexplored territory. No Hebbian network has been trained on language before. SoftHebb only did vision. The Forward-Forward paper showed MNIST/CIFAR-10 results but no language experiments.

Phase 3 asks: can a network that learns by "cells that fire together wire together" capture the statistical structure of human language?

The answer is probably "yes, but worse than backprop." The question is: HOW much worse, and is it still useful?

---

## 3.1 Hebbian Transformer Architecture

### TernaryHebbianLinear for All Projections

A standard Transformer block has 4 linear projections per attention layer + 2 in the FFN. All become `TernaryHebbianLinear`:

```
Transformer Block:
├── Attention:
│   ├── Q: TernaryHebbianLinear(d_model → d_model)
│   ├── K: TernaryHebbianLinear(d_model → d_model)
│   ├── V: TernaryHebbianLinear(d_model → d_model)
│   └── O: TernaryHebbianLinear(d_model → d_model)
└── FFN:
    ├── Up: TernaryHebbianLinear(d_model → 4*d_model)
    └── Down: TernaryHebbianLinear(4*d_model → d_model)
```

### Ternary Embeddings

```python
class TernaryHebbianEmbedding(nn.Module):
    """Ternary embedding table: vocab_size × d_model, values in {-1, 0, +1}."""
    def __init__(self, vocab_size, d_model, theta_upper=5.0, theta_lower=1.0):
        self.latent = LatentScoreTensor(vocab_size, d_model)
        self.weights = TernaryTensor(vocab_size, d_model)
        # Initialize some embeddings to non-zero (random ternary)
        self.weights.random_init_(sparsity=0.5)  # 50% zero, 25% +1, 25% -1
    
    def forward(self, token_ids):
        return self.weights[token_ids]  # (batch, seq_len, d_model), ternary
```

**Key question**: Should embeddings be trainable via Hebbian? Unlike other weights, embeddings don't have a clear "pre × post" pair — they're looked up by token ID.

Options:
- **Fixed random ternary**: Don't train embeddings at all. Surprisingly common in some architectures.
- **Train via co-occurrence Hebbian**: When two tokens appear in similar contexts, make their embeddings more similar. (Distributional hypothesis via Hebbian.)
- **Train on output**: The embedding matrix IS the output projection (weight tying). Train via the output Hebbian signal.

### Attention with Ternary Q, K, V

```python
def ternary_scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, heads, seq_len, d_head) — ternary {-1, 0, +1}
    
    Attention scores: Q @ K^T → integer values in [-d_head, +d_head]
    """
    # Attention scores are integers (popcount, not float multiply-add)
    scores = Q @ K.transpose(-2, -1)  # Integer matmul
    scores = scores / (d_head ** 0.5)  # Scale to prevent large values
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax on integer scores — standard, but explore alternatives
    attn_weights = F.softmax(scores.float(), dim=-1)
    
    # Weighted sum of ternary values → float output
    # Then re-ternarize for next layer
    output = attn_weights @ V.float()
    return ternary_sign(output)
```

**Alternative to softmax**: Hard attention (select top-k), sparsemax, or entmax. Softmax destroys the "integer purity" but may be necessary for gradient-free training.

### Position Encoding

RoPE is compatible with ternary Q/K: just rotate before attention. Or use ALiBi (additive bias, just integer addition).

---

## 3.2 Training Strategy

### The Core Challenge

Hebbian learning has no loss function. Language modeling's objective is "predict the next token." How do you provide a Hebbian teaching signal for this?

### Approach A: Next-Token Hebbian (Primary)

```python
for batch in train_loader:
    input_ids = batch[:, :-1]   # (batch, seq_len-1)
    target_ids = batch[:, 1:]   # (batch, seq_len-1)
    
    # Forward pass through all layers
    h = model.embedding(input_ids)  # (batch, seq_len, d_model), ternary
    for layer in model.layers:
        h = layer(h)  # Each layer: attention + FFN, output is ternary
    
    # Output: project to vocab space (TernaryHebbianLinear)
    logits = model.output_projection(h)  # (batch, seq_len, vocab_size)
    
    # Convert targets to {-1, +1} one-hot
    # +1 at correct token position, -1 elsewhere
    target = torch.full((batch, seq_len, vocab_size), -1, dtype=torch.int8)
    target.scatter_(-1, target_ids.unsqueeze(-1), 1)
    
    # Hebbian update for output layer
    # Strengthen connections that predict the correct token
    model.output_projection.hebbian_update(
        h,                    # pre: hidden state (ternary)
        target,               # post: one-hot target {-1, +1}
        lr=lr_output
    )
    
    # Hidden layers: self-organizing
    # Each layer's "post" is its own output
    for i, layer in enumerate(model.layers):
        layer.hebbian_update(
            layer_inputs[i],   # pre: input to this layer
            layer_outputs[i],  # post: this layer's output
            lr=lr_hidden
        )
    
    # Refresh weights
    model.refresh_all_weights()
```

### Approach B: Contrastive Hebbian for Sequences

```python
# Positive pass: real next token
h_pos = model(input_ids)
for layer, (inp, out) in zip(model.layers, layer_pairs_pos):
    layer.hebbian_update(inp, out, lr=+lr)

# Negative pass: random/wrong next token
wrong_ids = torch.randint(0, vocab_size, target_ids.shape)
# Or: use model's own prediction as negative
h_neg = model(wrong_ids)
for layer, (inp, out) in zip(model.layers, layer_pairs_neg):
    layer.hebbian_update(inp, out, lr=-lr)  # Anti-Hebbian
```

### Approach C: Layer-wise Next-Token Prediction

Each layer independently tries to predict the next token:

```python
for i, layer in enumerate(model.layers):
    h = layer(h_prev)
    
    # Each layer has its own output projection
    layer_logits = layer.output_head(h)
    
    # Train layer i to predict next token from its own representation
    layer.output_head.hebbian_update(h, target, lr)
```

This is like having a classifier at every layer (similar to how some interpretability work probes intermediate representations). The final model uses only the last layer's prediction, but training signal reaches all layers.

### Approach D: Embedding Co-occurrence as Training Signal

Instead of next-token prediction, train on token co-occurrence:

```python
# Within a window of size W, strengthen connections between co-occurring tokens
for i in range(seq_len):
    for j in range(max(0, i-W), min(seq_len, i+W+1)):
        if i != j:
            # Tokens i and j co-occur → make their embeddings more similar
            emb_i = model.embedding.weights[input_ids[i]]
            emb_j = model.embedding.weights[input_ids[j]]
            # Hebbian update: emb_i and emb_j should be correlated
```

This builds a distributional embedding space — the foundation for understanding word meaning. Combined with Approach A for generation.

---

## 3.3 TinyStories Experiment

### Model Config

```yaml
model:
  vocab_size: 5000  # TinyStories has small vocab
  d_model: 512
  n_heads: 8
  n_layers: 8
  d_ff: 2048
  max_seq_len: 256
  total_params: ~100M

training:
  dataset: TinyStories (~2M stories, ~500M tokens)
  batch_size: 32
  seq_len: 256
  lr_output: 0.01
  lr_hidden: 0.001
  theta_upper: 5.0
  theta_lower: 1.0
  decay: 1e-6
  steps: 50,000
```

### Evaluation

1. **Perplexity**: Standard metric. Expect higher than backprop (maybe 2-5×) but should be << random baseline.
2. **Generation quality**: Sample 100 continuations from 100 prompts. Human evaluation:
   - 1: Nonsense / word salad
   - 2: Some structure but incoherent
   - 3: Mostly coherent with occasional errors
   - 4: Coherent, grammatical, makes sense
   - 5: Indistinguishable from human-written
3. **Diversity**: Do different prompts produce different stories? Or does the model collapse to a few templates?
4. **Grammar**: Does the model learn basic syntax? Subject-verb agreement? Word order?

### Baselines

| Model | Perplexity | Generation Quality |
|-------|-----------|-------------------|
| Random ternary weights | ~vocab_size | 1 (nonsense) |
| PH-Neuro 100M (Hebbian) | ? | ? (target: ≥3) |
| GPT-2 Small (backprop, 124M) | ~20-25 | 4-5 |
| TinyStories-1M (backprop, ~1M) | ~30-40 | 3-4 |
| n-gram baseline (n=5) | ~40-50 | 2-3 |

### Success Criteria

- **Minimum**: Perplexity <100, generation quality ≥2 (some structure)
- **Good**: Perplexity <50, generation quality ≥3 (coherent)
- **Exceptional**: Perplexity <30, generation quality ≥3.5 — competitive with small backprop models

---

## 3.4 Analysis

### What Do Ternary Embeddings Look Like?

```python
# t-SNE of learned ternary embeddings
embeddings = model.embedding.weights.unpack()  # (vocab_size, d_model)
tsne = TSNE(n_components=2).fit_transform(embeddings.float())
# Are semantically similar words close? (cat near dog, run near walk?)
```

### Attention Pattern Analysis

```python
# For a given input, what do attention heads attend to?
# Are there:
#   - Positional heads (attend to previous token)?
#   - Syntactic heads (attend to subject/verb)?
#   - Semantic heads (attend to related concepts)?
```

### Layer-wise Representations

Probe each layer's hidden state with a linear classifier:
- Can we predict the next token from layer 1? Layer 4? Layer 8?
- Does representation quality improve monotonically with depth?
- Or do intermediate layers sometimes lose information?

### Weight Sparsity Over Time

```python
# Track % of ternary weights that are 0 during training
# Initial: ~50% (random init)
# After training: ? 
# Hypothesis: sparsity increases as model becomes more selective
```

---

## 3.5 Ablations

| Ablation | Question |
|----------|----------|
| No hidden layer updates | Does only training the output layer work? |
| Fixed random embeddings | Are learned embeddings necessary? |
| Without softmax (hard attention) | Is softmax needed or can we use hard winner-take-all? |
| Training approach A vs B vs C | Which Hebbian strategy works best for language? |
| With/without anti-Hebbian | Does negative signal help? |
| Different Hebbian variants | Basic vs Oja vs BCM for language? |

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model doesn't learn anything (random output) | Medium | High | Try contrastive approach, simplify to bigram Hebbian, verify on tiny synthetic data |
| Perplexity is very high (>200) | Medium | Medium | Accept this is inherent to Hebbian; focus on qualitative analysis |
| Training is slow (popcount not optimized) | High | Low | Use float MatMul for initial experiments, optimize later |
| Greedy layer-wise doesn't work for attention | Medium | High | Q/K/V/O can be trained with the same approach but attention mixes signals — may need different strategy |

---

## What's Next

After Phase 3 → Phase 4: Scale to 1B+ parameters, explore MoE, advanced Hebbian rules, continual language learning.
