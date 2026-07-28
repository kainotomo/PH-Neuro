# Phase 3 — Language: The Brain-Inspired Approach

> **Goal:** Show ternary Hebbian networks can learn sequential structure and language — not by mimicking Transformers, but by mimicking the brain's modular, temporally-aware architecture.  
> **Duration:** ~6-8 weeks (3a: 1-2 weeks, 3b: 2-3 weeks, 3c: 3-4 weeks)  
> **Hardware:** RTX 4060 8 GB — all models fit easily  
> **Success:** Coherent text generation from a brain-inspired architecture that never computes a gradient

---

## Overview

The brain doesn't use backpropagation. It doesn't use Transformers. It doesn't minimize cross-entropy loss. Yet a 5-year-old child, exposed to ~10-50 million words, speaks fluently with grammar, syntax, and meaning.

**How?** The brain uses:
1. **Predictive coding** — constantly predicting the next input, learning from prediction errors
2. **Working memory** — echo state / reservoir dynamics that maintain temporal context
3. **Hierarchical timescales** — different layers operate at different temporal resolutions (phoneme → word → phrase)
4. **Modular architecture** — specialized regions (Broca, Wernicke, hippocampus), not one monolithic network

Phase 3 asks: **can we replicate these principles with ternary Hebbian networks?**

This is NOT "put a Transformer with Hebbian." This is "build a brain-inspired language architecture from scratch."

---

## Phase 3a — Sequence Learning (SANITY CHECK)

**Goal:** Before attempting natural language, prove the Hebbian mechanism can learn sequential structure at all.

**Duration:** ~1-2 weeks

### 3a.1 n-gram Prediction

Train on synthetic sequences where P(next|context) follows a known distribution.

```python
# Generate sequences from a known n-gram model
# P(A|A)=0.6, P(B|A)=0.4, P(A|B)=0.3, P(B|B)=0.7 ...
def generate_ngram_sequence(length, transition_probs):
    seq = [random.choice(states)]
    for _ in range(length - 1):
        next_state = random.choices(states, 
            weights=[transition_probs[seq[-1]][s] for s in states])[0]
        seq.append(next_state)
    return seq
```

**Architecture**: Simple predictive Hebbian layer with working memory — no attention, no Transformer. Just: input → echo state → prediction → error → Hebbian update.

**Question**: Can a ternary Hebbian network learn P(next|previous)? This is the simplest possible sequential task.

**Success**: Matches count-based n-gram within 10%.

### 3a.2 Reber Grammar

Reber grammar is a classic artificial grammar — a finite-state automaton that generates strings like "BTSSXXVVE":

```
        ┌─── T ───┐
        │         ▼
   ┌─B─┴─P─────── X ── V ──┐
   │            │           │
   └──► S ──────┘           E
        │                   ▲
        └── T ── X ── S ────┘
```

Valid strings follow the grammar. Invalid strings violate it (e.g., "BTS" is invalid because S needs X after T).

**Question**: Can Hebbian learning discover abstract grammatical rules, not just surface correlations?

**Architecture**: 2-3 predictive Hebbian layers with echo state memory.

**Success**: >90% accuracy on valid/invalid classification after training only on valid strings.

### 3a.3 Toy Language

A miniature natural language: 100 words, 5 grammar rules.

```python
# Grammar rules:
# 1. S → NP VP
# 2. NP → Det Adj* N
# 3. VP → V NP | V
# 4. Adj agrees with N (gender/number)
# 5. Det agrees with N (definite/indefinite)

# Vocabulary (~100 words):
# Det: the, a, this, that
# Adj: big, small, red, blue, happy, sad
# N: cat, dog, bird, house, tree, book, boy, girl
# V: sees, chases, likes, eats, finds, gives
```

**Question**: Can Hebbian learning capture syntactic structure (word order, agreement)?

**Success**: Generates grammatically correct sentences >80% of the time.

---

## Phase 3b — Predictive Hebbian (THE MECHANISM)

**Goal:** Implement and validate predictive coding as the core learning mechanism.

**Duration:** ~2-3 weeks

### The Theory: Why Predictive Coding?

The brain doesn't say "this word follows that word." It says:
> "Based on everything I've heard so far, I PREDICT the next word will be X. ... Oh, it was Y instead. The error (Y-X) tells me how to update my connections."

| | Basic Hebbian | Predictive Hebbian |
|---|---|---|
| **What is learned** | Co-occurrence: "A and B happen together" | Causality: "A CAUSES B, and when it doesn't, something's wrong" |
| **Update signal** | `pre × post` (correlation) | `pre × (post_actual - post_predicted)` (prediction error) |
| **Temporal awareness** | None (stateless) | Implicit (prediction requires temporal model) |
| **Biological basis** | LTP/LTD from co-activation | Cortical predictive coding (Rao & Ballard, 1999) |
| **Language suitability** | Poor (language is sequential) | Good (prediction is the core of language processing) |

### 3b.1 Working Memory (Echo State)

```python
class EchoStateMemory(nn.Module):
    """
    Leaky integrator — maintains a running average of past inputs.
    No backprop-through-time needed. State evolves autonomously.
    """
    def __init__(self, dim, decay=0.9):
        super().__init__()
        self.decay = decay
        self.register_buffer('state', torch.zeros(1, dim))
    
    def forward(self, x):
        self.state = self.decay * self.state + (1 - self.decay) * x.mean(dim=0, keepdim=True)
        return torch.cat([x, self.state.expand(x.shape[0], -1)], dim=-1)
    
    def reset(self):
        self.state.zero_()
```

### 3b.2 Hierarchical Timescales

| Timescale | Duration | What it captures | Decay rate | Layer |
|-----------|----------|-----------------|------------|-------|
| Phonetic | ~50ms | Individual sounds | 0.3-0.5 | Layer 1 |
| Syllabic | ~200ms | Syllable patterns | 0.7-0.8 | Layer 2 |
| Lexical | ~500ms | Word identity | 0.9-0.95 | Layer 3 |
| Phrasal | ~2s | Multi-word phrases | 0.98-0.99 | Layer 4 |
| Sentential | ~5-10s | Sentence meaning | 0.995 | Global memory |

### 3b.3 Predictive Hebbian Layer

```python
class PredictiveHebbianLayer(nn.Module):
    """
    A layer that:
    1. Maintains temporal context via echo state memory
    2. PREDICTS its next output based on current input + memory
    3. Computes prediction error when actual next input arrives
    4. Updates weights via Hebbian rule on prediction error
    """
    def __init__(self, in_dim, out_dim, memory_decay=0.9,
                 theta_upper=5.0, theta_lower=1.0):
        super().__init__()
        self.linear = TernaryHebbianLinear(in_dim * 2, out_dim,
                                            theta_upper, theta_lower)
        self.predictor = TernaryHebbianLinear(out_dim, out_dim,
                                               theta_upper, theta_lower)
        self.memory = EchoStateMemory(out_dim, decay=memory_decay)
    
    def forward(self, x):
        x_with_memory = self.memory(x)
        h = ternary_sign(self.linear(x_with_memory))
        prediction = self.predictor(h)
        return h, prediction
    
    def learn(self, x_current, x_next, lr):
        _, prediction = self.forward(x_current)
        actual = ternary_sign(x_next)
        error = actual - prediction  # Prediction error
        
        x_with_memory = torch.cat([x_current, 
            self.memory.state.expand(x_current.shape[0], -1)], dim=-1)
        self.linear.hebbian_update(x_with_memory, error, lr)
        self.predictor.hebbian_update(h_current, error, lr)
```

### 3b.4 Experiment: Basic vs Predictive Hebbian

| Method | n-gram acc | Reber grammar | Toy language |
|--------|-----------|---------------|-------------|
| Basic Hebbian | ? | ? | ? |
| Predictive Hebbian | ? | ? | ? |

**Hypothesis**: Predictive Hebbian significantly outperforms basic Hebbian on sequential tasks.

---

## Phase 3c — TinyStories (NATURAL LANGUAGE)

**Goal:** Apply the predictive Hebbian architecture to real natural language.

**Duration:** ~3-4 weeks

**⚠️ Only proceed after 3a and 3b succeed.**

### 3c.1 Brain-Inspired Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PH-Neuro Language Model              │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ ENCODER  │    │ WORKING  │    │ DECODER  │          │
│  │(Wernicke)│───▶│ MEMORY   │───▶│ (Broca)  │──▶ output│
│  │          │    │(Hippocam)│    │          │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                                         │
│  All modules: ternary weights, Hebbian, no backprop     │
└─────────────────────────────────────────────────────────┘
```

| Module | Brain Region | Function | Timescale |
|--------|-------------|----------|-----------|
| **Encoder** | Wernicke's area | Input words → meaning | Fast (decay=0.85) |
| **Working Memory** | Hippocampus | Episodic context | Very slow (decay=0.995) |
| **Decoder** | Broca's area | Meaning → output words | Medium (decay=0.9) |

### 3c.2 Training Loop

```python
def train_step(model, batch, lr):
    model.working_memory.reset()
    
    for t in range(seq_len):
        encoded = model.encoder(batch[:, t])
        encoded_with_memory = model.working_memory(encoded)
        logits_t = model.decoder(encoded_with_memory)
        
        target = F.one_hot(batch[:, t+1], vocab_size).float() * 2 - 1
        prediction = ternary_sign(logits_t)
        error = target - prediction.float()
        
        # Hebbian updates on ALL layers based on prediction error
        model.encoder.update_all_layers(batch[:, t], error, lr)
        model.decoder.update_all_layers(encoded_with_memory, error, lr)
        
        model.refresh_all_weights()
```

### 3c.3 Model Config

```yaml
model:
  vocab_size: 5000
  d_model: 512
  encoder_layers: 4
  decoder_layers: 4
  memory_decay: 0.995    # Episodic — very slow
  encoder_decay: 0.85     # Fast — word level
  decoder_decay: 0.9      # Medium — phrase level
  theta_upper: 5.0
  theta_lower: 1.0
  total_params: ~100M

training:
  dataset: TinyStories (~500M tokens)
  batch_size: 1 (one story at a time)
  seq_len: 256
  lr: 0.01
  decay: 1e-6
  steps: 50,000
```

### 3c.4 Success Criteria

| Level | Perplexity | Generation Quality | Grammar |
|-------|-----------|-------------------|---------|
| Minimum | <200 | ≥2 (some structure) | — |
| Good | <100 | ≥3 (coherent) | Basic syntax |
| Exceptional | <60 | ≥3.5 | Proper agreement |

---

## Ablations

| Ablation | Question |
|----------|----------|
| No working memory | Is temporal context essential? |
| Single timescale | Are hierarchical timescales necessary? |
| Monolithic vs modular | Does brain-inspired architecture outperform one big Transformer? |
| Predictive vs basic Hebbian | Is prediction error better than co-occurrence? |

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Predictive Hebbian doesn't help | Medium | Revisit theory; basic Hebbian + echo state may be enough |
| Model produces word salad | Low-Medium | Fall back to toy language, simplify |
| Sequential processing too slow | High | Batch-sequential with parallel memory states |
| Working memory doesn't capture long context | Medium | Multiple memory slots, hierarchical memory |

---

## What's Next

After Phase 3 → Phase 4: Scale to 1B+, explore MoE, continual language learning (English → French → Code without forgetting).
