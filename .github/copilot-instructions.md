# PH-Neuro — Copilot Instructions

> These instructions tell GitHub Copilot how to write code for this project.
> PH-Neuro is a ternary Hebbian deep learning framework — NO backpropagation.

---

## Project Vision (KEEP THIS IN MIND)

PH-Neuro aims for **brain-like learning**: small, online, continual.

- **Small**: Ternary weights {-1, 0, +1} stored at 1 byte/weight (eventually 4 weights/byte). 
  No optimizer states, no gradient buffers, no replay buffers. Total memory is ~4× less than
  equivalent backprop networks.
- **Online**: Each sample updates the network once, on the fly. No full-dataset epoch training,
  no offline pre-training, no batch replay. The network learns while being used, same as a brain.
- **Continual**: No catastrophic forgetting. New tasks don't require retraining from scratch.
- **Local**: Every learning rule uses only pre-synaptic and post-synaptic activity at each synapse.
  No global loss signal, no backward pass, no gradient transport between layers.

### What this means for code decisions
- Prefer **online** rules (update per sample) over **batch** rules (update per epoch)
- Prefer **single-pass** learning over **multi-epoch** learning
- No replay buffers, no experience replay, no rehearsal
- Hidden layers must learn useful representations **without labels** (unsupervised)
- The output layer uses label information (WTA Hebbian), but hidden layers do not

---

## Core Rules (ALWAYS follow)

### 1. NEVER use backpropagation
- ❌ NEVER call `.backward()`
- ❌ NEVER use `torch.optim` (Adam, SGD, AdamW, etc.)
- ❌ NEVER use `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`
- ❌ NEVER use `torch.autograd` for learning
- ❌ NEVER compute gradients

### 2. Weights are TERNARY {-1, 0, +1}
- All weights use `TernaryTensor` — NOT `nn.Parameter`, NOT float tensors
- Forward pass: `ternary_sign(x)` activation, popcount MatMul
- Storage: naive int8 (1 byte/weight) during development, packed 2-bit later
- Weight values are ALWAYS in {-1, 0, +1} — never float

### 3. Learning is HEBBIAN (local, no global loss)
- Update rule: `Δlatent_score = lr × pre_activation × post_activation`
- Both pre and post are ternary {-1, 0, +1}
- If pre == post → strengthen (+lr)
- If pre != post → weaken (-lr)  
- If either is 0 → no update
- Learning is MANUAL (explicit weight updates in training loop)

### 4. Training loop pattern
```python
for x, y in dataloader:
    # Forward pass (ternary activations throughout)
    x_ternary = ternary_sign(x)
    h = model.layers(x_ternary)
    
    # Hebbian update (MANUAL, no loss.backward())
    target = one_hot(y) * 2 - 1  # {-1, +1}
    model.output_layer.hebbian_update(h, target, lr=0.01)
    
    # Refresh ternary weights (check thresholds)
    model.refresh_weights()
    
    # NO optimizer.step(), NO loss.backward()
```

### 5. No optimizer states, no gradient buffers
- Memory is ~4× less than backprop networks
- Each layer stores: ternary weights (int8) + latent scores (fp16) ONLY
- No Adam moments, no gradient accumulators, no activation checkpoints

---

## Project Structure

```
ph_neuro/
├── core/
│   ├── ternary_tensor.py      # TernaryTensor (int8 storage, pack/unpack)
│   ├── latent_scores.py        # LatentScoreTensor (fp16 scores)
│   └── hebbian_rules.py        # Basic, BCM, Oja, Anti-Hebbian, Predictive
├── layers/
│   ├── linear.py               # TernaryHebbianLinear
│   ├── conv.py                 # TernaryHebbianConv2d
│   └── attention.py            # TernaryHebbianAttention
├── training/
│   └── trainer.py              # HebbianTrainer (no optimizer!)
└── utils/
    └── popcount.py             # Popcount MatMul (future: CUDA kernel)
```

---

## Naming Conventions

| Concept | Class/Function Name |
|---------|-------------------|
| Ternary weight storage | `TernaryTensor` |
| Latent float scores | `LatentScoreTensor` |
| Linear layer (ternary + Hebbian) | `TernaryHebbianLinear` |
| Conv layer (ternary + Hebbian) | `TernaryHebbianConv2d` |
| Activation function (→ {-1,0,+1}) | `ternary_sign()` |
| Hebbian update method | `.hebbian_update(pre, post, lr)` |
| Threshold-based weight refresh | `.refresh_weights()` |
| Homeostatic decay | `decay_rate` parameter |
| Activation threshold (0→±1) | `theta_upper` |
| Deactivation threshold (±1→0) | `theta_lower` |
| Predictive Hebbian layer | `PredictiveHebbianLayer` |
| Working/echo state memory | `EchoStateMemory` |

---

## Key Patterns

### TernaryTensor usage
```python
# Create (starts all zeros)
w = TernaryTensor(shape=(out_dim, in_dim))

# Access as int8 {-1, 0, +1}
w_unpacked = w.unpack()  # torch.Tensor, dtype=int8

# Convert to float for MatMul
w_dense = w.to_dense()  # torch.Tensor, dtype=float32

# Pack (future optimization)
w_packed = w.pack()  # 4 weights per byte
```

### Hebbian update in a layer
```python
class TernaryHebbianLinear:
    def hebbian_update(self, pre, post, lr):
        # pre: (batch, in_features), ternary {-1,0,+1}
        # post: (batch, out_features), ternary {-1,0,+1}
        delta = lr * (pre.T @ post) / pre.shape[0]
        self.latent_scores += delta  # fp16 accumulation
```

### Hysteresis threshold logic
```python
def refresh_weights(self):
    # For each synapse:
    #   if weight==0 and |score| > θ_upper → flip to sign(score)
    #   if weight==±1 and |score| < θ_lower → flip to 0
    # Hysteresis gap (θ_upper - θ_lower) prevents oscillation
```

---

## What to AVOID

- `nn.Linear` → use `TernaryHebbianLinear`
- `nn.Conv2d` → use `TernaryHebbianConv2d`
- `F.linear()`, `F.conv2d()` → use ternary MatMul forward
- `nn.Parameter` for weights → use `TernaryTensor`
- `torch.optim.Adam` → NO optimizer
- `loss.backward()` → NO backward pass
- `model.train()` / `model.eval()` → not needed (no dropout, no batch norm in ternary mode)
- `gradient clipping` → no gradients exist
- `learning rate schedulers` → manual lr adjustment only

---

## Phase-Specific Context

### Phase 0 (complete): Core Mechanism
- Single-layer `TernaryHebbianLinear` on MNIST — **88.4% accuracy**
- Naive int8 storage (1 byte per weight)
- Float MatMul (not popcount yet)
- Winner-Take-All supervised Hebbian for output layer
- Verify: no `.backward()` anywhere

### Phase 1 (current): Multi-layer Vision POC
- Multi-layer MLP on MNIST (greedy layer-wise training)
- Hidden layers: **unsupervised competitive Hebbian** (winner-take-all with conscience mechanism)
  — neurons compete to represent input patterns, only the winner learns
  — this creates differentiated feature detectors, not PCA-like uniform features
- Output layer: supervised WTA Hebbian (same as Phase 0)
- `TernaryHebbianConv2d` with local Hebbian rule (later)
- Continual learning on split MNIST (later)

### Key Learning: Why basic Hebbian fails for hidden layers
Basic Hebbian (`ΔW = lr × postᵀ @ pre`) makes all hidden neurons learn the same
pattern (positive feedback loop). This is useless for hierarchical representations.
**Competitive Hebbian** (winner-take-all + conscience) is required to force different
neurons to specialize on different input patterns — analogous to cortical competition.

### Phase 3: Language
- `PredictiveHebbianLayer` with echo state memory
- Brain-inspired modular architecture (encoder + memory + decoder)
- Training: token-by-token sequential, NOT parallel

---

## Key References (for context)

- **PH-Neuro ROADMAP**: `docs/ROADMAP.md` — full project plan
- **Phase details**: `docs/phase-*.md`
- **PH-Net sibling project**: `/home/phalo/PH-Net/` — ternary weights + STE backprop (different approach)
- **SoftHebb** (ICLR 2023): float Hebbian baseline
- **Kim et al. 2017** (arXiv:1711.08679): only prior ternary+Hebbian work (single layer, MNIST only)

---

## Testing

- Tests NEVER import `torch.optim`
- `test_no_backward.py`: verifies autograd is never engaged
- `test_ternary_weights.py`: verifies all weights ∈ {-1, 0, +1} at all times
- `test_hebbian_update.py`: manual computation vs implementation
