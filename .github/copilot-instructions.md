# PH-Neuro — Copilot Instructions

> These instructions tell GitHub Copilot how to write code for this project.
> PH-Neuro is a **ternary deep learning framework** — weights are {-1, 0, +1}.
> We use **DQT (Direct Quantized Training)** with stochastic rounding — NO latent float scores.

---

## Project Vision (KEEP THIS IN MIND)

PH-Neuro aims for **the smallest deep learning models in the world.**

- **Ternary weights**: {-1, 0, +1} stored at 2 bits/weight. 8× smaller than FP16.
- **DQT training**: No latent float scores during training. 4.5× less training memory than STE.
- **MoE (Mixture of Experts)**: Sparse activation — only 50% of parameters active per input.
- **Edge-first**: Models train on consumer GPUs, run on phones/microcontrollers.

### What this means for code decisions
- Prefer **DQT** (stochastic rounding) over **STE** (latent scores) for memory efficiency
- Prefer **ternary weights** over float/int8/int4
- Prefer **MoE** over dense for scaling
- Memory efficiency > speed > accuracy (accuracy must be competitive but memory is the USP)
- Training infrastructure is PyTorch with custom autograd functions

---

## Core Rules (ALWAYS follow)

### 1. Weights are TERNARY {-1, 0, +1}
- Forward pass: `sign(W_latent)` for STE, or stored int8 for DQT
- Storage: int8 naive (1 byte/weight) for dev, packed 2-bit for production
- Weight values in the model are ALWAYS in {-1, 0, +1}

### 2. DQT is the PREFERRED training method
- `TernaryDQTLinear` — stochastic rounding, no latent float scores
- Forward: ternary weights as int8
- Backward: custom `_DQTGradFn` autograd function
- Update: optimizer step on float buffer → stochastic rounding → int8 ternary
- 4.5× less training memory than STE

### 3. STE is the FALLBACK
- `TernarySTELinear` — latent float scores + sign() in forward
- Use only when DQT fails to converge

### 4. MoE layers
- `TernaryDQTMoELayer` — multiple ternary experts + float router
- Top-K routing, load balancing loss required
- Only active experts run forward/backward
- Router must use lower learning rate (0.1× expert lr)

### 5. Training loop pattern (DQT)
```python
for x, y in train_loader:
    optimizer.zero_grad()
    out = model(x)           # forward with ternary weights
    loss = F.cross_entropy(out, y)
    loss.backward()          # STE through sign, custom grad for DQT
    optimizer.step()         # updates float buffer
    model.apply_stochastic_rounding()  # ternary weights updated
```

---

## Project Structure

```
ph_neuro/
├── layers/
│   ├── ste_dqt.py              # TernaryDQTLinear (stochastic rounding)
│   ├── ste_dqt_moe.py          # TernaryDQTMoELayer (Mixture of Experts)
│   ├── ste_linear.py           # TernarySTELinear (STE, fallback)
│   ├── ste_hysteresis.py       # HysteresisSTELinear (sparsity regularizer)
│   └── ste_conv.py             # TernarySTEConv2d
├── models/
│   ├── ste_models.py           # ste_mlp, ste_cnn factories
│   └── ste_models_lora.py      # QLoRA-enabled models
├── training/
│   ├── ewc.py                  # Elastic Weight Consolidation
│   ├── continual.py            # Continual learning evaluation
│   └── trainer.py              # HebbianTrainer (LEGACY — Hebbian era only)
└── analysis/
    └── continual.py            # Forgetting/accuracy metrics
```

---

## Key Patterns

### TernaryDQTLinear
```python
layer = TernaryDQTLinear(in_features=784, out_features=512)
# Weights stored as int8 {-1, 0, +1} — no latent scores
# Forward: cast to float, matmul
# Backward: stochastic rounding via custom autograd
```

### MoE Layer
```python
moe = TernaryDQTMoELayer(
    in_features=784, out_features=128,
    num_experts=4, top_k=2
)
# Router selects top-2 experts per input
# Only selected experts run forward/backward
# Load balancing loss required: lb_coef=0.1, router_lr=0.001
```

---

## What to AVOID

- ❌ Hebbian updates (manual weight changes) — that era is closed
- ❌ `TernaryHebbianLinear` — LEGACY, Hebbian era only
- ❌ Training without `.backward()` — we use autograd now
- ❌ Training without optimizer — AdamW is standard
- ❌ Latent float scores in DQT mode — defeats the purpose

### Hebbian era is CLOSED
All Hebbian-related code (`TernaryHebbianLinear`, `HebbianTrainer`, WTA Hebbian,
`ternary_sign` activations, conscience mechanisms, Forward-Forward, Equilibrium
Propagation) is in `research/`. Do NOT use or import it for new code.

---

## Research vs Product

| | Product (root) | Research (research/) |
|:--|:---------------|:--------------------|
| **Learning** | DQT / STE | Hebbian, FF, EP, NTH |
| **Backward** | `.backward()` (autograd) | Manual updates |
| **Optimizer** | AdamW | None |
| **Eras** | E017–E019 (DQT) | E001–E016 (Hebbian + early STE) |
| **Status** | Active | Archive — do not modify |

---

## Naming Conventions

| Concept | Class/Function Name |
|---------|-------------------|
| DQT linear layer | `TernaryDQTLinear` |
| DQT MoE layer | `TernaryDQTMoELayer` |
| STE linear layer | `TernarySTELinear` |
| STE conv layer | `TernarySTEConv2d` |
| Hysteresis STE layer | `HysteresisSTELinear` |
| Stochastic rounding function | `stochastic_round()` |
| DQT gradient function | `_DQTGradFn` |
| Load balancing loss | `load_balance_loss()` |

---

## Testing

- 564 tests, all must pass
- Tests in `tests/layers/` for each layer type
- Tests in `tests/integration/` for end-to-end experiments
- `test_ste_dqt.py` — DQT-specific: weight format, sparsity, convergence
- `test_ste_dqt_moe.py` — MoE: routing, load balancing, expert utilization

---

## Key References

- **DQT**: Zhao et al. (ACML 2025) — arXiv:2412.04787
- **BitNet b1.58**: Ma et al. (2024) — arXiv:2402.17764
- **BitNet v2**: Wang et al. (2025) — arXiv:2504.18415
- **CAT-Q**: Wang et al. (ICML 2026) — arXiv:2606.26650
- **Neutrino-8B**: Fermion Research (2026) — HuggingFace
- **Product goals**: [`GOALS.md`](GOALS.md)
- **Product roadmap**: [`ROADMAP.md`](ROADMAP.md)
- **Research archive**: [`research/`](research/)
