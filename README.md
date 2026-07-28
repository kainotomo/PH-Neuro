# PH-Neuro

> **Ternary Hebbian deep learning — no backpropagation.**
>
> Weights are {-1, 0, +1} like biological synapses: excitatory, inhibitory, or absent.
> Learning is local, brain-inspired, and continuous.
> No backward pass. No optimizer. No loss function.

---

## What is PH-Neuro?

PH-Neuro is a **research framework** exploring a radical hypothesis: can we build useful neural networks that learn **without backpropagation**?

Instead of gradient descent, PH-Neuro uses **Hebbian learning** — "neurons that fire together, wire together." Each synapse updates based only on the activity of the two neurons it connects. No global error signal. No chain rule. Just local correlation.

Combined with **ternary weights** {-1, 0, +1}, this creates networks that are:
- **Memory-efficient**: ~50× less training memory than backprop (no optimizer states, no gradient buffers)
- **Compute-efficient**: ~6.5× fewer FLOPs (no backward pass, popcount MatMul)
- **Continually learning**: No catastrophic forgetting — learn new tasks without erasing old ones
- **Brain-inspired**: Local learning rules, stable discrete weights, homeostatic regulation

### PH-Neuro vs PH-Net

| | PH-Net | PH-Neuro |
|---|---|---|
| **Learning** | STE + Backprop | Hebbian (no backprop) |
| **Weights** | Ternary (from float latents) | Native ternary |
| **Optimizer** | AdamW | None |
| **Goal** | Train ternary LLMs for deployment | Explore backprop-free learning |
| **Status** | Production path | Research project |

Both share the ternary weight philosophy. PH-Net uses proven methods (gradient descent). PH-Neuro explores the alternative.

---

## Why?

1. **Scientific curiosity**: The brain learns without backprop. How far can we push biologically plausible learning?
2. **Practical advantages**: If it works, Hebbian learning enables online learning on edge devices, continual adaptation, and training models on hardware that can't afford backprop's memory overhead.
3. **Unexplored territory**: No paper has combined ternary weights with Hebbian learning. This is genuinely new.

---

## Quick Start

```python
import torch
from ph_neuro import TernaryHebbianLinear, HebbianTrainer

model = torch.nn.Sequential(
    TernaryHebbianLinear(784, 256, theta_upper=5.0, theta_lower=1.0),
    torch.nn.Sign(),
    TernaryHebbianLinear(256, 10, theta_upper=5.0, theta_lower=1.0),
)

trainer = HebbianTrainer(model, lr=0.001, decay=1e-5)
trainer.fit(train_loader, epochs=10)

# No .backward(), no optimizer, no loss function.
```

---

## Testing

Run the full test suite from the project root:

```bash
.venv/bin/python -m pytest tests/ -v
```

Or individual test files:

```bash
# Ternary activation function (ternary_sign)
.venv/bin/python -m pytest tests/core/test_activation.py -v

# TernaryTensor storage (naive + packed modes)
.venv/bin/python -m pytest tests/core/test_ternary_tensor.py -v

# Weight packing utilities
.venv/bin/python -m pytest tests/utils/test_packing.py -v

# Latent scores and Hebbian rules
.venv/bin/python -m pytest tests/core/test_latent_scores.py -v
.venv/bin/python -m pytest tests/core/test_hebbian_rules.py -v

# Hebbian linear layer
.venv/bin/python -m pytest tests/layers/test_linear.py -v
```

---

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full development plan.

| Phase | Milestone | Status |
|-------|-----------|--------|
| 0 | Core Mechanism — MNIST >95% | � In progress |
| 1 | Vision POC — CIFAR-10 + Continual Learning | 🔴 Not started |
| 2 | Multi-Layer & Hierarchical Representations | 🔴 Not started |
| 3 | First Language Model — TinyStories | 🔴 Not started |
| 4 | Scale to 1B+ Parameters | 🔴 Not started |
| 5 | Package & Publish | 🔴 Not started |

---

## Key References

- **SoftHebb** (Journé et al., ICLR 2023) — SOTA Hebbian deep learning (float weights)
- **BitNet b1.58** (Wang et al., 2024) — Ternary LLMs at 3B scale
- **Forward-Forward** (Hinton, 2022) — Backprop-free learning with contrastive signals
- **Predictive Coding** (Whittington & Bogacz, 2017) — Learning through prediction error minimization

See [`docs/ROADMAP.md#key-references`](docs/ROADMAP.md#key-references) for the full reference list.

---

## License

MIT — see [LICENSE](LICENSE).

---

> *"The brain does not compute gradients. It learns by association. PH-Neuro explores whether machines can too."*
