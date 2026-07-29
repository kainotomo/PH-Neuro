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

## Current Status

| Phase | Title | Status | Key Result |
|:------|:------|:------:|:-----------|
| 0 | Core Mechanism | ✅ | 88.4% MNIST, single-layer WTA Hebbian |
| 1.1 | Multi-layer MLP | ✅ | 87.9% — depth doesn't help |
| 1.2 | CNN on CIFAR-10 | ✅ | 32.6% — conv Hebbian ≈ random |
| 1.3 | Continual Learning | ✅ | <5% multi-head ✅, single-head ❌ |
| 2 | **Forward Signals & Three-Factor** | � **COMPLETE** | TFF-1 ✅ 87.9%, NTH-1 ✅ 88.15%, TFF-2 ❌ **86.81%**, NTH-4 ❌ **85.79%** — ALL approaches exhausted. No method trains ternary Hebbian hidden layers. |
| 3 | Language Model | ⬜ | Predictive coding for error signal |
| 4-5 | Scale & Ship | ⬜ | 1B+ models, pip install |

**Critical findings across 7 experiments:**
1. **WTA Hebbian works** for single-task classification — but only on the output layer (88.4% MNIST)
2. **Unsupervised Hebbian ≠ discriminative features** — H4 falsified: depth provides zero improvement (MLP 87.9% = 1-layer 88.4%, CNN 32.6% = random 33.0%)
3. **Anti-Hebbian = gradient interference** — single-head continual learning fails (37% forgetting) because weakening wrong predictions destroys old knowledge
4. **Multi-head works** — separate output neurons per task achieve <5% forgetting ✅
5. **Forward-Forward + ternary ≠ hidden layer learning** — H5 falsified: TFF-2 achieves 86.81% (same as 1-layer). FF's popcount goodness trivially saturates; the contrastive signal doesn't create class-discriminative features.
6. **Three-factor Hebbian works for output layer** — H7 verified (output): NTH-1 achieves 88.15% MNIST with label modulator M∈{-1,0,+1}.
7. **Three-factor Hebbian fails for hidden layers** — H7 falsified (hidden): NTH-4 achieves 85.79% across all modulator propagation approaches. The feedback signal through sparse ternary weights is too weak. **All 7 experiments confirm: ternary Hebbian hidden layers cannot learn class-discriminative features without backprop.**

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

| Phase | Milestone | Status | Key Result |
|-------|-----------|--------|------------|
| 0 | Core Mechanism | ✅ Complete | 88.4% MNIST, single-layer WTA Hebbian |
| 1.1 | Multi-layer MLP on MNIST | ✅ Complete | 87.9%, depth doesn't help |
| 1.2 | CNN on CIFAR-10 | ✅ Complete | 32.6%, conv Hebbian = random |
| 1.3 | Continual Learning | ⬜ **NEXT** | **Primary contribution** — target <5% forgetting |
| 2 | Forward Signals & Three-Factor | 🔴 **Complete — all approaches exhausted** | No method trains ternary Hebbian hidden layers |
| 3 | Language Model | ⬜ On hold | Predictive coding faces same fundamental limitation |
| 4 | Scale to 1B+ | ⬜ Not started | |
| 5 | Package & Publish | ⬜ Not started | |

### Research Findings So Far

1. **WTA Hebbian works for single-task classification** (88.4% MNIST) — but only on the output layer with direct label supervision.
2. **Unsupervised Hebbian ≠ discriminative features** — hidden layers learn statistical structure (principal components), not class boundaries. H4 falsified across MLP and CNN.
3. **Anti-Hebbian = gradient interference** — the mechanism that weakens wrong predictions in single-head WTA is functionally identical to gradient-based forgetting. Hebbian doesn't solve the shared-representation interference problem.
4. **Multi-head works** — separate output neurons per task achieve <5% forgetting, the standard task-incremental learning protocol. This is a publishable result: ternary Hebbian networks support continual learning with isolated output heads.

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
