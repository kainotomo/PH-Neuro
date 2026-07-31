# PH-Neuro

> **Ternary Hebbian deep learning — no backpropagation.**
>
> Weights are {-1, 0, +1} like biological synapses: excitatory, inhibitory, or absent.
> Learning is local, brain-inspired, and continuous.
> No backward pass. No optimizer. No loss function.

---

## What is PH-Neuro?

> 📚 **Research phase closed.** See [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) for the definitive summary of all 9 experiments and [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md) for the paper outline.

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
| 2 | **Forward Signals & Three-Factor** | 🔴 **COMPLETE** | TFF-1 ✅ 87.9%, NTH-1 ✅ 88.15%, TFF-2 ❌ 86.81%, NTH-4 ❌ 85.79%, NTH-4b ❌ **86.68%**, TEP-1 ❌ **82.57%** — **ALL 9 approaches exhausted. No method trains ternary Hebbian hidden layers. Research phase closed.** |
| 3-5 | Language Model & Scale | ⬜ | Closed — requires backprop or predictive coding |

**Critical findings across 9 experiments:**
1. **WTA Hebbian works** for single-task classification — but only on the output layer (88.4% MNIST)
2. **Unsupervised Hebbian ≠ discriminative features** — H4 falsified: depth provides zero improvement (MLP 87.9% = 1-layer 88.4%, CNN 32.6% = random 33.0%)
3. **Anti-Hebbian = gradient interference** — single-head continual learning fails (37% forgetting) because weakening wrong predictions destroys old knowledge
4. **Multi-head works** — separate output neurons per task achieve <5% forgetting ✅
5. **Forward-Forward + ternary ≠ hidden layer learning** — H5 falsified: TFF-2 achieves 86.81% (same as 1-layer). FF's popcount goodness trivially saturates; the contrastive signal doesn't create class-discriminative features.
6. **Three-factor Hebbian works for output layer** — H7 verified (output): NTH-1 achieves 88.15% MNIST with label modulator M∈{-1,0,+1}.
7. **Three-factor Hebbian fails for hidden layers** — H7 falsified (hidden): NTH-4 achieves 85.79% across all modulator propagation approaches. NTH-4b reaches 86.68% with dense continuous latent score feedback, but hidden flip rate is still ~0.000%/step — **sparsity was not the bottleneck.** The Hebbian correlation-based update cannot create discriminative features even with a perfect dense feedback pathway.
8. **Equilibrium Propagation moves hidden weights but reduces accuracy** — TEP-1 achieves 0.005%/step hidden flip rate (first non-backprop method to do so) but accuracy drops to 82.57% — the EP signal pushes hidden representations in non-discriminative directions.

**Definitive conclusion: All 9 experiments confirm that ternary Hebbian hidden layers cannot learn class-discriminative features without backpropagation.** Research phase is closed. See [`docs/experiments/E008-equilibrium-propagation-mnist.md`](docs/experiments/E008-equilibrium-propagation-mnist.md) for the full report.

---

## STE Era (v2) — Current Experiments (July 2026 onward)

After the Hebbian research phase closed, PH-Neuro **pivoted to STE backpropagation with ternary weights** — the approach proven at scale by BitNet b1.58, CAT-Q, and Neutrino-8B. The infrastructure (packed tensors, hysteresis, flip-rate tracking, 200+ tests) now supports dual-track research into **low-memory supervised learning** and **continual learning with ternary weights**.

### Experiment Summary

| ID | Experiment | Status | Key Result |
|:---|:-----------|:------:|:-----------|
| **L1** | Ternary STE Baseline Suite (5 datasets × 5 variants) | ✅ | MNIST 98.17% — beats the 88% Hebbian ceiling by **+9.15 pp**. See [`E009`](docs/experiments/E009-ste-baseline-suite.md) |
| **L2** | Hysteresis-STE algorithm (dual-threshold regularizer) | 🟡 Code complete | Layers, runner, sweep script, 35 tests — awaiting full run |
| **L5** | BatchNorm → ElementWiseAffine fusion | ✅ | Accuracy preserved; CPU/edge inference win. See [`E011`](docs/experiments/E011-l5-batchnorm-fusion.md) |
| **L7** | Depth vs Width Scaling (fixed 530K budget) | ✅ | Depth helps ternary **more** than FP16; no STE gradient degradation. See [`E012`](docs/experiments/E012-l7-depth-vs-width.md) |
| **L8** | Forgetting Baseline (control for Track B) | ✅ | Ternary ≈ FP16 forgetting (gap <1 pp). See [`E010`](docs/experiments/E010-l8-forgetting-baseline.md) |

### L7 Depth vs Width Scaling (latest)

At a fixed parameter budget (~530K), 5 equal-width depth configs × Ternary STE vs FP16 × 3 seeds (30 runs) on MNIST:

| Depth | Ternary STE | FP16 | Ternary Gap |
|:-----:|:-----------:|:----:|:-----------:|
| D=1 | 97.86% | 98.53% | 0.67 pp |
| D=2 | 98.15% | 98.56% | 0.41 pp |
| **D=3** | **98.27%** | 98.68% | 0.41 pp |
| D=4 | 98.26% | 98.68% | 0.42 pp |
| D=5 | 98.24% | 98.69% | 0.45 pp |

**Key findings:**
1. **Depth scaling works for ternary STE** — the hypothesis that repeated STE sign ops cause gradient degradation is **falsified**. Ternary gains *more* from depth than FP16 (+0.41 pp vs +0.15 pp from D=1→D=3).
2. **Ternary gap is flat** (~0.4–0.7 pp) across all depths — no ternary depth penalty.
3. **Optimal config at this budget:** D=3 `[784, 353, 353, 353, 10]` → 98.27%.
4. **0% weight sparsity** at all depths (standard STE + AdamW → all weights ±1, no implicit regularization).

### Track B (Continual Learning) — Planned

| ID | Experiment | Status |
|:---|:-----------|:------:|
| B1 | EWC + Ternary STE on Split MNIST | ⬜ |
| B2 | QLoRA + Frozen Ternary Backbone | ⬜ |
| B3 | Ternary vs INT8/INT4/FP16 continual learning | ⬜ |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full plan.

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
| 1.3 | Continual Learning | ✅ Complete | **Primary contribution** — <5% multi-head forgetting |
| 2 | Forward Signals & Three-Factor | 🔴 **DEFINITIVELY CLOSED** | TFF-1 (87.9%), NTH-1 (88.15%), TFF-2 (86.81%), NTH-4 (85.79%), NTH-4b (86.68%), **TEP-1 (82.57%)** — All 9 approaches exhausted. |
| 3-5 | Language Model, Scale & Publish | ⬜ **On hold indefinitely** | Ternary Hebbian hidden layers cannot be trained without backprop. See research conclusion below. |

### Definitive Research Conclusion

After **9 experiments** across **4 fundamentally different approaches**, the evidence is conclusive:

| Approach | Experiments | Best Accuracy | Verdict |
|:---------|:-----------:|:------------:|:--------|
| Unsupervised Hebbian | Phase 1.1, 1.2 | 87.9% | ❌ Learns PCA, not classes |
| Forward-Forward | TFF-1, TFF-2 | 86.81% | ❌ Popcount goodness is trivial |
| Three-factor Hebbian | NTH-1, NTH-4 (B/C/D) | 86.68% | ❌ Sparse weights kill feedback; dense feedback gives diffuse modulators |
| Equilibrium Propagation | **TEP-1** | **82.57%** | ❌ Noisy targets; moving-target/stale-target instability |

**Ternary Hebbian hidden layers cannot learn class-discriminative features without backpropagation.** The research phase of PH-Neuro is officially closed. All plausible methods have been tested and exhausted.

**What remains:**
- **Multi-head continual learning** (<5% forgetting) is a publishable result
- **Predictive coding** is the only remaining non-backprop approach — but it requires floating-point error nodes and significant architectural changes

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
