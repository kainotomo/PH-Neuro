# PH-Neuro

> **Ternary deep learning — from Hebbian to STE.**
>
> Weights are {-1, 0, +1} like biological synapses: excitatory, inhibitory, or absent.
> Two eras: Hebbian (backprop-free, biological) → STE (backprop, scalable).
> Research question: what can ternary networks do that nobody has tried?

---

## What is PH-Neuro?

> 📚 **Phase 0-2 (Hebbian) closed.** See [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) for all 9 Hebbian experiments. **Phase 3 (STE) active** — see [`docs/ROADMAP.md`](docs/ROADMAP.md). See [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md) for paper outlines.

PH-Neuro started as a **research framework** exploring a radical hypothesis: can we build useful neural networks that learn **without backpropagation**, using only Hebbian plasticity and ternary weights?

After **9 experiments across 4 fundamentally different approaches**, the answer was definitive: **no.** Ternary Hebbian hidden layers hit a fundamental ~88% accuracy ceiling on MNIST regardless of depth. Hebbian learning optimizes for correlation, not classification — a fundamental limitation.

**But the journey continued.** A systematic literature scan (July 2026) revealed that ternary networks CAN learn deep representations — just with STE backpropagation (BitNet, CAT-Q, Neutrino-8B). So PH-Neuro **pivoted to STE-based ternary training**, asking a new question:

> *Given that ternary weights + STE backprop works, what else can we do that nobody has tried?*

The answer so far: **combine ternary networks with continual learning** — a research gap no one had explored.

### Key Results Across Both Eras

| Era | Key Finding | Best Result |
|:----|:------------|:-----------:|
| **Hebbian** | All 9 approaches hit ~88% MNIST ceiling | WTA Hebbian: 88.4% single-layer |
| **STE Supervised** | Ternary STE beats Hebbian by +10pp, gap to FP16 only 0.7pp | MNIST 98.0% (FP16: 98.7%) |
| **STE Continual** | QLoRA + frozen ternary = 0% forgetting, 99.2% accuracy | Split MNIST, rank=8 |
| **Hysteresis-STE** | Novel algorithm: 0%→95% sparsity at −0.25pp cost | MNIST 97.9% at 95.6% sparsity |

### PH-Neuro vs PH-Net

| | PH-Net | PH-Neuro v1 (Hebbian) | PH-Neuro v2 (STE) |
|---|---|---|---|
| **Learning** | STE + Backprop | Hebbian (no backprop) | STE + Backprop |
| **Optimizer** | AdamW | None | AdamW |
| **Deep learning** | ✅ | ❌ ~88% ceiling | ✅ 98% MNIST |
| **Continual learning** | ❌ (severe forgetting) | ⚠️ multi-head only | 🏆 0% via QLoRA |
| **Goal** | Train ternary LLMs | Explore backprop-free | Low-memory + continual |
| **Status** | Production path | Closed (July 2026) | Active |

---

## Why?

1. **Scientific**: The Hebbian era proved a fundamental negative result — ternary Hebbian hidden layers cannot learn. The STE era bridges two unexplored fields: ternary networks + continual learning.
2. **Practical**: Ternary weights = 2 bits/weight, popcount MatMul, 3.88 GB for an 8B model. Combined with QLoRA, you get zero-forgetting edge deployment.
3. **Unexplored**: No paper has combined ternary weights with continual learning. No paper has measured forgetting across FP16/INT8/INT4/ternary. PH-Neuro is first.

---

## Hebbian Era (v1) — Closed July 2026

| Phase | Title | Status | Key Result |
|:------|:------|:------:|:-----------|
| 0 | Core Mechanism | ✅ | 88.4% MNIST, single-layer WTA Hebbian |
| 1.1 | Multi-layer MLP | ✅ | 87.9% — depth doesn't help |
| 1.2 | CNN on CIFAR-10 | ✅ | 32.6% — conv Hebbian ≈ random |
| 1.3 | Continual Learning | ✅ | <5% multi-head ✅, single-head ❌ |
| 2 | Forward Signals & Three-Factor | 🔴 **CLOSED** | 9 approaches exhausted. TFF-1 87.9%, NTH-1 88.15%, TFF-2 86.81%, NTH-4b 86.68%, TEP-1 82.57%. **No method trains ternary Hebbian hidden layers.** |

**Definitive conclusion:** Ternary Hebbian hidden layers cannot learn class-discriminative features without backpropagation. See [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md).

### Hebbian Critical Findings

1. **WTA Hebbian works** — but only on output layer (88.4% MNIST)
2. **Unsupervised Hebbian ≠ discriminative features** — depth provides zero improvement
3. **Anti-Hebbian = gradient interference** — single-head continual learning fails (37% forgetting)
4. **Multi-head works** — separate output neurons per task achieve <5% forgetting
5. **Forward-Forward + ternary fails** — popcount goodness is not class-discriminative
6. **Three-factor Hebbian fails for hidden layers** — even dense continuous feedback doesn't work
7. **Equilibrium Propagation moves weights but hurts accuracy** — 82.57%, worst of all methods

### Hebbian Conclusion

| Approach | Best Accuracy | Verdict |
|:---------|:------------:|:--------|
| Unsupervised Hebbian | 87.9% | ❌ PCA, not classes |
| Forward-Forward | 86.81% | ❌ Popcount is trivial |
| Three-factor Hebbian | 86.68% | ❌ Correlation ≠ classification |
| Equilibrium Propagation | 82.57% | ❌ Noisy, non-discriminative |

---

## STE Era (v2) — Active

After the Hebbian research phase closed, PH-Neuro **pivoted to STE backpropagation with ternary weights** — the approach proven at scale by BitNet b1.58, CAT-Q, and Neutrino-8B. The infrastructure (packed tensors, hysteresis, flip-rate tracking, 200+ tests) now supports dual-track research into **low-memory supervised learning** and **continual learning with ternary weights**.

### Experiment Summary

| ID | Experiment | Status | Key Result |
|:---|:-----------|:------:|:-----------|
| **L1** | Ternary STE Baseline Suite (5 datasets × 5 variants) | ✅ | MNIST 98.17% — beats the 88% Hebbian ceiling by **+9.15 pp**. See [`E009`](docs/experiments/E009-ste-baseline-suite.md) |
| **L2** | Hysteresis-STE algorithm (dual-threshold regularizer) | ✅ | 36/36 runs. θ_u=0.3 → **0%→95% sparsity** at −0.25/−0.50/−1.60 pp; θ_u≥0.5 fails (deadzone). See [`E016`](docs/experiments/E016-l2-hysteresis-ste.md) |
| **L5** | BatchNorm → ElementWiseAffine fusion | ✅ | Accuracy preserved; CPU/edge inference win. See [`E011`](docs/experiments/E011-l5-batchnorm-fusion.md) |
| **L7** | Depth vs Width Scaling (fixed 530K budget) | ✅ | Depth helps ternary **more** than FP16; no STE gradient degradation. See [`E012`](docs/experiments/E012-l7-depth-vs-width.md) |
| **L8** | Forgetting Baseline (control for Track B) | ✅ | Ternary ≈ FP16 forgetting (gap <1 pp). See [`E010`](docs/experiments/E010-l8-forgetting-baseline.md) |
| **DQT** | Direct Quantized Training (Zhao et al., ACML 2025) | ✅ | **98.23% MNIST** (beats STE 98.17%), **56% sparsity**, **4.5× less training memory** — no latent float scores. See [`E017`](docs/experiments/E017-dqt-pilot.md) |
| **DQT+Hyst** | DQT + Hysteresis combination | ✅ | **98.09% MNIST**, **60.5% sparsity**. Stochastic rounding overrides hysteresis — DQT alone is better. See [`E018`](docs/experiments/E018-dqt-hysteresis.md) |
| **MoE DQT** | Mixture of Experts + DQT ternary | ✅ | **91.21% MNIST** (beats dense 88.73% by **+2.48pp**) with **50.5% active params**. First ternary MoE on vision. See [`E019`](docs/experiments/E019-moe-dqt-pilot.md) |

### DQT — Direct Quantized Training (new)

### DQT — Direct Quantized Training (new)

Trained ternary MLP on MNIST **without latent float scores** using stochastic rounding (Zhao et al., ACML 2025). First demonstration that DQT works for vision, not just LLMs.

| Method | Accuracy | Sparsity | Training Memory |
|:-------|:--------:|:--------:|:---------------:|
| Standard STE (L1) | 98.17% | 0% | ~9 bytes/param |
| Hysteresis-STE (L2) | 97.92% | 95% | ~9 bytes/param |
| **DQT (E017)** | **98.23%** | **56%** | **~2 bytes/param** |

**Key finding:** DQT beats STE accuracy while using 4.5× less training memory and producing naturally sparse weights — no explicit regularization needed. See [`E017`](docs/experiments/E017-dqt-pilot.md).

**MoE DQT — Mixture of Experts (new):** DQT + MoE with ternary experts achieves **91.21% vs dense 88.73% (+2.48pp)** using only **50.5% active parameters**. First ternary MoE on vision. Requires load-balancing loss + slow router to prevent expert collapse. See [`E019`](docs/experiments/E019-moe-dqt-pilot.md).

### L7 Depth vs Width Scaling

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

### Track B (Continual Learning) — Complete

All three Track B experiments are **completed** (2026-07-31): EWC, QLoRA, and precision comparison on Split/Permuted MNIST.

| ID | Experiment | Status | Key Result |
|:---|:-----------|:------:|:-----------|
| **B1** | EWC + Ternary STE | ✅ | Split forgetting 37.33%→**32.78%** (−4.55 pp), accuracy 62.16%→**66.65%** (+4.48 pp); no benefit on shared-head Permuted. See [`E013`](docs/experiments/E013-b1-ewc-ternary-ste.md) |
| **B2** | QLoRA + Frozen Ternary Backbone | ✅ | **Zero forgetting** (0.00% ± 0.00 in all 30 runs); r=64: Split **99.43%**, Permuted **92.55%** (task1) — beats L8/B1 by 32–53 pp. See [`E014`](docs/experiments/E014-b2-qlora-frozen-ternary.md) |
| **B3** | Ternary vs INT8/INT4/FP16 continual learning | ✅ | "When Less is More" holds but **weak** — quantization cuts forgetting only 0.2–1.2 pp; ranking **FP16 > Ternary > INT8 ≈ INT4**. See [`E015`](docs/experiments/E015-b3-precision-comparison.md) |

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full plan.

---

## Quick Start

### STE Era (v2) — Recommended

```python
from ph_neuro.layers import TernarySTELinear
from ph_neuro.models.ste_models import ste_mlp

# Ternary STE model with AdamW
model = ste_mlp(784, [512, 256], 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for x, y in train_loader:
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()      # STE backward through ternary weights
    optimizer.step()
```

### Hebbian Era (v1) — Legacy

```python
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

## Key References

### STE Era (v2)
- **BitNet b1.58** (Ma et al., 2024) — Ternary LLMs at scale via STE
- **BitNet v2** (Wang et al., 2025) — 4-bit activations with Hadamard transform
- **CAT-Q** (Wang et al., ICML 2026 Oral) — Post-training ternary quantization, 512 samples
- **Neutrino-8B** (Fermion Research, 2026) — 8B ternary model, 3.88 GB, Apache 2.0
- **"When Less is More"** (Zhang et al., 2025) — Quantization improves continual learning
- **TOM Accelerator** (Guan et al., 2026) — QLoRA on-device tunability for ternary

### Hebbian Era (v1)
- **SoftHebb** (Journé et al., ICLR 2023) — SOTA Hebbian deep learning (float)
- **Forward-Forward** (Hinton, 2022) — Backprop-free with contrastive signals
- **Equilibrium Propagation** (Scellier & Bengio, 2017) — Energy-based learning
- **Predictive Coding** (Whittington & Bogacz, 2017) — Prediction error minimization

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full plan and reference list.

---

## License

MIT — see [LICENSE](LICENSE).

---

> *"The Hebbian era proved what doesn't work. The STE era explores what does."*
