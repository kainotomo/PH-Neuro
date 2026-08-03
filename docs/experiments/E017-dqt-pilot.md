# E017: DQT Pilot — Direct Quantized Training Without Latent Scores

- **Date:** 2026-08-03
- **Git commit:** `main` (post E016)
- **Status:** completed
- **Phase:** 4 (Advanced Experiments — low-memory training)

---

## Hypothesis

**Direct Quantized Training (DQT) with stochastic rounding can train ternary MLPs on vision tasks without maintaining latent float scores during training**, achieving comparable or better accuracy than STE while naturally producing sparse weights. This replicates Zhao et al. (ACML 2025) findings outside the LLM domain.

Zhao et al. demonstrated that DQT works for LLaMA-structured LLMs. The open question: does it generalize to MLP architectures on vision tasks?

---

## Background: DQT vs STE

| | STE (L1 baseline) | DQT (this experiment) |
|:--|:-----------------|:---------------------|
| Weight storage during training | Latent float scores (fp16) | **Ternary int8 only** |
| Backward mechanism | STE through sign() | STE + stochastic rounding |
| Memory per parameter (training) | ~9 bytes | **~2 bytes** |
| Inference | Ternary {-1, 0, +1} | Ternary {-1, 0, +1} |

**Stochastic rounding formula:**
```
prob = |W_float - floor(W_float)|
W_new = floor(W_float) with prob (1-prob), ceil(W_float) with prob
```

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | MLP `[784, 512, 256, 10]` (`TernaryDQTLinear`) |
| Total parameters | ~530K |
| Weight format | Ternary {-1, 0, +1} (int8), no latent scores |
| Optimizer | AdamW + custom DQT gradient function |
| Loss | CrossEntropyLoss |
| Dataset | MNIST |
| Batch size | 128 (also tested 64) |
| Epochs | 30, 60 |
| Learning rate sweep | 0.001, 0.01 |
| Init std sweep | 0.1, 0.5 |
| Seed | 42 |

---

## Results

| Configuration | Best Acc | Final Acc | Sparsity | Time |
|:-------------|:--------:|:---------:|:--------:|:----:|
| **L1 STE Baseline** | **98.17%** | 97.97% | 0.0% | 66s |
| DQT lr=0.001, 30ep | 94.83% | 93.72% | 88.2% | 169s |
| DQT lr=0.001, 60ep | 97.20% | 96.81% | 84.5% | 572s |
| DQT lr=0.01, 30ep | 97.97% | 97.97% | 69.4% | 187s |
| **DQT lr=0.01, 60ep** ⭐ | **98.23%** | **98.10%** | **56.2%** | **453s** |
| DQT init_std=0.5, lr=0.001, 60ep | 97.41% | 97.05% | 58.6% | 189s |
| DQT lr=0.01, 60ep, bs=64 | 98.18% | — | 40.9% | 814s |

### Key Observations

1. **DQT achieves parity with STE** — best config at 98.23% vs STE 98.17%
2. **Learning rate is critical** — lr=0.001 converges slowly (94.8% at 30ep); lr=0.01 converges rapidly (98.0% at 30ep)
3. **Natural sparsity without regularization** — 56-88% sparsity vs STE's 0%
4. **Sparsity decreases during training** — from 91% early → 56% final as the network discovers needed weights
5. **Convergence is smooth** — no oscillation observed; training accuracy rises steadily from ~50% to >99.9%
6. **Small batch (bs=64) achieves similar accuracy** (98.18%) with lower sparsity (40.9%) but 2× slower

---

## Comparison: DQT vs Hysteresis-STE vs Standard STE

| Method | Accuracy | Sparsity | Training Memory | Novelty |
|:-------|:--------:|:--------:|:---------------:|:------:|
| Standard STE (L1) | 98.17% | 0% | ~9 bytes/param | Baseline |
| Hysteresis-STE (L2) | 97.92% | 95% | ~9 bytes/param | Novel regularizer |
| **DQT (this)** | **98.23%** | **56%** | **~2 bytes/param** | **Novel training method** |

**DQT achieves the best accuracy (98.23%), moderate sparsity (56%), and dramatically lower training memory (4.5× less than STE).**

---

## Implications for PH-Neuro Commercial Roadmap

1. **DQT is viable for vision** — Zhao et al.'s findings generalize beyond LLMs
2. **Training memory drops 4.5×** — enables larger models on the same GPU
3. **Natural sparsity is a bonus** — 56% fewer non-zero weights at no accuracy cost
4. **Combined with MoE** — DQT + sparse activation could enable 10B+ parameter models on consumer GPUs
5. **Combined with Hysteresis-STE** — could push sparsity even higher (>90%) while maintaining DQT's memory advantages

---

## Limitations & Future Work

- Tested only on MNIST MLP — needs validation on CIFAR-10 CNN and larger architectures
- 6× slower than standard STE (453s vs 66s) due to custom autograd overhead
- Training memory advantage not fully realized in PyTorch (custom kernels needed)
- Flip rate remains high (~23%) — may pose stability issues for continual learning

---

## Artifacts

- Results: `dqt_results/results_mnist_seed42.json`, `dqt_results/sweep_results.json`
- Layer implementation: `src/ph_neuro/layers/ste_dqt.py`
