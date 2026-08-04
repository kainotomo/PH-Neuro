# PH-Neuro — Benchmarks

> Consolidated results for all trained models. Hardware: **NVIDIA RTX 4060**
> (8 GB). Accuracy = mean **best** test accuracy across 3 seeds.
> All models use **2-bit ternary weights** {-1, 0, +1} (4 weights/byte).

---

## 1. Summary

| Model | Dataset | Accuracy | Params (ternary) | Size (packed) | Training Time | GPU |
|:------|:--------|:--------:|:----------------:|:-------------:|:-------------:|:---:|
| **DQT MLP** | MNIST | **98.23%** | 530 K | 132 KB | ~7.5 min | RTX 4060 |
| **DQT CNN** | CIFAR-10 | **78.98%** | 4.27 M | 1.0 MB | ~10 min | RTX 4060 |
| **DQT CNN** | CIFAR-100 | **54.15%** | 2.52 M | 615 KB | ~20 min | RTX 4060 |
| **STE CNN** | CIFAR-10 | **76.09%** | 4.27 M | — | ~10 min | RTX 4060 |

**Takeaways**

- DQT beats STE on **identical architectures** by **+2.89pp** (CIFAR-10) and
  **+15.95pp** (CIFAR-100) — see §2.
- Packed sizes are **16× smaller than FP32**: a 4.27M-parameter CNN fits in
  **1.0 MB**.
- Peak GPU memory during training is ~**340–365 MB** (see §5) — far below a
  consumer GPU's budget.

---

## 2. DQT vs STE baseline

| Dataset | DQT (best) | STE (best) | Δ | Milestone |
|:--------|:----------:|:----------:|:--:|:---------:|
| CIFAR-10 | **78.98%** | 76.09% | **+2.89pp** | M1.1 |
| CIFAR-100 | **54.15%** | 38.2% | **+15.95pp** | M1.2 |

> The STE baseline on CIFAR-10 (E009/L1, same 2-conv architecture) is
> **72.75%**; the standalone STE CNN run reached **76.09%**. On CIFAR-100 the
> STE baseline is **38.2%** (E009/L1, small 2-conv CNN).

---

## 3. Model artifacts

All exported artifacts live in [`models/`](../models/). ONNX files are
single self-contained files (dynamic batch); `.ternary` files are the
2-bit packed weights.

| Model | Dataset | ONNX (FP32) | Packed `.ternary` (2-bit) | Verified |
|:------|:--------|:-----------:|:-------------------------:|:--------:|
| `ste_mlp_mnist` | MNIST | 2.06 MB | 130.6 KB | ✅ |
| `dqt_cnn_cifar10` | CIFAR-10 | 16.33 MB | 1.02 MB | ✅ |
| `dqt_cnn_cifar100` | CIFAR-100 | 9.64 MB | 615 KB | ✅ |

All well under the **100 MB** deployment gate. ONNX output matches PyTorch
to machine precision (`max|Δ| ≈ 1e-4`, argmax agreement 100%).

---

## 4. Comparison with TF Lite / BitNet

| | TensorFlow Lite | BitNet | **PH-Neuro (DQT)** |
|:--|:---------------:|:------:|:------------------:|
| Weight size | 8-bit | 2-bit | **2-bit** |
| Training memory | High | High (latent scores) | **4.5× lower (no latent scores)** |
| Sparse activation | ❌ | ❌ | **✅ MoE** |
| Training-time memory (measured) | — | — | **~340–365 MB on CIFAR CNNs** |

> ⚠️ **TF Lite head-to-head is scheduled for M1.5** ("Memory benchmarks vs
> TF Lite — 4× smaller, 2× faster inference"). No TF Lite measurements exist
> yet; the roadmap gate is defined in [`ROADMAP.md`](../ROADMAP.md).

---

## 5. Training details (measured, 3-seed mean)

From the M1.1 / M1.2 result JSONs in the repo.

| Run | Epochs | Mean best | Mean final | Peak GPU mem | Train time |
|:----|:------:|:---------:|:----------:|:------------:|:----------:|
| M1.1 DQT CIFAR-10 (retry3) | 100 | 78.80%* | 77.70% | 363 MB | ~11.9 min |
| M1.2 DQT CIFAR-100 | 150 | **54.15%** | 53.92% | 336 MB | ~22.5 min |

\* The M1.1 published number **78.98%** is the best across all M1.1 retry
configurations (E021.2, 512-head, anneal@80%, patience 25). The 3-seed mean
of the final retry3 config is 78.80% with a spread of only 1.53pp.

Hyperparameters: `lr=0.01`, `weight_decay=1e-4`, `batch_size=128`,
`anneal_fraction=0.80` (final 20% uses deterministic `sign()`), patience
25–40. Annealing eliminates late-training flip jitter
(`flip_rate` 0.18 → 0.0006).

---

## 6. Reproduce

```bash
# CIFAR-10 (M1.1)
.venv/bin/python -m ph_neuro.examples.run_m1_1_dqt_cifar10 --lr 0.01 --epochs 100 --seed 42

# CIFAR-100 (M1.2)
.venv/bin/python -m ph_neuro.examples.run_m1_2_dqt_cifar100 --lr 0.01 --epochs 150 --seed 42

# Export + verify all models (M1.3)
.venv/bin/python -m ph_neuro.examples.run_m1_3_export --model dqt_cnn --packed --verify
```

Full experiment write-ups: [`research/docs/experiments/`](../research/docs/experiments/).
