# E009: Ternary STE Baseline Suite (L1)

- **Date:** 2026-07-30
- **Git commit:** `TBD`
- **Status:** completed
- **Phase:** 3A (Track A: Low-Memory Supervised Baselines)

---

## Hypothesis

Ternary STE backpropagation can match or closely approach float baselines on small vision tasks (MNIST, Fashion-MNIST, KMNIST, CIFAR-10, CIFAR-100), systematically establishing modern ternary vision benchmarks for the first time since ~2020. The ternary gap (FP16 accuracy − ternary accuracy) narrows as dataset complexity decreases, confirming that ternary weights are a viable memory-efficient alternative for vision.

---

## Experiment Design

### Variant Matrix

Each dataset is tested with **5 weight configurations** to enable systematic comparison:

| Variant ID | Weight Format | Training Method | Purpose |
|:-----------|:-------------|:----------------|:--------|
| **V1 — Ternary STE** | {-1, 0, +1} (2-bit packed) | STE backprop + AdamW on latent fp16 scores | **Our method** — primary benchmark |
| **V2 — FP16** | float16 | Standard backprop + AdamW | Upper bound (best possible accuracy) |
| **V3 — INT8 QAT** | int8 | Quantization-Aware Training (QAT) + STE | Established quantized baseline |
| **V4 — INT4 QAT** | int4 | Quantization-Aware Training (QAT) + STE | Aggressive quantization baseline |
| **V5 — Hebbian (v1)** | {-1, 0, +1} (2-bit packed) | WTA Hebbian (no backprop) | Legacy baseline — expected ~88% MNIST ceiling |

### Dataset × Architecture Matrix

| Dataset | Architecture | Params (ternary) | Input | Classes |
|:--------|:------------|:----------------:|:------|:-------:|
| **MNIST** | 784→512→256→10 | ~530K | 28×28 grayscale | 10 |
| **Fashion-MNIST** | 784→512→256→10 | ~530K | 28×28 grayscale | 10 |
| **KMNIST** | 784→512→256→10 | ~530K | 28×28 grayscale | 10 |
| **CIFAR-10** | Conv(3→32→64→128)→FC(2048→10) | ~350K | 32×32 RGB | 10 |
| **CIFAR-100** | Conv(3→64→128→256)→FC(4096→100) | ~1.2M | 32×32 RGB | 100 |

### Expected Results (before running)

| Dataset | Ternary STE | FP16 | INT8 QAT | INT4 QAT | Hebbian v1 | Ternary Gap |
|:--------|:-----------:|:----:|:--------:|:--------:|:----------:|:----------:|
| MNIST | 96-98% | ~98.5% | ~97% | ~95% | 88.4% | ~1-2pp |
| Fashion-MNIST | 88-91% | ~92% | ~90% | ~87% | — | ~2-3pp |
| KMNIST | 88-91% | ~93% | ~91% | ~88% | — | ~2-3pp |
| CIFAR-10 | 75-85% | ~90% | ~88% | ~82% | 32.6% | ~5-10pp |
| CIFAR-100 | 55-65% | ~72% | ~68% | ~60% | — | ~7-12pp |

---

## Configuration (per variant)

### V1: Ternary STE (Our Method)

| Parameter | Value |
|-----------|-------|
| Weight format | Ternary {-1, 0, +1} (packed 2-bit) |
| Latent scores | fp16 |
| Forward pass | `W_tern = sign(W_latent)` |
| Backward pass | STE: `∂L/∂W_latent = ∂L/∂W_tern` |
| Optimizer | AdamW on latent scores |
| Activation function | ReLU (float) — standard for STE training |
| Weight init | Latent scores ~ N(0, σ²), σ calibrated per layer |
| Batch size | 128 (MLP), 64 (CNN) |
| Epochs | 30 (MNIST/Fashion/KMNIST), 100 (CIFAR-10), 150 (CIFAR-100) |
| Learning rate | 0.001 (cosine schedule) |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss |

### V2: FP16 (Float Baseline)

| Parameter | Value |
|-----------|-------|
| Weight format | float16 |
| Forward pass | Standard float MatMul |
| Backward pass | Standard autograd |
| Optimizer | AdamW |
| Activation function | ReLU |
| Weight init | Kaiming uniform |
| Batch size | 128 (MLP), 64 (CNN) |
| Epochs | Same as V1 |
| Learning rate | 0.001 (cosine schedule) |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss |

### V3: INT8 QAT

| Parameter | Value |
|-----------|-------|
| Weight format | int8 (fake-quantized during training) |
| Quantization | `torch.quantization.fake_quantize` with per-tensor scale/zero_point |
| Forward pass | Fake-quantized float MatMul |
| Backward pass | STE through quantizer |
| Optimizer | AdamW on float weights |
| Activation function | ReLU |
| Weight init | Kaiming uniform |
| Batch size | 128 (MLP), 64 (CNN) |
| Epochs | Same as V1 |
| Learning rate | 0.001 (cosine schedule) |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss |

### V4: INT4 QAT

| Parameter | Value |
|-----------|-------|
| Weight format | int4 (fake-quantized during training) |
| Quantization | `torch.quantization.fake_quantize` with per-tensor scale/zero_point, qmin=-8, qmax=7 |
| Forward pass | Fake-quantized float MatMul |
| Backward pass | STE through quantizer |
| Optimizer | AdamW on float weights |
| Activation function | ReLU |
| Weight init | Kaiming uniform |
| Batch size | 128 (MLP), 64 (CNN) |
| Epochs | Same as V1 |
| Learning rate | 0.001 (cosine schedule) |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss |

### V5: Hebbian v1 (Legacy Baseline)

| Parameter | Value |
|-----------|-------|
| Weight format | Ternary {-1, 0, +1} |
| Latent scores | fp16 |
| Learning rule | WTA Hebbian: strengthen correct class, weaken wrong prediction |
| θ_upper / θ_lower | 1.0 / 0.3 (from E001) |
| Optimizer | None (manual Hebbian updates) |
| Backward pass | None — no `.backward()` called |
| Activation function | `ternary_sign()` |
| Batch size | 128 |
| Epochs | 10 (single layer only for Hebbian) |
| Learning rate | 0.01 |
| Loss | None (WTA Hebbian is the learning rule) |

---

## Common Settings (all variants)

| Parameter | Value |
|-----------|-------|
| Hardware | RTX 4060 8 GB VRAM, i7-14700K, 16 GB RAM |
| Framework | PyTorch 2.x |
| Data normalization | Standard per-dataset (MNIST: μ=0.1307 σ=0.3081, CIFAR: μ=0.5 σ=0.5 per channel) |
| Data augmentation (CIFAR) | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Data augmentation (MNIST/Fashion/KMNIST) | None |
| Validation split | 10% of training set |
| Early stopping | Patience=10 epochs on validation accuracy |
| Seeds | 1 (seed=42) — all 25 baseline runs completed on seed 42; multi-seed error bars not yet run |

---

## Results

### MNIST (784→512→256→10)

| Variant | Accuracy (test) | Training Time | Weight Memory | Ternary Gap |
|:--------|:---------------:|:------------:|:-------------:|:----------:|
| V1: Ternary STE | **98.17%** | ~1.5 min | ~0.13 MB | **0.56 pp** |
| V2: FP16 | **98.73%** | ~1.5 min | ~1.06 MB | — (upper bound) |
| V3: INT8 QAT | **97.58%** | ~2.0 min | ~0.53 MB | — |
| V4: INT4 QAT | **98.53%** | ~3.0 min | ~0.27 MB | — |
| V5: Hebbian v1 | **89.02%** | ~0.5 min | ~0.008 MB | 9.15 pp (vs V2) |

### Fashion-MNIST (784→512→256→10)

| Variant | Accuracy (test) | Training Time | Weight Memory | Ternary Gap |
|:--------|:---------------:|:------------:|:-------------:|:----------:|
| V1: Ternary STE | **89.13%** | ~1.5 min | ~0.13 MB | **1.06 pp** |
| V2: FP16 | **90.19%** | ~1.5 min | ~1.06 MB | — (upper bound) |
| V3: INT8 QAT | **90.14%** | ~2.0 min | ~0.53 MB | — |
| V4: INT4 QAT | **89.76%** | ~3.0 min | ~0.27 MB | — |
| V5: Hebbian v1 | **79.70%** | ~0.5 min | ~0.008 MB | 10.49 pp (vs V2) |

### KMNIST (784→512→256→10)

| Variant | Accuracy (test) | Training Time | Weight Memory | Ternary Gap |
|:--------|:---------------:|:------------:|:-------------:|:----------:|
| V1: Ternary STE | **91.26%** | ~1.5 min | ~0.13 MB | **2.32 pp** |
| V2: FP16 | **93.58%** | ~1.5 min | ~1.06 MB | — (upper bound) |
| V3: INT8 QAT | **93.41%** | ~2.0 min | ~0.53 MB | — |
| V4: INT4 QAT | **93.12%** | ~3 min | ~0.27 MB | — |
| V5: Hebbian v1 | **63.23%** | ~0.5 min | ~0.008 MB | 30.35 pp (vs V2) |

### CIFAR-10 (Conv→FC)

| Variant | Accuracy (test) | Training Time | Weight Memory | Ternary Gap |
|:--------|:---------------:|:------------:|:-------------:|:----------:|
| V1: Ternary STE | **72.75%** | ~20 min | ~0.09 MB | **13.58 pp** |
| V2: FP16 | **86.33%** | ~20 min | ~0.70 MB | — (upper bound) |
| V3: INT8 QAT | **86.82%** | ~20 min | ~0.35 MB | — |
| V4: INT4 QAT | **86.17%** | ~20 min | ~0.18 MB | — |
| V5: Hebbian v1 | **24.41%** | ~10 min | — | ~62 pp (vs V2) |

### CIFAR-100 (Conv→FC)

| Variant | Accuracy (test) | Training Time | Weight Memory | Ternary Gap |
|:--------|:---------------:|:------------:|:-------------:|:----------:|
| V1: Ternary STE | **39.00%** | ~30 min | ~0.30 MB | TBD (V2 running) |
| V2: FP16 | **57.50%** | ~30 min | ~2.40 MB | — (upper bound) |
| V3: INT8 QAT | **57.13%** | ~30 min | ~1.20 MB | — |
| V4: INT4 QAT | **55.33%** | ~30 min | ~0.60 MB | — |
| V5: Hebbian v1 | **6.13%** | ~10 min | — | ~51 pp (vs V2) |

---

## Analysis Dimensions

### 1. Ternary Gap per Dataset

```
Ternary Gap = FP16_accuracy − Ternary_STE_accuracy

Plot: Bar chart — 5 datasets × ternary gap (pp)
Expected: MNIST < Fashion-MNIST ≈ KMNIST < CIFAR-10 < CIFAR-100
```

### 2. Accuracy vs Weight Precision Trade-off

```
Plot: Accuracy vs bits-per-weight for each dataset
x-axis: 2 (ternary), 4 (INT4), 8 (INT8), 16 (FP16)
y-axis: Test accuracy
Expected: Monotonic increase — more bits → higher accuracy
```

### 3. Memory vs Accuracy Pareto Frontier

```
Plot: Scatter — Weight memory (MB) vs Accuracy
Each point = one variant × dataset combination
Pareto frontier shows optimal trade-offs
```

### 4. Training Efficiency

| Metric | FP16 | INT8 QAT | INT4 QAT | Ternary STE |
|:-------|:----:|:--------:|:--------:|:-----------:|
| Training time (relative) | 1.0× | ~0.8× | ~0.7× | TBD |
| Inference FLOPs (relative) | 1.0× | ~0.5× | ~0.3× | ~0.15× (popcount) |
| Training memory (relative) | 1.0× | ~0.7× | ~0.6× | ~0.5× |

### 5. Ternary Weight Dynamics (V1 only)

| Metric | MNIST | Fashion-MNIST | KMNIST | CIFAR-10 | CIFAR-100 |
|:-------|:-----:|:------------:|:------:|:--------:|:---------:|
| Final sparsity (% 0) | TBD | TBD | TBD | TBD | TBD |
| Final % +1 | TBD | TBD | TBD | TBD | TBD |
| Final % -1 | TBD | TBD | TBD | TBD | TBD |
| Flip rate (final epoch) | TBD | TBD | TBD | TBD | TBD |
| Mean |latent score| | TBD | TBD | TBD | TBD | TBD |

---

## Observations

### Key Findings

1. **STE BREAKS the ~88% Hebbian ceiling decisively**
   - MNIST: Ternary STE 98.17% vs Hebbian 89.02% → **+9.15 pp**
   - Fashion-MNIST: 89.13% vs 79.70% → **+9.43 pp**
   - KMNIST: 91.26% vs 63.23% → **+28.03 pp**
   - CIFAR-10: 72.75% vs 24.41% → **+48.34 pp**
   - CIFAR-100: 39.00% vs 6.13% → **+32.87 pp**

2. **Ternary gap scales with task difficulty**
   - MNIST: **0.56 pp** — barely measurable
   - Fashion-MNIST: **1.06 pp** — small
   - KMNIST: **2.32 pp** — moderate
   - CIFAR-10: **13.58 pp** — large (CNN architectural mismatch)
   - CIFAR-100: **18.50 pp** — largest (100 classes, harder task)
   - **Trend:** Gap ≈ 0.5-2 pp for simple MLPs, 13-18 pp for CNNs

3. **INT4 QAT is surprisingly strong**
   - On all MLP datasets, INT4 matches or exceeds FP16 and INT8
   - CIFAR-10: INT4 (86.17%) ≈ FP16 (86.33%) ≈ INT8 (86.82%)
   - MNIST: INT4 (98.53%) ≈ FP16 (98.73%) > INT8 (97.58%)

4. **Ternary STE is competitive with QAT on MLPs but lags on CNNs**
   - MLPs: Ternary ≈ INT4 ≈ INT8 (within 1-2 pp)
   - CNNs: Ternary (72.75%) lags behind INT4/INT8/FP16 (~86%)
   - This suggests CNN architectures need ternary-specific design (e.g., wider layers, no BN)

5. **Hebbian v1 is only viable for the simplest datasets**
   - MNIST: 89.02% (respectable)
   - Fashion-MNIST: 79.70% (struggling)
   - KMNIST: 63.23% (near random for 10 classes)
   - CIFAR-10: 24.41% (barely above random 10%)
   - CIFAR-100: 6.13% (random for 100 classes is 1%)

---

## Key Questions to Answer

1. **How does the ternary gap scale with task difficulty?** Is it consistently 1-2pp for simple tasks and 7-12pp for harder ones?
2. **Is ternary STE competitive with INT8/INT4 QAT?** Does ternary offer better accuracy-per-bit?
3. **Does ternary STE conclusively beat the ~88% Hebbian ceiling on MNIST?** This validates the strategic pivot.
4. **How much training memory does ternary STE actually use vs. FP16?** Empirical measurement, not estimate.
5. **What is the optimal architecture for ternary vision models?** Does width or depth matter more under ternary constraints?

---

## Bugs / Issues

- [ ] None yet

---

## L2: Hysteresis-STE Ablation

**Status:** ✅ **COMPLETED** (2026-08-02) — 36/36 runs done. Full report: [`E016`](E016-l2-hysteresis-ste.md)

**Core idea:** Apply PH-Neuro's dual-threshold hysteresis during STE training as a weight regularizer.

**Files:**
- `src/ph_neuro/layers/ste_hysteresis.py` — `HysteresisSTELinear`, `HysteresisSTEConv2d`, `ste_sign_hysteresis()`
- `src/ph_neuro/examples/run_l2_hysteresis_ste.py` — experiment runner with sparsity/flip/zone tracking
- `scripts/run_l2_ablation.sh` — sweep script (4× θ_upper × 3× θ_lower)
- `src/ph_neuro/examples/aggregate_l2_results.py` — results aggregator
- `tests/layers/test_ste_hysteresis.py` — 35 unit + integration tests ✅

**Key design:**
- Standard STE: `W_tern = sign(W_latent)`
- Hysteresis-STE: `W_tern = tern_hyst(W_latent, θ_upper, θ_lower, prev_ternary)`
  - `|W_latent| < θ_lower` → 0
  - `|W_latent| > θ_upper` → sign(W_latent)
  - otherwise → unchanged from previous step
- `prev_ternary` stored as `nn.Buffer` (persistent, not a Parameter)
- Backward: STE identity pass-through

**Smoke test (MNIST, 2 epochs, θ_u=0.3, θ_l=0.1):** 91.93% accuracy, 99% sparsity ✅

**Ablation results (2026-08-02, 36/36 runs):**

| Dataset | Control | Best Hyst (θ_u, θ_l) | Δ | Sparsity |
|:--------|:-------:|:--------------------:|:-:|:--------:|
| MNIST | 98.17% | 97.92% (0.3, 0.15) | −0.25 pp | 95.64% |
| Fashion-MNIST | 89.13% | 88.63% (0.3, 0.10) | −0.50 pp | 95.41% |
| KMNIST | 91.26% | 89.66% (0.3, 0.10) | −1.60 pp | 93.39% |

**Key findings:** (1) sparsity 0% → ~95% (hypothesis confirmed); (2) accuracy is *lower* at every working config (−0.25 to −1.60 pp); (3) **θ_u ≥ 0.5 fails completely** (deadzone barrier, 27/33 runs at ~10% accuracy) — latents init `N(0, 0.1)` never cross the activation threshold.

## Next Steps

- ✅ **L2 ablation sweep DONE** (2026-08-02): 36/36 runs. See [`E016`](E016-l2-hysteresis-ste.md)
- ✅ **L8 (Forgetting Baseline) DONE**: see [`E010`](E010-l8-forgetting-baseline.md)
- Use L1 baselines as reference for all Track B (continual learning) experiments
