# E009: Ternary STE Baseline Suite (L1)

- **Date:** 2026-07-30
- **Git commit:** `TBD`
- **Status:** planned
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
| Seeds | 3 random seeds (42, 123, 456) for error bars |

---

## Results

### MNIST (784→512→256→10)

| Variant | Accuracy (test) | Training Time | Memory (VRAM) | Weight Memory | Notes |
|:--------|:---------------:|:------------:|:-------------:|:-------------:|:------|
| V1: Ternary STE | TBD | TBD | TBD | ~0.13 MB | — |
| V2: FP16 | TBD | TBD | TBD | ~1.06 MB | Upper bound |
| V3: INT8 QAT | TBD | TBD | TBD | ~0.53 MB | — |
| V4: INT4 QAT | TBD | TBD | TBD | ~0.27 MB | — |
| V5: Hebbian v1 | 88.42% | 47 s | <100 MB | ~0.008 MB | From E001 |

### Fashion-MNIST (784→512→256→10)

| Variant | Accuracy (test) | Training Time | Memory (VRAM) | Weight Memory | Notes |
|:--------|:---------------:|:------------:|:-------------:|:-------------:|:------|
| V1: Ternary STE | TBD | TBD | TBD | ~0.13 MB | — |
| V2: FP16 | TBD | TBD | TBD | ~1.06 MB | Upper bound |
| V3: INT8 QAT | TBD | TBD | TBD | ~0.53 MB | — |
| V4: INT4 QAT | TBD | TBD | TBD | ~0.27 MB | — |
| V5: Hebbian v1 | TBD | TBD | TBD | — | New run |

### KMNIST (784→512→256→10)

| Variant | Accuracy (test) | Training Time | Memory (VRAM) | Weight Memory | Notes |
|:--------|:---------------:|:------------:|:-------------:|:-------------:|:------|
| V1: Ternary STE | TBD | TBD | TBD | ~0.13 MB | — |
| V2: FP16 | TBD | TBD | TBD | ~1.06 MB | Upper bound |
| V3: INT8 QAT | TBD | TBD | TBD | ~0.53 MB | — |
| V4: INT4 QAT | TBD | TBD | TBD | ~0.27 MB | — |
| V5: Hebbian v1 | TBD | TBD | TBD | — | New run |

### CIFAR-10 (Conv→FC)

| Variant | Accuracy (test) | Training Time | Memory (VRAM) | Weight Memory | Notes |
|:--------|:---------------:|:------------:|:-------------:|:-------------:|:------|
| V1: Ternary STE | TBD | TBD | TBD | ~0.09 MB | — |
| V2: FP16 | TBD | TBD | TBD | ~0.70 MB | Upper bound |
| V3: INT8 QAT | TBD | TBD | TBD | ~0.35 MB | — |
| V4: INT4 QAT | TBD | TBD | TBD | ~0.18 MB | — |
| V5: Hebbian v1 | 32.6% | TBD | TBD | — | From E003 |

### CIFAR-100 (Conv→FC)

| Variant | Accuracy (test) | Training Time | Memory (VRAM) | Weight Memory | Notes |
|:--------|:---------------:|:------------:|:-------------:|:-------------:|:------|
| V1: Ternary STE | TBD | TBD | TBD | ~0.30 MB | — |
| V2: FP16 | TBD | TBD | TBD | ~2.40 MB | Upper bound |
| V3: INT8 QAT | TBD | TBD | TBD | ~1.20 MB | — |
| V4: INT4 QAT | TBD | TBD | TBD | ~0.60 MB | — |
| V5: Hebbian v1 | TBD | TBD | TBD | — | New run |

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

- [To be filled after experiment runs]

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

## Next Steps

- After L1 completes → **L2 (Hysteresis-STE)**: Apply dual-threshold hysteresis as STE regularizer
- After L1 completes → **L8 (Forgetting Baseline)**: Measure forgetting with standard SGD on Split MNIST
- Use L1 baselines as reference for all Track B (continual learning) experiments
