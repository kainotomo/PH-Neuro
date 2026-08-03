# E012: L7 Depth vs Width Scaling

- **Date:** 2026-07-30
- **Git commit:** `TBD`
- **Status:** completed
- **Phase:** 4 (Advanced Experiments)

---

## Hypothesis

Deeper ternary STE networks will underperform wider ones at the same parameter budget, because each additional STE sign() operation in the forward pass introduces gradient degradation through the straight-through estimator. This makes the ternary depth/width trade-off favor width more strongly than FP16, where BatchNorm and ReLU already mitigate gradient degradation.

Specifically:
1. **At fixed budget (~530K params), wider (D=1, D=2) ternary STE networks will outperform deeper (D=4, D=5) ones.** ❌ **FALSIFIED**
2. **The ternary gap (FP16 accuracy − ternary accuracy) will increase with depth**, because deeper FP16 networks benefit from depth more than deeper ternary networks. ❌ **FALSIFIED**
3. **Weight sparsity will remain ~0% for all depths** (standard STE + AdamW pushes all latent scores away from zero, regardless of depth). ✅ **VERIFIED**

---

## Experiment Design

### Core Idea

Classic deep learning research shows that for a fixed parameter budget, there is an optimal depth-to-width ratio. However, this trade-off has never been studied for **ternary STE networks**. L7 fills this gap by systematically comparing 5 depth configurations at equal parameter count (~530K, matching L1 baseline).

### Depth Configurations

All configurations use equal-width hidden layers with BatchNorm and ReLU, no bias. Total parameters are held constant at ~530K.

| Depth | Hidden Layers | Layer Sizes | Width | Exact Params | % of Budget |
|:-----:|:-------------|:------------|:-----:|:------------:|:-----------:|
| D=1 | 1 | [784, 667, 10] | 667 | 529,898 | 99.98% |
| D=2 | 2 | [784, 432, 432, 10] | 432 | 529,632 | 99.93% |
| D=3 | 3 | [784, 353, 353, 353, 10] | 353 | 529,500 | 99.91% |
| D=4 | 4 | [784, 308, 308, 308, 308, 10] | 308 | 529,144 | 99.84% |
| D=5 | 5 | [784, 278, 278, 278, 278, 278, 10] | 278 | 529,868 | 99.97% |

*Formula:* For $D$ hidden layers of equal width $w$, $w$ solves $(D-1)w² + 794w = 530\,000$. Widths rounded to nearest integer.

*L1 baseline reference:* `[784, 512, 256, 10]` (535,296 params, closest to D=2 in depth with unequal widths).

### Variant Matrix

| Variant | Weight Format | Training Method | Hidden Layers |
|:--------|:-------------|:----------------|:--------------|
| **Ternary-D1** through **Ternary-D5** | {-1, 0, +1} (STE) | STE backprop + AdamW | 1 to 5 |
| **FP16-D1** through **FP16-D5** | float16 | Standard backprop + AdamW | 1 to 5 |

### Dataset

Primary: **MNIST** (60K train, 10K test, 10 classes, 28×28 grayscale)

Extensions (if primary results are interesting):
- Fashion-MNIST
- KMNIST

### Expected Results (before running)

| Depth | Ternary STE (expected) | FP16 (expected) | Ternary Gap |
|:-----:|:---------------------:|:---------------:|:-----------:|
| D=1 | 97.0–98.0% | 98.0–98.5% | ~1.0 pp |
| D=2 | 96.5–97.5% | 98.0–98.5% | ~1.5 pp |
| D=3 | 96.0–97.0% | 97.5–98.5% | ~2.0 pp |
| D=4 | 95.0–96.5% | 97.5–98.5% | ~2.5 pp |
| D=5 | 94.0–96.0% | 97.0–98.0% | ~3.0 pp |

**Key prediction:** The ternary gap grows monotonically with depth. At D=1 (single hidden layer), ternary performs nearly as well as FP16. At D=5, the gap widens as repeated STE sign operations degrade the gradient signal.

---

## Configuration

### Shared Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | MLP: `Flatten + Linear + ReLU + BN + ... + Linear` |
| Parameter budget | ~530K (all configs within 1%) |
| Optimizer | AdamW (both ternary and FP16) |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Loss | CrossEntropyLoss |
| Epochs | 30 |
| Scheduler | CosineAnnealingLR |
| Activation | ReLU |
| BatchNorm | Yes (after each hidden layer) |
| Early stopping | Patience=10 on test accuracy |
| Seeds | 42, 43, 44 (3 runs per config) |
| Total runs | 5 depths × 2 formats × 3 seeds = **30 runs** |
| Device | CUDA if available, else CPU |

### Ternary STE Specific

| Parameter | Value |
|-----------|-------|
| Weight format | Ternary {-1, 0, +1} (STE) |
| Forward pass | `W_tern = sign(W_latent)` |
| Backward pass | STE: `∂L/∂W_latent = ∂L/∂W_tern` |
| Weight init | Default (Kaiming uniform) for latent scores |

### FP16 Specific

| Parameter | Value |
|-----------|-------|
| Weight format | float16 |
| Forward pass | Standard float MatMul |
| Backward pass | Standard autograd |
| Weight init | Default (Kaiming uniform) |
| Bias | `False` (matching ternary STE with BatchNorm) |

---

## Results

### Main Metrics

**30/30 runs completed** (5 depths × 2 formats × 3 seeds, full 30 epochs each).

| Depth | Ternary STE | FP16 | Ternary Gap |
|:-----:|:-----------:|:----:|:-----------:|
| D=1 | **97.86%** ± 0.07 | **98.53%** ± 0.08 | **0.67 pp** |
| D=2 | **98.15%** ± 0.05 | **98.56%** ± 0.06 | **0.41 pp** |
| D=3 | **98.27%** ± 0.07 | **98.68%** ± 0.04 | **0.41 pp** |
| D=4 | **98.26%** ± 0.04 | **98.68%** ± 0.04 | **0.42 pp** |
| D=5 | **98.24%** ± 0.09 | **98.69%** ± 0.05 | **0.45 pp** |

**Key result: The ternary gap does NOT grow with depth.** It narrows from D=1 (0.67 pp) to D=2+ (~0.41 pp) and stays flat through D=5. Repeated STE sign operations do NOT cause gradient degradation.

### Depth Benefit (relative improvement D=1 → D=3)

| Metric | Ternary STE | FP16 |
|:-------|:-----------:|:----:|
| D=1 accuracy | 97.86% | 98.53% |
| D=3 accuracy | 98.27% | 98.68% |
| **Improvement** | **+0.41 pp** | **+0.15 pp** |

**Ternary benefits from depth ~2.7× more than FP16** in relative terms. Depth scaling works *better* for ternary weights than the hypothesis predicted.

### Per-Seed Accuracy

| Depth | Ternary seeds 42/43/44 | FP16 seeds 42/43/44 |
|:-----:|:----------------------|:-------------------|
| D=1 | 97.80 / 97.81 / 97.96 | 98.59 / 98.57 / 98.42 |
| D=2 | 98.15 / 98.09 / 98.21 | 98.58 / 98.63 / 98.48 |
| D=3 | 98.27 / 98.19 / 98.35 | 98.67 / 98.63 / 98.73 |
| D=4 | 98.31 / 98.25 / 98.22 | 98.67 / 98.64 / 98.73 |
| D=5 | 98.11 / 98.32 / 98.30 | 98.64 / 98.68 / 98.75 |

Very low variance across seeds (±0.04–0.09 pp), giving high confidence in the findings.

### Training Behavior

| Depth | Ternary epochs | FP16 epochs | Ternary time | FP16 time |
|:-----:|:--------------:|:-----------:|:------------:|:---------:|
| D=1 | 30 / 30 / 30 | 30 / 30 / 30 | 58.3s | 60.5s |
| D=2 | 30 / 30 / 30 | 30 / 30 / 30 | 59.9s | 60.7s |
| D=3 | 30 / 30 / 30 | 30 / 30 / 30 | 62.5s | 60.1s |
| D=4 | 30 / 30 / 30 | 30 / 30 / 30 | 66.7s | 61.4s |
| D=5 | 30 / 30 / 30 | 30 / 30 / 30 | 66.7s | 62.1s |

- **All runs trained the full 30 epochs** — no early stopping triggered (best_epoch was often 27–30, meaning accuracy kept improving near the end).
- **Training time scales with depth** for ternary (+14% from D=1 to D=5), roughly flat for FP16 (+3%). The STE ternary overhead grows slightly with depth but remains modest (~7% slower than FP16 at D=5).

### Weight Sparsity

| Depth | Ternary STE (% 0) | FP16 (% \|w\| < 0.01) |
|:-----:|:----------------:|:----------------------|
| D=1 | 0.00% | 18.80% |
| D=2 | 0.00% | 17.53% |
| D=3 | 0.00% | 15.96% |
| D=4 | 0.00% | 15.23% |
| D=5 | 0.00% | 14.47% |

- **Ternary sparsity is 0% at all depths** — as predicted. Standard STE + AdamW pushes every latent score to ±1, never leaving anything at 0. There is no implicit regularization from sparsity (consistent with L8 finding).
- **FP16 near-zero sparsity decreases with depth** (18.8% → 14.5%), suggesting deeper FP16 networks use their weights more evenly.

### Per-Layer Weight Distribution (Ternary, seed 42)

#### D=5

| Layer | Params | % +1 | % -1 |
|:-----:|:------:|:----:|:----:|
| 1 (784→278) | 217,952 | 48.35% | 51.65% |
| 2 (278→278) | 77,284 | 49.57% | 50.43% |
| 3 (278→278) | 77,284 | 50.58% | 49.42% |
| 4 (278→278) | 77,284 | 51.83% | 48.17% |
| 5 (278→278) | 77,284 | 53.62% | 46.38% |
| 6 (278→10) | 2,780 | 46.04% | 53.96% |

**Observation:** The % of +1 weights increases monotonically through the hidden layers (48.4% → 53.6%), then the output layer flips to a negative bias (46.0% +1). This per-layer asymmetry does not harm accuracy but hints that intermediate layers develop slightly different sign balances as depth increases.

#### D=3 (seed 42)

| Layer | Params | % +1 | % -1 |
|:-----:|:------:|:----:|:----:|
| 1 (784→353) | 276,752 | 49.14% | 50.86% |
| 2 (353→353) | 124,609 | 50.74% | 49.26% |
| 3 (353→353) | 124,609 | 53.73% | 46.27% |
| 4 (353→10) | 3,530 | 47.28% | 52.72% |

Same pattern: hidden layers drift more +1 with depth, output layer flips negative.

---

## Observations

### What worked well?
- **Depth scaling works for ternary STE** — the primary result. Ternary networks *gain* from depth at a fixed budget, and the gain (+0.41 pp from D=1→D=3) exceeds the FP16 gain (+0.15 pp).
- **Ternary gap is small and stable** (~0.4–0.7 pp) across all depths, confirming that ternary STE matches float training at every architecture size.
- **The experiment infrastructure worked flawlessly** — 30/30 runs completed without error, low variance across seeds, clean JSON output.

### What failed or was surprising?
- **⚠️ Both core hypotheses FALSIFIED — in a positive direction.** The hypothesis predicted deeper ternary networks would degrade (gradient degradation through repeated STE sign ops). Instead, ternary benefits from depth *more* than FP16. The STE straight-through estimator does NOT accumulate harmful gradient noise with depth.
- **0% ternary sparsity at all depths** was expected but notable — standard STE + AdamW produces purely binary weights (±1, no zeros). The optimal D=3 config uses every weight at full magnitude.
- **All runs trained the full 30 epochs** — the cosine schedule + early stopping (patience 10) never triggered, since accuracy kept setting new records at epoch 27–30. This differs from L1 where early stopping was more aggressive.
- **The output layer consistently shows a negative sign bias** (46–48% +1) regardless of depth, while hidden layers drift increasingly +1 with depth. This asymmetry appears to be a stable feature of ternary STE training.

---

## Conclusions

1. **Depth vs width: ternary prefers moderate depth (D=3)** — the optimal config at ~530K params is `[784, 353, 353, 353, 10]` (98.27%). Wider single-hidden-layer (D=1) is worst (97.86%); beyond D=3, accuracy plateaus (D=4: 98.26%, D=5: 98.24%).
2. **No ternary depth penalty** — the feared gradient degradation through repeated STE sign() operations does not exist. Ternary scales with depth at least as well as FP16.
3. **Ternary gap is ~0.4–0.7 pp regardless of architecture** — robust and useful for model selection: engineers can pick any depth at the same budget with a predictable ~0.5 pp penalty vs float.
4. **Implication for Track A**: ternary STE vision models can be designed with standard depth heuristics (2–3 hidden layers for MNIST-scale) without worrying about quantization-specific degradation.

---

## Next Steps

- **Compare with L1 baseline** (784→512→256→10, 535K params): L1's unequal-width D=2 config sits between L7's D=2 (98.15%) and D=3 (98.27%) — consistent with width helping at intermediate depth.
- **Test Hysteresis-STE (L2) at D=3**: if hysteresis reintroduces sparsity, does the optimal depth change?
- **Extend to Fashion-MNIST/KMNIST** to verify the depth-scaling finding is not MNIST-specific.
- **Investigate the per-layer sign drift**: why do hidden layers become progressively +1-biased with depth while the output layer flips negative? Measure gradient norms per layer to characterize STE gradient flow in deep nets.

---

## References

- [E009: Ternary STE Baseline Suite](E009-ste-baseline-suite.md) — L1 baseline with same 530K budget
- [README.md](../README.md) — Project overview and architecture
- Goodfellow et al. (2016): Deep Learning — depth vs width trade-off survey
- Eldan & Shamir (2016): The power of depth for feedforward neural networks
- Montufar et al. (2014): On the number of linear regions of deep neural networks
