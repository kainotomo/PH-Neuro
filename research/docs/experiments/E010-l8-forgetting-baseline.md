# E010: L8 Forgetting Baseline (Standard SGD)

- **Date:** 2026-07-30
- **Git commit:** `TBD`
- **Status:** completed
- **Phase:** 3A (Track A: Low-Memory Supervised Baselines)

---

## Hypothesis

Standard ternary STE training with a **single shared output head** (no multi-head, no EWC, no replay) will exhibit catastrophic forgetting comparable to FP16 on Split MNIST and Permuted MNIST. However, the ternary weight quantization noise may provide a mild implicit regularization effect, resulting in **slightly lower forgetting** (<5 pp difference) compared to FP16.

This is the **control experiment** for Track B (Continual Learning). All subsequent CL experiments (B1: EWC, B2: QLoRA, B3: multi-head) will be compared against these baselines.

---

## Experiment Design

### Variant Matrix

| Variant ID | Weight Format | Protocol | Architecture | Training |
|:-----------|:-------------|:--------|:-------------|:---------|
| **Ternary-Split** | {-1, 0, +1} (STE) | Split MNIST (5 tasks) | MLP 784→512→256→10 | AdamW, sequential SGD |
| **Ternary-Permuted** | {-1, 0, +1} (STE) | Permuted MNIST (10 tasks) | MLP 784→512→256→10 | AdamW, sequential SGD |
| **FP16-Split** | float16 | Split MNIST (5 tasks) | MLP 784→512→256→10 | AdamW, sequential SGD |
| **FP16-Permuted** | float16 | Permuted MNIST (10 tasks) | MLP 784→512→256→10 | AdamW, sequential SGD |

### Protocols

#### Split MNIST (5 Tasks)

| Task | Digit Pair | Train Samples | Type |
|:----|:-----------|:-------------|:-----|
| 1 | 0 vs 1 | ~14K | Binary classification |
| 2 | 2 vs 3 | ~14K | Binary classification |
| 3 | 4 vs 5 | ~14K | Binary classification |
| 4 | 6 vs 7 | ~14K | Binary classification |
| 5 | 8 vs 9 | ~13K | Binary classification |

**Evaluation:** After each task, evaluate on ALL 5 test sets. Single 10-neuron output head (no multi-head).

#### Permuted MNIST (10 Tasks)

| Task | Name | Train Samples | Type |
|:----|:-----|:-------------|:-----|
| 1-10 | Permute seed=0..9 | ~60K each | 10-class classification |

**Evaluation:** After each task, evaluate on all previous tasks' test sets. Single 10-neuron output head.

### Expected Results (before running)

| Protocol | Weight Format | Avg Forgetting | Final Avg Accuracy |
|:---------|:-------------|:--------------|:------------------|
| Split MNIST | Ternary STE | ~35-45% | ~70-80% |
| Split MNIST | FP16 | ~35-45% | ~75-85% |
| Permuted MNIST | Ternary STE | ~40-50% | ~30-40% |
| Permuted MNIST | FP16 | ~40-50% | ~35-45% |

**Ternary Gap (FP16 forgetting − Ternary forgetting):** Expected 0-5 pp (ternary may be slightly better or worse).

---

## Configuration

### Shared Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | MLP 784→512→256→10 |
| Total parameters | ~530K |
| Optimizer | AdamW (both ternary and FP16) |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Loss | CrossEntropyLoss |
| Epochs per task | 10 |
| Activation | ReLU |
| BatchNorm | Yes (after each hidden layer) |
| Seeds | 42, 43, 44 (3 runs for statistical significance) |
| Device | CUDA if available, else CPU |

### Ternary STE Specific

| Parameter | Value |
|-----------|-------|
| Weight format | Ternary {-1, 0, +1} (packed 2-bit) |
| Latent scores | fp16 |
| Forward pass | `W_tern = sign(W_latent)` |
| Backward pass | STE: `∂L/∂W_latent = ∂L/∂W_tern` |
| Weight init | Latent scores ~ N(0, σ²), σ calibrated per layer |

### FP16 Specific

| Parameter | Value |
|-----------|-------|
| Weight format | float16 |
| Forward pass | Standard float MatMul |
| Backward pass | Standard autograd |
| Weight init | Kaiming uniform |

---

## Results

### Main Metrics

| Protocol | Weight Format | Avg Forgetting | Final Avg Accuracy | Ternary Gap |
|:---------|:-------------|:--------------|:------------------|:-----------|
| Split MNIST | Ternary STE | **37.33% ± 2.32%** | **62.16% ± 2.39%** | — |
| Split MNIST | FP16 | **37.55% ± 0.48%** | **62.12% ± 0.46%** | **+0.22 pp** |
| Permuted MNIST | Ternary STE | **54.86% ± 2.63%** | **41.92% ± 2.63%** | — |
| Permuted MNIST | FP16 | **55.52% ± 0.25%** | **41.94% ± 0.17%** | **+0.66 pp** |

**Key result:** Ternary STE and FP16 forget at essentially identical rates. The ternary gap is <1 pp for both protocols, well within noise.

### Per-Task Forgetting (averaged across 3 seeds)

#### Split MNIST

| Task | Ternary STE | FP16 |
|:----|:-----------|:----|
| Task 1 (0 vs 1) | 44.81% | 46.62% |
| Task 2 (2 vs 3) | 32.19% | 34.18% |
| **Task 3 (4 vs 5)** | **87.57%** | **88.14%** |
| Task 4 (6 vs 7) | 22.09% | 18.82% |
| Task 5 (8 vs 9) | 0.00% | 0.00% |

#### Permuted MNIST

| Task | Ternary STE | FP16 |
|:----|:-----------|:----|
| Task 1 | 83.41% | 84.79% |
| Task 2 | 75.18% | 78.42% |
| Task 3 | 77.94% | 81.86% |
| Task 4 | 82.14% | 83.47% |
| Task 5 | 70.79% | 73.79% |
| Task 6 | 61.72% | 67.64% |
| Task 7 | 53.13% | 54.19% |
| Task 8 | 32.52% | 24.10% |
| Task 9 | 11.79% | 6.96% |
| Task 10 | 0.00% | 0.00% |

### Accuracy Matrix Example (Ternary STE, Split MNIST, seed=42)

```
After task 1:  99.91%
After task 2:  64.07%  99.02%
After task 3:  19.62%  83.59%  99.68%
After task 4:  72.39%  76.15%  48.56%  99.70%
After task 5:  53.62%  61.26%   8.80%  74.67%  98.84%
```

### Weight Analysis (Ternary Only)

Standard STE training with AdamW yields **0% weight sparsity** throughout — all ternary weights are either +1 or -1. The latent scores are pushed away from zero by AdamW's momentum and never cross the zero threshold. This means standard STE (unlike Hebbian or Hysteresis-STE) produces no implicit regularization from weight sparsity.

| Task | Weight Sparsity (% 0) | % +1 | % -1 |
|:----|---------------------|:----|:----|
| Init | 0.0% | 50.0% | 50.0% |
| After training | 0.0% | ~50% | ~50% |

### Training Time

| Protocol | Ternary STE | FP16 |
|:---------|:-----------|:----|
| Split MNIST (5 tasks × 10 epochs) | ~40s | ~40s |
| Permuted MNIST (10 tasks × 10 epochs) | ~400s | ~420s |

---

## Observations

### What worked well?
- The experiment infrastructure (runner, aggregator, shell script) worked flawlessly — 12/12 runs completed without error.
- Results are consistent across 3 seeds (low std dev), giving confidence in the findings.
- Training time on RTX 4060 is reasonable: ~40s for Split MNIST, ~400s for Permuted MNIST.

### What failed or was surprising?
- **⚠️ Hypothesis FALSIFIED: Ternary weights do NOT naturally forget less than FP16.** The forgetting gap is <1 pp for both protocols, well within statistical noise. Standard STE with AdamW produces ternary weights that are essentially binary (±1 with no zeros), and the forgetting dynamics are identical to float training.
- **0% weight sparsity throughout training** was surprising. Standard STE with AdamW never produces zero weights — momentum keeps latent scores well away from zero. This means the hypothesized "stiffness" from quantization noise does not exist in standard STE.
- Task 3 (4 vs 5) consistently shows the highest forgetting (~87%) in Split MNIST, regardless of weight format. This may be because digits 4 and 5 share more visual features with previously learned digits (0,1,2,3) or because of the specific ordering in the task sequence.
- Permuted MNIST forgetting (~55%) is higher than Split MNIST (~37%), as expected — the random permutations make all tasks equally interfering.
- **Baseline comparison with Hebbian:** Split MNIST Hebbian achieved ~37% forgetting (from Phase 1.3 findings), which is essentially identical to both ternary STE (37.33%) and FP16 (37.55%). **All three approaches forget at the same rate with a single shared output head.**

---

## Next Steps

- Use L8 results as baseline for Track B experiments:
  - **B1**: EWC + Ternary STE — does EWC reduce forgetting vs L8?
  - **B2**: QLoRA + Frozen Ternary — zero forgetting by design
  - **B3**: Multi-Head Ternary EWC — maximal protection
- Compare L8 results with Hebbian continual learning baseline (Phase 1.3: ~37% forgetting on Split MNIST)

---

## References

- [E009: Ternary STE Baseline Suite](../experiments/E009-ste-baseline-suite.md) — L1 baseline
- [Phase 1.3: Continual Learning](../phase-1-vision-poc.md) — Hebbian CL findings
- [ROADMAP.md](../ROADMAP.md) — Project roadmap
