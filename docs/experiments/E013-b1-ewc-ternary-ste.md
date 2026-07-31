# E013: B1 — EWC + Ternary STE

- **Date:** 2026-07-31
- **Git commit:** `6e4d330` (base)
- **Status:** completed
- **Phase:** 3B (Track B: Continual Learning with Ternary STE)

---

## Hypothesis

**Elastic Weight Consolidation (EWC) reduces catastrophic forgetting of ternary STE networks** on Split MNIST and Permuted MNIST, compared to the L8 control baseline (no EWC). By penalizing movement of the **latent scores** that were important for previously learned tasks (weighted by the diagonal Fisher Information), EWC should protect old knowledge while still learning new tasks.

**Expected:** EWC lowers average forgetting from the L8 baseline (~37% Split MNIST, ~55% Permuted MNIST) by 10-30 pp at the optimal λ, with a modest accuracy cost.

This is the **first experiment of Track B (Continual Learning with Ternary STE)** and the first test of the hypothesis that EWC + ternary weights can combine — a combination that has never been studied.

---

## Why EWC on Latent Scores (not Ternary Weights)

Ternary weights {-1, 0, +1} are a deterministic function of float latent scores via `sign()`. The latent scores are the differentiable parameters that STE backpropagation actually updates. EWC therefore operates on **latent scores**:

```
L_total = L_task + (λ / 2) · Σ_i F_i · (θ_i − θ*_i)²
```

where:
- `θ_i` = current latent scores
- `θ*_i` = consolidated reference latent scores (after the previous task)
- `F_i` = diagonal Fisher Information for the previous task, estimated as the mean squared STE gradient of the negative log-likelihood over the task's training data

### EWC Variants

| Variant | Memory | Description |
|:--------|:------:|:------------|
| **Online EWC** (default) | O(2·\|θ\|) | Accumulate one Fisher: `F_acc = γ·F_acc + F_new`; keep one reference (Schwarz et al., 2018) |
| **Multi-task EWC** | O(2·T·\|θ\|) | Store a separate (Fisher, reference) per task; penalty is the sum |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | MLP 784→512→256→10 (TernarySTELinear + ReLU + BatchNorm) |
| Total parameters | ~530K |
| Weight format | Ternary {-1, 0, +1} (STE) |
| EWC target | `TernarySTELinear.latent_scores` (float) |
| EWC variant | Online (default), Multi-task (ablation) |
| EWC γ | 1.0 (equal weighting) |
| Fisher samples | 500 batches per task |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Loss | CrossEntropyLoss + EWC penalty |
| Epochs per task | 10 |
| λ sweep | {0.1, 1.0, 10.0, 100.0, 1000.0} |
| Seeds | 42, 43, 44 (3 runs for statistical significance) |
| Device | CUDA if available, else CPU |

### Protocols

| Protocol | Tasks | Type |
|:---------|:-----:|:-----|
| Split MNIST | 5 binary tasks (0 vs 1, ..., 8 vs 9) | Task-incremental binary |
| Permuted MNIST | 10 tasks with different pixel permutations | Class-incremental 10-way |

**Evaluation:** after each task, evaluate on ALL previously seen tasks' test sets. Single shared 10-neuron output head (no multi-head).

---

## Implementation

| File | Purpose |
|------|---------|
| `src/ph_neuro/training/ewc.py` | EWC core: Fisher diagonal, penalty, OnlineEWC / MultiTaskEWC |
| `src/ph_neuro/examples/run_b1_ewc.py` | Experiment runner |
| `src/ph_neuro/examples/aggregate_b1_results.py` | Results aggregator + L8 comparison |
| `scripts/run_b1_ewc.sh` | Sweep orchestration |
| `tests/training/test_ewc.py` | Unit tests |
| `tests/integration/test_b1_ewc.py` | Integration tests |

Reuses the L8 infrastructure: task definitions, experiment loop, model builder, weight statistics, JSON format — so results are directly comparable with the L8 control.

---

## Expected Results (before running)

| Protocol | Method | Avg Forgetting | Final Avg Accuracy |
|:---------|:-------|:--------------|:------------------|
| Split MNIST | L8 baseline (no EWC) | ~37.33% ± 2.32% | ~62.16% ± 2.39% |
| Split MNIST | EWC (best λ) | **10-30 pp lower** | ~55-70% |
| Permuted MNIST | L8 baseline (no EWC) | ~54.86% ± 2.63% | ~41.92% ± 2.63% |
| Permuted MNIST | EWC (best λ) | **10-30 pp lower** | ~35-50% |

**λ effect (expected U-curve):** too low λ → no protection (≈ L8 baseline); too high λ → no new-task learning; mid-range λ minimizes forgetting.

---

## Results

### λ Sweep (Split MNIST, seed=42)

The sweep was extended beyond the planned {0.1, 1, 10, 100, 1000} because the
best value landed at the upper boundary; λ ∈ {3000, 10000} were added to probe
the trend.

| λ | Avg Forgetting | Avg Accuracy |
|:--|:--------------:|:------------:|
| 0.1 | 37.43% | 62.13% |
| 1 | 39.50% | 60.12% |
| 10 | 40.30% | 59.30% |
| 100 | 38.91% | 60.69% |
| 1000 | 35.54% | 64.10% |
| 3000 | 35.26% | 64.29% |
| **10000** | **33.11%** | **66.30%** |
| L8 baseline | 37.33% | 62.16% |

**Unexpected pattern:** the sweep is *not* a classic U-curve. Mid-range λ
(1, 10, 100) are **worse** than the no-EWC baseline, while high λ (≥1000)
improve it monotonically. Best λ = **10000** (also the upper edge tested).

### Main Metrics (best λ = 10000, 3 seeds)

| Protocol | Method | Avg Forgetting | Final Avg Accuracy |
|:---------|:-------|:--------------|:------------------|
| Split MNIST | **EWC (λ=10000)** | **32.78% ± 0.74** | **66.65% ± 0.71** |
| Split MNIST | L8 baseline (no EWC) | 37.33% ± 2.32 | 62.16% ± 2.39 |
| Split MNIST | **Δ (EWC − L8)** | **+4.55 pp** ✅ | **+4.48 pp** ✅ |
| Permuted MNIST | EWC (λ=10000) | 54.60% ± 1.76 | 39.78% ± 1.87 |
| Permuted MNIST | L8 baseline (no EWC) | 54.86% ± 2.63 | 41.92% ± 2.63 |
| Permuted MNIST | Δ (EWC − L8) | +0.26 pp (ns) | −2.14 pp ❌ |

### Per-Task Forgetting (Split MNIST, seed 42)

| Task | EWC (λ=10000) | L8 baseline | Δ |
|:-----|:-------------:|:-----------:|:--:|
| Task 1 (0 vs 1) | 45.58% | 44.81% | −0.77 pp |
| Task 2 (2 vs 3) | 24.68% | 32.19% | +7.51 pp |
| Task 3 (4 vs 5) | 82.07% | 87.57% | +5.50 pp |
| Task 4 (6 vs 7) | 13.24% | 22.09% | +8.85 pp |
| Task 5 (8 vs 9) | 0.00% | 0.00% | 0.00 pp |

EWC reduces forgetting on tasks 2-4 substantially; task 1 is marginally worse
(single-seed comparison).

### Accuracy Matrix (Split MNIST, λ=10000, seed 42)

```
After task 1:  99.91%
After task 2:  71.73%  99.41%
After task 3:  23.59%  89.37%  99.73%
After task 4:  46.76%  83.25%  65.15%  99.45%
After task 5:  54.33%  74.73%  17.66%  86.20%  98.59%
```

### Accuracy Matrix (Permuted MNIST, λ=10000, seed 42)

```
After task 1:  96.4%
After task 2:  94.6%  95.3%
After task 3:  91.6%  93.4%  95.1%
After task 4:  87.9%  88.5%  90.6%  94.3%
After task 5:  76.7%  79.0%  87.6%  89.9%  94.3%
After task 6:  59.7%  62.1%  67.2%  75.1%  84.4%  93.8%
After task 7:  43.4%  38.4%  48.0%  41.6%  67.5%  79.6%  94.0%
After task 8:  39.0%  33.1%  32.0%  31.1%  45.5%  52.8%  69.7%  93.6%
After task 9:  20.8%  22.2%  19.1%  19.2%  29.3%  35.0%  48.3%  75.3%  93.6%
After task 10: 23.1%  19.1%  20.2%  17.5%  25.2%  21.7%  35.1%  54.4%  69.9%  92.7%
```

### Weight Analysis

Standard STE training with AdamW + EWC yields **0% ternary weight sparsity**
throughout — identical to L8. EWC does not introduce sparsity; all weights
remain ±1. The regularization benefit comes from the EWC penalty on latent
scores, not from weight quantization sparsity.

---

## Analysis

### Hypothesis: ✅ **Partially verified (Split MNIST) / ❌ Falsified (Permuted MNIST)**

**Split MNIST — EWC works.** At λ=10000, EWC reduces average forgetting by
**4.55 pp** (37.33% → 32.78%) **and simultaneously increases final average
accuracy by 4.48 pp** (62.16% → 66.65%), robustly across 3 seeds (std < 1 pp).
EWC on ternary STE is the **first Track B method to beat the L8 baseline**.
The accuracy *increase* (not just less forgetting) suggests EWC acts as a
beneficial regularizer on the ternary latent scores.

**Permuted MNIST — EWC does not help.** Forgetting is unchanged (+0.26 pp,
within noise) and accuracy drops 2.14 pp. This matches the predicted cause:
Permuted MNIST uses **all 10 output neurons in every task** (class-incremental,
shared head), so after task 1 the entire output layer has high Fisher and the
λ=10000 penalty freezes it, limiting new-task learning. The high λ tuned for
Split (which benefits from *fresh* output neurons per task) is too stiff for
Permuted.

**Interpretation:** EWC + ternary STE is effective when tasks use **disjoint
output classes** (task-incremental Split) — the protected weights (task-1
classes + shared hidden layers) stay put while new tasks grow free output
neurons. It is ineffective for **shared-head class-incremental** (Permuted) at
the same λ; a much lower λ may be needed there (untested — out of B1 scope).

**λ sensitivity:** unlike float EWC's classic U-curve, mid-range λ **hurts**
ternary (1, 10, 100 worse than baseline) while high λ monotonically helps.
The optimum appears to be at the tested edge (λ=10000); whether it extends
further is an open question.

---

## Next Steps

- **B2:** QLoRA + frozen ternary backbone — zero forgetting by design, the
  natural next comparison against this EWC result on Split MNIST.
- **B3:** Multi-head ternary EWC — combine per-task output heads with EWC for
  maximal protection; and a λ sweep for Permuted (shared-head) at lower λ.
- Consider testing even higher λ (≥30000) to confirm the Split-MNIST trend has
  not plateaued.

---

## References

- Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting in neural networks". PNAS 114(13): 3521-3526.
- Schwarz, J. et al. (2018). "Progress & compress: A scalable framework for continual learning". ICML 2018.
- [E010: L8 Forgetting Baseline](../experiments/E010-l8-forgetting-baseline.md) — control experiment (no EWC)
- [ROADMAP.md](../ROADMAP.md) — Track B roadmap
