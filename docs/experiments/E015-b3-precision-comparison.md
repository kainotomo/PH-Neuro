# E015: B3 — Precision Comparison for Continual Learning

- **Date:** 2026-07-31
- **Git commit:** (base)
- **Status:** completed (12 new runs + 12 reused L8 runs)
- **Phase:** 3B (Track B: Continual Learning with Ternary STE)

---

## Hypothesis

**Stronger weight quantization reduces catastrophic forgetting in continual learning.** Replicating "When Less is More" (Zhang et al., arXiv:2512.18934 — INT8/INT4 quantization improves CL via quantization noise acting as implicit regularization) and **extending the precision ladder down to ternary**: if quantization noise is the mechanism, then ternary (2-bit, the strongest quantization) should forget the least.

| Precision | Expected Forgetting | Expected Accuracy |
|:----------|:------------------:|:-----------------:|
| FP16 | Highest (~36%) | Highest (~98%) |
| INT8 | Medium (~25%) | High (~97%) |
| INT4 | Low (~15%) | Medium (~95%) |
| **Ternary** | **Lowest? (~8%)** | **Good? (~94%)** |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | MLP 784→512→256→10 (ReLU + BatchNorm) |
| Total parameters | ~535–537K (all precisions) |
| Weight precisions | Ternary {-1,0,+1} (STE) · INT8 (QAT) · INT4 (QAT) · FP16 |
| Quantizer | Per-tensor symmetric fake-quantize + STE (`qat_helpers.fake_quantize_ste`) |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Epochs per task | 10 |
| Seeds | 42, 43, 44 |
| DataLoader workers | 2 (matches L8 baseline conditions; verified byte-identical to 0) |
| Device | RTX 4060 (CUDA) |
| Protocols | Split MNIST (5 tasks) · Permuted MNIST (10 tasks) |
| Source of ternary/fp16 | **Reused from L8** (`l8_results/`, identical hyperparameters + seeds) |
| New runs (B3) | INT8 + INT4 × 2 protocols × 3 seeds = **12 runs** → `b3_results/` |

### Protocol notes
- **Split MNIST** — 5 binary tasks (0v1, 2v3, ..., 8v9), 2 output classes/task, fresh output layer per task (multi-head by construction).
- **Permuted MNIST** — 10 tasks, all 10 classes with a different fixed pixel permutation, **shared 10-output head** (class-incremental).

**Evaluation:** standard continual-learning metrics via `evaluate_continual_learning`: `average_accuracy` (mean per-task accuracy after all tasks) and `average_forgetting` (mean over tasks of peak−final accuracy). Matches L8/B1/B2 protocol for direct comparability.

---

## Implementation

| File | Purpose |
|------|---------|
| `src/ph_neuro/examples/run_b3_precision_cl.py` | Runner: `--protocol`, `--weight-format {ternary,fp16,int8,int4}`; reuses L8 loop + `qat_helpers` builders + `ste_mlp` |
| `src/ph_neuro/examples/aggregate_b3_results.py` | Aggregator: merges B3 (int8/int4) + L8 (ternary/fp16), 4-way comparison table + per-task breakdown |
| `scripts/run_b3_precision.sh` | Orchestration: 12 runs, per-run logs `logs/b3/`, skip existing, FAILED + continue, NUM_WORKERS env |
| `tests/integration/test_b3_precision.py` | 20 tests (10 fast + 10 slow): builders, weight stats, end-to-end runs, JSON schema, determinism, aggregator |

**Reused (not rewritten):** `run_l8_forgetting_baseline` (`_build_fp16_mlp`, `_compute_ternary_weight_stats`, `train_task`), `qat_helpers` (`create_int8/int4_qat_mlp`), `models.ste_models.ste_mlp`, `training.continual` (`run_continual_experiment`, `make_backprop_predict_fn`), `analysis.continual.evaluate_continual_learning`.

---

## Results

### Main Comparison — Avg Forgetting & Avg Accuracy (3 seeds, mean ± std)

| Protocol | Precision | Avg Forgetting | Avg Accuracy | Δ Forgetting vs FP16 |
|:---------|:----------|:--------------:|:------------:|:--------------------:|
| Split | FP16 (L8) | 37.55% ± 0.48 | 62.12% ± 0.46 | — |
| Split | INT8 (QAT) | **37.06%** ± 0.66 | **62.54%** ± 0.63 | **+0.49 pp** |
| Split | INT4 (QAT) | 37.11% ± 0.61 | 62.50% ± 0.60 | +0.44 pp |
| Split | Ternary (L8) | 37.33% ± 2.32 | 62.16% ± 2.39 | +0.22 pp |
| Permuted | FP16 (L8) | 55.52% ± 0.25 | 41.94% ± 0.17 | — |
| Permuted | INT8 (QAT) | 54.40% ± 0.43 | **43.08%** ± 0.41 | +1.12 pp |
| Permuted | INT4 (QAT) | **54.36%** ± 0.52 | 43.07% ± 0.53 | **+1.17 pp** |
| Permuted | Ternary (L8) | 54.86% ± 2.63 | 41.92% ± 2.63 | +0.66 pp |

**Δ Forgetting: positive = precision forgets LESS than FP16.**

Ranking by forgetting (both protocols, consistent): **FP16 > Ternary > INT8 ≈ INT4**.

### Per-Task Detail (avg across 3 seeds)

**Split MNIST** — task 3 (4 vs 5) is the hardest for all precisions; quantization mainly helps task 4:

| Precision | T1 fgt | T2 fgt | T3 fgt | T4 fgt | T5 fgt |
|:----------|:------:|:------:|:------:|:------:|:------:|
| FP16 (L8) | 46.62% | 34.18% | 88.14% | 18.82% | 0.00% |
| Ternary (L8) | **44.81%** | **32.19%** | 87.57% | 22.09% | 0.00% |
| INT8 | 47.38% | 32.48% | 87.48% | **17.96%** | 0.00% |
| INT4 | 47.22% | 33.15% | **87.30%** | **17.88%** | 0.00% |

**Permuted MNIST** — quantization's win is spread over the mid/late tasks (e.g. task 8: INT4 23.61% vs FP16 24.10%; task 9: INT8/INT4 8.38% vs FP16 6.96% — not monotonic per-task):

| Precision | T1 | T3 | T6 | T8 | T9 | T10 |
|:----------|:--:|:--:|:--:|:--:|:--:|:---:|
| FP16 (L8) | 84.79% | 81.86% | 67.64% | 24.10% | 6.96% | 0.00% |
| Ternary (L8) | 83.41% | 77.94% | 61.72% | 32.52% | 11.79% | 0.00% |
| INT8 | 84.03% | 79.01% | 66.51% | 26.87% | 8.38% | 0.00% |
| INT4 | 86.27% | 78.43% | 60.19% | 23.61% | 8.38% | 0.00% |

---

## Analysis

1. **"When Less is More" is confirmed but WEAK at this scale.** Quantization reduces forgetting on **both** protocols (every quantized precision forgets less than FP16), but the effect is only **0.2–1.2 pp** — far below the 36→25→15→8% ladder the roadmap hypothesized. The benefit is ~2–5× larger on Permuted (shared 10-class head → more gradient interference to regularize) than on Split (fresh 2-class heads per task, already partially isolated). At MLP+MNIST scale the quantization-noise regularization is real but modest.
2. **INT8 ≈ INT4 — the benefit saturates at 8-bit.** INT8 and INT4 are statistically identical (Permuted 54.40% vs 54.36%; Split 37.06% vs 37.11% forgetting). Going from 8-bit to 4-bit adds no further forgetting reduction, so the effect is **not monotonic** with precision below 8-bit.
3. **Ternary does NOT sit at the bottom of the ladder.** Ternary lands **between FP16 and INT8/INT4** on both protocols (Permuted +0.66 pp vs INT +1.12/+1.17; Split +0.22 pp vs +0.44/+0.49). The hypothesis "strongest quantization (2-bit) → lowest forgetting" is **not supported**. Ternary still forgets *less* than FP16 (direction holds), but does not beat INT QAT. Note ternary's high seed variance (±2.3–2.6) makes its ordering vs INT borderline within noise.
4. **Accuracy: quantization wins too.** INT8/INT4 slightly **beat both FP16 and ternary** on accuracy (+0.3 to +1.1 pp; Permuted 43.08% vs FP16 41.94%, ternary 41.92%). Implicit regularization improves final accuracy marginally, not just forgetting.
5. **Seed stability.** INT runs are far more seed-stable (±0.4–0.7 pp) than the L8 ternary runs (±2.3–2.6 pp) and comparable to FP16 L8. The ternary results (both accuracy and forgetting) are noisier across seeds.
6. **Why ternary isn't the best regularizer (hypothesis).** STE `sign()` is a hard, deterministic quantization whose "noise" is fully correlated with the gradient (STE passes the gradient straight through), whereas QAT's per-tensor scale + rounding injects smoother, effectively independent rounding noise. On small models with shared-head interference, QAT's noise behaves like the implicit regularizer "When Less is More" describes; the hard ternary discretization does not provide the same benefit. This refines the L8 finding (ternary ≈ FP16 forgetting) into: **INT8/INT4 < Ternary ≈ FP16** in forgetting.

---

## Bugs & Issues

- **num_workers deviation from B2 rule:** `run_b3_precision.sh` defaults to `NUM_WORKERS=2` (not 0). Verified byte-identical results vs 0; matches L8 baseline conditions (L8 ran with default nw=2); 2.4× faster (PermutedMNIST permutes pixels in Python `__getitem__` → nw=0 is the bottleneck). Tiny 535K-param models have no OOM/corruption risk that motivated the B2 rule. Override: `NUM_WORKERS=0 bash scripts/run_b3_precision.sh`.

---

## Next Steps

- **B3.3:** Multi-head ternary EWC (per-task output heads + EWC).
- Consider r=128 QLoRA probe (Permuted saturation open from E014).
- **Open question:** does the "When Less is More" effect grow with model/dataset scale (CIFAR, deeper nets)? At MLP+MNIST it is small (≤1.2 pp).
- Position B3 alongside B1 (EWC) and B2 (QLoRA) in the Track B narrative.
