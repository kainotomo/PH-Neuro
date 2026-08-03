# E018: DQT + Hysteresis-STE Combined Experiment

- **Date:** 2026-08-03
- **Git commit:** `main` (post E017)
- **Status:** completed
- **Phase:** 4 (Advanced Experiments — low-memory training)

---

## Hypothesis

Combining two techniques validated independently should achieve **>97% accuracy with >90% sparsity AND no latent float scores** (4.5× less training memory):

1. **DQT (E017)** — ternary weights stored as int8, float buffer only for gradient accumulation, updated via **stochastic rounding**. Achieved 98.23% MNIST with 56% natural sparsity, no latent scores.
2. **Hysteresis-STE (E016/L2)** — dual-threshold hysteresis (`θ_upper` to activate, `θ_lower` to deactivate) as a sparsity regularizer. Achieved 97.92% MNIST with 95% sparsity.

The open question (flagged in E017's future work): **can the hysteresis regularizer push DQT's sparsity from 56% to >90% while keeping DQT's memory advantage and accuracy near 98%?**

---

## Method: Combined Layer

New layer `TernaryDQTHysteresisLinear` (`src/ph_neuro/layers/ste_dqt_hysteresis.py`):

- **Forward:** matmul uses the stored int8 ternary weights (`weight_ternary`); gradients route to the float accumulation buffer (`weight_float`) via a DQT-style STE autograd function. **No latent float scores** — the int8 ternary weights are the trained state.
- **After each `optimizer.step()`** — `apply_stochastic_rounding()` applies the combined discretisation rule:

```
|w_float| < theta_lower  ->  0                          (hysteresis deactivate)
|w_float| > theta_upper  ->  stochastic_round(w_float)  (DQT stochastic activate)
else (hysteresis gap)    ->  keep current ternary       (hysteresis memory)
```

The stochastic rounding in the upper zone is the key DQT mechanism that mitigates the L2 deadzone problem: weights just above `θ_upper` activate probabilistically instead of requiring a deterministic threshold crossing.

`explore_gap=True` is exposed as an ablation knob: stochastic-round the hysteresis gap too (removes hysteresis memory entirely).

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | MLP `[784, 512, 256, 10]` (`TernaryDQTHysteresisLinear`) |
| Total parameters | 535,040 (int8 ternary weights + float32 buffers) |
| Weight format | Ternary {-1, 0, +1} (int8), **no latent scores** |
| Hysteresis rule | \|w\| > θ_u → stochastic_round; \|w\| < θ_l → 0; gap → keep prev |
| θ_upper / θ_lower | **0.3 / 0.15** (L2 best config) |
| Optimizer | AdamW (lr=1e-2, weight_decay=1e-4) — DQT best lr |
| Scheduler | CosineAnnealingLR (T_max=60) |
| Batch size | 128 |
| Epochs | 60 (early stopping patience 10) |
| Init std | 0.1 (DQT best) |
| Seeds | 42, 43, 44 |
| Dataset | MNIST (60K train / 10K test) |
| Hardware | RTX 4060 8 GB (CUDA) |

**Hyperparameters are the exact best of each parent:** DQT best (lr=0.01, epochs=60, batch=128, init_std=0.1) + L2 best thresholds (θ_u=0.3, θ_l=0.15).

---

## Results

### Main config (θ_u=0.3, θ_l=0.15, lr=0.01, 3 seeds)

| Seed | Best Acc | Final Acc | Best Epoch | Sparsity (%) | Final Flip Rate | Time (s) |
|:----:|:--------:|:---------:|:----------:|:------------:|:---------------:|:--------:|
| 42 | 97.48% | 96.93% | 11 | 65.0% | 0.197 | 174 |
| 43 | 98.44% | 98.12% | 56 | 58.3% | 0.188 | 553 |
| 44 | 98.35% | 98.14% | 60 | 58.2% | 0.188 | 378 |
| **Mean ± std** | **98.09 ± 0.53%** | **97.73 ± 0.69%** | — | **60.5 ± 3.8%** | **0.191** | **368** |

> Note: seed 42 early-stopped at epoch 21 (best at 11) giving the low-accuracy /
> high-sparsity corner of the trade-off; seeds 43/44 trained the full 60 epochs
> and reached ~98.4% with ~58% sparsity.

### Deadzone ablation: `explore_gap=True` (seed 42)

Stochastic-rounding the hysteresis gap too (removes hysteresis memory entirely)
tests whether the gap memory is what retains what little sparsity exists:

| Config (seed 42) | Best Acc | Sparsity | Final Flip Rate |
|:-----------------|:--------:|:--------:|:---------------:|
| Main (gap = hysteresis memory) | 97.48% | 65.0% | 0.197 |
| **`explore_gap=True`** (stochastic everywhere) | **98.05%** | **57.7%** | **0.248** |

Result: **more stochastic rounding → less sparsity (57.7%), more weight churn
(0.248), and slightly higher accuracy (98.05%).** This confirms the densification
is driven by the DQT stochastic component — even the hysteresis gap memory only
buys ~7 pp of sparsity over full stochastic rounding, and neither approach comes
anywhere near 90%.

### Comparison with baselines

| Method | Accuracy | Sparsity | Latent Scores | Training Memory |
|:-------|:--------:|:--------:|:-------------:|:---------------:|
| STE (L1) | 98.17% | 0% | Yes | ~9 bytes/param |
| Hysteresis (L2) | 97.92% | 95% | Yes | ~9 bytes/param |
| DQT (E017) | 98.23% | 56% | No | ~2 bytes/param |
| **DQT+Hysteresis (E018)** | **98.09 ± 0.53%** | **60.5 ± 3.8%** | **No** | **~2 bytes/param** |

### Seed 42 trajectory (epoch → test acc / sparsity / flip)

| Epoch | Test Acc | Sparsity | Flip Rate |
|:-----:|:--------:|:--------:|:---------:|
| 1 | 91.78% | 93.9% | 0.058 |
| 5 | 96.48% | 88.3% | 0.124 |
| 11 (best) | **97.48%** | 78.5% | 0.173 |
| 15 | 97.40% | 71.7% | 0.191 |
| 21 (stop) | 96.93% | 65.0% | 0.199 |

---

## Analysis

### 1. Accuracy — target met, matches the baselines

Best **98.09 ± 0.53% > 97%** ✓. The mean is statistically indistinguishable from
DQT (98.23%), STE (98.17%) and slightly *above* L2 hysteresis (97.92%). Unlike
the seed-42-only snapshot, the combination does **not** lose accuracy to DQT —
it reproduces DQT's accuracy (seeds 43/44: 98.44/98.35%) while training full
epochs.

### 2. Sparsity — target MISSED (60.5%, not >90%) ✗ — THE HEADLINE NEGATIVE RESULT

**The DQT stochastic rounding completely erases the hysteresis sparsity:**

- Sparsity starts high (93.9% — hysteresis working) but **monotonically decreases**
  to a plateau of ~58–65%.
- Final sparsity (60.5%) is essentially **identical to pure DQT (56%)** — i.e. the
  hysteresis regularizer adds **zero** net sparsity on top of DQT.
- Root cause: with lr=0.01 the float buffer weights quickly grow past
  `θ_upper=0.3`. Once `|w| > θ_u`, the upper zone applies `stochastic_round`,
  which reactivates weights to ±1 with high probability (e.g. w=0.8 → +1 with
  80% prob). The hysteresis memory only protects the narrow gap
  `0.15 ≤ |w| ≤ 0.3`, which high-lr weights blow through in a few epochs.
- The trade-off is visible across seeds: seed 42 (early-stopped, more sparse at
  65%) is the *lowest* accuracy (97.48%); seeds 43/44 (58% sparse) are the
  highest (~98.4%). Keeping sparsity high costs accuracy, and even the sparse
  corner is far from 90%.

### 3. Does hysteresis block DQT? — No deadzone problem

**No.** Unlike pure L2 (where θ_u≥0.5 froze the network at ~10%), θ_u=0.3 +
stochastic rounding trains fine — no deadzone barrier, no collapse. The DQT
stochastic rounding in the upper zone provides the exploration that deterministic
hysteresis lacks, so the L2 deadzone is **resolved by construction**. Accuracy
never drops below ~96% after epoch 3 in any seed.

### 4. Stability concern — flip rate stays high and rising

Final flip rate **~0.19 and still climbing** in every seed (vs L2 hysteresis's
~0.005%/epoch). The hysteresis's stability benefit is **lost**: the stochastic
upper-zone rounding constantly re-randomises weights near the boundary, and the
network never reaches a fixed point (seeds 43/44 flip ~0.19 even at epoch 60).
This is a real concern for continual-learning uses (Track B).

---

## Answer to the research question

**No — the combination does not achieve >97% accuracy with >90% sparsity.** It
achieves **98.09% / 60.5% sparsity** (3 seeds). Accuracy target met ✅, sparsity
target missed ❌:

| | Desired | DQT+Hyst got | DQT alone | L2 alone |
|:--|:-------:|:------------:|:---------:|:--------:|
| Accuracy | >97% | ✅ 98.09% | 98.23% | 97.92% |
| Sparsity | >90% | ❌ 60.5% | 56% | 95% |
| No latent scores | yes | ✅ | ✅ | ❌ |

The sparsity target is the failure point: **stochastic rounding densifies faster
than hysteresis can prune at lr=0.01**, and the combination degenerates to plain
DQT in both accuracy and sparsity — the hysteresis contributes nothing
measurable. The two techniques are **not additive under DQT-best
hyperparameters**.

---

## Proposals for improvement

1. **Anneal / scale θ_upper to the growing float magnitudes.** Raise θ_upper over training (e.g. from 0.3 → 1.0) so the deactivation boundary tracks the weight growth, letting hysteresis prune later in training (the L2-recommended deadzone fix, applied here to sparsity retention).
2. **Make the upper zone deterministic `sign()` and move stochasticity to the gap only.** If `|w| > θ_u → sign(w)` (pure L2 rule) instead of `stochastic_round`, sparsity should rise toward L2's 95% while keeping some stochastic exploration in the gap. Trade-off: less boundary exploration. The `explore_gap` ablation already shows the opposite direction (full stochastic → 57.7%); the *reverse* ablation (full deterministic above θ_u) is the natural next test.
3. **Lower lr / add weight decay toward zero.** lr=0.01 is tuned for pure DQT's exploration; a smaller lr (or decay toward 0) would keep float weights in the hysteresis gap longer, preserving memory. Seeds 43/44 show full-epoch runs push sparsity down to ~58%; a lower lr is the cheapest lever to hold it higher.
4. **Post-hoc sparsification at inference** — train dense-ish (DQT) then threshold-prune to 90% at inference (retain-and-prune). Decouples training dynamics from deployment sparsity; DQT's accuracy is unaffected by pruning at this scale.
5. **Annealed stochastic rounding probability** — high exploration early → low late, to recover L2's convergence stability (fixes the rising flip rate ~0.19) while keeping DQT's training-memory advantage.
6. **`explore_gap=True` is NOT the answer** — confirmed to *reduce* sparsity (57.7%). The hysteresis memory is doing the (limited) work; removing it only helps accuracy marginally.

---

## Artifacts

- Layer: `src/ph_neuro/layers/ste_dqt_hysteresis.py` (`TernaryDQTHysteresisLinear`, `hysteresis_stochastic_round`)
- Runner: `src/ph_neuro/examples/run_dqt_hysteresis.py` (CLI: `--theta-upper --theta-lower --explore-gap --lr --epochs --seed ...`)
- Script: `scripts/run_dqt_hysteresis.sh`
- Tests: `tests/layers/test_ste_dqt_hysteresis.py` (15 tests ✅)
- Results: `dqt_hysteresis_results/results_mnist_th0.3_tl0.15_seed{42,43,44}.json`, `..._gap_seed42.json`
- Logs: `dqt_hysteresis_results/log_mnist_seed{42,43,44}.log`, `log_mnist_th0.3_tl0.15_gap_seed42.log`
