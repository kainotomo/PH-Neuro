# E022: M1.2 — DQT CNN on CIFAR-100 (GO/NO-GO >55%)

- **Date:** 2026-08-04
- **Git commit:** `main` (post E021.3)
- **Status:** completed — 🟡 MARGINAL (150 ep mean 54.15%; 200-ep retry mean 53.65% — no improvement)
- **Phase:** 4 (Advanced Experiments — low-memory training)
- **Milestone:** M1.2 — "DQT CNN CIFAR-100 >55% accuracy"

---

## Hypothesis

**Direct Quantized Training (DQT) — proven on a 2-conv CNN for CIFAR-10 in
M1.1 (E020/E021) — can be scaled to a larger 3-conv CNN and beat the CIFAR-100
ternary STE baseline (E009/L1: **38.2%**) by ~17 points, crossing **>55%**
test accuracy.** CIFAR-100 is far harder than CIFAR-10 (100 classes vs 10, 50K
train images), so M1.1's 2-conv model (which hit ~79% on CIFAR-10) lacks the
capacity; this milestone triples the conv depth and widens channels to
64→128→256 while keeping the DQT mechanic and annealing schedule identical.

The key lessons carried over from M1.1 (all reused unchanged):
1. DQT `TernaryDQTConv2d` backward is numerically exact (M1.1 ✅).
2. Annealing stochastic→deterministic `sign()` at 80% removes late-training
   flip noise (flip 0.18 → 0.0008) — `ANNEAL_FRACTION = 0.80` (M1.1-RETRY-2 ✅).
3. DQT beats STE by +2.89 pp on the same architecture (M1.1 ✅).
4. The 2-conv CNN has a ceiling ~79% on CIFAR-10 → a **larger 3-conv model is
   required** for CIFAR-100 (M1.1 ❌ lesson, addressed here).

---

## Background: M1.1 (CIFAR-10) vs this Milestone (CIFAR-100)

| | M1.1 (E020/E021) | M1.2 (this milestone) |
|:--|:------------------|:----------------------|
| Dataset | CIFAR-10 (10 classes) | CIFAR-100 (100 classes, fine-grained) |
| Architecture | 2-conv: 3→64→128, FC 8192→512→10 | 3-conv: 3→64→128→256, FC 4096→512→100 |
| Conv layers | 2 × TernaryDQTConv2d | 3 × TernaryDQTConv2d |
| Flat features | 8192 (after 2× pool) | 4096 (after 3× pool) |
| Ternary weights | ~2.18M | ~2.52M |
| STE baseline | 72.75% | **38.2%** |
| Target | >80% | **>55%** |
| Anneal fraction | 0.85 → 0.80 (retry) | 0.80 |
| Epochs | 100 | 150 |

DQT mechanic (identical to M1.1): ternary weights {-1, 0, +1} stored as int8,
updated via stochastic rounding of an accumulated float buffer after every
`optimizer.step()`. No persistent latent float scores during training.

---

## Architecture

`dqt_cnn_cifar100()` in `src/ph_neuro/models/dqt_models.py` — larger than
`dqt_cnn()` (M1.1):

```
Input (3, 32, 32)
  → TernaryDQTConv2d(3 → 64, k3, p1, no bias)      [1,728 ternary]
  → ReLU → BatchNorm2d(64) → MaxPool2d(2)          # 32 → 16
  → TernaryDQTConv2d(64 → 128, k3, p1, no bias)    [73,728 ternary]
  → ReLU → BatchNorm2d(128) → MaxPool2d(2)         # 16 → 8
  → TernaryDQTConv2d(128 → 256, k3, p1, no bias)   [294,912 ternary]
  → ReLU → BatchNorm2d(256) → MaxPool2d(2)         # 8 → 4
  → Flatten  (4 × 4 × 256 = 4096)
  → TernaryDQTLinear(4096 → 512)                    [2,097,152 ternary]
  → ReLU → BatchNorm1d(512)
  → TernaryDQTLinear(512 → 100)                     [51,200 ternary]
```

- **Total ternary weights: 2,518,720**; total params (incl. BN): 2,521,252.
- `n_classes` is parametric (default 100); `flat_features` is derived
  dynamically: `256 × (img_size // 8)² = 4096` for 32×32 inputs.
- Conv layers have no bias — BatchNorm handles the per-channel shift.
- MaxPool and ReLU stay float (not quantized in this milestone).
- The 4096→512 classifier dominates (2.1M of 2.52M ternary weights), same as
  M1.1 where the FC head dominated.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-100 (50K train / 10K test), fine-grained (100 classes) |
| Data augmentation | RandomCrop(32, pad 4) + RandomHorizontalFlip + Normalize `(0.5071, 0.4867, 0.4408) / (0.2675, 0.2565, 0.2761)` |
| Architecture | `dqt_cnn_cifar100()` — TernaryDQTConv2d(3→64→128→256) ×3 → Flatten → TernaryDQTLinear(4096→512) → TernaryDQTLinear(512→100) |
| Total parameters | 2,521,252 (float buffers + BN) |
| Ternary weights | 2,518,720 (int8, no latent float scores) |
| Weight format | Ternary {-1, 0, +1} (int8), no latent float scores |
| Optimizer | AdamW, weight_decay 1e-4 |
| Learning rate | 0.01 (sweep: 0.01, 0.005, 0.001) |
| Scheduler | CosineAnnealingLR, T_max = epochs |
| Batch size | 128 |
| Epochs | 150 |
| Seeds | 42, 43, 44 (full run); 42 (sweep) |
| DQT update | `apply_dqt_rounding()` after EVERY optimizer step |
| Annealing | `stochastic_round()` for epochs 1–119; deterministic `sign()` for epochs 120–150 (`ANNEAL_FRACTION = 0.80`, `anneal_start_epoch = 120`) |
| Early stopping | patience 30 (> anneal_start_epoch 120) |
| Hardware | RTX 4060 8 GB |
| STE baseline (for comparison) | E009/L1: 38.2% (2-conv CNN, 150 ep) |

---

## Method

Identical DQT training loop to M1.1 (reused verbatim), with the dataset,
model, and hyper-parameters swapped:

```python
# after EVERY optimizer.step():
use_stochastic = epoch < anneal_start_epoch          # anneal_start_epoch = 120
epoch_flips += apply_dqt_rounding(model, use_stochastic=use_stochastic)
```

- Epochs 1–119: `stochastic_round()` (exploration).
- Epochs 120–150: deterministic `sign()` — `weight_ternary = sign(weight_float)`
  (clean fine-tuning regime, no flip jitter). The switch is logged once
  (`🔒 Epoch 120: switching to DETERMINISTIC sign`).
- Per-epoch metrics: train/test top-1 accuracy, loss, LR, flip rate, epoch time.
- Final metrics: best accuracy + epoch, final accuracy, weight stats
  (pos/neg/zero %), final flip rate (mean of last 5 epochs), total training
  time, peak GPU memory (`torch.cuda.max_memory_allocated()`), anneal start epoch.
- Results saved to `results_dqt_cifar100_lr{lr}_seed{seed}.json`.

**Execution flow (scripts/run_m1_2_dqt_cifar100.sh):**
1. **LR sweep** — 1 seed (42), lr ∈ {0.01, 0.005, 0.001}, 150 ep each → `results/phase1/m1_2_sweep_results/`.
2. **Full run** — best LR from the sweep, 3 seeds (42/43/44), 150 ep each → `results/phase1/m1_2_results/`.

**GO/NO-GO:** mean best test accuracy over 3 seeds > 55% → GO;
50–55% → MARGINAL (evaluate vs STE 38.2%, try 200 ep / 4-conv);
≤ 50% → NO-GO (try lr=0.005 with 200 ep, add Dropout(0.25)).

---

## Results

### Sanity check (smoke test) — 2 epochs, lr=0.01, seed 42

| Metric | Value |
|--------|-------|
| Test accuracy (best) | 11.14% (epoch 1) |
| Train accuracy (ep 2) | 6.74% |
| Final flip rate | 0.0029 |
| Peak GPU memory | 334.2 MB |
| Epoch time | ~8 s |

Confirms the full pipeline runs end-to-end on CIFAR-100 (50K/10K), learns
quickly from random (11% in 2 epochs), and fits in well under the 8 GB VRAM
budget. Note: with only 2 epochs, `anneal_start_epoch = 1` so both epochs ran
in deterministic mode — the stochastic phase only matters for long runs.

### LR sweep — 150 epochs, seed 42, lr ∈ {0.01, 0.005, 0.001}

| LR | Best Acc | Final Acc | Best Epoch | Trained | Sparsity (%0) | Final Flip | Time |
|:---:|:--------:|:---------:|:----------:|:-------:|:-------------:|:----------:|:----:|
| **0.01** | **54.39%** | 54.36% | 148 | 150 | 0.0% | 0.0006 | 1170 s |
| 0.005 | 53.65% | 53.65% | 150 | 150 | 0.0% | 0.0004 | 1167 s |
| 0.001 | 47.30% | 47.21% | 140 | 150 | 0.0% | 0.0001 | 1176 s |

**Best LR: 0.01** (54.39%) — confirms the M1.1 finding that DQT wants a high
learning rate. All runs trained the full 150 epochs, annealed at epoch 120,
and reached a near-zero deterministic flip rate (0.0001–0.0006). Peak GPU
memory 336 MB across all runs.

### Main run — 3 seeds × 150 ep, lr=0.01

| Seed | Best Acc | Final Acc | Best Epoch | Trained | Sparsity (%0) | Final Flip | Time |
|:----:|:--------:|:---------:|:----------:|:-------:|:-------------:|:----------:|:----:|
| 42 | 54.39% | 54.36% | 148 | 150 | 0.0% | 0.0006 | 1189 s |
| 43 | 53.58% | 52.38% | 101 | 131 | 0.0% | 0.0006 | 1073 s |
| 44 | **54.48%** | 54.48% | 150 | 150 | 0.0% | 0.0005 | 1756 s |
| **mean** | **54.15%** | 53.74% | — | — | — | — | — |

> Seed 44's first attempt died silently at epoch 44 (~09:05) when another
> workload (a game) grabbed most of the 8 GB GPU → CUDA OOM (same pattern as
> the B2 r=32 probe). Re-run with `NUM_WORKERS=0` on a free GPU succeeded
> (54.48%, full 150 ep).

### Comparison with STE baseline (E009/L1, CIFAR-100)

| Method | Best Test Acc | Δ vs STE |
|:-------|:-------------:|:--------:|
| STE baseline (E009/L1, 2-conv, 150 ep) | 38.2% | — |
| **DQT M1.2 sweep lr=0.01 (3-conv, 150 ep)** | 54.39% | +16.2 pp |
| **DQT M1.2 full run (3 seeds, mean)** | **54.15%** | **+15.95 pp** |

**Verdict: 🟡 MARGINAL — mean best 54.15% (≤ 55% GO gate, > 50% NO-GO).**
DQT beats the CIFAR-100 STE baseline by +15.95 pp (38.2% → 54.15%), far
above the NO-GO line, just 0.85 pp short of GO.

---

## Observations

### What carried over from M1.1 (validated, unchanged)
- `apply_dqt_rounding()` + annealing at `ANNEAL_FRACTION = 0.80` reused
  verbatim — no changes to the layers or the training loop were needed.
- DQT layers (`TernaryDQTConv2d`, `TernaryDQTLinear`) unchanged — 16 unit
  tests still pass.
- Peak GPU memory 334 MB (smoke) — the 3-conv model is well within the 8 GB
  budget (M1.1 was 328–363 MB).

### What is new
- New model factory `dqt_cnn_cifar100()` (3-conv, parametric `n_classes`).
- New runner `run_m1_2_dqt_cifar100.py` (CIFAR-100 loaders, 150-ep defaults,
  patience 30, `--lr-sweep` in-process capability).
- New orchestration `scripts/run_m1_2_dqt_cifar100.sh` (sweep + full modes,
  skip-if-exists, logs to `logs/logs_m1_2/`).
- 8 new integration tests (build, forward, overfit, training loop, annealing,
  rounding modes) — all pass.

### LR sweep takeaways (seed 42)
- **lr=0.01 → 54.39%** is the best — DQT again prefers a high LR (matches
  E017/M1.1). lr=0.001 (47.3%) falls below the 50% gate, confirming LR is
  the critical knob.
- Best epoch (148) lands AFTER the deterministic switch (epoch 120) — the
  30-epoch deterministic tail lets the network fine-tune past its
  stochastic-phase peak (unlike M1.1 seed 43/44 which early-stopped before
  the switch). Patience 30 > anneal_start_epoch 120 is validated.
- Deterministic flip rate ~0.0001–0.0006 (≈ 0) — annealing fully removes
  flip jitter as designed.
- 0% weight sparsity in the deterministic phase, same as M1.1 (expected —
  ternary weights concentrate in ±1).

### Full-run observations (3/3 seeds)
- All three seeds land in **53.6–54.5%** (42: 54.39, 43: 53.58, 44: 54.48) — a
  tight, reproducible band → **MARGINAL**, +15.95 pp over the STE baseline,
  just 0.85 pp below the 55% gate.
- Seed 43 early-stopped at 131 (best ep 101, before the deterministic switch at
  120) — the same early-peak pattern as M1.1. Seeds 42/44 peaked inside the
  deterministic tail (ep 148/150) and hit 54.4-54.5%.
- All seeds: ~0 sparsity, near-zero deterministic flip (0.0005-0.0006) — the
  DQT + anneal pipeline is stable and reproducible.
- `num_workers=0` (seed 44) gave identical quality (54.48%, the best of the 3)
  at ~1.5× wall-clock (1756 s vs ~1180 s) — a safe option under GPU contention.
- **Ceiling analysis:** 150 epochs + 3-conv lands at ~54%. The best epochs
  (148-150) are AT the end of the run, so 150 epochs is not yet saturated.

---

## M1.2-RETRY (200 epochs) — completed, no improvement

Only runner defaults changed (150→200 ep, patience 30→40); everything else
identical (3-conv model, lr=0.01, anneal@80%, deterministic tail 160-200).

| Seed | Best Acc | Final Acc | Best Epoch | Trained | Flip | Time |
|:----:|:--------:|:---------:|:----------:|:-------:|:----:|:----:|
| 42 | 53.79% | 53.56% | 179 | 200 | 0.0006 | 1513 s |
| 43 | 53.05% | 52.00% | 128 | 168 | 0.0005 | 1287 s |
| 44 | 54.12% | 53.19% | 123 | 163 | 0.0299 | 1253 s |
| **mean** | **53.65%** | 52.92% | — | — | — | — |

**Verdict: 🟡 MARGINAL — mean 53.65%, −0.50 pp vs the 150-ep run (54.15%).**

### Why 200 epochs did NOT help (key finding)

1. **Seeds 43/44 peaked EARLY in the stochastic phase (ep 128/123) and
   early-stopped (patience 40) at ep 168/163 — just before/at the deterministic
   switch (160).** The longer schedule gave them a *shorter effective* tail, not
   a longer one. Seed 42 was the only one to peak in the tail (ep 179) and it
   still hit only 53.79% — below its 150-ep peak (54.39%).
2. **The ~54% ceiling is ARCHITECTURAL, not epoch-limited** — exactly the M1.1
   finding (2-conv ceiling ~79% was architectural, not tuning). Longer cosine
   simply slows LR decay, keeping the stochastic phase longer without raising
   the peak.
3. Flip rates stay ~0.0005-0.0006 (seed 44: 0.0299, slightly higher — its
   early-stop landed mid-tail). 0% sparsity as always.

**Conclusion: total epochs is NOT the lever. The 3-conv (64→128→256) model
saturates at ~54% on CIFAR-100; DQT beats STE by +15.5 pp but cannot reach
55% with this architecture at 150 OR 200 epochs.**

---

## Next steps (4-conv fallback from the brief)

1. **4-conv (64→128→256→512)** — more conv capacity is the remaining lever for
   the fine-grained 100-class task. Expected: more discriminative low-level
   features → higher ceiling. Need new `dqt_cnn_cifar100_v2()` (4 conv blocks)
   + flat = 512·(32//16)² = 2048 → FC(2048→512→100).
2. Dropout(0.25) after each MaxPool — only as a secondary regularizer if the
   bigger model overfits.

---

## Artifacts

- Model: `src/ph_neuro/models/dqt_models.py` (+`dqt_cnn_cifar100()`, 3-conv)
- Runner: `src/ph_neuro/examples/run_m1_2_dqt_cifar100.py` (defaults now 200 ep / patience 40 for M1.2-RETRY; annealing logic reused from M1.1)
- Script: `scripts/run_m1_2_dqt_cifar100.sh` (→ `research/scripts/run_m1_2_dqt_cifar100.sh`; hardened: `PYTHONUNBUFFERED=1`, `NUM_WORKERS` override, continue-past-failed-runs)
- Integration tests: `tests/integration/test_m1_2_dqt_cifar100.py` (8 tests, all pass)
- Results: `results/phase1/m1_2_sweep_results/` (3 JSON ✅), `results/phase1/m1_2_results/` (3 JSON ✅), `results/phase1/m1_2_retry_results/` (3 JSON ✅)
- Logs: `logs/logs_m1_2/`
