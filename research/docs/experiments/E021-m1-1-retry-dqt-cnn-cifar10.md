# E021: M1.1-RETRY — DQT CNN on CIFAR-10 (GO/NO-GO >80%)

- **Date:** 2026-08-03
- **Git commit:** `main` (post E020)
- **Status:** completed — 🔴 NO-GO (mean best 78.42% ≤ 80%)
- **Phase:** 4 (Advanced Experiments — low-memory training)
- **Milestone:** M1.1-RETRY — "DQT CNN CIFAR-10 >80% accuracy" (retry of E020)

---

## Hypothesis

The original M1.1 (E020) missed the 80% gate by 2.35 pp (mean best 77.65%).
Root cause identified: the **stochastic-rounding flip rate stays ~0.18 until
the end of training**, so the network never enters a clean fine-tuning regime
and test accuracy oscillates ±1-1.5 pp around the best epoch. Additionally,
>99.99% of the model's 4.27M parameters were in the 8192→512→10 classifier,
concentrating flip noise where it hurts the most.

**This retry combines three targeted changes (no new layers, no new runner)
to cross 80%:**

1. **Annealing stochastic → deterministic sign** for the final 15% of epochs
   (removes the late-training flip jitter → clean fine-tuning tail).
2. **Smaller FC head** — `8192→256→10` instead of `8192→512→10` (halves the
   classifier flip noise and parameter count: ~4.28M → ~2.18M).
3. Same 3 seeds (42/43/44), lr=0.01, 100 epochs.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (50K train / 10K test), RandomCrop(32, pad 4) + RandomHorizontalFlip + Normalize |
| Architecture | `dqt_cnn()` — TernaryDQTConv2d(3→64) → TernaryDQTConv2d(64→128) → Flatten → TernaryDQTLinear(8192→256) → TernaryDQTLinear(256→10) |
| Total parameters | ~2,176,330 (float buffers + BN) |
| Ternary weights | 2,175,168 (int8, no latent float scores) |
| Weight format | Ternary {-1, 0, +1} (int8), no latent float scores |
| Optimizer | AdamW, lr=0.01, weight_decay 1e-4 |
| Scheduler | CosineAnnealingLR, T_max = epochs |
| Batch size | 128 |
| Epochs | 100 (early stopping patience 15) |
| Seeds | 42, 43, 44 |
| DQT update | `apply_dqt_rounding()` after EVERY optimizer step |
| Annealing | `stochastic_round()` for epochs 1–84; deterministic `sign()` for epochs 85–100 (`ANNEAL_FRACTION = 0.85`, `anneal_start_epoch = int(0.85 × epochs)`) |
| Hardware | RTX 4060 8 GB |
| STE baseline (for comparison) | same architecture via STE runner (manual) |

---

## Method

The only functional change to the training loop is the rounding choice:

```python
# before the loop:
anneal_start_epoch = int(epochs * ANNEAL_FRACTION)   # 85 for 100 epochs

# inside the loop, replacing apply_stochastic_rounding(model):
use_stochastic = epoch < anneal_start_epoch
epoch_flips += apply_dqt_rounding(model, use_stochastic=use_stochastic)
```

When `use_stochastic=False`, every DQT layer calls the new
`apply_deterministic_rounding()`, which snaps `weight_ternary = sign(weight_float)`:

```python
@torch.no_grad()
def apply_deterministic_rounding(self) -> dict[str, float]:
    self._prev_ternary = self.weight_ternary.clone()
    w_new = self.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
    n_flips = (self.weight_ternary != w_new).sum().item()
    total = w_new.numel()
    self.weight_ternary.copy_(w_new)
    return {"flip_rate": n_flips / max(total, 1), "n_flips": n_flips}
```

The switch is logged once:
`🔒 Epoch 85: switching to DETERMINISTIC sign (no more stochastic rounding)`.

`anneal_start_epoch` and `anneal_fraction` are recorded in each result JSON.

**GO/NO-GO:** mean test accuracy over 3 seeds > 80% → GO; ≤ 80% → NO-GO.

---

## Results

### Main run — lr=0.01, 100 epochs, 3 seeds, anneal@85%

| Seed | Best Acc | Final Acc | Best Epoch | Trained | Sparsity (%0) | Final Flip | Time |
|:----:|:--------:|:---------:|:----------:|:-------:|:-------------:|:----------:|:----:|
| 42 | **79.49%** | 77.89% | 83 | 98 | 0.0% | 0.0006 | 624 s |
| 43 | 78.65% | 77.87% | 70 | 85 | 0.0% | 0.1389 | 556 s |
| 44 | 77.13% | 75.02% | 57 | 72 | 43.3% | 0.1790 | 465 s |
| **mean** | **78.42%** | 76.93% | 70 | 85 | 14.4% | 0.1062 | ~548 s |

**Verdict: 🔴 NO-GO — mean best 78.42% ≤ 80%** (missed by 1.58 pp).
Peak GPU memory: 328 MB (down from 363 MB — smaller FC head).

### Comparison with E020 (original M1.1)

| Seed | E020 (512 head) | **Retry (256 head + anneal)** | Δ |
|:----:|:---------------:|:-----------------------------:|:--:|
| 42 | 77.94% | **79.49%** | **+1.55 pp** |
| 43 | 76.15% | **78.65%** | **+2.50 pp** |
| 44 | **78.87%** | 77.13% | −1.74 pp |
| **mean** | **77.65%** | **78.42%** | **+0.77 pp** |

The retry improved the mean by +0.77 pp and — critically — seed 42/43 by
+1.5/+2.5 pp, but seed 44 *regressed* (−1.74 pp), keeping the mean below 80%.

### Annealing worked mechanically but didn't raise the peak

| Seed | Stochastic peak | Deterministic peak | Note |
|:----:|:---------------:|:------------------:|:-----|
| 42 | 79.49% (ep 83) | 79.27% (ep 90) | full 14-epoch deterministic tail, flip 0.173 → 0.0006 |
| 43 | 78.65% (ep 70) | — | early-stopped at ep 85 (right at the switch) |
| 44 | 77.13% (ep 57) | — | early-stopped at ep 72 (BEFORE the switch) |

- Seed 42's flip rate collapsed from ~0.173 to **0.0006** at the epoch-85
  switch and epoch-to-epoch oscillation dropped to ±0.5-0.8 pp — the 
  late-training jitter that plagued E020 is **gone**.
- **But the deterministic phase did NOT exceed the stochastic-phase peak.**
  Seed 42's best (79.49%) came at ep 83 (stochastic); the deterministic tail
  (85–98) stabilized around 77.5–79.3% without breaking through.

### Why still NO-GO

1. **Early stopping (patience 15) cut the deterministic tail short.** Seeds
   43 (best ep 70 → stopped ep 85) and 44 (best ep 57 → stopped ep 72) never
   got a meaningful deterministic fine-tuning phase — the exact mechanism the
   retry was built around. Only seed 42 reached a full 14-epoch deterministic
   tail.
2. **The 256-wide head is a capacity trade-off.** It helped seeds 42/43
   (less flip noise) but clearly hurt seed 44 (−1.74 pp). A smaller head caps
   the achievable ceiling for some seeds.
3. **Deterministic sign stabilizes but doesn't add accuracy.** Once weights
   freeze to `sign(float)`, the network stops exploring — the peak is set by
   the stochastic phase. Annealing alone cannot push past the ceiling; it
   only removes the jitter.

---

## Observations

### What worked well?
- The annealing mechanism is fully validated: deterministic `sign()` snaps
  the ternary weights, flip rate drops to ~0.0006, and late-training
  oscillation is eliminated (seed 42: stable 77.5–79.3% over ep 85–98 vs
  E020's ±1-1.5 pp swings).
- Mean improved 77.65% → 78.42% (+0.77 pp), with seeds 42/43 gaining
  +1.55/+2.50 pp.
- Training is ~3.7× faster (548 s vs ~1630 s mean per seed) with the
  half-size head, at lower peak memory (328 vs 363 MB).
- The smaller head halves classifier flip noise exactly as designed
  (4.28M → 2.18M params).

### What failed or was surprising?
- **Deterministic fine-tuning did not raise the peak above the stochastic
  phase.** The core hypothesis of the retry (remove jitter → fine-tune above
  80%) was only partially confirmed: jitter is gone, but the ceiling holds.
- **Early stopping undermines the anneal.** Seeds that peak early (43/44)
  stop before/at the switch, so they never benefit from the deterministic
  tail. The annealing schedule (ep 85) is too late relative to patience-15
  early stopping.
- **Seed 44 regressed** (77.13% vs 78.87% in E020) — the 256-wide head
  reduced capacity for that seed.

### Comparison to hypothesis
Partially confirmed: annealing removes late-training noise (flip 0.18 → 0.001)
and the smaller head cuts parameter count, but the combination did not cross
80% (mean 78.42%, −1.58 pp short). The ceiling is set by the stochastic phase
and the FC capacity trade-off, not by flip jitter alone.

### Recommended next steps (fallbacks from the brief, in order)
1. **Anneal from epoch 80 (ANNEAL_FRACTION 0.80)** — gives a 20-epoch
   deterministic tail and lets early-peaking seeds reach it. One-line change.
2. **Raise early-stopping patience or disable it** so every seed runs the
   full 100 epochs through the deterministic phase (seeds 43/44 never got it).
3. **lr=0.005** — slower, more stable exploration in the stochastic phase.
4. **Keep the 512-wide head + add annealing only** — recovers seed 44's
   capacity (+1.74 pp) while keeping the anneal gains on seeds 42/43.

---

## Artifacts

- Layer changes: `src/ph_neuro/layers/ste_dqt_conv.py`, `src/ph_neuro/layers/ste_dqt.py`
  (+`apply_deterministic_rounding()` on `TernaryDQTConv2d` and `TernaryDQTLinear`)
- Model: `src/ph_neuro/models/dqt_models.py` (FC head 8192→256→10)
- Runner: `src/ph_neuro/examples/run_m1_1_dqt_cifar10.py` (annealing logic)
- Script: `research/scripts/run_m1_1_dqt_cifar10.sh` (→ `results/phase1/m1_1_retry_results/`)
- Unit tests: `tests/layers/test_ste_dqt.py` (new, 11), `tests/layers/test_ste_dqt_conv.py` (+1)
- Integration tests: `tests/integration/test_m1_1_dqt_cifar10.py` (+2 annealing tests)
- Results: `results/phase1/m1_1_retry_results/results_dqt_cifar10_lr0.01_seed{42,43,44}.json`
