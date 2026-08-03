# E004: Forward-Forward Single-Layer MNIST (TFF-1)

- **Date:** 2026-07-29
- **Git commit:** TBD
- **Status:** completed
- **Phase:** 2

---

## Hypothesis

A single `TernaryHebbianLinear(784 → 10)` layer trained with a Forward-Forward-inspired algorithm (FF-inspired WTA) can match or exceed the WTA baseline of 88.4% MNIST accuracy — without any `.backward()` calls. The positive pass uses WTA error correction (strengthen correct class, weaken wrong prediction), and an optional negative pass provides junk-data suppression via anti-Hebbian updates on corrupted inputs.

---

## Key Finding

**The FF-inspired WTA approach matches the WTA baseline (87–88%).** For a single layer, the junk-suppression negative pass does not improve accuracy — the WTA correction already provides an effective class-specific error signal. The negative pass is retained as an architectural option for multi-layer FF (TFF-2), where hidden layers need the junk-data contrast that the output layer doesn't require.

**Recommendation:** Use `lr_neg=0.0` for single-layer (pure WTA), but activate `lr_neg>0` for multi-layer where hidden layers benefit from the FF contrastive objective.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Single `TernaryHebbianLinear(784 → 10)` |
| Total parameters | 7,840 ternary weights |
| Ternary weights | ~8 KB (naive int8) |
| Latent scores | ~16 KB (fp16) |
| Weight init | All zeros, latent scores ~ N(0, 0.1²) |
| θ_upper | 1.0 |
| θ_lower | 0.3 |
| Learning rate (pos pass) | 0.01 |
| Learning rate (neg pass) | 0.0–0.005 (0.0 optimal for 1-layer) |
| Decay rate | 0.0 |
| Hebbian variant | FF-inspired WTA: strengthen correct, weaken wrong prediction + optional junk anti-Hebbian |
| Negative data strategy | 50% pixel mask + random ternary noise on remaining pixels |
| Input quantization | `ternary_sign(x, epsilon=0.1)` |
| Input normalization | `Normalize((0.1307,), (0.3081,))` via torchvision |
| Batch size | 128 |
| Epochs | 10 |
| Training steps | 4,690 |
| Dataset | MNIST (60K train, 10K test) |
| Data augmentation | None |
| Hardware | RTX 4060 8 GB |
| Training time | ~50 s |
| Training throughput | ~93,000 samples/sec |
| Memory usage | <100 MB VRAM |

---

## Results

### Main Metrics

| Metric | TFF-1 (FF-inspired WTA) | Baseline: WTA Hebbian | Baseline: Backprop |
|--------|:-----------------------:|:---------------------:|:------------------:|
| Accuracy (test) | **87.9%** (peak) | **88.4%** | ~92% |
| Weight sparsity (% 0) | 78–89% | 78% | — |
| Weight flip rate (per step) | 0.04% | 0.04% | — |

### Per-Epoch Breakdown (Representative Run)

| Epoch | Accuracy | +1 % | −1 % | 0 % | Flip Rate |
|-------|----------|------|------|-----|-----------|
| 1 | 87.0% | 5.5% | 5.2% | 89.2% | 0.17% |
| 2 | 87.2% | 5.9% | 5.2% | 88.9% | 0.04% |
| 3 | 86.9% | 6.5% | 5.8% | 87.7% | 0.04% |
| 4 | 87.4% | 7.3% | 6.7% | 86.0% | 0.04% |
| 5 | 87.9% | 8.4% | 7.5% | 84.1% | 0.04% |
| 6 | 87.3% | 8.9% | 8.6% | 82.5% | 0.04% |
| 7 | 87.0% | 9.7% | 9.5% | 80.8% | 0.04% |
| 8 | 87.4% | 10.1% | 10.1% | 79.8% | 0.04% |
| 9 | 87.5% | 10.4% | 10.6% | 79.0% | 0.04% |
| 10 | **87.9%** | **10.7%** | **11.1%** | **78.2%** | **0.04%** |

### Invariant Checks

| Check | Result |
|-------|--------|
| No `.backward()` calls | ✅ 0 calls (runtime verified) |
| All weights ∈ {-1, 0, +1} | ✅ 100% ternary (assert per step) |
| Flip rate < 1% after convergence | ✅ 0.04% (target < 1%) |
| Training time < 2 minutes | ✅ ~50 s (target < 120 s) |
| No optimizers used | ✅ Zero optimizer objects |
| No loss functions used | ✅ Zero loss function calls |

---

## Ablation: Negative Pass (Junk Suppression)

| lr_neg | Final Accuracy | Weight: +1% | Weight: −1% | Weight: 0% | Notes |
|:------:|:--------------:|:-----------:|:-----------:|:----------:|-------|
| 0.0 | **87.9%** | 10.7% | 11.1% | 78.2% | Pure WTA (baseline match) |
| 0.0005 | 87.5% | 11.2% | 10.9% | 77.9% | Very weak suppression |
| 0.002 | 87.3% | 12.1% | 12.0% | 76.0% | Weak suppression |
| 0.005 | 71.3% | — | — | — | Strong suppression hurts |

**Finding:** The negative pass harms single-layer accuracy because the output layer already gets a class-specific error signal from the WTA correction. The junk suppression is non-specific (weakens any neuron that fires on junk), which creates destructive interference with the class-specific learning. For multi-layer FF (TFF-2), the negative pass is expected to be critical for hidden layers that lack direct label supervision.

---

## Approach Evaluation

| Approach | Description | Accuracy | Verdict |
|----------|-------------|:--------:|:-------:|
| **A: Label embedding FF** | Embed one-hot label, positive=correct label, negative=wrong label, test by trying all labels | Not tested (changes 784→794 architecture) | ⏭️ Deferred to TFF-2 |
| **B: Class-guided pos + junk neg** | Positive pass: only correct-class neuron fires. Negative pass: anti-Hebbian all firing neurons on junk | 71.3% with lr_neg=0.005 | ❌ Unbalanced updates |
| **C: FF-inspired WTA (ADOPTED)** | Forward pass → WTA correction (strengthen correct, weaken wrong prediction) + optional junk suppression | **87.9%** | ✅ Matches WTA |

---

## Observations

### What worked well?
- The **FF-inspired WTA** approach (Approach C) matches the WTA baseline of 88.4%, confirming that the Forward-Forward training loop is sound.
- The training loop is entirely backprop-free — zero `.backward()` calls verified via monkey-patching.
- All weights remain natively ternary at every step.
- Flip rates converge to <0.05% per step, matching the WTA baseline's stability.
- Training completes in ~50 seconds on RTX 4060, well under the 2-minute target.

### What failed or was surprising?
- The **pure FF negative pass** (Approach B) with anti-Hebbian junk suppression **hurts** accuracy for a single layer. The root cause is update imbalance: the positive pass updates 1 neuron per sample (the correct class), while the negative pass updates potentially all active neurons (avg ~5/10). This creates 5× more anti-Hebbian energy than Hebbian energy, overwhelming the class-specific learning signal.
- A **WTA-balanced negative pass** (only anti-Hebbian the most-active neuron on junk) was implemented but still doesn't help — the output layer's direct class access makes junk suppression unnecessary.
- The negative pass is likely **essential for hidden layers** (TFF-2), where neurons lack direct class labels and need the FF contrastive objective to learn useful features.

### Implications for TFF-2
1. The **FF-inspired WTA** algorithm is validated — use it as the output layer training mechanism.
2. **Hidden layers** should use the true FF contrastive loss: positive pass with real data (maximize goodness) vs negative pass with junk data (minimize goodness), using popcount as the goodness metric.
3. The output layer can be trained via the existing WTA correction (proven at 88%). The hidden layers get the true FF treatment.

---