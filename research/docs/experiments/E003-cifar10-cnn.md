# E003: Ternary Hebbian CNN on CIFAR-10

- **Date:** 2026-07-28
- **Git commit:** TBD
- **Status:** completed
- **Phase:** 1.2

---

## Hypothesis

A ternary Hebbian CNN with greedy layer-wise training can exceed 55% accuracy on CIFAR-10, demonstrating that ternary Hebbian learning works for real vision tasks with spatially-structured data.

---

## Key Finding

**Unsupervised Hebbian for conv layers does not improve over random projections for CIFAR-10 classification.** Three experimental variants were tested:

| Variant | Test Accuracy | Conv Sparsity | Problem |
|---------|:-----------:|:------------:|---------|
| Random conv weights (baseline) | **33.0%** | ~10% | Initial bootstrap |
| Competitive per-position Hebbian | **32.6%** | ~50% (conv1) | Learned features ≈ random |
| Class-guided Hebbian | **21.9%** | ~93% (saturated!) | Kills feature diversity |

The target of >55% was not reached. The convolutional architecture and greedy training pipeline are verified (no `.backward()`, ternary constraints, weight flip stabilization), but the Hebbian learning rules tested do not create class-relevant features beyond what random ternary projections provide.

This is consistent with Phase 1.1's finding that depth does not improve Hebbian MLP accuracy — unsupervised Hebbian learning does not create useful hidden representations.

---

## Configuration

### Architecture (same for all variants)

```
CIFAR-10 (3×32×32) → ternary_sign(ε=0.1)
  → Conv(3→64, 3×3, pad=1) → sign → MaxPool2d(2)
  → Conv(64→128, 3×3, pad=1) → sign → MaxPool2d(2)
  → Flatten (8192) → Linear(8192→10)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total ternary weights | 157,376 |
| Conv θ_upper | 2.0 |
| Conv θ_lower | 0.5 |
| Output θ_upper | 1.0 |
| Output θ_lower | 0.3 |
| Learning rate (conv) | 0.01 |
| Learning rate (output) | 0.01 |
| Batch size | 128 |
| Conv epochs | 5 each |
| Output epochs | 10 |
| Dataset | CIFAR-10 (50K train, 10K test) |
| Data augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Hardware | RTX 4060 8 GB |

---

## Results

### Main Metrics

| Metric | Random conv (baseline) | Competitive Hebbian | Class-guided Hebbian |
|--------|:--------------------:|:------------------:|:-------------------:|
| Test accuracy | **33.0%** | 32.6% | 21.9% |
| Train accuracy | 28.8% | 29.1% | 20.7% |
| Conv1 sparsity (% 0) | 90.0% | 51.0% | 7.0% |
| Conv2 sparsity (% 0) | 90.0% | 72.1% | 6.1% |
| Output sparsity (% 0) | 68.2% | 69.2% | 91.8% |
| Flip rate (converged) | — | <0.01%/step | <0.01%/step |

### Training Dynamics

**Competitive Hebbian:** Conv1 and Conv2 showed initial flips (0.10% and 0.05% in epoch 1) but rapidly converged. The learned feature density (~50% of conv1 weights active) was much higher than the ~10% bootstrap, but accuracy matched the random baseline.

**Class-guided Hebbian:** Conv layers learned aggressively (0.17% and 0.18% flips in epoch 1) but quickly saturated to >90% active weights. This loss of sparsity destroyed feature diversity and reduced accuracy below the random baseline.

### Evaluation of the Hebbian CNN Architecture

Despite not reaching the accuracy target, Phase 1.2 successfully demonstrated:

1. ✅ **TernaryHebbianConv2d** — correct forward pass (F.unfold + MatMul), spatial Hebbian update, hysteresis refresh
2. ✅ **HebbianCNN** — full model wiring, forward_through for greedy training
3. ✅ **No `.backward()`** — confirmed zero backward calls during training
4. ✅ **Ternary weight invariant** — all weights remain in {-1, 0, +1} at every step
5. ✅ **Weight flip stabilization** — flip rate converges to <0.01%/step
6. ✅ **Greedy layer-wise training** — conv1 → freeze → conv2 → freeze → output WTA
7. ✅ **End-to-end CIFAR-10 training** — completes in ~30 min on RTX 4060

---

## Comparison to Baselines

| Method | CIFAR-10 Expected |
|--------|:----------------:|
| PH-Neuro (ternary Hebbian CNN) — **this run** | **33%** |
| Single-layer Hebbian (Phase 0, 88% MNIST) | — |
| Float Hebbian (same arch, estimate) | ~75% |
| SoftHebb (Journé et al., 2023) | 80.3% |
| Backprop (same arch, float) | ~88% |
| Random ternary weights | ~10% |

---

## Lessons Learned

### What worked well
- **`TernaryHebbianConv2d` implementation** — F.unfold + F.linear forward is efficient and clean
- **Direct score updates** for conv layers (bypassing `hebbian_update`'s N*L division) — the competitive variant needed this to get any learning signal
- **Tests pass** — all 132 tests pass with no regressions
- **Architecture works** — the full CNN pipeline runs end-to-end on RTX 4060 in ~31min

### What failed or was surprising
- **Competitive Hebbian (per-position WTA) doesn't improve over random** — matches Phase 1.1 MLP finding
- **Class-guided Hebbian kills sparsity** — dense filters lose discriminative power
- **Anti-Hebbian for losers dominates** in multi-filter conv layers — 1 winner vs 63 losers means even -0.1× anti-Hebbian overwhelms the signal
- **Hebbian update divided by N*L** makes per-weight updates microscopically small for convs — required bypassing the standard `hebbian_update` method
- **The conv bottleneck**: 64 filters × 3×3×3 = 1,728 weights for conv1 — very few parameters to learn meaningful features for 10-class CIFAR-10

### Comparison to hypothesis
- **Disconfirmed**: Hebbian conv layers do not improve CIFAR-10 accuracy beyond random projections
- **Confirmed**: Hebbian learning rules can drive weight changes in conv layers (flips observed)
- **Confirmed**: The ternary Hebbian learning infrastructure works for CNNs

### Path Forward
The primary value of PH-Neuro is not raw accuracy but **continual learning**. Phase 1.3 will test whether the ternary Hebbian CNN, even at 33% accuracy, is significantly more resistant to catastrophic forgetting than a backprop baseline. If forgetting drops from >60% (backprop) to <5% (Hebbian), the trade-off is worth it.

---

## Run Command

```bash
# Default (class-guided conv training)
python -m ph_neuro.examples.cifar10_cnn

# Competitive (unsupervised, per-position WTA)
python -m ph_neuro.examples.cifar10_cnn --competitive

# With custom params
python -m ph_neuro.examples.cifar10_cnn \
    --conv-epochs 5 5 --output-epochs 10 \
    --lr 0.01 0.01 0.01 --batch-size 128
```

---

## Files Modified/Created

| File | Change |
|------|--------|
| `src/ph_neuro/layers/conv.py` | Implemented `TernaryHebbianConv2d` |
| `src/ph_neuro/models/cnn.py` | Implemented `HebbianCNN` |
| `src/ph_neuro/examples/cifar10_cnn.py` | Implemented experiment script |
| `src/ph_neuro/training/greedy.py` | Added CNN training functions |
| `tests/layers/test_conv.py` | 29 unit tests for conv layer |
| `tests/integration/test_phase1_cnn.py` | 10 integration tests |

---

## Next Steps

1. [x] Implement TernaryHebbianConv2d layer
2. [x] Implement HebbianCNN model
3. [x] Add conv training functions (competitive + class-guided)
4. [x] Write tests (29 unit + 10 integration, all passing)
5. [x] Run CIFAR-10 experiment — 33% (random baseline), 32.6% (competitive), 21.9% (class-guided)
6. [ ] **Proceed to Phase 1.3 — Continual Learning** (the primary contribution)
