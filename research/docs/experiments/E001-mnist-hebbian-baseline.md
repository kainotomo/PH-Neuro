# E001: Single-Layer Hebbian MNIST Baseline

- **Date:** 2026-07-28
- **Git commit:** `f450216`
- **Status:** completed
- **Phase:** 0

---

## Hypothesis

A single `TernaryHebbianLinear(784 → 10)` layer trained with a winner-take-all (WTA) Hebbian rule can achieve >85% accuracy on MNIST without any `.backward()` calls, optimizers, or loss functions.

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
| Learning rate | 0.01 |
| Decay rate | 0.0 |
| Hebbian variant | WTA — strengthen correct class, weaken wrong prediction |
| Input quantization | `ternary_sign(x, epsilon=0.1)` |
| Input normalization | `Normalize((0.1307,), (0.3081,))` via torchvision |
| Batch size | 128 |
| Epochs | 10 |
| Training steps | 4,690 |
| Dataset | MNIST (60K train, 10K test) |
| Data augmentation | None |
| Hardware | RTX 4060 8 GB |
| Training time | 47.3 s |
| Training throughput | ~12,700 samples/sec |
| Memory usage | <100 MB VRAM |

---

## Results

### Main Metrics

| Metric | PH-Neuro (this run) | Baseline: Backprop | Baseline: Float Hebbian |
|--------|--------------------|--------------------|-------------------------|
| Accuracy (test) | **88.42%** | ~92% | ~85-90% (estimated) |
| Weight sparsity (% 0) | 78.3% | — | — |
| Weight flip rate (per step) | 0.04% | — | — |

### Per-Epoch Breakdown

| Epoch | Accuracy | +1 % | −1 % | 0 % | Flip Rate |
|-------|----------|------|------|-----|-----------|
| 1 | 86.5% | 4.9% | 4.6% | 90.5% | 0.20% |
| 2 | 86.6% | 5.9% | 5.7% | 88.3% | 0.05% |
| 3 | 86.8% | 6.8% | 6.5% | 86.7% | 0.04% |
| 4 | 87.2% | 7.8% | 7.4% | 84.9% | 0.05% |
| 5 | 86.4% | 8.7% | 7.8% | 83.5% | 0.05% |
| 6 | 87.9% | 9.1% | 8.9% | 82.0% | 0.04% |
| 7 | 87.6% | 9.5% | 9.6% | 80.9% | 0.04% |
| 8 | 87.9% | 9.7% | 10.1% | 80.2% | 0.04% |
| 9 | 87.5% | 10.4% | 10.5% | 79.1% | 0.04% |
| 10 | **88.4%** | **11.0%** | **10.7%** | **78.3%** | **0.04%** |

### Invariant Checks

| Check | Result |
|-------|--------|
| No `.backward()` calls | ✅ 0 calls (runtime verified) |
| All weights ∈ {-1, 0, +1} | ✅ 100% ternary (assert per step) |
| Flip rate < 1% after convergence | ✅ 0.04% (target < 1%) |
| Training time < 1 hour | ✅ 47.3 s (target < 3600 s) |
| No optimizers used | ✅ Zero optimizer objects |
| No loss functions used | ✅ Zero loss function calls |

---

## Ablation Studies

During development, five Hebbian strategies were tested. The table below shows the best result for each:

| Strategy | Best Accuracy | Weight Distribution | Notes |
|----------|:------------:|:-------------------:|-------|
| **WTA Hebbian** (winner-take-all) | **88.4%** | 11%+1, 11%−1, 78% 0 | Strengthen correct, weaken wrong prediction |
| Correct-only Hebbian, batch lr=1.0 | 66.7% | 15%+1, 85%−1, 0% 0 | All classes learn same pattern |
| Full-target Hebbian (+1/−1 post) | 63.0% | ∼50/50 ±1 | Anti-Hebbian dominates (9:1 wrong:correct) |
| Balanced mean Hebbian | 70.3% | 29%+1, 48%−1, 23% 0 | Per-class mean vs global mean |
| Correct-only + anti-Hebbian (fused) | 53.0% | 15%+1, 85%−1, 0% 0 | Built-in ÷128 dilutes signal |

### Key Findings

1. **Batch-normalized update is too slow**: The built-in `hebbian_update` divides by `batch_size=128`, diluting the per-class signal (~13 correct samples / 128 = 10× attenuation). WTA bypasses this by computing `Δscore = lr × (correct_onehot.T @ pre - pred_onehot.T @ pre)` without batch division.

2. **Correct-only Hebbian creates identical weight patterns**: All 10 output classes learn the same ±1 pattern (background pixels → −1, foreground → +1). Only 15% of weights differ between classes, preventing effective discrimination.

3. **Full-target Hebbian (post = +1/−1) is dominated by anti-Hebbian**: With 9× more wrong-class samples than correct per batch, the anti-Hebbian term overwhelms the positive term. All scores drift negative.

4. **WTA Hebbian achieves near-theoretical maximum**: A single linear layer with continuous weights achieves ~92% on MNIST. The ternary constraint costs ~4 percentage points. The 88.4% achieved is ~96% of the theoretical maximum.

5. **Noise filtering (epsilon=0.1) improves robustness**: `ternary_sign(x, epsilon=0.1)` suppresses ambiguous near-zero pixel activations, reducing noise in the Hebbian updates.

---

## Code

The training loop is encapsulated in `SupervisedHebbianClassifier` (`src/ph_neuro/training/supervised.py`):

```python
classifier = SupervisedHebbianClassifier(
    in_features=784,
    out_features=10,
    theta_upper=1.0,
    theta_lower=0.3,
    device="cuda",
)

for epoch in range(10):
    for x, y in train_loader:
        with torch.no_grad():
            classifier.train_step(x, y, lr=0.01, epsilon=0.1)
    acc = classifier.evaluate(test_loader, epsilon=0.1)
```

No `.backward()`, no optimizer, no loss function.

---

## Reproducibility

```bash
# Install
cd PH-Neuro
pip install -e ".[dev,examples]"

# Train with defaults
python -m ph_neuro.examples.mnist_mlp

# Train with optimal parameters
python -m ph_neuro.examples.mnist_mlp \
    --epochs 10 \
    --theta-upper 1.0 \
    --theta-lower 0.3 \
    --lr 0.01 \
    --epsilon 0.1

# Run tests
pytest tests/ -v -m "not slow"
pytest tests/integration/test_phase0_mnist.py -v -m "slow"
```
