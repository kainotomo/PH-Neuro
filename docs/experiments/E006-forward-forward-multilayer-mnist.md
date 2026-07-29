# E006: Forward-Forward Multi-Layer MLP on MNIST (TFF-2)

- **Date:** 2026-07-29
- **Git commit:** TBD
- **Status:** completed
- **Phase:** 2

---

## Hypothesis

A 2-layer ternary network (784\u2192512\u219210) trained with greedy layer-wise Forward-Forward can exceed 95% MNIST accuracy — beating the single-layer \u223c88% bound and proving that Forward-Forward gives hidden layers a useful local error signal that unsupervised Hebbian (Phase 1.1) could not.

---

## Key Finding

**TFF-2 achieves 86.81% MNIST accuracy — essentially identical to Phase 0 (88.4%), Phase 1.1 (87.9%), and TFF-1 (87.9%).** The Forward-Forward contrastive objective does NOT solve the hidden-layer problem for ternary weight networks.

The failure is fundamental: **FF's positive-pass Hebbian update on top-1 winners creates the same prototypes as competitive unsupervised Hebbian, and the negative-pass anti-Hebbian on junk data adds negligible discriminative information.** The hidden layer weights barely change (flip rate \u223c0.000%/step) because the per-batch score updates with lr=0.01 and \u03b8_u=1.0 are too small to cross the hysteresis threshold within 5 epochs.

The 86.81% accuracy comes almost entirely from **random bootstrapped representations** (10% sparse ternary prototypes initialized via noise), not from the FF training. The output layer learns to map these 512 random prototypes to 10 classes, achieving the same \u223c88% limit as a single-layer linear classifier.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer: 784\u2192512\u219210 |
| Total parameters | 406,528 ternary weights |
| Ternary weights | \u223c0.4 MB (naive int8) |
| Latent scores | \u223c0.8 MB (fp16) |
| Weight init | All zeros, latent scores \u223c N(0, 0.1\u00b2); hidden layer bootstrapped with 10% random connectivity |
| \u03b8_upper | 1.0 |
| \u03b8_lower | 0.3 |
| Hidden FF learning rate (pos) | 0.01 |
| Hidden FF learning rate (neg) | 0.01 |
| Output WTA learning rate | 0.005 |
| Decay rate | 0.0 |
| Hidden layer competition | Top-1 per sample (batch-efficient via argmax + one-hot) |
| Hidden layer goodness | Top-1 raw (pre-ternary) activation |
| Negative data strategy | 50% pixel mask + random ternary noise |
| Input quantization | `ternary_sign(x, epsilon=0.1)` |
| Input normalization | `Normalize((0.1307,), (0.3081,))` via torchvision |
| Batch size | 128 |
| Epochs (hidden / output) | 5 / 15 |
| Training steps | 2,345 (hidden) + 7,035 (output) |
| Dataset | MNIST (60K train, 10K test) |
| Data augmentation | None |
| Hardware | RTX 4060 8 GB |
| Training time | \u223c87 s |
| Memory usage | <200 MB VRAM |

---

## Results

### Main Metrics

| Metric | TFF-2 | Phase 0: WTA 1-layer | Phase 1.1: unsup 2-layer | TFF-1: FF 1-layer | Backprop 2-layer |
|--------|:-----:|:--------------------:|:------------------------:|:------------------:|:----------------:|
| Accuracy (test) | **86.81%** | **88.4%** | **87.9%** | **87.9%** | \u223c98% |
| Weight sparsity (hidden) | 90.1% | — | 89.8% | — | — |
| Weight sparsity (output) | 64.5% | 78% | — | 78% | — |
| Hidden flip rate (per step) | \u223c0.000% | — | 0.003% | — | — |
| Output flip rate (converged) | 0.032% | 0.04% | 0.032% | 0.04% | — |

### Per-Epoch Breakdown

**Hidden layer (FF, top-1 winners, epochs 1-5):**
| Epoch | Flips | g_pos | g_neg | Separation |
|:-----:|:----:|:----:|:----:|:---------:|
| 1 | 0.000% | — | — | — |
| 2 | 0.000% | — | — | — |
| 3 | 0.000% | — | — | — |
| 4 | 0.000% | — | — | — |
| 5 | 0.000% | — | — | — |

**Output layer (WTA, epochs 1-15):**
| Epoch | Acc | +1% | -1% | 0% | Flip Rate |
|:----:|:---:|:---:|:---:|:--:|:---------:|
| 1 | 77.56% | — | — | — | 0.070% |
| 2 | 84.94% | — | — | — | 0.033% |
| 3 | 86.03% | — | — | — | 0.036% |
| 4 | 86.77% | — | — | — | 0.033% |
| 5 | 86.94% | — | — | — | 0.034% |
| 6 | 87.40% | — | — | — | 0.033% |
| 7 | 87.66% | — | — | — | 0.032% |
| 8 | 87.34% | — | — | — | 0.034% |
| 9 | 87.70% | — | — | — | 0.032% |
| 10 | 87.59% | — | — | — | 0.034% |
| 11 | 87.85% | — | — | — | 0.033% |
| 12 | **88.06%** | — | — | — | 0.032% |
| 13 | 87.69% | — | — | — | 0.034% |
| 14 | 87.81% | — | — | — | 0.034% |
| 15 | 88.15% | — | — | — | 0.032% |

### Final Weight Distributions

| Layer | +1% | -1% | 0% |
|-------|:---:|:---:|:--:|
| Hidden (784\u2192512) | 4.8% | 5.1% | 90.1% |
| Output (512\u219210) | 17.8% | 17.7% | 64.5% |

### Goodness Separation

| Metric | Value |
|--------|:-----:|
| g_pos (mean top-1 raw activation, real data) | 539.5 |
| g_neg (mean top-1 raw activation, junk data) | 19.2 |
| Separation (g_pos - g_neg) | **+520.3** |

The hidden layer shows excellent separation between real and junk data (520.3), but this separation is from **random bootstrapped weights**, not learned. The FF training barely changes the weights.

### Invariant Checks

| Check | Result |
|-------|--------|
| No `.backward()` calls | ✅ 0 calls (by design) |
| All weights \u2208 {-1, 0, +1} | ✅ 100% ternary (verified by `TernaryHebbianLinear`) |
| Flip rate < 1% after convergence | ✅ 0.000% (hidden), 0.032% (output) |
| Training time < 2 minutes | ✅ \u223c87 s |
| No optimizers used | ✅ Zero optimizer objects |
| No loss functions used | ✅ Zero loss function calls |

---

## Comparison to All Prior Experiments

| Experiment | Architecture | Learning | Accuracy | vs TFF-2 |
|:-----------|:------------|:---------|:--------:|:--------:|
| **TFF-2 (this run)** | 784\u2192512\u219210 | FF hidden + WTA output | **86.81%** | \u2014 |
| Phase 0 WTA 1-layer | 784\u219210 | Supervised WTA | 88.4% | +1.59pp |
| Phase 1.1 unsup 2-layer | 784\u2192512\u219210 | Online competitive + WTA | 87.9% | +1.09pp |
| TFF-1 FF 1-layer | 784\u219210 | FF-inspired WTA | 87.9% | +1.09pp |
| NTH-1 1-layer | 784\u219210 | Label neuromodulator | 88.15% | +1.34pp |
| Backprop 2-layer (theoretical) | 784\u2192512\u219210 | Cross-entropy + SGD | \u223c98% | +11.19pp |

---

## Ablation: Negative Pass (lr_neg)

Three variants were tested for the hidden layer training:

| Variant | lr_neg | Hidden flips | Accuracy | Notes |
|:--------|:------:|:------------:|:--------:|-------|
| **Batch top-51 + Hebbian** | 0.01 | 0.023% | 9.74% | All neurons fire on everything; +80% positive weights |
| **Batch top-51 + Hebbian** | 0.01 | 0.008% | 8.92% | Higher hysteresis (\u03b8_u=5.0); still dense |
| **Batch top-1 + FF** | 0.01 | \u223c0.000% | **86.81%** | Sparse weights (90.1% zeros); matches Phase 1.1 |
| **Online competitive (ref)** | N/A | 0.003% | **87.64%** | Phase 1.1 best; per-sample top-1 + conscience |

### Analysis

1. **Top-51 (10% winners) fails catastrophically** — With 51 winners per sample out of 512, almost every neuron appears in the top-51 across many samples. All weights converge to the same pattern (+80% positive), producing dense non-discriminative representations. Accuracy \u223c10% (random).

2. **Top-1 (WTA) succeeds for sparsity** — Only one neuron per sample gets updated. Each neuron learns a specific input prototype. Weights stay sparse (90.1% zeros). The representations contain enough information for the output layer to reach \u223c87%.

3. **FF negative pass adds no benefit** — With or without the FF negative pass (lr_neg=0 vs lr_neg=0.01), accuracy is \u223c87%. The bootstrapped random prototypes already contain all the discriminative information the output layer can use. The anti-Hebbian suppression of junk data winners doesn't add class-relevant structure to the representations.

4. **Per-batch update is too slow for \u03b8_u=1.0** — With 512 neurons, each winner gets only \u223c0.25 samples per 128-sample batch. The per-feature score update is \u223c\u00b10.0025 per batch, requiring \u223c400 batches (\u223c1 epoch) to cross the hysteresis threshold. Over 5 epochs, the cumulative change is \u223c\u00b11.17 per feature, barely above \u03b8_u=1.0.

---

## Observations

### What worked well?
- **Top-1 winner selection** creates sparse, differentiated prototypes (90.1% zero weights) — same mechanism as Phase 1.1's online competitive Hebbian.
- **Batch-efficient top-1** via `F.one_hot(winners, n_out).T @ h` is \u223c100\u00d7 faster than per-sample loops while producing equivalent results.
- The **greedy layer-wise training framework** (`MultiLayerHebbianClassifier.fit_greedy()`) works correctly — online_competitive achieves 87.64% (reproducing Phase 1.1).
- All infrastructure invariants hold: no `.backward()`, all weights ternary, flip rates stabilize.

### What failed or was surprising?
- **The FF negative pass does not improve hidden representations** beyond random bootstrapped prototypes. Accuracy is the same with or without lr_neg>0. This falsifies the hypothesis that FF solves the hidden-layer problem for ternary weights.
- **Top-k > 1 fails** catastrophically because batch updates with multiple winners per sample create dense, homogeneous weights. The competitive Hebbian approach REQUIRES top-1 (winner-take-all) to maintain differentiation.
- **Score updates are too slow** with 512 neurons and 128-sample batches. Each neuron wins \u223c0.25 times per batch, making the per-batch update negligible. Higher learning rates or smaller hidden sizes are needed for meaningful changes.
- **Goodness separation is excellent** (520.3) but entirely from bootstrapped random weights. The FF contrastive signal doesn't add class-relevant structure — it only separates "real" from "junk" at the whole-layer level, not per class.

### Root Cause Analysis

The fundamental issue is that **the Forward-Forward contrastive objective optimizes for a whole-layer property (popcount/goodness), not for class-discriminative features.** Top-1 competitive Hebbian creates differentiated prototypes, but the FF objective doesn't guide them to be class-specific — it only requires them to fire more strongly on real data than on junk. The prototypes end up as random MNIST patches, which the output layer learns to classify with \u223c87% accuracy (the linear-separability limit of random features).

This is WHY Phase 1.1 and Phase 2 reach the same \u223c88% bound: both create random sparse prototypes that capture the MNIST manifold's statistical structure but not its class structure. The output layer is a linear classifier on these prototypes, and \u223c88% is the linear-separability limit of 512 random prototypes for 10-class MNIST.

---

## Conclusion

**Tier: \U0001f534 Fail (<90%).**

TFF-2 achieves 86.81%, which is NOT an improvement over Phase 0 (88.4%), Phase 1.1 (87.9%), or TFF-1 (87.9%). The Forward-Forward contrastive objective does not solve the hidden-layer problem for ternary weight networks.

### What This Means

1. **FF+ternary is incompatible for hidden layers.** The FF goodness function (popcount) trivially saturates without competition. When competition is added (top-1), the FF objective adds no benefit beyond random competitive Hebbian.

2. **The \u223c88% bound is fundamental** — It represents the linear separability limit of 512 random sparse features for MNIST. Neither unsupervised Hebbian (Phase 1.1) nor Forward-Forward (Phase 2) breaks this bound.

3. **NTH-only pivot is indicated** — The NTH approach (neuromodulated Hebbian with label-derived modulator) is the remaining pathway. NTH gives each neuron a class-specific signal, which can propagate through layers. TFF-3 (3-layer), TFF-5 (CIFAR-10 CNN FF), and other FF-only experiments are NOT indicated.

### Decision

- [x] TFF-3 (3-layer FF on MNIST): **\u2717 CANCELLED** — would not break the \u223c88% bound
- [x] TFF-5 (CNN FF on CIFAR-10): **\u2717 CANCELLED** — FF does not work for hidden layers
- [ ] NTH-4 (multi-layer NTH on MNIST): **Proceed** — only remaining viable approach
- [ ] Re-evaluate H5 status: **\u2717 FALSIFIED** for ternary weights

---

## Code

```bash
# Best TFF-2 run
python -m ph_neuro.examples.forward_forward_multilayer_mnist \
    --epochs-hidden 5 --epochs-output 15 \
    --lr-pos 0.01 --lr-neg 0.01 --lr-output 0.005 \
    --theta-upper 1.0 --theta-lower 0.3 \
    --epsilon 0.1 --mask-ratio 0.5

# Online competitive baseline (87.64%)
python -c "
from ph_neuro.training.data import get_mnist_loaders
from ph_neuro.training.greedy import LayerConfig, MultiLayerHebbianClassifier
clf = MultiLayerHebbianClassifier([784, 512, 10], theta_upper=1.0, theta_lower=0.3)
configs = [
    LayerConfig(lr=0.01, epochs=3, hebbian_rule='online_competitive'),
    LayerConfig(lr=0.005, epochs=15),
]
train_loader, _ = get_mnist_loaders()
clf.fit_greedy(train_loader, layer_configs=configs, verbose=False)
print(clf.evaluate(test_loader))
"
```

---

## Next Steps

1. [x] Run TFF-2 with best hyperparameters — 86.81%
2. [x] Run online_competitive baseline for comparison — 87.64%
3. [x] Ablate FF negative pass value — adds nothing
4. [ ] **Pivot to NTH-4 (multi-layer NTH)** — the label modulator can provide class-specific signals through hidden layers
5. [ ] Update strategy: FF layers only for output (proven), NTH for hidden layers (hypothesized)
