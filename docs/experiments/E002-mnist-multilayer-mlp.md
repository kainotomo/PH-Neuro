# E002: Multi-Layer Hebbian MLP on MNIST

- **Date:** 2026-07-28
- **Git commit:** TBD
- **Status:** planned
- **Phase:** 1

---

## Hypothesis

Greedy layer-wise Hebbian learning with 2+ hidden layers can exceed 95% MNIST accuracy, surpassing the single-layer theoretical maximum of ~92% for a linear model. The hidden layers build hierarchical representations that improve class separability beyond what a single layer can achieve.

---

## Configuration

### Architecture

```
Input (784) -> TernaryHebbianLinear(784 -> 256) -> sign()
            -> TernaryHebbianLinear(256 -> 128) -> sign()
            -> TernaryHebbianLinear(128 -> 10) -> sign()
```

| Parameter | Default | Range Explored |
|-----------|---------|----------------|
| Architecture | 784 -> 256 -> 128 -> 10 | 2-layer (784->256->10), 3-layer (default), single-layer baseline |
| Total parameters | ~221,440 ternary weights | — |
| Ternary weights | ~221 KB (naive int8) | — |
| Latent scores | ~443 KB (fp16) | — |
| Weight init | All zeros, latent scores ~ N(0, 0.1²) | — |
| θ_upper (hidden) | 5.0 | 2.0, 5.0, 10.0 |
| θ_lower (hidden) | 1.0 | 0.3, 1.0, 5.0 |
| θ_upper (output) | 1.0 | 0.5, 1.0, 3.0 |
| θ_lower (output) | 0.3 | 0.1, 0.3, 1.0 |
| LR (hidden layers) | 0.01 | 0.001, 0.005, 0.01, 0.02 |
| LR (output layer) | 0.01 | 0.005, 0.01, 0.02 |
| Decay rate | 0.0 | 0, 1e-5, 1e-4 |
| Hebbian variant (hidden) | Basic | Basic, Oja, BCM |
| Anti-Hebbian (output) | No | Yes / No |
| Input quantization | `ternary_sign(x, epsilon=0.1)` | epsilon 0.0, 0.05, 0.1 |
| Input normalization | `Normalize((0.1307,), (0.3081,))` | — |
| Batch size | 128 | — |
| Epochs per hidden layer | 5 | 3, 5, 10, 20 |
| Epochs per output layer | 10 | 5, 10, 20 |
| Dataset | MNIST (60K train, 10K test) | — |
| Data augmentation | None | — |
| Hardware | RTX 4060 8 GB | — |

---

## Training Strategy: Greedy Layer-Wise

### Step 1: Train Layer 1 (unsupervised, self-organizing)

Layer 1 (784 -> 256) learns from its own ternary output, capturing statistical structure in the input pixels:

```python
for epoch in range(epochs):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784), epsilon=0.1)
        h1 = layer1(x_ternary)                    # forward pass
        post = ternary_sign(h1, epsilon=0.1)       # quantize output
        layer1.hebbian_update(x_ternary, post, lr) # self-organizing
        layer1.refresh_weights()                   # hysteresis
```

### Step 2: Freeze Layer 1, Train Layer 2

```python
layer1.requires_hebbian_(False)
for epoch in range(epochs):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        with torch.no_grad():
            h1 = layer1(x_ternary)
        h1_ternary = ternary_sign(h1, epsilon=0.1)
        h2 = layer2(h1_ternary)
        post = ternary_sign(h2, epsilon=0.1)
        layer2.hebbian_update(h1_ternary, post, lr)
        layer2.refresh_weights()
```

### Step 3: Freeze Layer 2, Train Output Layer (supervised WTA)

Same WTA strategy from Phase 0: strengthen correct class, weaken wrong prediction.

---

## Results

*To be filled after implementation.*

### Main Metrics

| Metric | PH-Neuro (3-layer) | Baseline: Single-layer | Baseline: Backprop |
|--------|:------------------:|:---------------------:|:------------------:|
| Accuracy (test) | **>95% (target)** | 88.4% | ~98% |
| Weight sparsity (% 0) | — | 78.3% | — |

### Ablations

| Variation | Accuracy | Notes |
|-----------|:--------:|-------|
| Single-layer (784->10) | 88.4% | Phase 0 baseline |
| 2-layer (784->256->10) | — | Is 1 hidden layer enough? |
| 3-layer (784->256->128->10) | — | Full stack |
| Without anti-Hebbian | — | Only strengthen correct |
| With anti-Hebbian | — | Weaken all non-target classes |
| With decay (1e-5) | — | Homeostatic regularization |
| Oja rule (hidden) | — | Weight normalization |
| BCM rule (hidden) | — | Sliding threshold |
| θ_upper=2.0 (hidden) | — | Faster activation |
| θ_upper=10.0 (hidden) | — | Slower, more selective |

---

## Observations

*To be filled after implementation.*

### What worked well?
- 

### What failed or was surprising?
- 

### Comparison to hypothesis
- 

---

## Code

```bash
# Default 3-layer
python -m ph_neuro.examples.mnist_multilayer

# Single-layer baseline (replicates Phase 0)
python -m ph_neuro.examples.mnist_multilayer --single-layer

# 2-layer
python -m ph_neuro.examples.mnist_multilayer --n-layers 2 --hidden-sizes 256

# Oja rule for hidden layers, anti-Hebbian for output
python -m ph_neuro.examples.mnist_multilayer \
    --hebbian-rules oja oja \
    --anti-hebbian

# Ablation sweep
python -m ph_neuro.examples.ablation_multilayer
```

---

## Next Steps

1. [ ] Run 3-layer default config — does it beat 88.4%?
2. [ ] Run ablation: 2-layer vs 3-layer — does depth help?
3. [ ] Run ablation: Hebbian variants for hidden layers
4. [ ] Run ablation: anti-Hebbian on/off
5. [ ] Run ablation: decay rates
6. [ ] Run ablation: theta thresholds
7. [ ] If >95%: proceed to Phase 1.2 (CNN on CIFAR-10)
