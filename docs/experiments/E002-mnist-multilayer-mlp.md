# E002: Multi-Layer Hebbian MLP on MNIST

- **Date:** 2026-07-28
- **Git commit:** TBD
- **Status:** completed
- **Phase:** 1

---

## Hypothesis

Greedy layer-wise Hebbian learning with 2+ hidden layers can exceed 95% MNIST accuracy, surpassing the single-layer theoretical maximum of ~92% for a linear model. The hidden layers build hierarchical representations that improve class separability beyond what a single layer can achieve.

---

## Key Finding

**Online competitive Hebbian (winner-take-all + conscience)** is the correct self-organizing rule for ternary hidden layers. Basic Hebbian (`ΔW = lr × postᵀ @ pre`) makes all hidden neurons learn the same pattern (useless). Oja's rule creates balanced 50/50 weights but only random projections (~60% accuracy).

Online competitive Hebbian creates **sparse prototypes**: each hidden neuron learns a distinct input pattern, with ~10% active weights. This matches the single-layer baseline (87.9% vs 87.5%) with an unsupervised hidden layer.

**Depth does not yet provide meaningful improvement** beyond the single-layer baseline with the current competitive rule. The 2-layer matches 1-layer accuracy, suggesting the prototypes capture similar information to the direct WTA weights.

---

## Configuration

### Architecture

```
Input (784) -> TernaryHebbianLinear(784 -> 512) -> TernaryHebbianLinear(512 -> 10)
```

Training: greedy layer-wise with **online competitive Hebbian** for hidden layer (unsupervised, WTA + conscience).

| Parameter | Best Value Found |
|-----------|-----------------|
| Architecture | 784 -> 512 -> 10 (2-layer) |
| Total parameters | ~407,048 ternary weights |
| Hidden rule | Online competitive (WTA + conscience) |
| Output rule | Supervised WTA |
| Hidden LR | 0.01 |
| Output LR | 0.005 |
| Hidden epochs | 3 (online, per-sample) |
| Output epochs | 15 |
| θ_upper (output) | 1.0 |
| θ_lower (output) | 0.3 |
| Dead-zone (epsilon) | 0.1 |
| Batch size | 128 |
| Dataset | MNIST (60K train, 10K test) |
| Hardware | RTX 4060 8 GB |
| Training time | ~10 min (2-layer, online per-sample) — |

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

### Main Metrics

| Metric | PH-Neuro (2-layer) | Baseline: Single-layer | Baseline: Backprop |
|--------|:------------------:|:---------------------:|:------------------:|
| Accuracy (test) | **87.9%** | 87.5% | ~98% |
| Hidden weight sparsity (% 0) | 89.8% | — | — |
| Output flip rate (converged) | 0.032%/step | 0.044%/step | — |

### Depth Ablation

| Depth | Architecture | Accuracy | Improvement |
|:-----|:-------------|:--------:|:----------:|
| 1-layer | 784 -> 10 (WTA) | **87.5%** | baseline |
| 2-layer | 784 -> 512 -> 10 (competitive + WTA) | **87.9%** | +0.4% |
| 3-layer | 784 -> 512 -> 256 -> 10 | *pending* | — |

### Ablation: Hidden Layer Hebbian Variants

| Variant | 2-layer Accuracy | Problem |
|---------|:---------------:|---------|
| Basic Hebbian | ~10% | All weights same sign, zero information |
| Oja's rule | ~60% | Balanced 50/50 weights, random projections |
| BCM rule | ~10% | Sliding threshold collapses with ternary |
| Competitive top-k | ~10% | Anti-Hebbian kills all neurons |
| Class-guided | ~10% | 87%+ positive weights, all output -1 |
| Reward-modulated | ~10% | Near-zero mean reward |
| **Online competitive (WTA+conscience)** | **87.9%** | Sparse prototypes, ~10% active weights |

### Key Insight

Online competitive Hebbian with conscience works because it creates **sparse, differentiated prototypes** — each hidden neuron learns a distinct input pattern. This is analogous to how the brain's cortex uses competitive learning to develop specialized feature detectors. The conscience mechanism (penalizing over-frequent winners) ensures full codebook utilization.

The 2-layer matches the single-layer baseline because both learn similar information: the 784→512 competitive layer learns 512 prototypes covering the MNIST manifold, and the 512→10 WTA layer maps these prototypes to classes. With 512 prototypes and 10 classes, each class has ~51 prototypes — enough to cover intra-class variation.

---

## Observations

### What worked well?
- **Online competitive Hebbian** — the only approach that creates useful hidden representations
- **Conscience mechanism** — prevents prototype collapse (all neurons winning equally)
- **Sparse prototypes** (~10% active weights) — each neuron learns a distinct pattern
- **Per-sample online update** — aligns with the brain-like vision

### What failed or was surprising?
- **Basic Hebbian** (`ΔW = lr × postᵀ @ pre`) makes all hidden neurons identical — fundamental limitation
- **Oja's rule** creates balanced weights but no class signal — caps at ~60%
- **Class-guided methods** paradoxically make all weights positive (background pixels dominate)
- **95% target not reached** — depth provides no improvement over single layer with 512 prototypes
- The 2-layer accuracy (87.9%) closely matches single-layer (87.5%) — neither exceeds the ~92% theoretical max for linear MNIST features

### Comparison to hypothesis
- **Partial confirmation**: Depth provides a very small improvement (87.5% → 87.9%) but far from the 95% target
- The 3-layer result will confirm whether deeper stacking helps

---

## Code

```bash
# 2-layer online competitive (best config)
python -m ph_neuro.examples.mnist_multilayer \
    --n-layers 2 --hidden-sizes 512 \
    --hebbian-rules online_competitive \
    --lr 0.01 0.005 \
    --epochs 3 15 \
    --theta-upper 1.0

# Single-layer baseline
python -m ph_neuro.examples.mnist_multilayer --single-layer

# Ablation sweep
python -m ph_neuro.examples.ablation_multilayer
```

---

## Next Steps

1. [x] Run 2-layer default config — 87.9% (matches single-layer)
2. [x] Run ablation: depth (1L vs 2L) — depth helps slightly (+0.4%)
3. [x] Run ablation: Hebbian variants for hidden layers — online competitive wins
4. [/] Run 3-layer experiment — *pending completion*
