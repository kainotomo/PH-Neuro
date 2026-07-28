# Phase 2 — Multi-Layer & Hierarchical Representations

> **Goal:** Understand how Hebbian learning behaves in deep networks. Explore alternative multi-layer strategies.  
> **Duration:** ~3-4 weeks  
> **Hardware:** RTX 4060 8 GB — easy  
> **Success:** 5+ layer Hebbian CNN >65% CIFAR-10, feature hierarchy visualizable

---

## Overview

Phase 2 digs deeper into the question: "What happens when we stack many Hebbian layers?" 

Greedy layer-wise training (Phase 1) is simple but potentially limiting — each layer only sees the previous layer's output, never gets any "top-down" signal. This phase explores whether that's a fundamental limitation or if Hebbian self-organization is sufficient for deep hierarchies.

---

## 2.1 Deep Networks — Does Depth Help?

### Experiment: Depth Scaling on CIFAR-10

Train identical architectures with varying depth, all Hebbian:

| Layers | Architecture | Params |
|--------|-------------|--------|
| 1 | Conv(3→128) → FC(128→10) | ~200K |
| 2 | Conv(3→64) → Conv(64→128) → FC(128→10) | ~300K |
| 3 | Conv(3→64) → Conv(64→128) → Conv(128→256) → FC(256→10) | ~500K |
| 5 | Conv(3→32) → Conv(32→64) → Conv(64→128) → Conv(128→256) → Conv(256→10) → FC(256→10) | ~600K |
| 10 | [Ten conv layers with skip connections] → FC(256→10) | ~2M |

### Hypothesis

Backprop benefits greatly from depth because the global error signal can coordinate layers. Hebbian learning may benefit less because:
1. Each layer is independent — no coordination between layers
2. Information loss at each ternary activation (sign function drops magnitude)
3. Later layers can't "ask" earlier layers for better features

### Expected Outcome

Accuracy should improve from 1→3 layers, then plateau or decline. Finding the plateau point tells us the "effective depth" of Hebbian learning.

### Ablation: Skip Connections

Do residual connections help Hebbian learning? They preserve information across layers and may help with the ternary information bottleneck.

```python
class TernaryHebbianResBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = ternary_sign(out)
        out = self.conv2(out)
        return ternary_sign(out + residual)  # Ternary addition
```

---

## 2.2 Feature Visualization

### What Does Each Layer Learn?

For each conv layer, visualize:
1. **Weight filters**: Reshape each filter to an image. Are they Gabor-like (edges, orientations)?
2. **Activation maximization**: Find input images that maximize specific neuron activations
3. **Receptive fields**: What part of the input image affects each neuron?

### Visualization Tools

```python
def visualize_filters(layer, n_cols=8):
    """Show first N filters as grayscale images."""
    weights = layer.weights.unpack()  # (out_ch, in_ch, h, w)
    # For ternary, show +1 as white, -1 as black, 0 as gray
    ...

def activation_maximization(model, layer_idx, neuron_idx, steps=100):
    """Optimize input to maximize neuron activation."""
    # Even though we don't use backprop for learning, 
    # we can use it for visualization
    ...

def plot_filter_similarity(layer):
    """Correlation matrix between filters — do they diversify?"""
    ...
```

### Comparison: Hebbian vs Backprop Features

- Backprop features: Task-specific, discriminative (e.g., "dog ears")
- Hebbian features (expected): Generic, statistical (e.g., "edges at 45°")
- This is NOT a bug — it's why Hebbian networks transfer better between tasks!

---

## 2.3 Alternative Multi-Layer Strategies

### A. Simultaneous Hebbian (Baseline: Greedy Layer-Wise)

All layers update simultaneously. Hidden layers use their own output as "post":

```python
for x, y in train_loader:
    h = [ternary_sign(x)]
    for layer in model.hidden_layers:
        h.append(layer(h[-1]))
    
    # Update all layers simultaneously
    for i, layer in enumerate(model.hidden_layers):
        layer.hebbian_update(h[i], h[i+1], lr)
        layer.refresh_weights()
    
    # Output layer (supervised)
    target = one_hot(y) * 2 - 1
    model.output_layer.hebbian_update(h[-1], target, lr)
```

**Risk**: All layers may learn the same thing (collapse). Mitigation: add diversity regularization.

### B. Contrastive Hebbian (Forward-Forward Inspired)

Hinton's Forward-Forward algorithm uses two forward passes:
- **Positive pass**: Real data → Hebbian update ("this is good")
- **Negative pass**: Generated/junk data → Anti-Hebbian ("this is bad")

```python
# Positive pass
h_pos = model(x_real)
for layer in model.layers:
    layer.hebbian_update(h_pos[i], h_pos[i+1], lr=+lr)  # Strengthen

# Negative pass  
x_neg = generate_negative_sample(x_real)  # Shuffle pixels, mix images, etc.
h_neg = model(x_neg)
for layer in model.layers:
    layer.hebbian_update(h_neg[i], h_neg[i+1], lr=-lr)  # Weaken
```

**Layer-wise goodness**: Each layer computes a "goodness" score (e.g., sum of squared activations). The layer learns to give high goodness to real data and low goodness to negative data.

### C. Difference Target Propagation (Lightweight)

Instead of gradients, propagate target activations backward:

```python
# Forward pass: compute activations
h = model.forward(x)

# Backward pass: compute targets (not gradients!)
# Target for layer L is "what activation would have been better?"
# Can be approximated by: target_L = h_L + feedback(h_{L+1}, target_{L+1})

# Update: minimize difference between h_L and target_L
for layer in model.layers:
    delta = target_L - h_L
    layer.hebbian_update(h_{L-1}, delta, lr)  # Learn to produce better targets
```

This is still local (each layer only uses its own input and a target), but provides a "teaching signal" that greedy layer-wise lacks.

### Experiment Plan

| Strategy | MNIST Test | CIFAR-10 Test | Notes |
|----------|-----------|---------------|-------|
| Greedy layer-wise (baseline) | ✓ | ✓ | From Phase 1 |
| Simultaneous Hebbian | ✓ | ✓ | Risk of collapse |
| Contrastive Hebbian | ✓ | ✓ | Most promising alternative |
| Difference Target Prop | ✓ | — | Complex, try on MNIST first |

---

## 2.4 Continual Learning at Depth

### Does Depth Help or Hurt?

**Hypothesis**: Deeper networks may have MORE forgetting because:
- Early layers drift → affects ALL subsequent layers
- More layers = more places for interference
- Ternary constraint at each layer compounds errors

**Counter-hypothesis**: Deeper networks may have LESS forgetting because:
- Early layers learn generic features that transfer across tasks
- More capacity = less interference
- Late layers can specialize per task while early layers stay stable

### Experiment

Train 1-layer, 3-layer, and 5-layer networks on split MNIST. Measure forgetting.

```python
for depth in [1, 3, 5]:
    model = make_hebbian_mlp(depth, hidden_dim=256)
    results, forgetting = evaluate_continual_learning(model, split_mnist_tasks)
    print(f"Depth {depth}: avg_acc={results.mean():.1%}, forgetting={forgetting:.1%}")
```

### Layer Freezing Strategy

For continual learning, maybe we should freeze early layers after initial training:

```python
# After task 1: freeze layer 1
# After task 2: freeze layer 2
# ...
# This "crystallizes" knowledge in early layers
```

Compare: freeze-all vs freeze-nothing vs progressive-freeze.

---

## 2.5 Weight Dynamics Deep Dive

### Synapse Lifetime Tracking

How long does a ternary weight persist before flipping?

```python
class SynapseTracker:
    def __init__(self, layer):
        self.birth_step = {}  # (i, j) → step when weight became ±1
        self.death_step = {}  # (i, j) → step when weight became 0
        self.lifetimes = []   # death_step - birth_step
    
    def update(self, old_weights, new_weights, step):
        # New activations (0 → ±1)
        activated = (old_weights == 0) & (new_weights != 0)
        for idx in activated.nonzero():
            self.birth_step[tuple(idx)] = step
        
        # Deactivations (±1 → 0)
        deactivated = (old_weights != 0) & (new_weights == 0)
        for idx in deactivated.nonzero():
            if tuple(idx) in self.birth_step:
                lifetime = step - self.birth_step[tuple(idx)]
                self.lifetimes.append(lifetime)
        
        # Flips (+1 → -1 or vice versa)
        flipped = (old_weights != 0) & (new_weights != 0) & (old_weights != new_weights)
        # These are "resets" — treat as death + rebirth
```

### Critical Period Analysis

Do early training steps have disproportionate impact?

- Freeze weights after K steps, continue training only on output layer
- Vary K from 10 to 1000
- Does performance saturate quickly? (Yes → early steps matter most)

### Weight Entropy

How diverse are the learned weight patterns?

```python
def weight_entropy(weights):
    """Entropy of weight distribution. Higher = more diverse."""
    pos = (weights == 1).float().mean()
    neg = (weights == -1).float().mean()
    zero = (weights == 0).float().mean()
    probs = torch.tensor([pos, neg, zero])
    return -(probs * torch.log(probs + 1e-8)).sum()
```

---

## Deliverables

- [ ] Depth scaling experiment (1-10 layers on CIFAR-10)
- [ ] Feature visualization tools + analysis
- [ ] Comparison table: greedy vs simultaneous vs contrastive Hebbian
- [ ] Continual learning at depth experiment
- [ ] Synapse lifetime analysis
- [ ] Ablation: skip connections, normalization, activation functions

---

## What's Next

After Phase 2 → Phase 3: First language model. Take what we've learned about multi-layer Hebbian dynamics and apply it to Transformers on TinyStories.
