# Phase 1 — Vision Proof-of-Concept (Status: In Progress)

> **Goal:** Show ternary Hebbian learning works on real vision tasks. Demonstrate continual learning.  
> **Duration:** ~2-3 weeks  
> **Hardware:** RTX 4060 8 GB — easy  
> **Success:** CNN >55% CIFAR-10, <5% forgetting on split MNIST
>
> **Note (2026-07-28):** Targets adjusted based on Phase 0-1.1 findings. Ternary Hebbian operates ~10-15pp below backprop in raw accuracy. The ~88-89% MNIST ceiling appears to be the practical limit for Hebbian MLPs. CNN target lowered from >60% to >55% — the real differentiator is continual learning (Phase 1.3), not raw accuracy.

---

## Overview

Phase 1 takes the core mechanism from Phase 0 and applies it to:
1. Multi-layer MLP on MNIST (>85%) — **Section 1.1 complete (87.9%)**
2. CNN on CIFAR-10 (>55%, adjusted from >60%) — **not yet started**
3. **Continual learning** (<5% forgetting) — **not yet started, primary contribution**

---

## 1.1 Multi-Layer MLP on MNIST (COMPLETE)

**Status:** Delivered. See `docs/experiments/E002-mnist-multilayer-mlp.md` for full results.

**Key finding:** Basic Hebbian (`ΔW = lr × postᵀ @ pre`) does **not** work for unsupervised hidden layers — all weights collapse to the same sign. The correct rule is **online competitive Hebbian with conscience** (winner-take-all + fairness bias), which creates sparse prototypes (~10% active weights).

**Best accuracy:** 87.9% (2-layer, 784→512→10) — matches the single-layer Phase 0 baseline, but depth does not provide significant improvement beyond it.

### Architecture
```
Input (784) → TernaryHebbianLinear(784→512) → sign() 
           → TernaryHebbianLinear(512→10) → sign()
```

### Training Strategy: Greedy Layer-Wise with Online Competitive Hebbian

```python
# Step 1: Train Layer 1 (unsupervised, online competitive)
# Each hidden neuron acts as a prototype (representative pattern)
# Conscience mechanism prevents any one neuron from dominating
layer1.requires_hebbian_(True)
for epoch in range(3):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        h = x_ternary.float()
        
        # Process each sample one at a time (online / brain-like)
        for s in range(h.shape[0]):
            out = layer1(h[s:s+1])
            
            # Conscience bias: penalize over-frequent winners
            if total_steps > 0:
                freq = win_counts / total_steps
                conscience = 0.1 * (freq - 1/n_neurons)
            winner = (out - conscience).argmax()
            
            # Move winner's weights toward input
            layer1._latent_scores.scores[winner] += lr * h[s]
            win_counts[winner] += 1
            total_steps += 1
            
            layer1.refresh_weights()

# Step 2: Freeze Layer 1, Train Layer 2 (supervised WTA)
layer1.requires_hebbian_(False)
for epoch in range(epochs):
    for x, y in train_loader:
        with torch.no_grad():
            h1 = layer1(ternary_sign(x.view(-1, 784)).float())
        out = layer2(h1.float())
        pred = out.argmax(dim=1)
        
        # WTA: strengthen correct class, weaken wrong prediction
        if (pred != y).any():
            correct_hot = one_hot(y[wrong], 10)
            pred_hot = one_hot(pred[wrong], 10)
            scores += lr * (correct_hot.T @ h1[wrong] - pred_hot.T @ h1[wrong])
        
        layer2.refresh_weights()
```

### Results

| Depth | Architecture | Accuracy |
|:------|:-------------|:--------:|
| 1-layer | 784 → 10 | 87.5% |
| 2-layer | 784 → 512 → 10 | **87.9%** |
| 3-layer | 784 → 512 → 256 → 10 | *pending* |

### Key Lessons

1. **Basic Hebbian fails**: `ΔW = lr × postᵀ @ pre` makes all hidden neurons identical.
2. **Oja's rule**: Creates balanced 50/50 weights but random projections (~60% max).
3. **Online competitive + conscience**: The only approach that works. Creates sparse, differentiated prototypes.
4. **Depth doesn't improve** beyond the ~88% single-layer bound for MNIST with linear layers.
5. **The entire pipeline** (no `.backward()`, no optimizer, no loss function, ternary weights) is verified.

### Ablations

See `docs/experiments/E002-mnist-multilayer-mlp.md` for the complete ablation study across 7 Hebbian variants.

---

## 1.2 CNN on CIFAR-10 (NOT YET STARTED)
        h2 = model.layer2(h1)
        
        model.layer2.hebbian_update(h1, h2, lr=0.01)
        model.layer2.refresh_weights()

# Step 3: Train output layer (supervised)
model.layer2.requires_hebbian_(False)  # Freeze
for epoch in range(epochs):
    for x, y in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        with torch.no_grad():
            h1 = model.layer1(x_ternary)
            h2 = model.layer2(h1)
        out = model.layer3(h2)
        
        target = one_hot(y) * 2 - 1  # {-1, +1}
        model.layer3.hebbian_update(h2, target, lr=0.01)
        model.layer3.refresh_weights()
```

### Ablations
- [ ] Single layer vs 2-layer vs 3-layer — does depth help for MNIST?
- [ ] With/without anti-Hebbian on wrong output classes
- [ ] With/without homeostatic decay
- [ ] Different θ_upper / θ_lower values
- [ ] Different Hebbian variants: basic, Oja's rule, BCM

### Target
- 2-layer: >95% accuracy
- 3-layer: >96% accuracy (should match or beat single layer)

---

## 1.2 CNN on CIFAR-10

### HebbianConv2d

The Hebbian rule for convolutions is naturally local — each filter weight connects a local patch of the input to one output neuron:

```python
class TernaryHebbianConv2d(nn.Module):
    def hebbian_update(self, pre_patches, post_activation, lr):
        """
        pre_patches: (batch, out_h, out_w, kernel_h, kernel_w, in_channels)
                     — the input patch for each output position
        post: (batch, out_channels, out_h, out_w) — ternary output
        
        For each filter f at position (i,j):
            ΔW[f, :, :, :] = lr × pre_patches[:, i, j, :, :, :] × post[:, f, i, j]
        
        This is a local outer product, summed over batch and spatial positions.
        """
```

Implementation note: Use `torch.nn.functional.unfold` to extract all patches efficiently, then batch matmul.

### Architecture Options

| Option | Layers | Params | Notes |
|--------|--------|--------|-------|
| A | Conv(3→64, 3×3) → Conv(64→128, 3×3) → Linear(128×8×8→10) | ~300K | Simple baseline |
| B | Conv(3→64) → Conv(64→128) → Conv(128→256) → Linear(256×4×4→10) | ~1M | Deeper |
| C | Same as B + MaxPool between layers | ~1M | Standard CNN |

Start with Option A, iterate if needed.

### Training

```python
# Greedy layer-wise, same as MLP
# Conv1: self-organizing (unsupervised)
# Conv2: self-organizing on Conv1's output
# Output Linear: supervised Hebbian
```

### Baselines

| Method | Expected CIFAR-10 |
|--------|-------------------|
| PH-Neuro (ternary Hebbian) | >55% (target, adjusted from >60%) |
| Float Hebbian (same arch, float weights) | ~75% (estimate) |
| SoftHebb (Journé et al., 2023) | 80.3% (published) |
| Backprop (same arch, float) | ~88% |
| Backprop + STE (PH-Net style, ternary) | ~85% |
| Random ternary weights | ~10% |

### Success Interpretation

- >65%: Exceptional — ternary Hebbian approaches float Hebbian
- 55-65%: Good — ternary costs ~20pp vs float Hebbian, gains 50× memory and continual learning
- 45-55%: Acceptable — mechanism works but needs architectural improvement
- <45%: Concerning — ternary constraint is too severe for vision; revisit approach

---

## 1.3 Continual Learning — The Key Experiment

This is the experiment that makes PH-Neuro publishable.

### Split MNIST Protocol

5 binary classification tasks presented sequentially:

```
Task 1: 0 vs 1
Task 2: 2 vs 3  
Task 3: 4 vs 5
Task 4: 6 vs 7
Task 5: 8 vs 9
```

**Evaluation**: After training on task T, test on ALL previous tasks 1..T.

**Metrics**:
- **Average accuracy**: Mean accuracy across all 5 tasks after training completes
- **Forgetting**: For task i, forgetting = max_accuracy_i - final_accuracy_i
- **Average forgetting**: Mean forgetting across tasks 1-4 (task 5 has no "after")

```python
def evaluate_continual_learning(model, task_sequence):
    """
    Returns:
        accuracies: dict[task_id, list[float]] — accuracy after each training phase
        forgetting: float — average forgetting across tasks
    """
    results = {task_id: [] for task_id in task_sequence}
    
    for t, (train_loader, test_loaders) in enumerate(task_sequence):
        # Train on current task
        train_on_task(model, train_loader)
        
        # Evaluate on ALL tasks seen so far
        for task_id in task_sequence[:t+1]:
            acc = evaluate(model, test_loaders[task_id])
            results[task_id].append(acc)
    
    # Compute forgetting
    forgetting = 0
    for task_id in task_sequence[:-1]:
        max_acc = max(results[task_id])
        final_acc = results[task_id][-1]
        forgetting += (max_acc - final_acc)
    forgetting /= len(task_sequence) - 1
    
    return results, forgetting
```

### Permuted MNIST Protocol

Same as split MNIST but each task uses a random pixel permutation:

```python
def permute_mnist(x, seed):
    """Apply a fixed random permutation to pixel positions."""
    torch.manual_seed(seed)
    perm = torch.randperm(784)
    return x[:, perm]
```

5 tasks, each with a different permutation seed. Tests whether the network can learn 5 completely different mappings from the same input distribution.

### Expected Results

| Method | Split MNIST Avg Acc | Split MNIST Forgetting | Permuted MNIST Forgetting |
|--------|---------------------|----------------------|--------------------------|
| PH-Neuro (ternary Hebbian) | >90% | <5% | <10% |
| Backprop (no replay) | ~60% | >40% | >50% |
| Backprop + EWC | ~85% | ~15% | ~20% |
| Backprop + Replay (1%) | ~90% | ~10% | ~15% |

### Why This Matters

Backprop without replay suffers catastrophic forgetting because:
1. New gradients overwrite old weight configurations
2. The loss landscape for new tasks has different minima
3. There's no mechanism to "protect" important weights

PH-Neuro naturally avoids this because:
1. Hebbian updates are local — new task updates don't affect weights used by old tasks unless those neurons fire
2. Ternary weights are discrete — they don't "drift" continuously like float weights
3. Hysteresis makes activated synapses resistant to change
4. Unused synapses slowly decay but can be reactivated

---

## 1.4 Analysis & Visualization

### Weight Distribution Tracking

```python
def log_weight_stats(model, step):
    """Log per-layer weight distribution."""
    for name, layer in model.named_modules():
        if isinstance(layer, TernaryHebbianLinear):
            w = layer.weights.unpack()
            pos = (w == 1).float().mean().item()
            neg = (w == -1).float().mean().item()
            zero = (w == 0).float().mean().item()
            wandb.log({
                f"{name}/weight_pos": pos,
                f"{name}/weight_neg": neg,
                f"{name}/weight_zero": zero,
            }, step=step)
```

### Activation Sparsity

```python
def log_activation_stats(model, step):
    """What fraction of neurons output 0 for each layer?"""
    ...
```

### Forgetting Curve

Plot accuracy on task 1 as tasks 2, 3, 4, 5 are learned. The flatter the curve, the better.

### Confusion Matrix per Task

After training on all 5 split MNIST tasks, what does the confusion matrix look like? Are digits confused across tasks or within tasks?

---

## Phase 1 Conclusions (2026-07-29)

### What We Learned

1. **Unsupervised Hebbian does NOT build useful hidden representations.** Across 7 variants (basic, Oja, BCM, competitive, class-guided, reward-modulated, online competitive+conscience), only competitive Hebbian with conscience created differentiated prototypes — but even those provided ZERO accuracy improvement over the single-layer baseline (87.9% vs 87.5%).

2. **The root cause is mathematical, not architectural.** Hebbian learning optimizes $\max \text{corr(pre, post)}$ — this captures statistical structure (PCA), not discriminative structure. Each hidden layer compounds the problem: correlations of correlations diverge from class-relevant features.

3. **Depth does not help without error signals.** Both MLP and CNN experiments confirm: more layers ≠ better accuracy. The CNN conv layers match random projections (32.6% vs 33.0%).

4. **Anti-Hebbian interference = gradient interference.** Single-head continual learning fails (~37% forgetting) because weakening wrong predictions destroys old knowledge. This is a fundamental insight: the mechanism that backprop uses for discrimination is the same mechanism that causes forgetting.

5. **Multi-head continual learning works perfectly** (<5% forgetting). Separate output neurons per task is standard in the continual learning literature and validates the ternary Hebbian infrastructure.

6. **The infrastructure is solid.** 132+ tests, no `.backward()` anywhere, ternary weight invariant verified, flip rate stabilization confirmed. The problem is the learning algorithm, not the implementation.

### What This Means for the Project

The original goal of "deep Hebbian networks" with unsupervised hidden layers is **not viable.** The project pivots to:

- **Phase 2 (NEW): Forward Signals & Three-Factor Learning** — Forward-Forward (Hinton, 2022) and neuromodulated Hebbian (Frémaux & Gerstner, 2016) provide local error signals to hidden layers without backprop.
- **Phase 3: Language & Predictive Coding** — Proceeds after Phase 2 validates the error-signal mechanism.

### Key Literature

- Hinton, G. "The Forward-Forward Algorithm." arXiv:2212.13345, 2022. (98.6% MNIST without backprop)
- Frémaux, N. & Gerstner, W. "Three-Factor Learning Rules." Frontiers in Neural Circuits, 2016. (659 citations)
- Whittington, J.C.R. & Bogacz, R. "Predictive Coding ≈ Backprop." Neural Computation, 2017. (494 citations)

---

## What's Next

After Phase 1 → **Phase 2: Forward Signals & Three-Factor Learning.** Implement Forward-Forward with ternary weights and neuromodulated Hebbian to give hidden layers the local error signal they've been missing.
