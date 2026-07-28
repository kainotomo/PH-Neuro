# Phase 1 — Vision Proof-of-Concept

> **Goal:** Show ternary Hebbian learning works on real vision tasks. Demonstrate continual learning.  
> **Duration:** ~2-3 weeks  
> **Hardware:** RTX 4060 8 GB — easy  
> **Success:** CNN >60% CIFAR-10, <5% forgetting on split MNIST

---

## Overview

Phase 1 takes the core mechanism from Phase 0 and applies it to:
1. Multi-layer MLP on MNIST (>95%)
2. CNN on CIFAR-10 (>60%)
3. **Continual learning** — the core differentiator (<5% forgetting)

This phase produces the key results for the first paper.

---

## 1.1 Multi-Layer MLP on MNIST

### Architecture
```
Input (784) → TernaryHebbianLinear(784→256) → sign() 
           → TernaryHebbianLinear(256→128) → sign() 
           → TernaryHebbianLinear(128→10) → sign()
```

### Training Strategy: Greedy Layer-Wise

```python
# Step 1: Train Layer 1 (unsupervised, self-organizing)
for epoch in range(epochs):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        h1 = model.layer1(x_ternary)  # ternary output
        
        # Layer 1 learns from its own output (self-organizing)
        # This captures statistical structure in the input
        model.layer1.hebbian_update(x_ternary, h1, lr=0.01)
        model.layer1.refresh_weights()

# Step 2: Train Layer 2 on Layer 1's frozen output
model.layer1.requires_hebbian_(False)  # Freeze
for epoch in range(epochs):
    for x, _ in train_loader:
        x_ternary = ternary_sign(x.view(-1, 784))
        with torch.no_grad():
            h1 = model.layer1(x_ternary)
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
| PH-Neuro (ternary Hebbian) | >60% (target) |
| Float Hebbian (same arch, float weights) | ~75% (estimate) |
| SoftHebb (Journé et al., 2023) | 80.3% (published) |
| Backprop (same arch, float) | ~88% |
| Backprop + STE (PH-Net style, ternary) | ~85% |
| Random ternary weights | ~10% |

### Success Interpretation

- >70%: Exceptional — ternary Hebbian is near float Hebbian
- 60-70%: Good — ternary costs ~15% accuracy vs float Hebbian, gains 50× memory
- 50-60%: Acceptable — mechanism works but needs improvement
- <50%: Concerning — ternary constraint is too severe for vision

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

## Deliverables

- [ ] `TernaryHebbianConv2d` with correct Hebbian rule
- [ ] Multi-layer MNIST: >95% accuracy
- [ ] CIFAR-10 CNN: >60% accuracy
- [ ] Split MNIST continual learning: <5% forgetting
- [ ] Permuted MNIST continual learning: <10% forgetting
- [ ] Backprop baselines for all experiments
- [ ] Weight/activation analysis tools
- [ ] Experiment logs for all runs

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| CIFAR-10 <50% | Medium | Try different architectures, Hebbian variants (BCM, Oja), data augmentation |
| Forgetting >20% | Low-Medium | This would challenge H2 — investigate why weights are being overwritten |
| Greedy layer-wise doesn't scale to 3+ layers | Medium | Explore simultaneous Hebbian or contrastive approaches earlier |

---

## What's Next

After Phase 1 succeeds → Phase 2: Deep networks (5-10 layers), feature visualization, alternative multi-layer strategies.
