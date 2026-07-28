# Phase 0 — Core Hebbian-Ternary Mechanism

> **Goal:** Build the fundamental `TernaryHebbianLinear` layer. Prove it works on MNIST in under an hour.  
> **Duration:** ~1 week  
> **Hardware:** RTX 4060 8 GB — trivial  
> **Success:** >90% MNIST with single layer, no `.backward()` anywhere

---

## Overview

This phase establishes the atomic unit of PH-Neuro: a linear layer that:
1. Stores weights as {-1, 0, +1} (ternary)
2. Maintains latent float scores for each weight (fp16)
3. Updates scores via Hebbian rule: `Δscore = lr × pre × post`
4. Flips ternary weights when scores cross hysteresis thresholds
5. Performs forward pass via popcount MatMul

If this doesn't work on MNIST in a single layer, nothing else matters. This is the sanity check.

---

## Deliverables

### 0.1 Ternary Tensor Representation

```python
class TernaryTensor:
    """{-1, 0, +1} weight storage — two modes:
    
    Naive (Phases 0-2): 1 byte per weight (int8). Simple, debuggable.
    Packed (Phases 3+):  4 weights per byte (2-bit encoding). Memory efficient.
    
    The API is identical — only internal storage changes.
    """
    def __init__(self, shape, packed=False):
        if packed:
            self.data: torch.Tensor  # int8, 4 weights/byte
        else:
            self.data: torch.Tensor  # int8, 1 weight/byte (naive)
    
    @staticmethod
    def pack(weights: torch.Tensor) -> 'TernaryTensor': ...
    
    def unpack(self) -> torch.Tensor:  # Returns {-1, 0, +1} as int8
        ...
    
    def to_dense(self) -> torch.Tensor:  # Returns float for MatMul
        ...
```

**Implementation notes:**
- **2-bit encoding**: 00 = 0, 01 = +1, 10 = -1, 11 = unused
- **Start naive (int8)**: 1 byte/weight for Phases 0-2 — correctness first
- **Pack later**: 4 weights/byte for Phases 3-4 — efficiency when scaling
- PyTorch doesn't have native int2 — we implement packing/unpacking ourselves
- The Hebbian update logic doesn't care about packing — it always works with unpacked {-1, 0, +1} tensors

### 0.2 Latent Score Storage

```python
class LatentScoreTensor:
    """fp16 scores paired with ternary weights."""
    def __init__(self, shape):
        self.scores: torch.Tensor  # fp16, same shape as weights
    
    def get_ternary(self, theta_upper: float, theta_lower: float) -> TernaryTensor:
        """Convert scores to ternary weights using hysteresis."""
        ...
    
    def apply_hebbian(self, pre: torch.Tensor, post: torch.Tensor, lr: float) -> None:
        """Δscore = lr × pre × post (in-place)."""
        ...
    
    def apply_decay(self, decay_rate: float) -> None:
        """Homeostatic decay: score -= decay_rate × score."""
        ...
```

### 0.3 TernaryHebbianLinear Layer

```python
class TernaryHebbianLinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 theta_upper=5.0, theta_lower=1.0, 
                 bias=False):
        self.latent = LatentScoreTensor(out_features, in_features)
        self.weights = TernaryTensor(out_features, in_features)  # all zeros initially
        self.theta_upper = theta_upper
        self.theta_lower = theta_lower
        self.has_bias = bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ternary MatMul + sign activation."""
        # 1. Quantize input to ternary
        x_ternary = ternary_sign(x)
        # 2. MatMul with ternary weights (float for debugging, popcount later)
        w_dense = self.weights.to_dense()
        out = F.linear(x_ternary, w_dense)
        # 3. Ternary activation
        return ternary_sign(out)
    
    def hebbian_update(self, pre_activation: torch.Tensor, 
                        post_activation: torch.Tensor, lr: float):
        """Apply Hebbian rule: Δscore = lr × pre × post."""
        # pre: (batch, in_features), post: (batch, out_features)
        # Outer product per sample, averaged over batch
        delta = lr * (pre_activation.T @ post_activation) / pre_activation.shape[0]
        self.latent.scores += delta
    
    def refresh_weights(self):
        """Check latent scores against thresholds, update ternary weights."""
        # For each synapse:
        #   if weight == 0 and |score| > theta_upper → flip to sign(score)
        #   if weight == ±1 and |score| < theta_lower → flip to 0
        ...
```

### 0.4 Activation Function

```python
def ternary_sign(x: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """Map to {-1, 0, +1}.
    
    With epsilon=0: sign(x) → ±1, 0 stays 0.
    With epsilon>0: values in (-epsilon, +epsilon) → 0.
    """
    if epsilon > 0:
        x = torch.where(torch.abs(x) < epsilon, torch.zeros_like(x), x)
    return torch.sign(x).to(torch.int8)
```

### 0.5 MNIST Experiment

```python
# Single layer: 784 → 10
model = TernaryHebbianLinear(784, 10, theta_upper=5.0, theta_lower=1.0)

for epoch in range(epochs):
    for x, y in train_loader:
        x = x.view(-1, 784)  # flatten
        
        # Forward pass
        x_ternary = ternary_sign(x)
        out = model(x_ternary)
        
        # Hebbian update on output layer
        target = F.one_hot(y, num_classes=10).to(torch.int8)  # {0, 1} → {0, 1}
        target = target * 2 - 1  # {0, 1} → {-1, +1}
        
        # Positive Hebbian for correct class
        correct_mask = target == 1  # shape: (batch, 10), True for correct class
        # Anti-Hebbian for wrong classes (optional, try both)
        wrong_mask = target == -1
        
        model.hebbian_update(x_ternary, target * correct_mask, lr=0.01)
        # Optional: anti-Hebbian
        # model.hebbian_update(x_ternary, target * wrong_mask, lr=-0.005)
        
        model.refresh_weights()
```

### 0.6 Tests

```python
# test_ternary_representation.py
def test_pack_unpack():
    w = torch.tensor([1, 0, -1, 1, 0, -1, 1, 0], dtype=torch.int8)
    packed = TernaryTensor.pack(w)
    unpacked = packed.unpack()
    assert torch.equal(w, unpacked)

# test_hebbian_update.py
def test_correlated_fire():
    """If pre and post are both +1, score should increase."""
    scores = LatentScoreTensor(2, 2)
    pre = torch.tensor([[1, 1]], dtype=torch.int8)
    post = torch.tensor([[1, 0]], dtype=torch.int8)
    old_score = scores.scores[0, 0].clone()
    scores.apply_hebbian(pre, post, lr=0.1)
    assert scores.scores[0, 0] > old_score  # correlated → strengthen

def test_anticorrelated_weaken():
    """If pre=+1 and post=-1, score should decrease."""
    ...

def test_silent_no_update():
    """If either pre or post is 0, no update."""
    ...

# test_hysteresis.py
def test_activation_threshold():
    """Score must exceed theta_upper to activate a synapse."""
    ...

def test_deactivation_threshold():
    """Score must fall below theta_lower to deactivate."""
    ...

def test_no_oscillation():
    """Constant input should not cause weight flips after convergence."""
    ...

# test_no_backward.py
def test_no_autograd():
    """Verify torch.autograd.backward is never called."""
    ...

# test_mnist.py
def test_mnist_minimal():
    """Single layer 784→10, 5 epochs, >90% accuracy."""
    ...
```

---

## Success Criteria

| Criterion | Target | Verification |
|-----------|--------|-------------|
| MNIST accuracy (single layer) | >90% | `test_mnist_minimal.py` |
| No `.backward()` calls | 0 calls | Runtime check |
| Weight ternary constraint | All weights ∈ {-1, 0, +1} | Assert at each step |
| Hysteresis stability | <1% weight flips/step after convergence | Log during training |
| Training time | <1 hour on RTX 4060 | Wall clock |

---

## Design Decisions

### Why fp16 latent scores?
- fp16 is half the memory of fp32
- Hebbian updates are small (±lr), fp16 has enough precision for score accumulation
- Threshold comparison doesn't need high precision
- If gradient noise becomes an issue, fall back to fp32

### Why dual threshold (θ_upper ≠ θ_lower)?
- Single threshold → weights oscillate at the boundary
- Hysteresis gap → once activated, weights are "sticky"
- Biological analogy: LTP requires strong stimulation, LTD is gradual

### Why naive int8 first (not packed 2-bit)?
- Packing 4 weights/byte adds complexity — bit shifts, masking
- Naive int8 is trivial to debug (each byte is one weight, visible in a hex dump)
- Memory difference is negligible for Phases 0-2 models (<1M params: ~1 MB vs ~0.25 MB)
- Packed storage is a pure optimization — same API, same logic, just denser bytes
- Switch to packed in Phase 3 when 100M+ models make the 4× savings worthwhile

### Why start with float MatMul (not popcount)?
- PyTorch doesn't have native popcount MatMul
- Float MatMul is fine for small models (MNIST: 784×10 = 7840 weights)
- Popcount optimization is premature at this stage
- We can verify correctness against float MatMul

### Why sign() activation (not ReLU)?
- We need {-1, 0, +1} activations for Hebbian rule to be simple
- ReLU gives {0, +} — loses inhibitory signal
- sign() preserves both excitatory (+1) and inhibitory (-1) signals
- 0 stays 0 (subthreshold activity → silent neuron)

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Hebbian rule doesn't converge | Medium | Try different lr, thresholds; add Oja's normalization |
| All weights stay at 0 | Low | Lower θ_upper, increase lr, ensure input has variance |
| All weights become ±1 (no sparsity) | Medium | Add stronger decay, raise θ_upper, use anti-Hebbian for wrong outputs |
| MNIST <90% (single layer should be easy) | Low | If this fails, the core hypothesis is wrong — revisit fundamentals |

---

## What's Next

After Phase 0 succeeds → Phase 1: Multi-layer MLP, CNN on CIFAR-10, continual learning experiments.
