# Step 0.4 — Brain Wrapper Architecture Design

> **Status:** ⬜ Not Started
> **Goal:** Design the software architecture for wrapping any HuggingFace model with local plasticity. Define the public API, the hooking mechanism, and the plasticity injection points.

---

## Design Goals

1. **Model-agnostic.** Work with any HuggingFace `AutoModelForCausalLM` (any decoder-only transformer family) without model-specific code.
2. **Non-invasive.** Don't modify the original model's source code. Use PyTorch hooks or wrapper modules.
3. **Minimal overhead.** Plastic weights are tiny (~0.1–1% of model params). Inference with plastic weights disabled should be identical to the frozen model.
4. **Clean API.** The user experience should be:
   ```python
   model = AutoModelForCausalLM.from_pretrained("<user-selected model id>")
   brain = BrainWrapper(model)
   brain.learn(texts)        # adapt plastic weights
   brain.generate(prompt)    # use model + plastic weights
   brain.save("my_brain.pt") # save plastic weights only
   ```

---

## Hooking Mechanism

### Option A: PyTorch Forward Hooks

```python
def hook_fn(module, input, output):
    # input: tuple of inputs to this module
    # output: the module's output
    # Store for plasticity update
```

**Pros:** Works with any `nn.Module`. No model modification. Standard PyTorch.

**Cons:** Hooks fire on every forward pass — need to manage state carefully. Cannot easily inject plastic weights INTO the module's computation (only observe/modify outputs).

**Verdict:** Best for capturing pre/post activations. For injecting plastic weights, we need something else.

### Option B: Module Replacement (Wrapper)

Replace each transformer block with a wrapped version:

```python
class PlasticBlock(nn.Module):
    def __init__(self, frozen_block, plastic_dim):
        self.frozen = frozen_block       # original, frozen
        self.plastic_bias = nn.Parameter(torch.zeros(plastic_dim))

    def forward(self, hidden_states, **kwargs):
        out = self.frozen(hidden_states, **kwargs)
        return out + self.plastic_bias    # inject plasticity
```

**Pros:** Full control over the forward pass. Clean integration. Plastic weights are proper `nn.Parameter`s.

**Cons:** Requires knowing the module structure. Different per model family (e.g. `transformer.h[i]` vs `model.layers[i]`). Need per-architecture adaptation.

**Verdict:** Best for controlling the forward pass. The architecture-specific mapping is manageable (we only need to support a few architectures).

### Option C: Output Modification via Hooks

Use forward hooks to modify the output of each block:

```python
def make_plastic_hook(plastic_bias):
    def hook(module, input, output):
        # output is a tuple for transformer blocks
        if isinstance(output, tuple):
            return (output[0] + plastic_bias,) + output[1:]
        return output + plastic_bias
    return hook
```

**Pros:** Model-agnostic. No module replacement needed.

**Cons:** Hooks modifying output can be fragile. The plastic bias must match the hidden dimension (architecture-specific). Less explicit than wrapper modules.

**Verdict:** Good for rapid prototyping. May be less robust for production.

### Recommendation: Option B (Module Replacement) for production, Option A (hooks) for initial exploration.

---

## Plasticity Injection Points

For a standard decoder-only transformer block:

```
Input → LayerNorm → Self-Attention → Residual → LayerNorm → MLP → Residual → Output
```

Where to inject plastic weights?

### 1. Post-Block Bias (simplest)
After the entire transformer block, add a learnable bias vector:
```
output = frozen_block(hidden_states) + plastic_bias  # shape: (batch, seq, d_model)
```
**Capacity:** d_model parameters per block. For a 12-block, d_model=768 model: 12 × 768 = 9,216 params (36 KB float32, 2.3 KB ternary).

### 2. Post-Attention Bias
After the self-attention sublayer, before the second residual and LayerNorm:
```
attn_out = frozen_attention(normed_input)
hidden_states = hidden_states + attn_out + plastic_attn_bias
```
**Capacity:** Same as post-block. 2× if we do both attention and MLP.

### 3. Post-MLP Bias
After the MLP sublayer:
```
mlp_out = frozen_mlp(normed_hidden)
hidden_states = hidden_states + mlp_out + plastic_mlp_bias
```

### 4. Key/Value Bias (attention modulation)
Add plastic bias to the key and value projections before attention:
```
key = frozen_key_proj(x) + plastic_key_bias
value = frozen_value_proj(x) + plastic_value_bias
```
This modulates WHAT the layer attends to, rather than the output.

### 5. Low-Rank Matrices (LoRA-style)
```
attn_out = frozen_attention(x) + plastic_B @ plastic_A @ x
```
Where A: (d_model, rank), B: (rank, d_model), rank ≪ d_model.
**Capacity:** 2 × d_model × rank per injection point. For d_model=768, rank=4: 2 × 768 × 4 = 6,144 params per injection.

### Injection Strategy by Phase

| Phase | Injection | Plastic Capacity (12 blocks × d_model=768) | Rationale |
|:------|:----------|:------------------------:|:----------|
| 1.1 | Post-block vector bias | 9,216 params (2.3 KB ternary) | Simplest. Test the core hypothesis. |
| 1.2 | Same as 1.1 | Same | Ablation experiments. |
| 1.3 | Same as 1.1 | Same (e.g. 30 blocks × d_model=576 = 17,280) | Architectural generalization. |
| 2.1 | Post-attention + post-MLP low-rank (rank 4) | ~300K params (75 KB ternary) | More capacity. |
| 2.2 | Same, with ternary weights | Same, but 2-bit | Memory efficiency. |
| 2.3 | Same + consolidation store | 2× (fast + slow) | Long-term memory. |

---

## Public API Design

```python
class BrainWrapper:
    """Wraps a pre-trained model with local plasticity for continual learning.

    The frozen model acts as the "born brain" — its weights never change.
    Plastic weights (tiny, ternary) are injected at each layer and updated
    via local Hebbian rules modulated by a surprise signal. No backprop
    through frozen layers.

    Args:
        model: A HuggingFace AutoModelForCausalLM (any decoder-only transformer)
        plasticity: Type of plastic weights — "vector_bias", "low_rank", or "ternary"
        capacity: Plastic capacity — for low_rank, the rank. For others, ignored.
        modulator: Surprise signal type — "prediction_error", "uncertainty", "constant"

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("<user-selected model id>")
        >>> brain = BrainWrapper(model, plasticity="vector_bias")
        >>> brain.learn(["The patient presented with acute..."])
        >>> brain.generate("The patient", max_length=50)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        plasticity: str = "vector_bias",
        capacity: int = 4,
        modulator: str = "prediction_error",
    ): ...

    def learn(
        self,
        texts: list[str],
        steps: int = 1,
    ) -> dict[str, float]:
        """Adapt plastic weights on new text using local updates.

        Returns metrics dict with loss, surprise, and weight statistics.
        """
        ...

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using frozen model + active plastic weights."""
        ...

    def without_plasticity(self) -> Iterator[None]:
        """Context manager that temporarily disables plastic weights.

        Use for baseline evaluation:
            with brain.without_plasticity():
                baseline_output = brain.generate(prompt)
        """
        ...

    def consolidate(self) -> None:
        """Sleep-like consolidation: transfer important plastic changes
        to long-term store, reset short-term plastic weights."""
        ...

    def save(self, path: str) -> None:
        """Save plastic weights only (not the frozen model)."""
        ...

    def load(self, path: str) -> None:
        """Load plastic weights."""
        ...

    def eval(self) -> None:
        """Set model to eval mode. Plastic weights frozen during eval."""
        ...
```

---

## Memory Budget

| Component | Parameter Count | Memory (float32) | Memory (ternary 2-bit) |
|:----------|:---------------:|:----------------:|:----------------------:|
| Frozen model (user-selected) | 124M | ~500 MB | N/A (on disk only) |
| Vector bias (12 blocks × 768) | 9,216 | 36 KB | 2.3 KB |
| Low-rank (rank 4, attn+MLP, 12 blocks) | ~300K | ~1.2 MB | ~75 KB |
| Low-rank (rank 8, full injection, 12 blocks) | ~1.2M | ~4.8 MB | ~300 KB |
| Consolidation store | 2× plastic | 2× plastic | 2× plastic |

Total overhead (Phase 2.2, ternary, rank 8): **~600 KB** of plastic weights for a 500 MB model — **0.12% overhead.**

---

## Next Steps

- [ ] Map the transformer block structure for the user-selected model family
- [ ] Implement Option A (forward hooks) for rapid prototyping
- [ ] Verify that hooking doesn't break generation or double memory
- [ ] Implement Option B (module replacement) for the clean API
- [ ] Write architecture-specific block wrappers for each supported model family
