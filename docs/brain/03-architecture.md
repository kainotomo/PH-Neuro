# Step 0.4 — Brain Wrapper Architecture Design

> **Status:** ✅ Complete (2026-08-12)
> **Goal:** Design the software architecture for wrapping any HuggingFace model with local plasticity. Define the public API, the hooking mechanism, and the plasticity injection points.
> **This document is the implementable spec for Step 1.1.** The Step 1.1 chat should implement the Brain Wrapper from this document without ambiguity.

---

## Locked Design (Steps 0.1–0.3, recapped)

| Component | Decision |
|:----------|:---------|
| **Primary model** | SmolLM2-1.7B (`HuggingFaceTB/SmolLM2-1.7B`, Apache-2.0) — LLaMA-modern, 24 blocks, d_model 2048, RoPE, SwiGLU, RMSNorm, full MHA (32/32) |
| **Gen-test model** | GPT-2 124M (`openai-community/gpt2`, MIT) — classic pre-norm, 12 blocks, d_model 768, LayerNorm, GELU, no RoPE |
| **Plasticity mechanism** | 3-factor Hebbian, **vector bias**: `Δb = η · M · mean(post)` at `down_proj` + `o_proj` per block |
| **Surprise signal** | Global **float32** scalar `M = M_max / (1 + exp(−k·(s − s₀)))`, `s = (L − L̂)/L̂`, `L̂ ← α·L̂ + (1−α)·L`; defaults α=0.99, s₀=0.05, k=60, M_max=1.0 |
| **Hard constraints** | No backprop through frozen layers; frozen backbone never modified; plastic weights float32; L, L̂, s, M computed in float32 |
| **Verification** | All module paths below verified 2026-08-12 against the installed transformers **5.15.0** modeling source **and** the locally cached weights (SmolLM2-1.7B, GPT-2 124M) |

---

## Design Goals

1. **Model-agnostic.** Work with any HuggingFace `AutoModelForCausalLM`. Architecture-specific knowledge is confined to a thin per-architecture adapter (~20 lines), not scattered through the wrapper.
2. **Non-invasive.** Don't modify the original model's source code and don't wrap/replace transformer blocks. Use PyTorch forward hooks on projection submodules only.
3. **Identity guarantee.** Inference with plasticity disabled (or all plastic weights 0) must be **bit-identical** to the frozen unwrapped model. This is a testable invariant (I1), not an aspiration.
4. **Minimal overhead.** Plastic weights are tiny (~0.01% of model params in Phase 1.1). One vector-add per injection point per forward pass.
5. **No backprop.** Plastic updates use locally captured activations + a global scalar. Zero `.backward()` calls in the training loop.
6. **Clean API.** The user experience should be:
   ```python
   model = AutoModelForCausalLM.from_pretrained("<model id>")
   brain = BrainWrapper(model)
   brain.learn(texts, steps=1000)  # adapt plastic weights
   brain.generate(prompt)          # use model + plastic weights
   brain.save("my_brain.pt")       # save plastic weights only
   ```

---

## Hooking Mechanism — Selection & Justification

Three mechanisms were analyzed. They are not mutually exclusive; the final design uses one as primary and reserves another as a Phase 2 fallback.

### Option A: PyTorch Forward Hooks (observe-only)

```python
def hook_fn(module, input, output):
    # input: tuple of inputs to this module
    # output: the module's output
    # Store for plasticity update
```

**Pros:** Works with any `nn.Module`. No model modification. Standard PyTorch. Zero risk to the frozen model.
**Cons:** Hooks fire on every forward pass — state must be managed carefully. Cannot inject plastic weights INTO the module's computation on their own.
**Verdict:** Used only for **capture** (pre/post activations), never for injection on its own.

### Option B: Module Replacement (wrap each block)

```python
class PlasticBlock(nn.Module):
    def __init__(self, frozen_block, plastic_dim):
        self.frozen = frozen_block       # original, frozen
        self.plastic_bias = nn.Parameter(torch.zeros(plastic_dim))

    def forward(self, hidden_states, **kwargs):
        out = self.frozen(hidden_states, **kwargs)
        return out + self.plastic_bias    # inject plasticity
```

**Pros:** Full control over the forward pass. Plastic bias is a proper `nn.Parameter`.
**Cons:** Must replicate the block's forward signature **exactly**. In transformers 5.x the decoder-block forward takes a long, version-unstable set of kwargs (`attention_mask`, `position_ids`, `past_key_values`/`Cache`, `position_embeddings`, `**kwargs: Unpack[TransformersKwargs]`, …). Any replication bug silently breaks the identity guarantee; any transformers upgrade can break the replica. Per-architecture cost is ~100–200 lines of brittle code.
**Verdict:** **Rejected for Phase 1.1.** Revisit only in Phase 2 if we need to inject *inside* attention internals (e.g. pre-RoPE Q/K bias), which a hook on the outer projection cannot reach.

### Option C: Output Modification via Hooks (inject + capture)

```python
def make_inject_hook(bias_ref, active_ref):
    def hook(module, args, output):
        # output is a plain tensor (B, S, d_out) for ALL our injection points
        if active_ref():
            return output + bias_ref().to(output.dtype)
        return output
    return hook
```

**Pros:** Model-agnostic — works identically on `nn.Linear` (SmolLM2) and `Conv1D` (GPT-2). No block forward pass to replicate. Identity guarantee is structural (`active=False` → return output unchanged). Bias can be a plain float32 tensor owned by the wrapper (no `nn.Parameter` needed — there is no autograd anyway).
**Cons:** Only sees module input/output (no access to attention internals). Bias dimension must equal the projection's out_features (architecture-specific — but *any* mechanism has this requirement).
**Verdict:** **PRIMARY for Phase 1.1 (and Phase 2).**

### Final Recommendation: Option C

**Why C is the right production choice here** (the draft's "B for production" verdict changes once the actual model structure is verified):

1. **Both architectures converge at the projection module.** SmolLM2's `model.model.layers[i]` vs GPT-2's `model.transformer.h[i]` differ at the *block* level, but the injection targets — `down_proj`/`o_proj` (`nn.Linear`) and `c_proj` (`Conv1D`) — are all plain modules with a single-tensor input and single-tensor output. One hook implementation covers both; the only per-architecture code is *which module to hook*, which lives in the ~20-line `BlockWrapper` adapter (see below). → **minimal code duplication.**
2. **Phase 1.1 needs only the post-activation.** `Δb = η·M·mean(post)` consumes the projection output — exactly what an output hook sees. No pre capture, no weight surgery. (Pre capture is still added cheaply for diagnostics and Phase 2 low-rank — the hook receives `args`.)
3. **No autograd ⇒ no `nn.Parameter` requirement.** Plastic biases are updated by hand-written Hebbian rules, never by an optimizer. They are plain float32 tensors owned by `BrainWrapper`; Option B's "proper Parameter" advantage is moot.
4. **Identity guarantee is structural, not tested-by-luck.** With C, `active=False` returns the module output untouched, so the wrapped model *is* the frozen model by construction. With B you must prove your block replica is byte-identical, across transformers versions.
5. **Robust to HF internals.** Hooks attach to stable, named submodules (`self_attn.o_proj`, `mlp.down_proj`, `attn.c_proj`, `mlp.c_proj`). If HF changes the block wrapper but keeps these submodules, hooks keep working; module replacement must track every forward-signature change.
6. **Phase 2 compatible.** Low-rank plastic matrices also need only module input + output — both available to hooks. Ternary and consolidation are hook-side too.

**When to switch to B:** only if a later phase must inject *inside* the attention computation (e.g. plastic bias on `q_proj`/`k_proj` *before* RoPE). That is explicitly **not** a Phase 1.1/1.3 requirement.

**Failure mode:** if a future transformers version renames a target submodule, hook attach raises `AttributeError` loudly — never a silent no-op.

---

## Verified Injection Points (Phase 1.1)

Verified 2026-08-12 against the installed transformers **5.15.0** modeling source and the locally cached weights. All four injection-point paths below exist exactly as written.

### SmolLM2-1.7B (`LlamaForCausalLM`, `config.model_type == "llama"`)

Block path: **`model.model.layers[i]`** (i ∈ 0..23, `LlamaDecoderLayer`).

| Full module path | Module type | In→Out | Role | Injected? |
|:-----------------|:------------|:------:|:-----|:---------:|
| `model.model.layers[i].self_attn.q_proj` | `nn.Linear` | 2048→2048 | Q proj (pre-RoPE) | — |
| `model.model.layers[i].self_attn.k_proj` | `nn.Linear` | 2048→2048 | K proj (pre-RoPE) | — |
| `model.model.layers[i].self_attn.v_proj` | `nn.Linear` | 2048→2048 | V proj | — |
| `model.model.layers[i].self_attn.o_proj` | `nn.Linear` | 2048→2048 | **attention output** | ✅ **inject** |
| `model.model.layers[i].mlp.gate_proj` | `nn.Linear` | 2048→8192 | SwiGLU gate | — |
| `model.model.layers[i].mlp.up_proj` | `nn.Linear` | 2048→8192 | SwiGLU up | — |
| `model.model.layers[i].mlp.down_proj` | `nn.Linear` | 8192→2048 | **MLP output** | ✅ **inject** |
| `model.model.layers[i].input_layernorm` | `LlamaRMSNorm` | 2048 | pre-attn norm | — |
| `model.model.layers[i].post_attention_layernorm` | `LlamaRMSNorm` | 2048 | pre-MLP norm | — |

Forward flow (from `LlamaDecoderLayer.forward`):

```
residual = hidden
hidden   = input_layernorm(hidden)
hidden   = self_attn(hidden)          # q/k/v → RoPE → attn → o_proj   ← O_PROJ POST CAPTURED HERE
hidden   = residual + hidden          # post-attention residual
residual = hidden
hidden   = post_attention_layernorm(hidden)
hidden   = mlp(hidden)                # gate_proj→SiLU ⊙ up_proj → down_proj   ← DOWN_PROJ POST CAPTURED HERE
hidden   = residual + hidden          # post-block residual
return hidden
```

`LlamaMLP.forward`: `down_proj(act_fn(gate_proj(x)) * up_proj(x))` — the plastic bias on `down_proj` operates on the already-gated SwiGLU hidden state, avoiding the gate-zeroing risk (Step 0.2).

### GPT-2 124M (`GPT2LMHeadModel`, `config.model_type == "gpt2"`)

Block path: **`model.transformer.h[i]`** (i ∈ 0..11, `GPT2Block`).

| Full module path | Module type | In→Out | Role | Injected? |
|:-----------------|:------------|:------:|:-----|:---------:|
| `model.transformer.h[i].attn.c_attn` | `Conv1D` | 768→2304 | combined QKV | — |
| `model.transformer.h[i].attn.c_proj` | `Conv1D` | 768→768 | **attention output** | ✅ **inject** |
| `model.transformer.h[i].mlp.c_fc` | `Conv1D` | 768→3072 | MLP in | — |
| `model.transformer.h[i].mlp.c_proj` | `Conv1D` | 3072→768 | **MLP output** | ✅ **inject** |
| `model.transformer.h[i].ln_1` | `nn.LayerNorm` | 768 | pre-attn norm | — |
| `model.transformer.h[i].ln_2` | `nn.LayerNorm` | 768 | pre-MLP norm | — |

Forward flow (from `GPT2Block.forward`):

```
residual = hidden
hidden   = ln_1(hidden)
hidden   = attn(hidden)               # c_attn→split QKV→attn→c_proj→resid_dropout   ← ATTN C_PROJ POST CAPTURED HERE
hidden   = hidden + residual          # post-attention residual
residual = hidden
hidden   = ln_2(hidden)
hidden   = mlp(hidden)                # c_fc→GELU→c_proj→dropout   ← MLP C_PROJ POST CAPTURED HERE
hidden   = residual + hidden          # post-block residual
return hidden
```

### Key verification notes (implementer must not re-derive)

1. **`Conv1D` vs `nn.Linear`.** GPT-2's `Conv1D(nf, nx)` computes `x @ W` with weight shape `(nx, nf)` — the transpose of `nn.Linear`'s `x @ Wᵀ`. To a forward hook this is invisible: input `(B, S, in)` → output `(B, S, out)`. The same inject hook works on both. Plastic bias shape is always `(out_features,)` = `hidden_size`. To read `out_features` robustly: `nn.Linear` → `.out_features`; `Conv1D` → `.nf` (it has **no** `.out_features`/`.in_features` attributes — verified).
2. **All four injection modules return plain tensors** (not tuples) — the "fragile tuple-output hook" case never occurs at these points.
3. **Dropout after GPT-2 `c_proj`** is inert because `BrainWrapper` runs the frozen model in **eval()** mode during `learn()` and `generate()`.
4. **Pre/post capture semantics (per Step 0.2):**
   - `pre` = input to the projection, shape `(B, S, d_in)`. **Not used by the vector-bias update**; captured for diagnostics + Phase 2 (low-rank).
   - `post` = output **after** bias injection, shape `(B, S, d_out)`; the update uses `post.mean(dim=(0,1))`.
   - Note `mean(post) = mean(frozen_output) + b` (b is constant over B·S). The spec uses the *combined* post. Ablation variant (Step 1.2): use the pre-bias frozen output — a one-line hook change.
5. **Layer ordering is stable**: `o_proj` fires before `down_proj` in every block (attention precedes MLP). No ordering surprises.

### Injection design space (retained for reference — full analysis in Step 0.2)

Vector-bias injection at the sublayer *output* projections (`o_proj`/`down_proj`/`c_proj`) is the Phase 1.1 choice. Post-block, KV-bias, and LoRA-style low-rank injections remain the Phase 2 design space (low-rank: `y + B @ (A @ x)`, both `A`, `B` hook-side).

### Injection Strategy by Phase

| Phase | Injection | Plastic params (SmolLM2-1.7B) | Rationale |
|:------|:----------|:----------------------------:|:----------|
| 1.1/1.3 | Vector bias at `o_proj` + `down_proj` | 98,304 (393 KB fp32) | Minimal. Test the core hypothesis. |
| 2.1 | Low-rank (rank r) at same sites | 48 · 2·2048·r (r=4 → 786K, 3.1 MB) | More capacity. |
| 2.2 | Same, ternary 2-bit | same ÷ 16 | DQT/hysteresis reuse (Phase 0 infra). |
| 2.3 | Same + consolidation store | 2× (fast + slow) | Long-term memory. |

---

## Architecture Abstraction Layer

SmolLM2 and GPT-2 differ only in *which modules get hooked*, so the architecture-specific knowledge is confined to a thin adapter. Spec (protocol, not implementation — implement in Step 1.1):

```python
@dataclass
class InjectionPoint:
    name: str                      # stable id, e.g. "L03.o_proj", "L07.mlp_c_proj"
    module: nn.Module              # frozen projection (nn.Linear | Conv1D)
    bias: torch.Tensor             # plastic bias, shape (out_features,), float32, zeros-init
    pre_handle: object | None = None    # registered forward_pre_hook (pre capture)
    post_handle: object | None = None   # registered forward_hook (bias inject + post capture)
    out_features: int | None = None     # d_out, read at construction (Linear.out_features | Conv1D.nf)

class BlockWrapper(Protocol):
    block_paths: tuple[str, ...]        # submodule paths to hook, e.g. ("self_attn.o_proj", "mlp.down_proj")
    def get_injection_points(self, block: nn.Module, layer_idx: int) -> list[InjectionPoint]: ...

class SmolLM2BlockWrapper(BlockWrapper):
    # block type: LlamaDecoderLayer; covered by model_type "llama" (incl. SmolLM2 135M/360M)
    block_paths = ("self_attn.o_proj", "mlp.down_proj")
    def get_injection_points(self, block, layer_idx):
        return [
            InjectionPoint(f"L{layer_idx:02d}.o_proj",    block.self_attn.o_proj, zeros(block.hidden_size)),
            InjectionPoint(f"L{layer_idx:02d}.down_proj", block.mlp.down_proj,    zeros(block.hidden_size)),
        ]

class GPT2BlockWrapper(BlockWrapper):
    # block type: GPT2Block; covered by model_type "gpt2"
    block_paths = ("attn.c_proj", "mlp.c_proj")
    def get_injection_points(self, block, layer_idx):
        return [
            InjectionPoint(f"L{layer_idx:02d}.attn_c_proj", block.attn.c_proj, zeros(768)),
            InjectionPoint(f"L{layer_idx:02d}.mlp_c_proj",  block.mlp.c_proj,  zeros(768)),
        ]

def get_block_wrapper(model) -> type[BlockWrapper]:
    """Auto-detect from config.model_type (NOT the Python class — SmolLM2 is LlamaForCausalLM)."""
    t = model.config.model_type
    if t == "llama": return SmolLM2BlockWrapper     # covers the whole SmolLM2 scaling ladder
    if t == "gpt2":  return GPT2BlockWrapper
    raise NotImplementedError(f"model_type={t!r} not supported (supported: llama, gpt2)")

def get_block_container(model) -> nn.ModuleList:
    t = model.config.model_type
    if t == "llama": return model.model.layers
    if t == "gpt2":  return model.transformer.h
    raise NotImplementedError(f"model_type={t!r} not supported (supported: llama, gpt2)")
```

Design points:
- **Detection key = `config.model_type`.** SmolLM2 (all three tiers) is `"llama"`; GPT-2 is `"gpt2"`. The Step 0.1 scaling ladder (135M→1.7B) then works with zero extra code.
- `InjectionPoint.bias` is **not** an `nn.Parameter` and **not** part of `model.state_dict()` — it lives in the wrapper, keeping the frozen model's `state_dict()` clean.
- Full log path is derived: `model.model.layers[{i}].self_attn.o_proj` (llama) / `model.transformer.h[{i}].attn.c_proj` (gpt2).

---

## BrainWrapper Public API (final)

```python
class BrainWrapper:
    """Wrap a frozen pre-trained CausalLM with local, surprise-modulated
    3-factor Hebbian plasticity. The frozen backbone never changes; tiny
    plastic bias vectors are injected at each block's `o_proj`+`down_proj`
    (SmolLM2) / `attn.c_proj`+`mlp.c_proj` (GPT-2) and updated from locally
    captured activations and a global surprise scalar. No backprop.

    Step 0.4 spec — Step 1.1 implements exactly this surface.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        plasticity: str = "vector_bias",
        modulator_cfg: dict | None = None,
        *,
        lr: float = 1e-3,                  # η
        decay_rate: float = 0.0,           # λ; 0.0 = off
        dtype: torch.dtype = torch.float32,
        tokenizer: PreTrainedTokenizer | None = None,
        device: str | torch.device | None = None,
        checkpoint_dir: str | None = None, # None → checkpointing disabled
        checkpoint_every: int = 100,       # N steps between checkpoints
        min_free_gb: float | None = None,  # GPU gate; auto from model size
    ) -> None: ...
```

**Constructor behavior:** detects the architecture via `get_block_wrapper` /
`get_block_container`, builds every `InjectionPoint`, zero-inits all biases on
`device`, registers pre/post hooks, sets `model.eval()`, freezes the model
(`model.requires_grad_(False)`), inits EMA state (`L̂` unset sentinel), and
validates the `modulator_cfg` keys. `plasticity != "vector_bias"` raises
`NotImplementedError` until Phase 2. `tokenizer` is only required for
`learn(list[str])` / `generate(str)`; if None it is resolved via
`model.get_tokenizer()` then `AutoTokenizer.from_pretrained(model.config._name_or_path)`.

### `learn(texts_or_dataloader, steps, ...) -> list[dict]`

```python
def learn(
    self,
    texts_or_dataloader: list[str] | Iterable[dict],
    steps: int,
    *,
    batch_size: int = 4,
    seq_len: int = 256,
    gpu_policy: str = "exit",      # "exit" | "wait" | "warn"
    warmup_steps: int = 0,         # optional M=0 warm-up to settle EMA
    seed: int | None = None,
) -> list[dict]:
```

**Behavior:** GPU check → resume-from-checkpoint (skip-if-exists) → install
signal handlers → run the loop (§Training Loop) → save final checkpoint →
restore handlers. Returns one metric dict per step: `{step, loss, ema_loss,
surprise_s, modulator_M, mean_abs_delta_b, mean_abs_b, tokens_seen}`.
**Input:** `list[str]` is tokenized/packed/cycled internally (needs
`tokenizer`); an `Iterable` must yield `{"input_ids": LongTensor (B,S),
"attention_mask": LongTensor (B,S)}` (mask optional, defaults to all-ones).
**Edge cases:** `steps <= 0` → `[]`; empty texts / missing tokenizer →
`ValueError`; batch longer than `seq_len` → truncate; a checkpoint with
`step >= steps` → log `already complete`, return `[]` (never restart).

### `generate(prompt, **kwargs) -> str`

```python
def generate(
    self,
    prompt: str,
    *,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float | None = None,
    top_k: int | None = None,
    **generate_kwargs,
) -> str:
```

Tokenizes, moves to device, runs `model.generate(...)` under `torch.no_grad()`
with hooks **active** (plastic ON), decodes. Empty prompt → uses the model's
BOS/start token; warns if `pad_token_id`/`eos_token_id` are missing.
`with brain.without_plasticity(): brain.generate(p)` returns the frozen output.

### `without_plasticity() -> contextmanager`

Sets `active=False` on enter (hooks return output unchanged) and restores the
previous value on exit; a nesting counter makes nested use safe. Does **not**
touch plastic state or EMA. This is the mechanism for baseline eval.

### `consolidate() -> dict`

Phase 2.3 placeholder. Logs a warning and returns
`{"status": "not_implemented", "phase": "2.3"}`. Signature is stable so Phase
1.1 callers and checkpoint formats won't break when Phase 2.3 lands.

### `save(path) / load(path)`

- `save(path)`: atomic `torch.save(state_dict())` (+ config + timestamp) via
  temp file then `os.replace`; creates parent dirs. Saves **plastic weights
  only** — never the frozen model.
- `load(path)`: restores biases into the current instance, shape-validated
  against every injection point; wrong architecture → `ValueError`. Returns
  `self` (chainable).

### `state_dict() / load_state_dict()`

- `state_dict()` → flat `OrderedDict`: `f"plastic.{ip.name}" → bias` (float32),
  one entry per injection point. Compatible with `torch.save`/`torch.load` and
  standard tooling.
- `load_state_dict(state_dict, strict=True)`: restores biases; validates
  keys + shapes. `strict=True` (default) raises on missing/extra keys;
  `strict=False` ignores extras, still errors on shape mismatch.

### Public helpers

`plastic_parameter_count()`, `plastic_memory_bytes()` (float32), `injection_point_names()`,
`summary()`, `to(device)`, `set_lr(η)`, `set_decay_rate(λ)`.

---

## Training Loop (learn) — Pseudocode

```python
def learn(texts_or_dataloader, steps, *, batch_size=4, seq_len=256,
          gpu_policy="exit", warmup_steps=0, seed=None):
    # 0. Setup
    self.model.eval()                                    # no dropout → deterministic activations
    self.model.requires_grad_(False)
    self._check_gpu(gpu_policy, self.min_free_gb)        # nvidia-smi; exit/wait if free < min_free_gb
    start_step = self._resume(steps)                     # skip-if-exists; restore plastic+EMA+step
    if start_step >= steps:
        log(f"already complete (step {start_step} >= {steps}); skipping")
        return []
    _install_signal_handlers(on_signal=lambda: self._save_checkpoint(current_step))
    data_iter = self._make_batch_iter(texts_or_dataloader, batch_size, seq_len, seed)
    metrics, current_step = [], start_step
    L_hat = self._ema_loss if self._ema_initialized else None

    for step in range(start_step, steps):
        current_step = step
        batch = next(data_iter)                          # {"input_ids": (B,S), "attention_mask": (B,S)}
        ids, mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

        # 1. Forward (frozen); hooks capture post at every injection point
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=mask).logits   # (B,S,V) bf16

        # 2. Loss L — float32, computed manually (not the model's internal CE)
        L = F.cross_entropy(
            logits[..., :-1, :].float().reshape(-1, V), ids[..., 1:].reshape(-1)
        )                                                                    # float32 scalar

        # 3. EMA  L̂ ← α·L̂ + (1−α)·L   (first step: L̂ ← L so s=0, M≈0.018)
        L_hat = L.detach().clone() if L_hat is None else alpha * L_hat + (1 - alpha) * L

        # 4. Surprise s = (L − L̂)/L̂   5. Modulator M = sigmoid(s)  (both float32)
        s = (L - L_hat) / L_hat
        M = M_max / (1 + math.exp(-k * (s - s0)))
        if warmup_steps and step < warmup_steps:
            M = 0.0                                            # optional EMA warm-up

        # 6. Plastic update per injection point (float32)
        mean_abs_delta = 0.0
        for ip in self._injection_points:
            post = self._last_post[ip.name]                    # (B,S,d) float32, from hook
            delta_b = eta * M * post.mean(dim=(0, 1))          # (d,) float32
            ip.bias.add_(delta_b)                              # b ← b + Δb
            if decay_rate > 0:
                ip.bias.mul_(1.0 - decay_rate)                 # b ← b·(1−λ)
            mean_abs_delta += delta_b.abs().mean().item()

        # 7. Metrics
        metrics.append({
            "step": step, "loss": L.item(), "ema_loss": L_hat.item(),
            "surprise_s": s.item(), "modulator_M": float(M),
            "mean_abs_delta_b": mean_abs_delta / len(self._injection_points),
            "mean_abs_b": self._mean_abs_bias(),
            "tokens_seen": (step + 1) * ids.numel(),
        })
        self._ema_loss, self._ema_initialized = L_hat, True

        # 8. Checkpoint every N steps
        if (step + 1) % self.checkpoint_every == 0:
            self._save_checkpoint(step + 1)

    self._save_checkpoint(steps)                             # final
    _restore_signal_handlers()
    return metrics
```

**Notes:** `mean_abs_delta_b` is the pre-decay `|Δb|` mean over all plastic
params; `mean_abs_b` is the post-decay `|b|` mean. EMA is updated **before**
`L̂` is used in `s` (per Step 0.3; with α=0.99 the ordering effect is ~1%).
All math is float32. The loop is pure inference + local updates — no autograd
graph, so batch 4 × seq 256 on SmolLM2-1.7B is cheap (~5–6 GB peak incl.
weights, see Memory Budget).

---

## Precision Design

| Value | Precision | Where |
|:------|:----------|:------|
| Frozen backbone weights/activations | bf16 | unchanged (model dtype) |
| Plastic bias state `b` | **float32** | stored in the wrapper |
| Forward injection | `output + b.to(bf16)` | per-forward cast of 2048 elems — negligible |
| `post` capture | upcast to float32 | `combined.detach().float()` in the hook |
| `L`, `L̂`, `s`, `M` | **float32** | Step 0.3 requirement (bf16 underflows M ≈ 1e-3 → 0) |
| `Δb`, decay | float32 | accumulation precision |

## Memory Budget (Phase 1.1, float32 plastic)

| Model | Blocks | Inj. pts | Bias dim | Plastic params | Plastic bytes | % of frozen (bf16) |
|:------|:------:|:--------:|:--------:|:--------------:|:-------------:|:------------------:|
| SmolLM2-1.7B | 24 | 48 | 2048 | 98,304 | 393 KB | ~0.011% |
| GPT-2 124M | 12 | 24 | 768 | 18,432 | 73.7 KB | ~0.029% |

Phase 2 low-rank (rank r): 48 · 2·2048·r plastic params on SmolLM2 — r=4 → 786K (3.1 MB float32). Ternary 2-bit divides the same by 16 (DQT/hysteresis infra from Phase 0).

**GPU footprint (SmolLM2-1.7B, the 8 GB constraint):** bf16 weights ≈ 3.42 GB; activations at batch 4 × seq 256 ≈ 1–1.5 GB; CUDA context + workspace ≈ 0.5–1 GB → **~5–6 GB peak**, comfortably inside the RTX 4060's 8 GB. Plastic biases add ≪ 1 MB.

---

## Operational Design — GPU check, pause/resume, checkpoints, signals, logging

### GPU check (every GPU run)

`_check_gpu(policy, min_free_gb)`: if the model device is not CUDA, skip (CPU runs are allowed — Step 0.1 used CPU). Else run `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`, parse free MiB, and if `free < min_free_gb·1024`:
- `policy="exit"` (default): log `WARNING: only {free} GB free, need ≥ {min_free_gb} GB for {model}` then `sys.exit(1)` — supervisor-friendly (retry later).
- `policy="wait"`: poll every 60 s until free ≥ threshold (interactive co-use).
- `policy="warn"`: log and proceed (accepting the risk).

**Thresholds (documented defaults):** SmolLM2-1.7B → `min_free_gb = 6.0` (~3.4 GB weights + ~1–1.5 GB activations + ~0.5–1 GB context ≈ 5–6 GB peak; the 6 GB gate leaves headroom and prevents the project's documented silent-OOM failure mode — M1.2/M2.1: shared GPU, game holds ~6 GB, training dies silently). GPT-2 124M → `2.0`. Auto default: `min_free_gb = clamp(params_bytes_bf16 · 2.2 / 1e9, 2.0, 8.0)`; explicit override wins.

### Checkpoints — save / resume / skip-if-exists

Checkpoint file (one dict via `torch.save`):
```
{
  "format": "ph_neuro_brain_checkpoint", "version": 1,
  "step": N,                              # steps completed
  "plastic": OrderedDict,                  # state_dict(): plastic.{name} → (d,) float32
  "ema_loss": float, "ema_initialized": bool,
  "config": {"alpha","s0","k","M_max","eta","decay_rate","model_type","n_layers","hidden_size","plasticity"},
  "saved_at": ISO-8601,
}
```
- **Every N steps** (`checkpoint_every`, default 100): write `{checkpoint_dir}/brain_ckpt_step{step}.pt`; also (re)write `{checkpoint_dir}/brain_latest.pt`.
- **Atomic write:** `torch.save(state, f"{path}.tmp.{os.getpid()}")` then `os.replace(tmp, path)` — atomic on POSIX, no torn files.
- **Resume:** on `learn()`, scan `{checkpoint_dir}/brain_ckpt_step*.pt`, pick the highest `step < steps`, restore plastic + EMA + step, log `resumed from step {N}`, continue.
- **Skip-if-exists:** if the highest saved step ≥ `steps`, log `already complete` and return `[]` — a completed run is never restarted.
- Also save on SIGINT/SIGTERM and at the final step.

### Signal handling (SIGINT / SIGTERM)

During `learn()`, install handlers for SIGINT and SIGTERM. First signal: log `signal {name} received — saving checkpoint at step {current}`, call `_save_checkpoint(current_step)`, then exit 130 (supervisor resumes later). Second signal: exit immediately without saving (force). Restore prior handlers on normal exit / `KeyboardInterrupt`.

### Logging

Launch scripts must set `PYTHONUNBUFFERED=1` (repo convention — block-buffered logs look frozen). `logging` module with two handlers: file `{logs_dir}/brain_learn_{model_type}_{run_id}.log` (repo convention `logs/logs_brain/`) formatted `%(asctime)s %(levelname)s %(message)s` (timestamps required), and console INFO. Log per-step metrics every 10 steps; always log start, resume, each checkpoint save, signal, completion.

---

## Edge Cases & Invariants

**Invariants (must be tested in Step 1.1):**
- **I1 (frozen identity):** with `active=False`, the wrapped model's logits are bit-identical to the unwrapped frozen model's logits — also with `active=True` and all biases 0.
- **I2 (zero init):** immediately after construction, `generate()` == frozen `generate()`.
- **I3 (no autograd):** `learn()` runs under `torch.no_grad()`; tests monkey-patch `.backward()` and assert zero calls (repo convention).
- **I4 (eval mode):** `learn()`/`generate()` always run the frozen model in `eval()` — GPT-2's post-`c_proj` dropout is a no-op.
- **I5 (shapes):** every plastic bias has shape `(d_out,)` = hidden_size; `_get_out_features` handles `nn.Linear.out_features` and `Conv1D.nf` (verified: `Conv1D(nf, nx)` stores weight `(nx, nf)` and has no `out_features`/`in_features` attrs).

**Edge cases:** `steps <= 0` → `[]` without touching checkpoints; empty `texts` / missing tokenizer → `ValueError`; dataloader batches missing `attention_mask` → all-ones; batch longer than `seq_len` → truncate (SmolLM2 max 8192, GPT-2 max 1024; Phase 1.1 uses 256); `load()` into a different architecture → shape `ValueError`; `save()` to a non-existent dir → create parents; `M` underflow impossible in float32 (sigmoid bounded [0,1]); `L̂` warm-up sets `s=0, M≈0.018` on step 0 — no spurious first-step update.

---

## Implementation Checklist (Step 1.1 — from this spec)

- [ ] `src/ph_neuro/brain/` package: `injection.py` — `InjectionPoint`, `BlockWrapper` protocol, `SmolLM2BlockWrapper`, `GPT2BlockWrapper`, `get_block_wrapper`, `get_block_container`, `_get_out_features`
- [ ] `surprise.py` — pure helpers for EMA + `s` + `M` (unit-testable against Step 0.3 defaults α=0.99, s₀=0.05, k=60, M_max=1.0)
- [ ] `wrapper.py` — `BrainWrapper` (constructor, hooks, `learn`, `generate`, `without_plasticity`, `consolidate`, `save`/`load`, `state_dict`/`load_state_dict`, `_check_gpu`, `_save_checkpoint`/`_resume`, signal handlers)
- [ ] Tests: I1–I5 above + resume round-trip + atomic write + skip-if-exists + GPU check (mocked `nvidia-smi`)
- [ ] Phase 1.1 runner wires WikiText-2 → PubMed data, sets `checkpoint_dir` and `PYTHONUNBUFFERED=1`
