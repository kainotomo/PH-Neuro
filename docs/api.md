# PH-Neuro — API Reference

> Complete reference for the public `ph_neuro` API. Every example below is
> copy-paste runnable. Signatures were verified against the source.

**Quick index**

| Symbol | Module |
|:-------|:-------|
| [`TernaryDQTLinear`](#1-ternarydqtlinear) | `ph_neuro.layers.ste_dqt` |
| [`TernaryDQTConv2d`](#2-ternarydqtconv2d) | `ph_neuro.layers.ste_dqt_conv` |
| [`TernaryDQTMoELayer`](#3-ternarydqtmoelayer) | `ph_neuro.layers.ste_dqt_moe` |
| [`TernarySTELinear`](#4-ternarystelinear-fallback) | `ph_neuro.layers.ste_linear` |
| [`TernarySTEConv2d`](#5-ternarysteconv2d-fallback) | `ph_neuro.layers.ste_conv` |
| [`stochastic_round()`](#6-stochastic_round) | `ph_neuro.layers.ste_dqt` |
| [`ste_sign()`](#7-ste_sign) | `ph_neuro.layers.ste_linear` |
| [`dqt_to_inference_model()`](#8-dqt_to_inference_model) | `ph_neuro.models.export` |
| [`export_to_onnx()`](#9-export_to_onnx) | `ph_neuro.models.export` |
| [`verify_onnx()`](#10-verify_onnx) | `ph_neuro.models.export` |
| [`pack_ternary()` / `unpack_ternary()`](#11-pack_ternary--unpack_ternary) | `ph_neuro.utils.packing` |
| [`fuse_bn_layers()`](#12-fuse_bn_layers) | `ph_neuro.models.fuse_bn` |
| [Bonus: export helpers](#bonus-export-helpers) | `ph_neuro.models.export` |

> **DQT vs STE in one line.** DQT layers (`TernaryDQTLinear`,
> `TernaryDQTConv2d`) store the ternary weights directly as an int8 buffer
> and update them via **stochastic rounding** of a float accumulation buffer —
> no persistent latent float scores, 4.5× less training memory. STE layers
> (`TernarySTELinear`, `TernarySTEConv2d`) keep persistent float latent
> scores and derive ternary weights with `sign()`. Use **DQT** for new work;
> STE exists as the validated fallback/classic path.

---

## 1. `TernaryDQTLinear`

Linear layer with ternary weights trained via **DQT + stochastic rounding**.
The primary layer for MLPs.

**Module:** `ph_neuro.layers.ste_dqt`

```python
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
```

### Signature

```python
TernaryDQTLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
)
```

### Parameters

| Arg | Type | Default | Description |
|:----|:-----|:-------:|:------------|
| `in_features` | `int` | — | Size of each input sample. |
| `out_features` | `int` | — | Size of each output sample. |
| `bias` | `bool` | `True` | Adds a learnable bias. |
| `device` | `torch.device \| str \| None` | `None` | Torch device. |
| `dtype` | `torch.dtype` | `torch.float32` | Dtype of the float accumulation buffer. |

### Attributes

| Attribute | Kind | Description |
|:----------|:-----|:------------|
| `weight_float` | `nn.Parameter` | Float accumulation buffer — the **only** learnable parameter. |
| `weight_ternary` | buffer (`int8`) | The actual ternary weights in {-1, 0, +1}. |
| `in_features` / `out_features` | property | Dimension accessors. |

### Methods

- `forward(x)` — output shape `(batch, *, out_features)`.
- `apply_stochastic_rounding()` → `dict` — call **after** `optimizer.step()`.
  Returns `{"flip_rate": float, "n_flips": int}`.
- `apply_deterministic_rounding()` → `dict` — annealing phase (`sign()`
  instead of stochastic). Same return shape.
- `get_flip_rate()` → `float` — fraction of ternary weights that changed
  since the last rounding step.
- `get_weight_stats()` → `dict` — `{"pos_pct", "neg_pct", "zero_pct"}`.

### Example

```python
import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear

model = nn.Sequential(
    TernaryDQTLinear(784, 512),
    nn.ReLU(),
    TernaryDQTLinear(512, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

x, y = torch.randn(128, 784), torch.randint(0, 10, (128,))
optimizer.zero_grad()
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step()

# ★ THE DQT STEP: discretize float buffer -> ternary after every optimizer step
for module in model.modules():
    if isinstance(module, TernaryDQTLinear):
        module.apply_stochastic_rounding()
```

### Notes

- ⚠️ **Always call `apply_stochastic_rounding()` after `optimizer.step()`** —
  otherwise the ternary weights never update. This is the #1 DQT gotcha.
- `weight_float` is the only parameter the optimizer sees; `weight_ternary`
  is a non-trainable buffer.
- For the annealing (fine-tuning) tail, switch to
  `apply_deterministic_rounding()` to eliminate flip jitter.
- `forward` auto-flattens inputs of dim > 2 (like `(batch, *, in_features)`).

---

## 2. `TernaryDQTConv2d`

2D convolution with ternary weights trained via DQT + stochastic rounding.
The conv counterpart of `TernaryDQTLinear`.

**Module:** `ph_neuro.layers.ste_dqt_conv`

```python
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
```

### Signature

```python
TernaryDQTConv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    bias: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
)
```

### Parameters

| Arg | Type | Default | Description |
|:----|:-----|:-------:|:------------|
| `in_channels` | `int` | — | Number of input channels. |
| `out_channels` | `int` | — | Number of output filters. |
| `kernel_size` | `int \| tuple` | — | Kernel size. |
| `stride` | `int \| tuple` | `1` | Convolution stride. |
| `padding` | `int \| tuple` | `0` | Padding on both sides of the input. |
| `dilation` | `int \| tuple` | `1` | Spacing between kernel elements. |
| `bias` | `bool` | `False` | **Default `False`** — BatchNorm handles the shift. |
| `device` | `torch.device \| str \| None` | `None` | Torch device. |
| `dtype` | `torch.dtype` | `torch.float32` | Dtype of the float accumulation buffer. |

### Attributes

| Attribute | Kind | Description |
|:----------|:-----|:------------|
| `weight_float` | `nn.Parameter` | Float accumulation buffer — the only learnable parameter. |
| `weight_ternary` | buffer (`int8`) | Ternary weights `(out, in, kH, kW)` in {-1, 0, +1}. |
| `in_channels` / `out_channels` / `kernel_size` | property | Accessors. |

### Methods

Same rounding/flip/stat methods as `TernaryDQTLinear`, plus
`ternary_weight()` → the int8 ternary weight tensor.

### Example

```python
import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d

model = nn.Sequential(
    TernaryDQTConv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(2),
    TernaryDQTConv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(128),
    nn.MaxPool2d(2),
    nn.Flatten(),
    TernaryDQTLinear(128 * 8 * 8, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    TernaryDQTLinear(512, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
optimizer.zero_grad()
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step()

from ph_neuro.layers.ste_dqt import TernaryDQTLinear as _DQTLinear
for module in model.modules():
    if isinstance(module, (TernaryDQTConv2d, _DQTLinear)):
        module.apply_stochastic_rounding()
```

### Notes

- Use the ready-made `dqt_cnn()` / `dqt_cnn_cifar100()` factories from
  `ph_neuro.models.dqt_models` instead of hand-building CNNs.
- Backward is numerically exact vs PyTorch autograd (uses
  `torch.nn.grad.conv2d_input` + the im2col correlation identity) — verified
  by 16 unit + 6 integration tests in M1.1.
- The `bias=False` default matters: BatchNorm provides the per-channel shift.

---

## 3. `TernaryDQTMoELayer`

Top-K **Mixture of Experts** layer where each expert is a
`TernaryDQTLinear` and a tiny float router selects the top-K experts per
sample. Only the selected experts run — active params scale with
`top_k / n_experts`.

**Module:** `ph_neuro.layers.ste_dqt_moe`

```python
from ph_neuro.layers.ste_dqt_moe import TernaryDQTMoELayer
```

### Signature

```python
TernaryDQTMoELayer(
    in_features: int,
    expert_width: int,
    n_experts: int,
    top_k: int,
    init_std: float = 0.1,
    router_init_std: float = 0.02,
)
```

### Parameters

| Arg | Type | Default | Description |
|:----|:-----|:-------:|:------------|
| `in_features` | `int` | — | Input dim (expert input dim). |
| `expert_width` | `int` | — | Output width of each expert. |
| `n_experts` | `int` | — | Total number of experts. |
| `top_k` | `int` | — | Active experts per sample; must be `1 <= top_k <= n_experts`. |
| `init_std` | `float` | `0.1` | Init std of expert float buffers. |
| `router_init_std` | `float` | `0.02` | Init std of the (float) router weights. |

### Attributes

| Attribute | Description |
|:----------|:------------|
| `router` | Float `nn.Linear(in_features, n_experts)` — the router (not quantized). |
| `experts` | `ModuleList` of `n_experts` `TernaryDQTLinear` layers. |
| `selection_counts`, `n_selections`, `coverage_counts`, `n_samples` | Load-balancing buffers. |

### Methods

- `forward(x, return_routing=False)` — output `(batch, expert_width)`; if
  `return_routing=True`, returns `(out, logits, indices, weights)`.
- `selection_fractions()` → `Tensor` — per-expert selection share (sums to 1;
  ideal uniform = `1 / n_experts`).
- `coverage_fractions()` → `Tensor` — per-expert coverage share (ideal
  uniform = `top_k / n_experts`).
- `aux_load_balance_loss()` → `Tensor` — Switch-Transformer style aux loss
  (≥ 1.0; lower = more balanced). Call **right after forward** while the
  graph is alive.
- `reset_usage_stats()` — zero the load-balancing counters.
- `get_weight_stats()` → `dict` — aggregated ternary stats across experts.
- `count_parameters()` → `dict` — `{"router", "experts", "total"}`.

### Example

```python
import torch

from ph_neuro.layers.ste_dqt_moe import TernaryDQTMoELayer

moe = TernaryDQTMoELayer(784, 128, n_experts=4, top_k=2)
x = torch.randn(32, 784)

out = moe(x)                       # (32, 128) — weighted sum of active experts
out, logits, indices, weights = moe(x, return_routing=True)

print(moe.selection_fractions())   # load balance metric, sums to 1
aux = moe.aux_load_balance_loss()  # add to the total loss: loss = ce + 0.01 * aux
```

### Notes

- The router is intentionally **not** quantized — it is ~0.8% of params and
  needs full precision for stable top-K selection.
- Experts are DQT layers: you still must call `apply_stochastic_rounding()`
  on every expert after `optimizer.step()` (iterate `moe.experts`).
- Load-balancing buffers accumulate across forwards; call
  `reset_usage_stats()` between train/eval if you need clean metrics.
- Raises `ValueError` if `top_k` is not in `[1, n_experts]`.

---

## 4. `TernarySTELinear` (fallback)

Classic STE linear layer: persistent float **latent scores**, ternary weights
derived via `sign()`, gradients pass through the sign via the STE trick.
The validated fallback path — use DQT for new work.

**Module:** `ph_neuro.layers.ste_linear`

```python
from ph_neuro.layers.ste_linear import TernarySTELinear
```

### Signature

```python
TernarySTELinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
)
```

### Parameters

Same as `TernaryDQTLinear` (see [§1](#1-ternarydqtlinear)).

### Attributes

| Attribute | Kind | Description |
|:----------|:-----|:------------|
| `latent_scores` | `nn.Parameter` | The float latent scores — the learnable parameters. |
| `in_features` / `out_features` | property | Accessors. |

### Methods

- `forward(x)` — output `(batch, *, out_features)`.
- `ternary_weight()` → `Tensor` — int8 ternary matrix `(out, in)` from
  `sign(latent_scores)`.

### Example

```python
import torch
import torch.nn as nn

from ph_neuro.layers.ste_linear import TernarySTELinear

model = nn.Sequential(
    TernarySTELinear(784, 512), nn.ReLU(),
    TernarySTELinear(512, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

x, y = torch.randn(128, 784), torch.randint(0, 10, (128,))
optimizer.zero_grad()
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step()                  # ← no extra rounding step needed for STE

w = model[0].ternary_weight()     # extract int8 ternary weights for inference
```

### Notes

- **No rounding step required** — the optimizer updates `latent_scores`
  directly; ternary weights are re-derived every forward.
- The exported/quantized form uses `ternary_weight()` (frozen), so latent
  scores are only needed during training.

---

## 5. `TernarySTEConv2d` (fallback)

Classic STE 2D convolution — the conv counterpart of `TernarySTELinear`.

**Module:** `ph_neuro.layers.ste_conv`

```python
from ph_neuro.layers.ste_conv import TernarySTEConv2d
```

### Signature

```python
TernarySTEConv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    bias: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
)
```

### Parameters

Same as `TernaryDQTConv2d`, except `bias` defaults to `True` (see
[§2](#2-ternarydqtconv2d)).

### Methods

- `forward(x)` — output `(batch, out_channels, H_out, W_out)`.
- `ternary_weight()` → `Tensor` — int8 ternary `(out, in, kH, kW)`.

### Example

```python
import torch
import torch.nn as nn

from ph_neuro.layers.ste_conv import TernarySTEConv2d

model = nn.Sequential(
    TernarySTEConv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(), nn.MaxPool2d(2),
    TernarySTEConv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 10),
)
out = model(torch.randn(8, 3, 32, 32))
print(out.shape)  # (8, 10)
```

---

## 6. `stochastic_round()`

The core DQT mechanism: stochastically round a float tensor to
{-1, 0, +1}. Rounds toward `ceil` with probability equal to the fractional
part — lets the optimizer explore the weight space without getting stuck at
zero.

**Module:** `ph_neuro.layers.ste_dqt`

```python
from ph_neuro.layers.ste_dqt import stochastic_round
```

### Signature

```python
stochastic_round(x: torch.Tensor) -> torch.Tensor
```

### Parameters

| Arg | Type | Description |
|:----|:-----|:------------|
| `x` | `torch.Tensor` | Float tensor of any shape. |

### Returns

`torch.Tensor` — **int8** tensor of the same shape, values in {-1, 0, +1}.

### Example

```python
import torch

from ph_neuro.layers.ste_dqt import stochastic_round

x = torch.tensor([0.7, -0.2, 0.0, 1.0, -0.9])
print(stochastic_round(x))   # int8 values in {-1, 0, +1}
print(stochastic_round(x).dtype)  # torch.int8
```

### Notes

- The input is clamped to [-1, 1] first.
- Non-deterministic (uses `torch.rand_like`); set a seed for reproducibility.
- Prefer the layer method `apply_stochastic_rounding()` over calling this
  directly during training.

---

## 7. `ste_sign()`

Straight-Through Estimator for `sign()`: forward returns `sign(x)`
(values in {-1, 0, +1}); backward passes the gradient through unchanged.

**Module:** `ph_neuro.layers.ste_linear`

```python
from ph_neuro.layers.ste_linear import ste_sign
```

### Signature

```python
ste_sign(x: torch.Tensor) -> torch.Tensor
```

### Parameters

| Arg | Type | Description |
|:----|:-----|:------------|
| `x` | `torch.Tensor` | Input tensor. |

### Returns

`torch.Tensor` — same shape, values in {-1, 0, +1} in the forward pass;
identity gradient in the backward pass.

### Example

```python
import torch

from ph_neuro.layers.ste_linear import ste_sign

x = torch.tensor([0.5, -1.2, 0.0], requires_grad=True)
y = ste_sign(x)            # [1, -1, 0]
y.sum().backward()
print(x.grad)              # [1, 1, 1] — STE passes gradient through
```

### Notes

- This is the mechanism behind `TernarySTELinear` / `TernarySTEConv2d`.
- `sign(0)` returns `0`, so ternary weights can be exactly sparse.

---

## 8. `dqt_to_inference_model()`

Convert a trained DQT/STE **training** model into an inference-only
`nn.Sequential` built from standard `nn.Conv2d` / `nn.Linear` with frozen
ternary weights (int8 → float). Drops the training-only float buffers.

**Module:** `ph_neuro.models.export`

```python
from ph_neuro.models.export import dqt_to_inference_model
```

### Signature

```python
dqt_to_inference_model(dqt_model: nn.Sequential) -> nn.Sequential
```

### Parameters

| Arg | Type | Description |
|:----|:-----|:------------|
| `dqt_model` | `nn.Sequential` | DQT/STE model with conv/linear layers, BN, ReLU, MaxPool, Flatten, Dropout. |

### Returns

`nn.Sequential` — standard PyTorch layers, frozen, in `eval()` mode, on CPU.

### Raises

`TypeError` — if a module type is not supported (neither a ternary layer nor
a pass-through layer).

### Example

```python
import torch

from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import dqt_to_inference_model

model = dqt_cnn()  # trained DQT CNN
inf = dqt_to_inference_model(model)

with torch.no_grad():
    out = inf(torch.randn(1, 3, 32, 32))   # (1, 10)
```

### Notes

- Supports DQT layers (`TernaryDQTConv2d`, `TernaryDQTLinear`) **and** STE
  layers (`TernarySTEConv2d`, `TernarySTELinear`).
- The result is forced to **CPU** (ONNX tracing is a CPU operation).
- BatchNorm is NOT fused here — that happens inside `export_to_onnx`.

---

## 9. `export_to_onnx()`

Export an inference model to a single self-contained `.onnx` file. Fuses
BatchNorm to element-wise affine first, uses a dynamic batch axis, and
embeds weights inline (`external_data=False`).

**Module:** `ph_neuro.models.export`

```python
from ph_neuro.models.export import export_to_onnx
```

### Signature

```python
export_to_onnx(
    inference_model: nn.Sequential,
    input_shape: tuple[int, ...],
    output_path: str | os.PathLike,
    opset_version: int = 18,
) -> None
```

### Parameters

| Arg | Type | Default | Description |
|:----|:-----|:-------:|:------------|
| `inference_model` | `nn.Sequential` | — | Output of `dqt_to_inference_model()`. |
| `input_shape` | `tuple[int, ...]` | — | E.g. `(1, 3, 32, 32)`. |
| `output_path` | `str \| os.PathLike` | — | Where to write the `.onnx`. |
| `opset_version` | `int` | `18` | ONNX opset (onnxruntime ≥1.16 on ARM supports 18). |

### Returns

`None` — writes the `.onnx` file (creates parent dirs).

### Example

```python
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import dqt_to_inference_model, export_to_onnx

model = dqt_cnn()                       # trained model
inf = dqt_to_inference_model(model)     # 1) rebuild with standard layers
export_to_onnx(inf, (1, 3, 32, 32), "models/dqt_cnn_cifar10.onnx")
```

### Notes

- The result is a single self-contained file — copy it to a Raspberry Pi
  with no companion `.onnx.data`.
- The batch axis is dynamic (`input`/`output` names are fixed).
- Use the one-call helper `export_model_to_onnx()` for the full pipeline
  (convert + export + verify + pack) — see [Bonus helpers](#bonus-export-helpers).

---

## 10. `verify_onnx()`

Verify an exported ONNX model matches its PyTorch source. Runs the same
fused graph through onnxruntime and asserts the outputs match.

**Module:** `ph_neuro.models.export`

```python
from ph_neuro.models.export import verify_onnx
```

### Signature

```python
verify_onnx(
    inference_model: nn.Sequential,
    onnx_path: str | os.PathLike,
    dummy_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool
```

### Parameters

| Arg | Type | Default | Description |
|:----|:-----|:-------:|:------------|
| `inference_model` | `nn.Sequential` | — | The same model passed to `export_to_onnx()`. |
| `onnx_path` | `str \| os.PathLike` | — | Path to the exported `.onnx`. |
| `dummy_input` | `torch.Tensor` | — | Input tensor (any batch size). |
| `rtol` | `float` | `1e-3` | Relative tolerance for `assert_allclose`. |
| `atol` | `float` | `1e-5` | Absolute tolerance for `assert_allclose`. |

### Returns

`bool` — `True` if the ONNX output matches the PyTorch output.

### Raises

- `AssertionError` — outputs differ beyond tolerance.
- `ImportError` — `onnxruntime` not installed (`pip install onnxruntime`).

### Example

```python
import torch

from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import (
    dqt_to_inference_model,
    export_to_onnx,
    verify_onnx,
)

model = dqt_cnn()
inf = dqt_to_inference_model(model)
export_to_onnx(inf, (1, 3, 32, 32), "models/dqt_cnn_cifar10.onnx")
assert verify_onnx(inf, "models/dqt_cnn_cifar10.onnx", torch.randn(2, 3, 32, 32))
print("ONNX verified ✅")
```

### Notes

- On Ampere+ GPUs, disable TF32 before comparing CUDA vs CPU ONNX output
  (the M1.3 export pipeline does this internally) — otherwise reduced-precision
  TF32 can shift borderline logits.
- In practice max|Δ| ≈ 1e-4 on real data with argmax agreement 100%.

---

## 11. `pack_ternary()` / `unpack_ternary()`

2-bit packing for ternary weights: 4 weights per byte (16× smaller than
FP32, 4× smaller than int8). Encoding: `00`=0, `01`=+1, `10`=-1, `11`=unused.

**Module:** `ph_neuro.utils.packing`

```python
from ph_neuro.utils.packing import pack_ternary, unpack_ternary
```

### Signatures

```python
pack_ternary(weights: torch.Tensor) -> torch.Tensor
unpack_ternary(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor
```

### Parameters — `pack_ternary`

| Arg | Type | Description |
|:----|:-----|:------------|
| `weights` | `torch.Tensor` | int8 tensor with values in {-1, 0, +1}. |

**Returns:** int8 tensor with `ceil(numel / 4)` elements (2-bit packed).
**Raises:** `ValueError` if values are outside {-1, 0, +1}.

### Parameters — `unpack_ternary`

| Arg | Type | Description |
|:----|:-----|:------------|
| `packed` | `torch.Tensor` | int8 packed tensor (4 weights/byte). |
| `shape` | `tuple[int, ...]` | Desired output shape (`numel <= packed.numel() * 4`). |

**Returns:** int8 tensor of `shape` with values in {-1, 0, +1}.

### Example

```python
import torch

from ph_neuro.utils.packing import pack_ternary, unpack_ternary

w = torch.tensor([1, 0, -1, 1, -1, 0, 0, 1], dtype=torch.int8)
packed = pack_ternary(w)                 # 2 bytes (8 weights / 4)
restored = unpack_ternary(packed, w.shape)
assert torch.equal(restored, w)          # lossless round-trip ✅
print(packed, restored)
```

### Notes

- Pure-Python bit manipulation — fine for model files, but for **training**
  use the popcount-optimized path (`ph_neuro.utils.popcount`) where relevant.
- The M1.3 export pipeline uses these to write the companion `.ternary` file.

---

## 12. `fuse_bn_layers()`

Replace every `BatchNorm1d` / `BatchNorm2d` with a cheaper
`ElementWiseAffine` layer (`scale * x + bias`) — mathematically identical at
inference, faster, and fully ONNX-compatible. Called internally by
`export_to_onnx()`.

**Module:** `ph_neuro.models.fuse_bn`

```python
from ph_neuro.models.fuse_bn import fuse_bn_layers
```

### Signature

```python
fuse_bn_layers(model: nn.Sequential, inplace: bool = False) -> nn.Sequential
```

### Parameters

| Arg | Type | Default | Description |
|:----|:-----|:-------:|:------------|
| `model` | `nn.Sequential` | — | Model with ternary layers + optional BN. Must be in `eval()` mode. |
| `inplace` | `bool` | `False` | If `True`, modify the original model; if `False`, return a new one. |

### Returns

`nn.Sequential` — same architecture with BN replaced by `ElementWiseAffine`.

### Raises

- `RuntimeError` — model in training mode (call `model.eval()` first).
- `TypeError` — model is not an `nn.Sequential`.

### Example

```python
import torch

from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.fuse_bn import fuse_bn_layers

model = dqt_cnn().eval()
fused = fuse_bn_layers(model)   # BN → element-wise affine

with torch.no_grad():
    print(fused(torch.randn(1, 3, 32, 32)).shape)  # (1, 10)
```

### Notes

- BN in this codebase typically appears after `TernarySTELinear → ReLU →
  BN`, so it cannot be fused *into* the linear layer — the affine replacement
  is the correct approach.
- Outputs may differ from the unfused model by up to ~1e-2 due to
  floating-point reordering; accuracy is preserved within tolerance.

---

## Bonus: export helpers

Additional public helpers in `ph_neuro.models.export` used by the M1.3
pipeline:

| Function | Signature | Returns |
|:---------|:----------|:--------|
| `export_model_to_onnx` | `(dqt_model, input_shape, output_path, opset_version=17, verify=True, packed_path=None, device="cpu")` | Summary `dict` (paths, sizes, verified, max diff). One-call: convert + export + verify + pack. |
| `export_packed_ternary` | `(model, output_path)` | Writes a 2-bit `.ternary` file (magic `PHN3`, per-layer shapes + payloads). |
| `load_packed_ternary` | `(path)` | `list[(class_name, shape, weights)]` — lossless restore. |
| `get_model_params_count` | `(model)` | `int` — number of ternary weights. |
| `estimate_packed_size` | `(model)` | `int` — bytes = `ceil(n_weights / 4)`. |
| `get_model_size_mb` | `(model_or_path)` | `float` — MB of a module or file. |
| `is_ternary_layer` | `(module)` | `bool` — is it a DQT/STE weight layer. |

### One-call export example

```python
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import export_model_to_onnx

summary = export_model_to_onnx(
    dqt_cnn(),
    (1, 3, 32, 32),
    "models/dqt_cnn_cifar10.onnx",
    packed_path="models/dqt_cnn_cifar10.ternary",
    verify=True,
)
print(summary["onnx_size_mb"], summary["packed_size_mb"], summary["verified"])
```

> 📦 Full deployment guide (Raspberry Pi, C API, packing): see
> [`docs/export_guide.md`](export_guide.md).
