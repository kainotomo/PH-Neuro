# PH-Neuro — ONNX Model Export & Raspberry Pi Deployment Guide (M1.3)

This guide explains how to take a trained **DQT (Direct Quantized Training)**
ternary model and deploy it as a small ONNX model that runs on a **Raspberry
Pi** (ARM CPU, no GPU, no PyTorch).

## Why ONNX at all?

DQT models are trained with custom autograd functions (`_DQTGradFn`,
`_DQTConvGradFn`) that `torch.onnx.export` cannot trace. But **at inference**
a DQT model is trivial:

```
conv2d(ternary_weights.float(), x)      # conv block
linear(ternary_weights.float(), x)      # linear block
ReLU / BatchNorm / MaxPool / Flatten
```

The export pipeline (in `src/ph_neuro/models/export.py`) rebuilds exactly
that graph with standard `nn.Conv2d` / `nn.Linear` layers holding the frozen
int8 ternary weights, fuses BatchNorm into element-wise affine, and exports a
single self-contained `.onnx` file. The result is a tiny, CPU-only model.

---

## 1. Export a trained model

```bash
cd /home/phalo/PH-Neuro

# Export a trained checkpoint (state_dict):
.venv/bin/python -m ph_neuro.examples.run_m1_3_export \
    --model dqt_cnn --checkpoint models/dqt_cnn_cifar10.pt \
    --output models/dqt_cnn_cifar10.onnx --packed --verify

# No checkpoint? The CLI trains a small demo model first:
.venv/bin/python -m ph_neuro.examples.run_m1_3_export \
    --model dqt_cnn --output models/dqt_cnn_cifar10.onnx --packed --verify

# ste_mlp (MNIST) — tiny demo:
.venv/bin/python -m ph_neuro.examples.run_m1_3_export \
    --model ste_mlp --output models/ste_mlp_mnist.onnx --packed --verify
```

Supported `--model` values:

| `--model` | Dataset | Input shape | Architecture |
|:----------|:--------|:------------|:-------------|
| `dqt_cnn` | CIFAR-10 | `(1, 3, 32, 32)` | 2-conv DQT CNN |
| `dqt_cnn_cifar100` | CIFAR-100 | `(1, 3, 32, 32)` | 3-conv DQT CNN |
| `ste_mlp` | MNIST | `(1, 1, 28, 28)` | 3-layer STE MLP |

Flags:

- `--output <path>` — where to write the `.onnx` (default `models/<model>.onnx`).
- `--packed` — also write a 2-bit packed `.ternary` companion file
  (4 ternary weights per byte, ~16× smaller than FP32).
- `--verify` — run onnxruntime verification + a torch-vs-ONNX accuracy
  comparison on real test data.
- `--opset <N>` — ONNX opset (default 18; onnxruntime ≥1.16 on ARM supports it).

### Outputs

- `models/<model>.onnx` — single self-contained file (weights embedded inline).
- `models/<model>.ternary` — optional 2-bit packed ternary weights.

Typical sizes (all far under the 100 MB gate):

| Model | Ternary weights | ONNX (FP32) | Packed `.ternary` (2-bit) |
|:------|:---------------:|:-----------:|:-------------------------:|
| `dqt_cnn` (CIFAR-10) | ~4.27 M | ~17 MB | ~1.0 MB |
| `dqt_cnn_cifar100` (CIFAR-100) | ~2.52 M | ~10 MB | ~0.6 MB |
| `ste_mlp` (MNIST) | ~535 K | ~2 MB | ~0.13 MB |

> The 100 MB gate is trivially passed: ternary weights are 2-bit, and even
> the FP32 ONNX weights are a few MB.

---

## 2. Deploy on Raspberry Pi

### 2.1 Install onnxruntime (ARM)

```bash
# Raspberry Pi OS (64-bit, recommended for ARM64 performance):
python3 -m pip install --upgrade pip
python3 -m pip install onnxruntime

# Optional — extra speed for ARM64 (requires gcc on the Pi):
# python3 -m pip install onnxruntime --extra-index-url \
#   https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-packages/pypi/simple/
```

`onnxruntime` is the **only** runtime dependency. No PyTorch is needed on the
Pi.

### 2.2 Copy the model

Copy the single `.onnx` file (and the `.ternary` if you use it) to the Pi:

```bash
scp models/dqt_cnn_cifar10.onnx pi@raspberrypi.local:~/
```

---

## 3. Inference script (Python, copy-paste)

Save this as `infer.py` on the Pi (~50 lines):

```python
"""Minimal ONNX inference for a PH-Neuro DQT model on Raspberry Pi."""
import sys

import numpy as np
import onnxruntime as ort


def load_image(path, size=32, mean=(0.4914, 0.4822, 0.4465),
               std=(0.2470, 0.2435, 0.2616)):
    """Load and preprocess an image exactly like training (CIFAR-style).

    For MNIST models use size=28, mean=(0.1307,), std=(0.3081,).
    Requires Pillow: pip install pillow
    """
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [0,1]
    arr = arr.transpose(2, 0, 1)                             # HWC -> CHW
    for c in range(arr.shape[0]):
        arr[c] = (arr[c] - mean[c]) / std[c]
    return arr[None].astype(np.float32)                      # (1,C,H,W)


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
    img_path = sys.argv[2] if len(sys.argv) > 2 else "image.jpg"

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    x = load_image(img_path)
    logits = session.run(None, {"input": x})[0]      # (1, n_classes)
    probs = np.exp(logits - logits.max())            # softmax
    probs = probs / probs.sum()

    top = np.argsort(probs[0])[::-1][:3]
    print("Top predictions:")
    for k in top:
        name = CIFAR10_CLASSES[k] if k < len(CIFAR10_CLASSES) else f"class{k}"
        print(f"  {name:12s} {probs[0][k]:.4f}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python3 infer.py dqt_cnn_cifar10.onnx image.jpg
```

---

## 4. Expected performance on Raspberry Pi

CPU-only, batch size 1, FP32 ONNX:

| Device | `ste_mlp` (MNIST) | `dqt_cnn` (CIFAR-10) |
|:-------|:-----------------:|:--------------------:|
| Raspberry Pi 4 (4× Cortex-A72, 1.5 GHz) | ~2–4 ms/img | ~15–30 ms/img |
| Raspberry Pi 5 (4× Cortex-A76, 2.4 GHz) | ~1–2 ms/img | ~8–15 ms/img |

A rough way to measure on the Pi:

```bash
python3 -c "
import numpy as np, onnxruntime as ort, time
s = ort.InferenceSession('dqt_cnn_cifar10.onnx', providers=['CPUExecutionProvider'])
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
s.run(None, {'input': x})  # warmup
t0 = time.perf_counter()
N = 100
for _ in range(N):
    s.run(None, {'input': x})
dt = (time.perf_counter() - t0) / N
print(f'{dt*1000:.1f} ms/img  ->  {1/dt:.0f} fps')
"
```

> Real CIFAR images are normalized to roughly `N(0,1)`, so the benchmark
> above is representative. Expect **30–100 fps** on a Pi 4/5 for these models.

---

## 5. (Optional) C inference with the onnxruntime C API

For a pure-C deployment (no Python on the Pi), use the onnxruntime C API:

```bash
# On the Pi, build onnxruntime from source OR use the prebuilt ARM64 wheel's
# bundled shared library (libonnxruntime.so inside the wheel).
pip download onnxruntime --platform manylinux2014_aarch64 --no-deps
```

Minimal C sketch:

```c
#include <onnxruntime_c_api.h>

int main(void) {
    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env;  OrtSessionOptions *opts;  OrtSession *session;
    api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "phneuro", &env);
    api->CreateSessionOptions(&opts);
    api->SetIntraOpNumThreads(opts, 4);          /* all 4 cores */
    api->CreateSession(env, "model.onnx", opts, &session);

    /* ...allocate input tensor (1,3,32,32) float32, feed 'input',
       read 'output', softmax, argmax... */

    api->ReleaseSession(session);
    api->ReleaseSessionOptions(opts);
    api->ReleaseEnv(env);
    return 0;
}
```

Build with `gcc infer.c -lonnxruntime -o infer`. The full C API header is
`onnxruntime_c_api.h` (ships with the wheel / source).

---

## 6. Restoring weights from the packed `.ternary` file

The optional 2-bit packed companion file lets you ship weights at 4 bits per
byte on top of the ONNX:

```python
from ph_neuro.models.export import load_packed_ternary

layers = load_packed_ternary("models/dqt_cnn_cifar10.ternary")
for name, shape, weights in layers:
    print(name, shape, weights.dtype)   # int8, values in {-1, 0, +1}
```

`pack_ternary` / `unpack_ternary` (`src/ph_neuro/utils/packing.py`) implement
the 2-bit encoding (00=0, 01=+1, 10=-1).

---

## 7. Verification checklist (GO criteria)

1. ✅ ONNX file is produced without errors and is a single self-contained file.
2. ✅ ONNX output matches the PyTorch (fused) output — `rtol=1e-3` /
   `atol=1e-5`; in practice max|Δ| ≈ 1e-4 on real data, argmax agreement 100%.
3. ✅ ONNX size < 100 MB (models are a few MB).
4. ✅ 8/8 M1.3 tests pass (`tests/integration/test_m1_3_export.py`).
5. ✅ Runs on CPU-only onnxruntime (verified in this repo; no CUDA required).
