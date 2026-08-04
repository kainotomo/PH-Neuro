# PH-Neuro — Quickstart

> Get from zero to a trained + exported ternary model in ~5 minutes.
> Every example below is **copy-paste runnable** (CPU is fine for the
> MNIST and export examples).

---

## 1. Installation

### Option A — install the package (editable, for development)

```bash
cd /home/phalo/PH-Neuro
python -m venv .venv && source .venv/bin/activate   # optional, recommended
pip install -e .                                    # core: torch, torchvision, numpy
pip install -e ".[dev,examples]"                    # + pytest, ruff, mypy, matplotlib, tqdm
pip install onnxruntime                             # needed for ONNX export/verify
```

### Option B — just run from source

```bash
cd /home/phalo/PH-Neuro
export PYTHONPATH=src
python your_script.py
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0.

**Verify the install:**

```python
import ph_neuro
print(ph_neuro.__version__)   # 0.1.0.dev0
```

---

## 2. MNIST DQT MLP — full training example

A 2-layer DQT MLP that reaches **98.23%** on MNIST. The only DQT-specific
mechanic is `apply_stochastic_rounding()` after every `optimizer.step()`.

Save as `train_mnist_dqt.py` and run:

```bash
python train_mnist_dqt.py
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.training.data import get_mnist_loaders

# ── Model: DQT MLP (ternary weights, no latent float scores) ──────────
model = nn.Sequential(
    TernaryDQTLinear(784, 512),
    nn.ReLU(),
    TernaryDQTLinear(512, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
train_loader, test_loader = get_mnist_loaders(batch_size=128, root="data")


def dqt_round(model):
    """★ THE DQT STEP: discretize float buffer -> ternary after each step."""
    for module in model.modules():
        if isinstance(module, TernaryDQTLinear):
            module.apply_stochastic_rounding()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).argmax(1).eq(y).sum().item()
            total += y.size(0)
    return correct / total


# ── Train ──────────────────────────────────────────────────────────────
for epoch in range(1, 6):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        dqt_round(model)                      # ← DQT discretization
    print(f"epoch {epoch}/5  test acc {evaluate(model, test_loader):.2%}")
```

> 🔁 For full accuracy, train ~150 epochs with a cosine LR schedule and
> anneal to deterministic rounding for the final 20% — see
> [`examples/run_m1_1_dqt_cifar10.py`](../src/ph_neuro/examples/run_m1_1_dqt_cifar10.py).

---

## 3. CIFAR-10 DQT CNN — full training example

Uses the ready-made `dqt_cnn()` factory (2-conv DQT CNN, **78.98%** on
CIFAR-10). Save as `train_cifar10_dqt.py`:

```bash
python train_cifar10_dqt.py
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.training.data import get_cifar10_loaders

model = dqt_cnn()                                   # 2-conv DQT CNN (4.27M ternary weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
train_loader, test_loader = get_cifar10_loaders(batch_size=128, root="data")

DQT_LAYERS = (TernaryDQTConv2d, TernaryDQTLinear)


def dqt_round(model, stochastic=True):
    for module in model.modules():
        if isinstance(module, DQT_LAYERS):
            module.apply_stochastic_rounding() if stochastic else module.apply_deterministic_rounding()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).argmax(1).eq(y).sum().item()
            total += y.size(0)
    return correct / total


for epoch in range(1, 21):                          # 20 epochs ≈ a few minutes on GPU
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        dqt_round(model, stochastic=epoch < 16)     # anneal: deterministic for last 20%
    scheduler.step()
    print(f"epoch {epoch}/20  test acc {evaluate(model, test_loader):.2%}")
```

> For CIFAR-100 use `dqt_cnn_cifar100()` from the same module (3-conv,
> **54.15%** — +15.95pp over the STE baseline).

---

## 4. Export to ONNX — 5 lines

Convert a trained DQT model to a CPU-runnable ONNX file and verify it:

```python
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import dqt_to_inference_model, export_to_onnx, verify_onnx
import torch

inf = dqt_to_inference_model(dqt_cnn())                    # rebuild with standard layers
export_to_onnx(inf, (1, 3, 32, 32), "models/dqt_cnn_cifar10.onnx")  # single self-contained file
assert verify_onnx(inf, "models/dqt_cnn_cifar10.onnx", torch.randn(2, 3, 32, 32))
```

> 📦 Full guide (packed `.ternary`, Raspberry Pi, C API): see
> [`docs/export_guide.md`](export_guide.md).

---

## 5. Inference on CPU — 5 lines

Run the exported ONNX with `onnxruntime` — no PyTorch required:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("models/dqt_cnn_cifar10.onnx", providers=["CPUExecutionProvider"])
x = np.random.randn(1, 3, 32, 32).astype(np.float32)      # normalized input image
logits = session.run(None, {"input": x})[0]               # (1, 10)
print(logits.argmax(1), np.exp(logits).sum())             # class + sanity check
```

Expected CPU performance on a Raspberry Pi 5: **~8–15 ms/image**
(~30–100 fps) for the CIFAR-10 CNN. See
[`docs/export_guide.md`](export_guide.md) for the full deployment script.

---

## Next steps

- 📖 **API reference** for every layer/function: [`docs/api.md`](api.md)
- 📊 **Benchmarks** (accuracy, params, sizes, training time): [`docs/benchmarks.md`](benchmarks.md)
- 🧠 **Research archive** (19 experiments): [`research/docs/RESEARCH_SUMMARY.md`](../research/docs/RESEARCH_SUMMARY.md)
- 🗺️ **Product roadmap**: [`ROADMAP.md`](../ROADMAP.md)
