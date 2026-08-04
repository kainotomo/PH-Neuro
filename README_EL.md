# PH-Neuro — Tiny Ternary AI (Ελληνικά)

> **Τα μικρότερα μοντέλα βαθιάς μάθησης στον κόσμο. 2-bit βάρη. Εκπαίδευση σε μία GPU. Τρέχουν σε κινητό.**

Τα βάρη είναι **{-1, 0, +1}** (2 bits, 4 ανά byte). Εκπαίδευση χωρίς latent float
scores (**DQT**) — 4.5× λιγότερη μνήμη εκπαίδευσης από το BitNet. Αραιή ενεργοποίηση
(**MoE**). Ένα CNN με 4.27M παραμέτρους εκπαιδεύεται με ~350 MB μνήμης GPU και
"χωράει" σε αρχείο **1 MB**.

**Διάβασε πρώτα:** [🚀 Quickstart](docs/quickstart.md) · [📖 API Reference](docs/api.md) · [📊 Benchmarks](docs/benchmarks.md) · [English README](README.md)

---

## 🔥 Τελευταία Αποτελέσματα (Αύγουστος 2026)

| Milestone | Αποτέλεσμα | Κατάσταση | Λεπτομέρειες |
|:----------|:-----------|:---------:|:-------------|
| **M1.1** — DQT CNN σε CIFAR-10 | **78.98%** (+2.89pp vs STE) | ✅ CONDITIONAL GO | [E020–E021.3](research/docs/RESEARCH_SUMMARY.md) |
| **M1.2** — DQT CNN σε CIFAR-100 | **54.15%** (+15.95pp vs STE) | 🟡 CONDITIONAL GO | [E022](research/docs/RESEARCH_SUMMARY.md) |
| **M1.3** — Εξαγωγή ONNX | 3 μοντέλα, <1 MB packed, verified | ✅ GO | [export guide](docs/export_guide.md) |
| **M1.4** — Production README + docs | **← εδώ είμαστε** | 🟡 | αυτό το repo |

**Σταθερό μοτίβο:** σε ίδια αρχιτεκτονική, το DQT νικά το STE baseline κατά
**+2.89pp** (CIFAR-10) και **+15.95pp** (CIFAR-100).

---

## ⚡ Quickstart σε 5 λεπτά

```bash
pip install -e . && pip install onnxruntime   # από τη ρίζα του repo
```

```python
import torch, torch.nn as nn, torch.nn.functional as F
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.training.data import get_mnist_loaders

model = nn.Sequential(TernaryDQTLinear(784, 512), nn.ReLU(), TernaryDQTLinear(512, 10))
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
train, test = get_mnist_loaders(batch_size=128)

for x, y in train:
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    for m in model.modules():          # ★ ΤΟ ΒΗΜΑ ΤΟΥ DQT
        if isinstance(m, TernaryDQTLinear):
            m.apply_stochastic_rounding()
```

Αυτός είναι όλος ο κύκλος εκπαίδευσης DQT — **98.23% στο MNIST**. Πλήρη
παραδείγματα (MLP, CNN, export, CPU inference): [docs/quickstart.md](docs/quickstart.md).

---

## 🧠 Πυρήνας Τεχνολογίας

| Πυλώνας | Τι κάνει | Κατάσταση | Μετρική |
|:--------|:---------|:---------:|:--------|
| **DQT** | Εκπαίδευση τριμικών βαρών *χωρίς* latent float scores | ✅ | 4.5× λιγότερη μνήμη, 98.2% MNIST |
| **MoE** | Αραιή ενεργοποίηση — μόνο `top_k/n` experts τρέχουν | ✅ | +2.5pp vs dense |
| **Ternary** | Βάρη {-1, 0, +1}, 2 bits/βάρος | ✅ | 8× μικρότερα από FP16 |

Στόχος: **1B παράμετροι → 200 MB σε δίσκο → τρέχει σε κινητό.** Βλέπε [GOALS.md](GOALS.md).

---

## 📦 Model Zoo

| Μοντέλο | Dataset | Ακρίβεια | Weights (2-bit) | ONNX | Packed Size |
|:--------|:--------|:--------:|:----------------|:-----|:-----------:|
| `dqt_cnn` | CIFAR-10 | 78.98% | [.ternary](models/dqt_cnn_cifar10.ternary) | [.onnx](models/dqt_cnn_cifar10.onnx) | 1.0 MB |
| `dqt_cnn_cifar100` | CIFAR-100 | 54.15% | [.ternary](models/dqt_cnn_cifar100.ternary) | [.onnx](models/dqt_cnn_cifar100.onnx) | 615 KB |
| `ste_mlp` | MNIST | 98.23% | [.ternary](models/ste_mlp_mnist.ternary) | [.onnx](models/ste_mlp_mnist.onnx) | 132 KB |

Περισσότερα benchmarks: [docs/benchmarks.md](docs/benchmarks.md).

---

## 📖 API Reference

Κάθε public layer/συνάρτηση τεκμηριώνεται (signature, args, runnable
examples, gotchas) στο **[docs/api.md](docs/api.md)**:

`TernaryDQTLinear` · `TernaryDQTConv2d` · `TernaryDQTMoELayer` ·
`TernarySTELinear` · `TernarySTEConv2d` · `stochastic_round()` ·
`ste_sign()` · `dqt_to_inference_model()` · `export_to_onnx()` ·
`verify_onnx()` · `pack_ternary()` / `unpack_ternary()` · `fuse_bn_layers()`

---

## 🚀 Εξαγωγή σε ONNX

```python
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import dqt_to_inference_model, export_to_onnx, verify_onnx
import torch

inf = dqt_to_inference_model(dqt_cnn())
export_to_onnx(inf, (1, 3, 32, 32), "models/dqt_cnn_cifar10.onnx")
assert verify_onnx(inf, "models/dqt_cnn_cifar10.onnx", torch.randn(2, 3, 32, 32))
```

Πλήρης οδηγός (packed `.ternary`, Raspberry Pi, C API):
[docs/export_guide.md](docs/export_guide.md).

---

## 🧪 Testing & Roadmap

```bash
.venv/bin/python -m pytest tests/ -v        # 616 tests
```

Φάσεις: **0** Research ✅ · **1** Production DQT 🟡 (M1.4 σε εξέλιξη) ·
**2** Tiny Transformer ⬜ · **3** MVP ⬜ · **4** Scale ⬜ · **5** Platform ⬜.
Αναλυτικό roadmap: [ROADMAP.md](ROADMAP.md) · Όραμα: [GOALS.md](GOALS.md).

---

## License

MIT — βλέπε [LICENSE](LICENSE).

---

> *"Τα μικρότερα μοντέλα βαθιάς μάθησης στον κόσμο."*
