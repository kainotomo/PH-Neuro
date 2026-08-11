# PH-Neuro — Tiny Ternary AI

> **The smallest deep learning models in the world. 2-bit weights. Train on one GPU. Run on a phone.**

Weights are **{-1, 0, +1}** (2 bits, 4 per byte). Train without latent float
scores (**DQT**) — 4.5× less training memory than BitNet. Sparse activation
(**MoE**). A 4.27M-parameter CNN trains on ~350 MB of GPU memory and ships
as a **1 MB** file.

**Read this first:** [🚀 Quickstart](docs/quickstart.md) · [📖 API Reference](docs/api.md) · [📊 Benchmarks](docs/benchmarks.md) · [🇬🇷 Ελληνικά](README_EL.md)

---

## 🔥 Latest Results (August 2026)

| Milestone | Result | Status | Details |
|:----------|:-------|:------:|:--------|
| **M2.1** — DQT Transformer 102M | **ppl 11.35** TinyStories | ✅ GO | [E025](research/docs/RESEARCH_SUMMARY.md) |
| **M2.2** — DQT Transformer 253M | **Stable ✅** WikiText-2 | 🟡 SCIENTIFIC GO | [E026](research/docs/RESEARCH_SUMMARY.md) |
| **M2.3** — MoE DQT Transformer | **ppl 14.08**, 265M/190M active | ✅ GO | [E027](research/docs/RESEARCH_SUMMARY.md) |
| **M2.4** — On-device demo | **21-25 tok/s CPU**, 11 MB packed | ✅ GO | [E028](research/docs/RESEARCH_SUMMARY.md) |
| **M2.5** — Public demo | **Gradio app**, 3 models, 26 MB total | ✅ GO | [E029](research/docs/RESEARCH_SUMMARY.md) |
| **M2.6** — 8-bit AdamW + bf16 | 10 scripts converted, MNIST smoke OK | ✅ DONE | [ROADMAP §2.5](ROADMAP.md) |
| **M2.7** — Flash Attention / SDPA | Transformer attention, tests pass | ✅ DONE | [ROADMAP §2.5](ROADMAP.md) |
| **M2.8** — 1B DQT Transformer | **1.02B ternary, stable, 8.04 GB peak** | ✅ GO | [E030](research/docs/experiments/E030-m2-9-memory-benchmark.md) |

**Phase 2 COMPLETE ✅** — 5/5 gate milestones closed. **Phase 2.5 (Memory
Optimization Sprint) COMPLETE ✅** — 8-bit AdamW + bf16 + Flash Attention
cut training memory ~22–31% (253M: 6.5 → 5.0 GB) and scaled DQT to the
first **1B-param model on an RTX 4060 8 GB** (8.04 GB peak) — 3.4× the old
300M ceiling, zero DQT-autograd changes.

---

## ⚡ 5-Minute Quickstart

```bash
pip install -e . && pip install onnxruntime bitsandbytes  # from repo root
```

```python
import torch, torch.nn as nn, torch.nn.functional as F, bitsandbytes as bnb
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.training.data import get_mnist_loaders

model = nn.Sequential(TernaryDQTLinear(784, 512), nn.ReLU(), TernaryDQTLinear(512, 10))
opt = bnb.optim.AdamW8bit(model.parameters(), lr=0.01)  # ★ 8-bit Adam = 75% less VRAM
train, test = get_mnist_loaders(batch_size=128)

for x, y in train:
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    for m in model.modules():          # ★ THE DQT STEP
        if isinstance(m, TernaryDQTLinear):
            m.apply_stochastic_rounding()
```

That is the whole DQT training loop — **98.23% on MNIST**. Full examples
(MLP, CNN, export, CPU inference): [docs/quickstart.md](docs/quickstart.md).

---

## 🎮 Public Demo (M2.5)

> **3 models, <30 MB total, running in your browser on CPU.** No GPU. No cloud.

```bash
pip install -e . gradio onnxruntime
python scripts/run_m2_5_demo.py            # → http://localhost:7860
```

Three tabs: 📝 **Text Generation** (DQT Transformer 102M — TinyStories,
live token streaming, temperature / top-k) · 🖼️ **Image Classification**
(DQT CNNs, CIFAR-10 & CIFAR-100, upload or webcam, top-3 with confidence) ·
📊 **Benchmarks** (all models vs GPT-2 / TF-Lite).

The 102M-parameter transformer ships as a **25 MB** 2-bit packed file and
generates ~20+ tokens/sec on CPU. Retrain + export everything yourself:

```bash
bash scripts/run_m2_5_demo.sh              # full: train → export → demo
bash scripts/run_m2_5_demo.sh train        # 3 models (~2 h on RTX 4060)
bash scripts/run_m2_5_demo.sh export       # ONNX + 2-bit packed
bash scripts/run_m2_5_demo.sh demo         # gradio @ :7860
```

Launch blog post: [docs/blog.md](docs/blog.md).

---

## 🧠 Core Technology

| Pillar | What it does | Status | Metric |
|:-------|:-------------|:------:|:-------|
| **DQT** | Train ternary weights *without* latent float scores | ✅ | 4.5× less training memory, 98.2% MNIST |
| **MoE** | Sparse activation — only `top_k/n` experts run | ✅ | +2.5pp vs dense |
| **Ternary** | {-1, 0, +1} weights, 2 bits/weight | ✅ | 8× smaller than FP16 |

Combined target: **1B parameters → 200 MB on disk → runs on a phone.**
See [GOALS.md](GOALS.md) for the full vision.

---

## 📦 Model Zoo

All artifacts are in [`models/`](models/). ONNX files are self-contained
(dynamic batch); `.ternary` files are the 2-bit packed weights (4/byte).

| Model | Dataset | Accuracy | Weights (2-bit) | ONNX | Packed Size |
|:------|:--------|:--------:|:----------------|:-----|:-----------:|
| `dqt_cnn` | CIFAR-10 | 78.98% | [.ternary](models/dqt_cnn_cifar10.ternary) | [.onnx](models/dqt_cnn_cifar10.onnx) | 1.0 MB |
| `dqt_cnn_cifar100` | CIFAR-100 | 54.15% | [.ternary](models/dqt_cnn_cifar100.ternary) | [.onnx](models/dqt_cnn_cifar100.onnx) | 615 KB |
| `ste_mlp` | MNIST | 98.23% | [.ternary](models/ste_mlp_mnist.ternary) | [.onnx](models/ste_mlp_mnist.onnx) | 132 KB |

More benchmarks (training time, GPU memory, STE comparison):
[docs/benchmarks.md](docs/benchmarks.md).

---

## 📖 API Reference

Every public layer/function is documented with signatures, args, runnable
examples and gotchas in **[docs/api.md](docs/api.md)**.

| Layer / Function | Module | Use |
|:-----------------|:-------|:----|
| `TernaryDQTLinear` | `ph_neuro.layers.ste_dqt` | ⭐ DQT MLP layer |
| `TernaryDQTConv2d` | `ph_neuro.layers.ste_dqt_conv` | ⭐ DQT CNN layer |
| `TernaryDQTMoELayer` | `ph_neuro.layers.ste_dqt_moe` | ⭐ Sparse MoE (top-K experts) |
| `TernarySTELinear` | `ph_neuro.layers.ste_linear` | STE fallback (latent scores) |
| `TernarySTEConv2d` | `ph_neuro.layers.ste_conv` | STE fallback conv |
| `stochastic_round()` | `ph_neuro.layers.ste_dqt` | Core DQT discretization |
| `ste_sign()` | `ph_neuro.layers.ste_linear` | STE sign for `sign()` backward |
| `dqt_to_inference_model()` | `ph_neuro.models.export` | Training → inference graph |
| `export_to_onnx()` | `ph_neuro.models.export` | Export to `.onnx` |
| `verify_onnx()` | `ph_neuro.models.export` | ONNX ≡ PyTorch check |
| `pack_ternary()` / `unpack_ternary()` | `ph_neuro.utils.packing` | 2-bit packing |
| `fuse_bn_layers()` | `ph_neuro.models.fuse_bn` | BN → element-wise affine |

Model factories: `dqt_cnn()`, `dqt_cnn_cifar100()` in
`ph_neuro.models.dqt_models`; `ste_mlp()`, `ste_cnn()` in
`ph_neuro.models.ste_models`.

---

## 🚀 Export to ONNX

Trained DQT models ship as tiny CPU-runnable ONNX files (Raspberry Pi-ready):

```python
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import dqt_to_inference_model, export_to_onnx, verify_onnx
import torch

inf = dqt_to_inference_model(dqt_cnn())
export_to_onnx(inf, (1, 3, 32, 32), "models/dqt_cnn_cifar10.onnx")
assert verify_onnx(inf, "models/dqt_cnn_cifar10.onnx", torch.randn(2, 3, 32, 32))
```

Full guide — packed `.ternary`, Raspberry Pi deployment, C API:
[docs/export_guide.md](docs/export_guide.md).

---

## 🧪 Testing

```bash
.venv/bin/python -m pytest tests/ -v        # 616 tests
make test-quick                              # skip slow tests
```

Test coverage: layers (DQT, STE, MoE, hysteresis, LoRA, fused-BN), core
(Hebbian rules, ternary tensor), training, and M1.1–M1.3 integration suites.

---

## 📚 Research Archive

19 experiments across three eras — Hebbian, STE, DQT:

- Summary: [research/docs/RESEARCH_SUMMARY.md](research/docs/RESEARCH_SUMMARY.md)
- Full write-ups: [research/docs/experiments/](research/docs/experiments/)

**Headline findings:** Hebbian learning cannot train hidden layers (88%
single-layer ceiling) → STE solves it (98% MNIST) → DQT beats STE and removes
latent scores (the current production path).

---

## 🗺️ Roadmap

| Phase | Milestone | Status |
|:------|:----------|:------:|
| 0 | Research foundation (19 experiments) | ✅ Closed |
| 1 | Production DQT (M1.1–M1.5) | ✅ Closed |
| 2 | Tiny Transformer (M2.1–M2.5) | ✅ Closed |
| **2.5** | **Memory Optimization Sprint (8-bit Adam + bf16 → 1B+)** | ✅ **Complete** |
| 3 | MVP & first customers — `pip install ph-neuro` | ⬜ |
| 4 | Scale — pre-seed, 10B MoE model | ⬜ |
| 5 | Commercial platform | ⬜ |

Full roadmap with go/no-go gates: [ROADMAP.md](ROADMAP.md) ·
Vision & goals: [GOALS.md](GOALS.md).

---

## License

MIT — see [LICENSE](LICENSE).

---

> *"The smallest deep learning models in the world."*
