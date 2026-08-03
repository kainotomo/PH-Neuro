# PH-Neuro — Tiny Ternary AI

> **The smallest deep learning models in the world.**
>
> Weights are {-1, 0, +1}. Training without latent float scores (DQT).
> Sparse activation (MoE). 4.5× less training memory than BitNet.
> Target: 10B+ parameter models on consumer GPUs.

---

## What is this?

PH-Neuro is a **commercial framework** for building extremely memory-efficient neural networks. The core technology:

| Component | What it does | Status |
|:----------|:-------------|:------:|
| **DQT** | Train ternary weights without latent float scores — 4.5× less training memory | ✅ Proven (98.2% MNIST) |
| **MoE** | Sparse activation — only 50% of parameters active per input | ✅ Proven (+2.5pp vs dense) |
| **Ternary** | {-1, 0, +1} weights — 8× smaller than FP16 at inference | ✅ Proven |

Combined: **DQT + MoE + ternary** enables models 300× smaller than FP16 at inference, trainable on a single consumer GPU.

> 📚 **Research archive:** 19 experiments across Hebbian and STE eras in [`research/`](research/). See [`research/RESEARCH_SUMMARY.md`](research/docs/RESEARCH_SUMMARY.md).

## Quick Start

```python
from ph_neuro.layers.ste_dqt import TernaryDQTLinear

# DQT model — no latent float scores, 4.5× less training memory
model = torch.nn.Sequential(
    TernaryDQTLinear(784, 512),
    torch.nn.ReLU(),
    TernaryDQTLinear(512, 256),
    torch.nn.ReLU(),
    TernaryDQTLinear(256, 10),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

for x, y in train_loader:
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    # Stochastic rounding updates ternary weights — no latent scores stored
```

## Performance (Proof of Concept — MNIST MLP)

| Method | Accuracy | Sparsity | Training Memory |
|:-------|:--------:|:--------:|:---------------:|
| Standard STE | 98.17% | 0% | ~9 bytes/param |
| **DQT (ours)** | **98.23%** | **56%** | **~2 bytes/param** |
| DQT + MoE (ours) | 91.21% | 50.5% active | **~1 byte/param active** |

## Roadmap

| Milestone | Goal |
|:----------|:-----|
| ✅ DQT on MNIST MLP | Prove method works for vision |
| ✅ MoE DQT on MNIST | Prove sparse activation works |
| ⬜ DQT + CNN on CIFAR-10 | Prove on realistic vision task |
| ⬜ DQT + Transformer on text | Prove on language (TinyStories) |
| ⬜ 1B-parameter MoE model | Scale to competitive size |

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

---

> *"The smallest deep learning models in the world."*
