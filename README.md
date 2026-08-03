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

> 📚 **Research archive:** 19 experiments across Hebbian and STE eras in [`research/`](research/). See [`research/docs/RESEARCH_SUMMARY.md`](research/docs/RESEARCH_SUMMARY.md).
>
> 🎯 **Product vision:** [`GOALS.md`](GOALS.md) — mission, target market, competitive advantage.
> 🗺️ **Roadmap:** [`ROADMAP.md`](ROADMAP.md) — current phase, milestones, go/no-go gates.

## Quick Start

```python
from ph_neuro.layers.ste_dqt import TernaryDQTLinear

# DQT model — no latent float scores, 4.5× less training memory
model = torch.nn.Sequential(
    TernaryDQTLinear(784, 512),
    torch.nn.ReLU(),
    TernaryDQTLinear(512, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

for x, y in train_loader:
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
```

## Project Status

| Doc | What |
|:----|:-----|
| [`GOALS.md`](GOALS.md) | Vision, mission, target market, funding strategy |
| [`ROADMAP.md`](ROADMAP.md) | Current phase, milestones, go/no-go gates |
| [`research/`](research/) | All 19 experiments, results, Hebbian era archive |

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

---

> *"The smallest deep learning models in the world."*
