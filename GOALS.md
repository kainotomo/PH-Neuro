# PH-Neuro — Goals & Vision

> **Give any AI model a brain. Continual learning without backpropagation. No forgetting.**
> Last updated: 2026-08-12

---

## Vision

**AI that learns like a brain — born with innate knowledge, adapts throughout life, never forgets.**

Every AI model today is a static snapshot. You train it once, ship it, and it never learns again. Fine-tuning requires backpropagation through the entire model and causes catastrophic forgetting of previous knowledge. This is not how intelligence works.

The brain is born pre-wired (evolution's pre-training) and learns continuously through local synaptic plasticity — no global backpropagation, no catastrophic forgetting. PH-Neuro Brain brings this paradigm to deep learning: wrap any pre-trained open-source model with local plasticity, and it continues learning from experience without forgetting what it already knows.

---

## Mission

Build a **brain-like continual learning platform** where:

1. **Any pre-trained model becomes a "born brain"** — GPT-2, SmolLM, Llama, ViT, or your own
2. **Learning is local, not global** — no backpropagation through frozen layers, only local Hebbian/neuromodulated updates
3. **No catastrophic forgetting** — plastic weights adapt to new domains while structural weights preserve original knowledge
4. **Runs on-device** — ternary plastic weights (2-bit), popcount math, single GPU or CPU
5. **Biologically grounded** — surprise-modulated plasticity, memory consolidation, natural decay

---

## Core Technology

### The Product: Brain Wrapper

| Pillar | What | Status | Key Metric |
|:-------|:-----|:------:|:-----------|
| **Brain Wrapper** | Local plasticity on frozen pre-trained models | 🔬 Research | Continual adaptation, zero forgetting |

### The Pre-Training Toolkit (Infrastructure)

| Pillar | What | Status | Key Metric |
|:-------|:-----|:------:|:-----------|
| **DQT** | Train ternary weights without latent float scores | ✅ Proven | 98.2% MNIST, 4.5× less training memory |
| **MoE** | Sparse activation — only top_k/n experts run | ✅ Proven | +2.5pp accuracy vs dense |
| **Ternary** | {-1, 0, +1} weights, 2 bits/weight | ✅ Proven | 8× smaller than FP16 |

Combined target: **1B parameters → 200MB on disk → runs on a phone.**

---

## Target Product

### Brain Wrapper — The Real Product

A Python library that wraps any HuggingFace model and gives it a brain:

```
model = AutoModelForCausalLM.from_pretrained("gpt2")
brain = BrainWrapper(model, plasticity="ternary_hebbian")
brain.learn(new_experiences)       # local updates, no backprop
brain.generate(prompt)             # inference with plastic weights
brain.consolidate()                # sleep-like memory transfer
```

- **Input:** Any pre-trained open-source model (GPT-2, SmolLM, Llama, ViT, etc.)
- **Output:** The same model, but now it continues learning from experience
- **Mechanism:** Local Hebbian/neuromodulated plasticity injected at each layer
- **Memory:** Plastic weights are ternary {-1, 0, +1}, 2-bit, ~0.1–1% of model size
- **Hardware:** Runs on the same device as the frozen model — no extra GPU needed

### Use Cases

- **Personal AI assistants** that learn your writing style, preferences, and knowledge over time — without sending your data to the cloud
- **Edge devices** that adapt to their environment (cameras, sensors, robots)
- **Privacy-first applications** where models must learn on-device and never share data
- **Scientific models** that stay current with new research without full retraining

---

## Competitive Advantage

| | Fine-tuning | LoRA | EWC | **PH-Neuro Brain** |
|:--|:-----------:|:----:|:---:|:------------------:|
| Continual learning | ❌ Forgets | ❌ Adapter per task | ⚠️ Limited | **✅ By design** |
| Local updates (no backprop) | ❌ | ❌ | ❌ | **✅ Hebbian/neuromodulated** |
| Works with any pre-trained model | ✅ | ✅ | ✅ | **✅** |
| Biological plausibility | ❌ | ❌ | ❌ | **✅ Surprise, consolidation, decay** |
| On-device | ❌ | ⚠️ | ⚠️ | **✅ Ternary plastic weights (2-bit)** |
| No forgetting of original knowledge | ❌ | ✅ (frozen) | ⚠️ | **✅ Structural weights frozen** |
| Open source | ✅ | ✅ | ✅ | ✅ |

---

## Infrastructure Milestones — COMPLETE ✅

These are the pre-training toolkit milestones. All completed. They enable
building efficient "born networks" — but they are NOT the product.

### Phase 1 (Vision DQT)

| Metric | Result | Target | Status |
|:-------|:------:|:------:|:------:|
| DQT CNN CIFAR-10 | **78.98%** (+6.23pp vs STE) | >80% | 🟡 MARGINAL |
| DQT CNN CIFAR-100 | **54.15%** (+15.95pp vs STE) | >55% | 🟡 MARGINAL |
| Model export ONNX | **<17 MB** (all models) | <100MB | ✅ GO |
| Production docs | **README + API + quickstart** | Complete | ✅ GO |
| Memory vs TF Lite | **4.00× smaller, 2× faster** | 4× / 2× | ✅ GO |

### Phase 2 (Transformer DQT)

| Metric | Result | Target | Status |
|:-------|:------:|:------:|:------:|
| DQT Transformer 100M | **11.35 ppl** TinyStories | <30 ppl | ✅ GO |
| DQT Transformer 250M | **Stable ✅** | <20 ppl | 🟡 SCIENTIFIC GO |
| MoE DQT Transformer | **14.08 ppl** | <20 ppl | ✅ GO |
| On-device demo | **21-25 tok/s CPU** | Token generation | ✅ GO |
| Public launch | **Gradio app + blog**, 3 models, 26 MB | Demo live | ✅ GO |

### Phase 2.5 (Memory Sprint)

| Metric | Result | Target | Status |
|:-------|:------:|:------:|:------:|
| 8-bit AdamW + bf16 | All scripts converted, MNIST smoke OK | — | ✅ DONE |
| Flash Attention / SDPA | Transformer attention, tests pass | — | ✅ DONE |
| 1B-param DQT Transformer | **1.02B ternary, stable, 8.04 GB peak** | — | ✅ GO |
| Memory benchmark | M2.2 −22/−31%, 1B fits in 8.04 GB | — | ✅ DONE |

> DQT training budget: **~5 bytes/param** (8-bit AdamW + bf16).
> **1B+ ternary params train stably on RTX 4060 8 GB.** This is the
> pre-training toolkit that builds the "born networks" for Brain Wrapper.
> See [ROADMAP.md § Phase 2.5](ROADMAP.md) and
> [E030](research/docs/experiments/E030-m2-9-memory-benchmark.md).

---

## Brain Milestones — THE PRODUCT

| Phase | Milestone | Target | Status |
|:------|:----------|:------|:------:|
| **0** | Foundation Research | Model selection, plasticity survey, architecture design | 🔬 In progress |
| **1** | Proof of Concept | GPT-2 + vector bias plasticity, WikiText-2 → PubMed adaptation | ⬜ |
| **2** | Scaling Plasticity | Low-rank + ternary plastic weights, consolidation mechanism | ⬜ |
| **3** | Continual Learning at Scale | Multi-domain adaptation, scaling laws, vision extension | ⬜ |

See [ROADMAP.md](ROADMAP.md) for the full 13-step plan.

---

## Non-Goals (What We Are NOT)

- ❌ NOT an LLM company (we don't train models from scratch — we wrap existing ones)
- ❌ NOT competing with OpenAI/DeepSeek/Meta on model quality
- ❌ NOT a model compression framework (that's infrastructure, not the product)
- ❌ NOT selling training services or cloud compute
- ❌ NOT a research lab — we're building a product, grounded in science

---

## Funding Strategy

1. **Bootstrap Phase 0–1** (€0 — using existing RTX 4060, open-source models are free)
2. **Public demo after Phase 1** (GPT-2 + Brain Wrapper, self-hosted)
3. **Pre-seed after Phase 2 demo** (€50K–200K from EU/Cyprus startup grants or angels)
4. **Seed after Phase 3 validation** (€500K–2M for team + compute)

---

## References

- Research archive: [`research/`](research/)
- Product roadmap: [`ROADMAP.md`](ROADMAP.md)
- Brain investigation docs: [`docs/brain/`](docs/brain/)
- Original research summary: [`research/docs/RESEARCH_SUMMARY.md`](research/docs/RESEARCH_SUMMARY.md)
