# PH-Neuro — Goals & Vision

> **The smallest useful AI models in the world.**
> Last updated: 2026-08-04

---

## Vision

**Democratize AI by making deep learning models so small they run anywhere.**

Every device — phone, watch, sensor, drone — should be able to run and train its own AI. No cloud. No data center. No compromise on privacy.

---

## Mission

Build the most memory-efficient deep learning framework that:

1. **Trains on a single consumer GPU** — no data center required
2. **Runs on edge devices** — smartphones, microcontrollers, wearables
3. **Learns continuously** — adapts to new data without forgetting (QLoRA)
4. **Uses ternary weights** — {-1, 0, +1}, 2 bits/weight, popcount math

---

## Core Technology (3 Pillars)

| Pillar | What | Status | Key Metric |
|:-------|:-----|:------:|:-----------|
| **DQT** | Train ternary weights without latent float scores | ✅ Proven | 98.2% MNIST, 4.5× less training memory |
| **MoE** | Sparse activation — only 50% params active | ✅ Proven | +2.5pp accuracy vs dense |
| **Ternary** | {-1, 0, +1} weights, 2 bits/weight | ✅ Proven | 8× smaller than FP16 |

Combined target: **1B parameters → 200MB on disk → runs on a phone.**

---

## Target Product

### Phase 1: Tiny Vision (6 months)
- Image classifier: 100MB, >80% CIFAR-10, runs on Raspberry Pi
- Object detector: person/face detection for security cameras
- Target customers: IoT manufacturers, security companies

### Phase 2: Tiny LLM (12 months)
- Language model: 1B params, 200MB, perplexity <30
- Chat/assistant that runs entirely on-device
- Target customers: mobile app developers, privacy-focused companies

### Phase 3: Platform (18-24 months)
- SDK for custom on-device models
- Training pipeline for non-ML engineers
- Target: any company that needs AI on edge devices

---

## Competitive Advantage

| | TensorFlow Lite | ONNX Runtime | BitNet | **PH-Neuro** |
|:--|:---------------:|:------------:|:------:|:------------:|
| Weight size | 8-bit | 8-bit | 2-bit | **2-bit** |
| Training memory | High | High | High (latent scores) | **4.5× lower (DQT)** |
| Sparse activation | ❌ | ❌ | ❌ | **✅ MoE** |
| On-device training | ❌ | ❌ | ❌ | **✅ QLoRA** |
| Open source | ✅ | ✅ | ✅ | ✅ |

---

## Success Metrics (Phase 1)

| Metric | Current | Target | Deadline |
|:-------|:------:|:------:|:--------:|
| DQT CNN CIFAR-10 accuracy | **78.98%** (DQT, +6.23pp vs STE 72.75%) | >80% | Aug 2026 |
| DQT CNN CIFAR-100 accuracy | **54.15%** (DQT, +15.95pp vs STE 38.2%) | >55% | Aug 2026 |
| Model size (on disk) | — | <100MB | Sep 2026 |
| Inference speed (Raspberry Pi) | — | >10 fps | Oct 2026 |
| GitHub stars | — | >500 | Dec 2026 |
| First paying customer | — | 1 | Mar 2027 |

---

## Non-Goals (What We Are NOT)

- ❌ NOT an LLM company (we're infrastructure, not models)
- ❌ NOT competing with OpenAI/DeepSeek on scale
- ❌ NOT selling training services
- ❌ NOT targeting data centers (we're edge-first)
- ❌ NOT a research lab anymore (product-first from Aug 2026)

---

## Funding Strategy

1. **Bootstrap Phase 1** (€0 — using existing RTX 4060)
2. **Cloud for Phase 2** (~€2,000 — personal investment)
3. **Pre-seed after Phase 2 demo** (€50K–200K from EU/Cyprus startup grants or angels)
4. **Seed after Phase 3 MVP** (€500K–2M for team + compute)

---

## References

- Research archive: [`research/`](research/)
- Product roadmap: [`ROADMAP.md`](ROADMAP.md)
- Original research summary: [`research/docs/RESEARCH_SUMMARY.md`](research/docs/RESEARCH_SUMMARY.md)
