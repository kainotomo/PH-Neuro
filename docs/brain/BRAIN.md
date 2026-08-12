# PH-Neuro Brain — Overview

> **Status:** Phase 0 — Foundation Research (August 2026)
> **Principle:** Investigate → Decide → Implement. No code before investigation is documented.

---

## The Core Hypothesis

**A frozen pre-trained model + local Hebbian/neuromodulated plasticity can achieve meaningful continual domain adaptation without catastrophic forgetting.**

The brain is not trained from scratch. It's born pre-wired by evolution (pre-training) and learns throughout life via local synaptic plasticity. We replicate this: take any open-source pre-trained model, freeze it, inject tiny plastic weights at each layer, and update them via local, brain-like rules.

---

## Why Previous Approaches Failed

Our 19 Hebbian experiments (E001–E019, see [`research/docs/RESEARCH_SUMMARY.md`](../research/docs/RESEARCH_SUMMARY.md)) proved definitively that local Hebbian rules **cannot train deep ternary networks from scratch** — every method hits the ~88% MNIST ceiling. The reason: Hebbian rules optimize for statistical correlation (pre × post), not classification error. Without backprop to assign credit across layers, hidden layers cannot learn class-discriminative features.

**But this does NOT mean local rules are useless.** The brain doesn't use local rules to build its architecture from scratch — evolution does that. Local rules handle *adaptation* of an already-structured system. Our hypothesis is that a pre-trained backbone provides enough representational structure that local error signals can drive useful adaptation.

---

## The Plan (13 Steps)

### Phase 0: Foundation Research (No Code)
| Step | Document | Question |
|:----:|:---------|:---------|
| 0.1 | [Model Selection](00-model-selection.md) | Which pre-trained model is the best "born brain"? |
| 0.2 | [Plasticity Mechanisms](01-plasticity-mechanisms.md) | Which local learning rule could work on a frozen backbone? |
| 0.3 | [Surprise Signal](02-surprise-signal.md) | What tells the brain "learn now"? |
| 0.4 | [Architecture Design](03-architecture.md) | How does the Brain Wrapper hook into any model? |
| 0.5 | [Evaluation Protocol](04-evaluation-protocol.md) | How do we know if it worked? |

### Phase 1: Proof of Concept
| Step | Document | Question |
|:----:|:---------|:---------|
| 1.1 | [Minimal Viable Experiment](05-e031-minimal-viable.md) | Frozen [user-selected model] + vector bias + surprise-modulated Hebbian. Does it work? |
| 1.2 | [Ablation Experiments](06-e032-ablation.md) | Which components are necessary? |
| 1.3 | [Architectural Generalization](07-e033-generalization.md) | Does it work on a different architecture (e.g. RoPE/SwiGLU-style)? |

### Phase 2: Scaling Plasticity
| Step | Document | Question |
|:----:|:---------|:---------|
| 2.1 | [Low-Rank Plastic Matrices](08-e034-low-rank.md) | Does more capacity → better adaptation? |
| 2.2 | [Ternary Plastic Weights](09-e035-ternary-plasticity.md) | Can we make plasticity tiny AND effective? |
| 2.3 | [Consolidation Mechanism](10-e036-consolidation.md) | Can we build long-term memory from short-term plasticity? |

### Phase 3: Continual Learning at Scale
| Step | Document | Question |
|:----:|:---------|:---------|
| 3.1 | [Multi-Domain Adaptation](11-e037-multi-domain.md) | Does it scale to many domains without forgetting? |
| 3.2 | [Scaling Laws](12-e038-scaling.md) | How does it behave as models and data grow? |
| 3.3 | [Vision Extension](13-e039-vision.md) | Is this a general principle, not just a language trick? |

---

## Key Design Principles

1. **No backprop through frozen layers.** Each layer's plastic weights update from locally available information only: pre-activation, post-activation, and a global surprise/modulator signal.

2. **Plastic weights are tiny.** Target: 0.1–1% of model parameters. For a ~124M-parameter model that is at most ~1.2M plastic parameters; with ternary (2-bit) storage, ~300 KB.

3. **Forgetting is a feature, not a bug.** Plastic weights decay toward zero. Only surprising, repeated, or important experiences persist. This is biologically accurate — most things we experience are forgotten.

4. **Consolidation happens during "sleep."** Periodically, important plastic changes are transferred to a slower-decaying long-term store. The fast plastic store is then cleared, keeping it responsive.

5. **The frozen model is sacred.** Structural weights never change. This guarantees zero catastrophic forgetting of original capabilities. The model can always be used without plastic weights active.

---

## Success Criteria

### Minimum Viable (Phase 1.1)
- Any measurable perplexity improvement on a target domain over frozen baseline
- <1% perplexity degradation on source domain (no forgetting)
- Surprise modulation outperforms constant learning rate

### Strong Signal (Phase 2)
- Low-rank plasticity > vector bias plasticity (capacity matters)
- Ternary plastic weights ≥90% of float adaptation quality
- Consolidation reduces forgetting across sequential domains

### Breakthrough (Phase 3)
- A single model adapts across 5+ domains with improvement on ALL vs frozen
- Scaling law: larger models benefit more from plasticity
- Method transfers to vision (ViT class-incremental learning)

---

## References

- [PH-Neuro Research Summary](../research/docs/RESEARCH_SUMMARY.md) — 19 Hebbian/STE/DQT experiments
- [GOALS.md](../../GOALS.md) — Project vision and competitive landscape
- [ROADMAP.md](../../ROADMAP.md) — Full roadmap with milestone status
