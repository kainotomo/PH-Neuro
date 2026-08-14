# PH-Neuro Brain — Overview

> **Status:** Phase 0 ✅ COMPLETE (2026-08-12) · **Phase 1.1 (E031) ✅ COMPLETE — PARTIAL SUCCESS** (2026-08-12) · **Phase 1.2 (E032) ✅ COMPLETE — NEGATIVE / LOW-RANK HEBBIAN REJECTED** (2026-08-13)
> **Principle:** Investigate → Decide → Implement. No code before investigation is documented.
> **Phase 1.1 verdict:** 5/6 pre-registered checks pass at 100K. Surprise-modulated plasticity **works** and is validated as **essential** (constant-M control: +10.7% catastrophic forgetting, −0.57 target; surprise: +0.03 target, +0.37% forgetting, p=0.003) — but Δppl = +0.034 ≪ the 0.5-practical bar → **partial success, not a full GO**. See [05-e031-minimal-viable.md](05-e031-minimal-viable.md).
> **Phase 1.2 verdict:** the two E031 bottleneck hypotheses are answered **decisively in the negative** for local plasticity. Capacity (rank) **destroys** (best rank-1 Δppl = −1.35; rank 2/4 ≈ −1650/−3170); every gain knob (η↑, s₀↓, k↓, M_max↑) is monotonically destructive; decay is neutral; 1M budget compounds damage to Δppl ≈ −7381. **Matched-budget backprop LoRA exceeds the 0.5 bar at every lr** (+0.86/+1.32/+1.52, source *improved*). → **The missing ingredient is credit assignment, not capacity/gain/decay.** Low-rank Hebbian **rejected**; vector-bias (+0.034) remains the only stable local config. See [06-e032-capacity-gain.md](06-e032-capacity-gain.md).

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
| Step | Document | Question | Status |
|:----:|:---------|:---------|:------:|
| 0.1 | [Model Selection](00-model-selection.md) | Which pre-trained model is the best "born brain"? | ✅ |
| 0.2 | [Plasticity Mechanisms](01-plasticity-mechanisms.md) | Which local learning rule could work on a frozen backbone? | ✅ |
| 0.3 | [Surprise Signal](02-surprise-signal.md) | What tells the brain "learn now"? **→ Sequence-mean loss dev. from EMA → sigmoid → global float32 scalar M.** | ✅ |
| 0.4 | [Architecture Design](03-architecture.md) | How does the Brain Wrapper hook into any model? **→ Output-modification forward hooks on `o_proj`/`down_proj` (SmolLM2) & `c_proj` (GPT-2); thin BlockWrapper adapter; full `BrainWrapper` API + learn/generate/pause-resume spec.** | ✅ |
| 0.5 | [Evaluation Protocol](04-evaluation-protocol.md) | How do we know if it worked? **→ LOCKED: WikiText-2→PubMed (verified, 10.65→11.67 ppl); budgets 1K/10K/100K/1M (100K=primary surprise point, EMA τ≈102K tok); baselines frozen/random/const-M/LoRA (r=1 o_proj = exact 98,304-param budget match; full-FT infeasible on 8 GB); fixed 512-token window, stride 256; d=0.5@80% needs 16,074 tok (test sets 19–31× larger); success = Δppl≥0.5 ppl, p<0.05, <1% forgetting.** | ✅ |

### Phase 1: Proof of Concept
| Step | Document | Question |
|:----:|:---------|:---------|
| 1.1 | [Minimal Viable Experiment](05-e031-minimal-viable.md) | Frozen SmolLM2-1.7B (primary) + vector bias + surprise-modulated Hebbian. Does it work? | 🟡 **PARTIAL SUCCESS** — mechanism works, surprise modulator essential (vs constM's +10.7% forgetting), but Δppl=+0.034 ≪ 0.5 bar |
| 1.2 | [Capacity & Gain Experiment](06-e032-capacity-gain.md) | Low-rank capacity + gain sweep + decay ablation + matched-budget LoRA upper bound. Does capacity/gain rescue local plasticity? | ❌ **NEGATIVE — LOW-RANK HEBBIAN REJECTED.** Capacity destroys (rank-1 −1.35 → rank-4 −3172); all gain knobs destructive; decay neutral; 1M compounds to −7381. Backprop LoRA at the same 344K-param budget **exceeds 0.5 bar** (+1.52, source improved). Missing ingredient = **credit assignment**, not capacity/gain/decay. |
| 1.3 | [Architectural Generalization](07-e033-generalization.md) | Does it work on a different architecture — GPT-2 124M (gen-test, classic pre-norm, no RoPE/GQA)? |

### Phase 2: Scaling Plasticity
| Step | Document | Question | Status |
|:----:|:---------|:---------|:------:|
| 2.1 | [Low-Rank Plastic Matrices](08-e034-low-rank.md) | Does more capacity → better adaptation? | ❌ **ANSWERED NEGATIVE in E032** — local low-rank Hebbian is catastrophically unstable (rank-1 −1.35 … rank-4 −3172, 1M −7381). Re-scope required: error-based local rules (credit assignment without backprop) or promote **backprop LoRA** (the proven +1.52 bound) as the scaling mechanism |
| 2.2 | [Ternary Plastic Weights](09-e035-ternary-plasticity.md) | Can we make plasticity tiny AND effective? | ⬜ |
| 2.3 | [Consolidation Mechanism](10-e036-consolidation.md) | Can we build long-term memory from short-term plasticity? | ⬜ |

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

### Minimum Viable (Phase 1.1) — verdict (2026-08-12)
| Criterion | Result |
|:----------|:------:|
| Any measurable ppl improvement on target over frozen | ✅ Δppl = +0.034, p = 0.003 |
| <1% source degradation (no forgetting) | ✅ +0.37% (vs constM's +10.7%) |
| Surprise modulation outperforms constant learning rate | ✅ +0.034 vs **−0.573** |
| Practical bar: Δppl ≥ 0.5 ppl (locked §7) | ❌ +0.034 ≪ 0.5 |

**PARTIAL SUCCESS:** mechanism works and the surprise modulator is essential
(it prevents constant-M's catastrophic forgetting), but the effect is too
small to be practically meaningful at vector-bias capacity. Phase 1.2
(low-rank + gain) was then run and answered this decisively.

### Capacity & Gain (Phase 1.2) — verdict (2026-08-13)
| Criterion | Result |
|:----------|:------:|
| Δppl ≥ 0.5 at 100K (practical bar) | ❌ no local config (best low-rank −1.35; E031 vector-bias +0.034); ✅ **every** LoRA config |
| Capacity (rank) rescues | ❌ rank 1/2/4 = −1.35 / −1648 / −3172 (p=0.014 at rank 4 — destruction is significant) |
| Gain (η↑, s₀↓, k↓, M_max↑) rescues | ❌ monotonic in the destructive direction (−1.35 → −126K at η=1e-2) |
| Decay (λ 1e-5/1e-4) rescues | ❌ neutral (−1.36 / −1.28 ≈ λ=0 −1.35) |
| 1M anneal saturates | ❌ compounds: Δppl ≈ **−7381**, +201,000% forgetting |
| LoRA (same 344K budget) beats local | ✅ Δppl = +0.86/+1.32/+1.52, source **improved** (−6.5 to −8.5%) — ratio ≈ −0.84 (opposite sign) |

**NEGATIVE:** surprise-modulated Hebbian with matrix capacity is **structurally
unstable** (surprise positive feedback + Hebbian concentration); the missing
ingredient is **credit assignment** (backprop), not capacity/gain/decay.
Low-rank Hebbian is **rejected**; vector-bias (+0.034) remains the safe
default. Next step options in §12 of the report (error-based local rules /
GPT-2 replication).

### Strong Signal (Phase 2)
- ~~Low-rank plasticity > vector bias plasticity (capacity matters)~~ **FALSIFIED (E032)** — capacity destroys local Hebbian
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
