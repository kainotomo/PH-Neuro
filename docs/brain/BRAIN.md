# PH-Neuro Brain — Overview

> **Status:** Phase 0 ✅ COMPLETE (2026-08-12) · **Phase 1.1 (E031) ✅ COMPLETE — PARTIAL SUCCESS** (2026-08-12) · **Phase 1.2 (E032) ✅ COMPLETE — NEGATIVE / LOW-RANK HEBBIAN REJECTED** (2026-08-13) · **Phase 1.3 (E033) ✅ COMPLETE — NEGATIVE / PREDICTIVE CODING INERT — LOCAL-RULE QUESTION CLOSED** (2026-08-14) · **Phase 2.1 (E034) ✅ COMPLETE — SURPRISE GATE ADDS VALUE ON BACKPROP** (2026-08-14) · **Phase 2.2 (E035) ✅ COMPLETE — TERNARY LORA: 2-BIT PRESERVES ≥90% OF FLOAT QUALITY AT 16× STORAGE** (2026-08-15)
> **Principle:** Investigate → Decide → Implement. No code before investigation is documented.
> **Phase 1.1 verdict:** 5/6 pre-registered checks pass at 100K. Surprise-modulated plasticity **works** and is validated as **essential** (constant-M control: +10.7% catastrophic forgetting, −0.57 target; surprise: +0.03 target, +0.37% forgetting, p=0.003) — but Δppl = +0.034 ≪ the 0.5-practical bar → **partial success, not a full GO**. See [05-e031-minimal-viable.md](05-e031-minimal-viable.md).
> **Phase 1.2 verdict:** the two E031 bottleneck hypotheses are answered **decisively in the negative** for local plasticity. Capacity (rank) **destroys** (best rank-1 Δppl = −1.35; rank 2/4 ≈ −1650/−3170); every gain knob (η↑, s₀↓, k↓, M_max↑) is monotonically destructive; decay is neutral; 1M budget compounds damage to Δppl ≈ −7381. **Matched-budget backprop LoRA exceeds the 0.5 bar at every lr** (+0.86/+1.32/+1.52, source *improved*). → **The missing ingredient is credit assignment, not capacity/gain/decay.** Low-rank Hebbian **rejected**; vector-bias (+0.034) remains the only stable local config. See [06-e032-capacity-gain.md](06-e032-capacity-gain.md).
> **Phase 1.3 verdict:** the final local-rule test — **error-based predictive coding** at the same matched 344K budget. Δppl_PC = **+0.001 ± 0.003** (p = 0.737, per-seed +0.001/+0.004/−0.003) — statistically null, ~500× below the 0.5 bar. **The error-driven update is stable** (source *improved* −0.012%, no forgetting — it fixes E032's instability) **but inert** (plastic weights barely move; mean|B| ≈ 0.0016; the reconstruction error carries no usable adaptation direction). → **Pre-registered kill criterion fired: the local-rule scientific question is CLOSED; the project pivots to the backprop-LoRA product path.** See [07-e033-predictive-coding.md](07-e033-predictive-coding.md).
> **Phase 2.1 verdict (E034 — Surprise-Gated LoRA):** the first value-add of the brain machinery on top of backprop LoRA is **confirmed**. Reusing E032's LoRA (o_proj+down_proj, rank 1 = 344,064 params, AdamW) and replacing the constant lr with the **surprise-gated lr** `η·M_t` (E031's modulator). **Single-domain** (WikiText-2 → PubMed, 100K, 3 seeds): gated Δppl = **+0.902 ± 0.182** (p=0.013; +0.693/+1.025/+0.988) — **≥ 0.5 practical bar**, source *improved* **−2.66%** (the gate preserves a practical effect while learning at ~1/10th the total rate). **Sequential two-domain** (WikiText → PubMed → CNN/DailyMail): gated backward transfer on PubMed **BT = −0.009 ≈ 0** vs plain **+1.854** (3/3 seeds — plain's PubMed gains wiped by CNN training, ending worse than frozen; gated preserves PubMed at +0.911 over frozen while still adapting to CNN) → **the surprise gate makes LoRA selective**. **Optional control** (const_reduced, single-domain, 3 seeds): gated ≈ constant reduced lr (Δppl **+0.902** vs **+0.947**, p≈0.14) — the single-domain gain is explained by the *lower average lr*; the gate's distinctive value remains the two-domain selectivity. **Verdict: the gate adds value on top of backprop → Step 2.2 (ternary LoRA) uses gated LoRA.** See [08-e034-surprise-gated-lora.md](08-e034-surprise-gated-lora.md).
> **Phase 2.2 verdict (E035 — Ternary LoRA):** **2-bit ternary adaptation PRESERVES ≥90% of float gated-LoRA quality at 16× smaller storage — CONFIRMED for T-C (STE with latent scores).** Three ternarization paths on the same 344,064-param budget / gated protocol. **Single-domain** (WikiText-2 → PubMed, 100K, 3 seeds): **T-C Δppl = +0.892 ± 0.206 = 99% of float (+0.902)**, source *improved* −2.26%, storage **15.93×** (1.38 MB → **86,016 B** on disk) — **meets all pre-registered criteria**. **T-A** post-training quantize preserves 69–76% (+0.618/+0.689, below the 0.81 bar; 20-step calibration recovers some). **T-B (DQT stochastic rounding) is inert** (+0.000 ± 0.001, flips ~7%/step never coalesce at rank-1). **Sequential two-domain** (best variant T-C, → CNN/DailyMail): backward transfer **BT = −0.0118 < 0.1** (3/3 seeds), PubMed still +0.903 while adapting to CNN (+0.130) — **selectivity survives quantization** (matches float's BT −0.009). **Verdict: the on-device product stack HOLDS — surprise-gated, ternary, continually-learning adapters.** See [09-e035-ternary-lora.md](09-e035-ternary-lora.md).

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
| 1.3 | [Predictive Coding (re-scoped)](07-e033-predictive-coding.md) | The last local-rule family with a credit-assignment story: error-based predictive coding (per-site reconstruction error) at the matched 344K LoRA budget. Can a no-backprop error signal adapt a frozen LM? | ❌ **NEGATIVE — INERT / LOCAL-RULE QUESTION CLOSED.** Δppl_PC = +0.001 ± 0.003 (p = 0.737) — statistically null. The error-driven update is **stable** (source improved −0.012%, no forgetting — fixes E032's instability) but **inert** (plastic weights barely move). Pre-registered kill criterion fired → **pivot to the backprop-LoRA product path**. |

### Phase 2: Scaling Plasticity
| Step | Document | Question | Status |
|:----:|:---------|:---------|:------:|
| 2.1 | [Surprise-Gated LoRA](08-e034-surprise-gated-lora.md) | Does the surprise gate add value on top of backprop LoRA (the product path)? | ✅ **GATE ADDS VALUE ON BACKPROP (E034).** Single-domain gated Δppl = **+0.902 ± 0.182** ≥ 0.5 bar, source improved −2.66%. Two-domain: gated backward transfer on PubMed **BT = −0.009** vs plain **+1.854** (3/3 seeds) → **selectivity confirmed**. **Step 2.2 uses gated LoRA.** |
| 2.2 | [Ternary Plastic Weights](09-e035-ternary-lora.md) | Can we make plasticity tiny AND effective? | ✅ **TERNARY LORA (E035): 2-BIT PRESERVES ≥90% OF FLOAT AT 16× STORAGE.** Three paths on the gated-LoRA protocol: **T-C (STE)** Δppl = **+0.892 ± 0.206 = 99% of float** (+0.902) ≥ 0.81 bar, source improved −2.26%, storage **15.93×** (86,016 B), two-domain **BT −0.0118 < 0.1** — selectivity survives quantization. **T-A** (post-train quantize) 69–76% (below bar). **T-B (DQT) inert** (+0.000). Product adapter = STE ternary gated LoRA. |
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

### Predictive Coding (Phase 1.3, re-scoped) — verdict (2026-08-14)
| Criterion | Result |
|:----------|:------:|
| Δppl ≥ 0.5 at 100K (practical bar) | ❌ Δppl_PC = **+0.001 ± 0.003** (p = 0.737) — ~500× below, below the 0.2 noise floor |
| Sign agreement with LoRA (Δppl > 0) | ❌ mean nominally +, but statistically 0 (p=0.737); seed 44 **negative** (−0.003) |
| Forgetting < 1% (stability) | ✅ mean **−0.012%** — source *improved*; PC fixes E032's instability |
| Δppl_PC / Δppl_LoRA ratio | ≈ **0.000** (vs E032 Hebbian's −0.84) |
| Plastic weights adapt | ❌ inert — mean\|A\| ≈ init, mean\|B\| ≈ 0.0016; ε carries no usable direction |

**NEGATIVE — INERT:** the error-driven reconstruction-error update is the
first local matrix-capacity rule that is fully **stable** (no destruction, no
forgetting) but it is also **inert** — the plastic weights do not move in a
direction that reduces ppl (Δppl ≈ 0). The per-site linear-inverse
reconstruction error is underdetermined and carries no usable credit signal
on a 1.7B residual transformer. **Pre-registered kill criterion fired: the
local-rule scientific question is CLOSED; the project pivots to the
backprop-LoRA product path** (E032's proven +1.52 bound).

### Strong Signal (Phase 2)
- ~~Low-rank plasticity > vector bias plasticity (capacity matters)~~ **FALSIFIED (E032)** — capacity destroys local Hebbian
- ~~Low-rank plasticity > vector bias plasticity (capacity matters)~~ **FALSIFIED twice (E032 + E033)** — local rules at matched budget are destructive (Hebbian) or inert (predictive coding); the Phase 2 scaling mechanism is **backprop LoRA** (proven +1.52, product path)
- ~~Ternary plastic weights ≥90% of float adaptation quality~~ **ACHIEVED (E035)** — T-C (STE ternary gated LoRA) = **+0.892 = 99% of float** at 16× storage; selectivity survives (two-domain BT −0.0118). Confirmed for the STE representation; T-A (post-train quantize) 69–76%, T-B (DQT) inert.
- Consolidation reduces forgetting across sequential domains (Step 2.3, E036 — uses T-C as the short-term store)

### Breakthrough (Phase 3)
- A single model adapts across 5+ domains with improvement on ALL vs frozen
- Scaling law: larger models benefit more from plasticity
- Method transfers to vision (ViT class-incremental learning)

---

## References

- [PH-Neuro Research Summary](../research/docs/RESEARCH_SUMMARY.md) — 19 Hebbian/STE/DQT experiments
- [GOALS.md](../../GOALS.md) — Project vision and competitive landscape
- [ROADMAP.md](../../ROADMAP.md) — Full roadmap with milestone status
