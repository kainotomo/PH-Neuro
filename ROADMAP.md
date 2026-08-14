# PH-Neuro — Product Roadmap

> **Last updated:** 2026-08-14
> **Status:** Infrastructure (Phases 0–2.5) COMPLETE ✅ → **Brain Phase 0 COMPLETE ✅ → Brain Phase 1.1 COMPLETE 🟡 (PARTIAL SUCCESS) → Brain Phase 1.2 COMPLETE ❌ (LOW-RANK HEBBIAN REJECTED; LoRA bound met) → Brain Phase 1.3 COMPLETE ❌ (PREDICTIVE CODING INERT; LOCAL-RULE QUESTION CLOSED) → next: pivot to backprop-LoRA product path**

---

## Phase 0: Research Foundation ✅ (July 2026)

All 19 experiments completed. See [`research/`](research/).

| Milestone | Status | Key Result |
|:----------|:------:|:-----------|
| Hebbian era (E001–E008) | ✅ | 9 experiments, all hit ~88% ceiling — Hebbian cannot train deep ternary networks |
| STE era (E009–E015) | ✅ | Ternary STE works (98% MNIST), QLoRA solves continual learning (0% forgetting) |
| DQT era (E016–E019) | ✅ | DQT beats STE (98.2%), MoE works (+2.5pp), Hysteresis overridden by DQT |

---

## Phase 1: Production-Ready DQT (Aug–Sep 2026)

**Goal:** Prove DQT works on realistic problems, not just MNIST toy tasks.

| Milestone | Target | Priority | Status |
|:----------|:------:|:--------:|:------:|
| **M1.1** DQT CNN on CIFAR-10 | >80% accuracy | 🔴 Critical | ✅ CONDITIONAL GO |
| **M1.2** DQT CNN on CIFAR-100 | >55% accuracy | 🟡 High | 🟡 CONDITIONAL GO |
| **M1.3** Model export (ONNX/C) | <100MB, runs on Raspberry Pi | 🟡 High | ✅ GO |
| **M1.4** Production README + docs | Clear quickstart + API docs | 🟡 High | ✅ GO |
| **M1.5** Memory benchmarks vs TF Lite | 4× smaller, 2× faster inference | 🟢 Medium | ✅ GO |

**Go/No-go gate:** M1.1 — accuracy gate 78.98% (missed 80% by 1.02pp), but **scientific goal achieved**: DQT Conv2d layer validated, DQT > STE by +2.89pp on identical architecture. Closed as CONDITIONAL GO. See [E020–E021.3](research/docs/RESEARCH_SUMMARY.md).

---

## Phase 2: Tiny Transformer (Sep–Nov 2026)

**Goal:** First ternary language model — the smallest useful LLM.

> ⚠️ **VRAM CONSTRAINT (Aug 2026):** M2.1 measured 7.3/8.0 GB on RTX 4060 for 141M total params.
> DQT training state = ~13 bytes/param (weight_float 4B + AdamW 8B + ternary 1B).
> **Max trainable on 8GB: ~300M ternary params.** Original 500M/1B targets adjusted accordingly.

| Milestone | Target | Priority | Status |
|:----------|:------:|:--------:|:------:|
| **M2.1** DQT Transformer 100M params | Perplexity <30 on TinyStories (**mean 11.35** ✅) | 🔴 Critical | ✅ GO |
| **M2.2** DQT Transformer 250M params | Perplexity <20 on WikiText-2 | 🟡 High | 🟡 SCIENTIFIC GO |
| **M2.3** MoE DQT Transformer | **265M total, 190M active**, ppl <20 | 🟡 High | ✅ GO |
| **M2.4** On-device inference demo | Token generation on CPU, 11 MB packed | 🟡 High | ✅ GO |
| **M2.5** Public demo + blog post | Gradio app + launch blog | 🟢 Medium | ✅ GO |

**Go/No-go gate:** M2.1 — ✅ **GO** (ppl **11.35**). M2.2 — 🟡 **SCIENTIFIC GO**
(250M stable, data-limited). M2.3 — ✅ **GO** (mean ppl **14.08** < 20,
first MoE DQT Transformer, 265M total/190M active, 3/3 seeds stable).
M2.4 — ✅ **GO** (CPU inference, 21-25 tok/s, 11 MB packed).
M2.5 — ✅ **GO** (Gradio demo: text + vision + benchmarks, 3 models,
26 MB total packed, blog post).

> 🎉 **Phase 2 COMPLETE.** All 5 milestones closed. See Phase 3.
(already validated on MLP vision — E019).

---

## Phase 2.5: Memory Optimization Sprint (Aug 2026) ✅ COMPLETE

**Goal:** Break the 300M-param VRAM ceiling. Scale DQT training to **1B+ ternary
params on a single RTX 4060 (8 GB VRAM + 32 GB RAM)** — without rewriting the
DQT autograd.

### Why Now

PH-Neuro has hit the **VRAM ceiling**: ~13 bytes/param (fp32 weight + AdamW
m/v + int8 ternary). Max ~300M ternary params on 8 GB. The 100M–250M models
validated Phase 2, but to build a competitive 1B+ LLM we need **more params
without more GPU**.

### Research (2026-08-11): 13 Techniques Evaluated

Full analysis in the project session log. Summary:

| # | Technique | VRAM Saving | Speed | DQT Compat | Verdict |
|:--|:----------|:-----------:|:-----:|:----------:|:-------:|
| 1 | **8-bit AdamW (bitsandbytes)** | **75% opt** | 1.0× | ✅ | 🔴 **#1 priority** |
| 2 | **bf16 weight_float + autocast** | **50% weights** | 1.2× | ✅ | 🔴 **#2 priority** |
| 3 | **Fused AdamW** (PyTorch native) | ~15% temps | 1.1× | ✅ | 🟡 Quick win |
| 4 | **Flash Attention / SDPA** | Attention mem | 2-3× attn | ✅ | 🟡 Quick win |
| 5 | Gradient Checkpointing | ~50% activ | 0.7× | ✅ | ✅ Already active |
| 6 | Embed-SGD (no AdamW for embedding) | 0.4 GB | 1.0× | ✅ | ✅ Already active |
| 7 | `expandable_segments` | ~10% frag | 1.0× | ✅ | ✅ Already active |
| 8 | **accelerate cpu_offload** | Opt states → RAM | 0.85× | ✅ | 🟢 Fallback |
| 9 | Adafactor | ~50% opt | 1.0× | ✅ | ⚠️ Convergence risk |
| 10 | torch.compile (JIT) | ~15% temps | 1.3× | ⚠️ | ⚠️ Test first |
| 11 | GaLore (ICML 2024 Oral) | 65% opt | 0.85× | ⚠️ | 🔬 Research |
| 12 | DeepSpeed ZeRO-3 | 90% | 0.2× | ⚠️ | ⏳ Later |
| 13 | NVMe Offloading | ~100% | 0.1× | ⚠️ | ⏳ Later |

### New Memory Budget (projected)

| Scenario | GPU bytes/param | Max Params (8 GB) |
|:---------|:---------------:|:-----------------:|
| Current (fp32 all-GPU) | 13 B | **~300M** |
| + 8-bit AdamW | 7 B | **~1.1B** |
| + 8-bit AdamW + bf16 | **5 B** | **~1.5B** |
| + all above + Flash Attn | ~4.5 B | **~1.7B** |

**Target:** **1B ternary params** (≈ 600M fp16 equivalent quality), trainable
in ~7 GB VRAM. This is **5× the current 300M ceiling.**

### Implementation Sprint

| Step | What | Time | Priority | Status |
|:-----|:-----|:----:|:--------:|:------:|
| **OPT-1** | Install `bitsandbytes`, test 8-bit AdamW on MNIST DQT | 10 min | 🔴 | ✅ **DONE** (acc 94.45% vs 92.99% fp32; resume round-trip Δ=0.0) |
| **OPT-2** | Convert all training scripts: AdamW → `AdamW8bit` | 30 min | 🔴 | ✅ **DONE** (10 scripts + `utils/optimizers.py::make_adamw`) |
| **OPT-3** | bf16 weight_float + autocast in training loop | 30 min | 🔴 | ✅ **DONE** (`--dtype bf16`; DQT backward dtype-agnostic) |
| **OPT-4** | Replace manual attention with `F.scaled_dot_product_attention` | 30 min | 🟡 | ✅ **DONE** (layer + M2.1 integration tests pass) |
| **OPT-5** | Test `torch.compile` (skip if it breaks custom autograd) | 15 min | 🟢 | ⏸️ **SKIPPED** (no C compiler for Triton in this env) |
| **OPT-6** | Integrate all optimizations, run M2.2 smoke test | 1 h | 🔴 | ✅ **DONE** — 341 integration ✅; **M2.2 smoke peak 5.03 GB (batch 4)** vs 6.5 GB baseline (−23%) |
| **OPT-7** | Increase batch size (4→8) / seq length where VRAM allows | 30 min | 🟡 | ✅ **DONE** — batch-8 smoke peak **5.23 GB** < 7.5 GB gate → default bumped 4→8 |
| **OPT-DOC** | Update docs (ROADMAP, GOALS, README, benchmarks) | 1 h | 🟡 | ✅ **DONE** — E030 report, benchmarks §6 measured, GOALS/README updated |

**Go/No-go gate:** OPT-1 (8-bit AdamW + DQT MNIST accuracy == fp32 baseline)
— ✅ **PASSED** (8-bit accuracy ≥ fp32; `state_dict()` round-trips exactly).

### Milestones

| Milestone | Target | Priority | Status |
|:----------|:------:|:--------:|:------:|
| **M2.6** 8-bit AdamW + bf16 on all training scripts | All scripts converted, MNIST smoke OK | 🔴 Critical | ✅ **DONE** |
| **M2.7** Flash Attention / SDPA in transformer attention | SDPA passes transformer layer tests | 🟡 High | ✅ **DONE** |
| **M2.8** 1B-param DQT Transformer smoke test | Stable training at 1B ternary params | 🔴 Critical | ✅ **GO** — **1.02B ternary, 20 steps, loss 3.96 (< random baseline), peak 8.04 GB** on RTX 4060 |
| **M2.9** Memory benchmark report | Measured: VRAM, speed, accuracy impact | 🟡 High | ✅ **DONE** — [E030](research/docs/experiments/E030-m2-9-memory-benchmark.md): M2.2 −22/−31%, 1B fits in 8.04 GB |

---

# 🧠 PH-Neuro Brain — The Real Product

> The infrastructure above (Phases 0–2.5) is the **pre-training toolkit** —
> it builds efficient "born networks." The product is what comes next.

## Brain Phase 0: Foundation Research (Aug 2026) ✅ COMPLETE

**Goal:** Answer every open question before writing a single line of new code.
**Rule:** Investigate → Decide → Implement. No implementation before investigation is documented.
**Status (2026-08-12):** All 5 foundation steps closed. Evaluation protocol LOCKED. Phase 1.1 implementation is next.

| Step | What | Question It Answers | Status |
|:----:|:-----|:--------------------|:------:|
| **0.1** | Model Selection | Which pre-trained model is the best "born brain"? Survey (HuggingFace, 2026-08-12) → **primary SmolLM2-1.7B · gen-test GPT-2 124M · scaling SmolLM2 135M→1.7B · bench BitNet b1.58 2B4T**. [Report](docs/brain/00-model-selection.md) | ✅ |
| **0.2** | Plasticity Mechanism Survey | Which local learning rule could work on a frozen backbone? Catalog Hebbian, Oja, BCM, STDP, 3-factor, predictive coding, target propagation, Forward-Forward, Equilibrium Propagation. **→ 3-factor Hebbian with global surprise modulator selected.** | ✅ |
| **0.3** | Surprise Signal Design | What tells the brain "learn now"? Survey prediction error, Bayesian surprise, information content, novelty, uncertainty, free energy. **→ Sequence-mean loss dev. from EMA, sigmoid-modulated, global float32 scalar M.** [Report](docs/brain/02-surprise-signal.md) | ✅ |
| **0.4** | Architecture Design | How does the Brain Wrapper hook into any HuggingFace model? Forward hooks, monkey-patching, custom wrapper. Design the public API. **→ Output-modification forward hooks + thin `BlockWrapper` adapter; full `BrainWrapper` API + learn/checkpoint/GPU spec.** [Report](docs/brain/03-architecture.md) | ✅ |
| **0.5** | Evaluation Protocol | What does success look like? Domain adaptation ppl, forgetting resistance, forward/backward transfer. Baselines: frozen, random plastic, full fine-tune, LoRA. **→ LOCKED: WikiText-2→PubMed (10.65→11.67 ppl, verified); budgets 1K/10K/100K/1M (100K=primary, EMA τ≈102K tok); baselines frozen/random/const-M/LoRA-r1 (98,304-param exact match); full-FT infeasible on 8 GB; window 512/stride 256; d=0.5@80%→16,074 tok; success=Δppl≥0.5 ppl, p<0.05, <1% forgetting.** [Report](docs/brain/04-evaluation-protocol.md) | ✅ |

**Deliverables:** `docs/brain/00-model-selection.md` through `docs/brain/04-evaluation-protocol.md`

---

## Brain Phase 1: Proof of Concept — Vector Bias Plasticity

**Goal:** The simplest possible experiment that tests the core hypothesis.
**Hypothesis:** Local surprise-modulated Hebbian updates on a frozen pre-trained backbone can produce measurable domain adaptation without catastrophic forgetting.

| Step | What | Key Question | Status |
|:----:|:-----|:-------------|:------:|
| **1.1** | Minimal Viable Experiment | Frozen SmolLM2-1.7B (primary) + vector bias per transformer block + surprise-modulated Hebbian. WikiText-2 → PubMed, 100K tokens (primary), 10K go/no-go. Does ppl improve? | 🟡 **PARTIAL SUCCESS** — mechanism works; surprise modulator ESSENTIAL (constM control: +10.7% catastrophic forgetting, −0.57 tgt; surprise: +0.03 tgt, +0.37% forgetting, p=0.003); Δppl=+0.034 ≪ 0.5 practical bar. [Report](docs/brain/05-e031-minimal-viable.md) |
| **1.2** | Capacity & Gain Experiment (re-scoped; protocol §11 deviation) | E031 identified capacity + modulator conservatism as bottlenecks. E032 tests low-rank capacity (r=1/2/4), a full gain sweep (η, s₀, k, M_max), the untested decay axis, and a matched-budget **backprop LoRA upper bound**; plus a 1M anneal. [Report](docs/brain/06-e032-capacity-gain.md) | ❌ **NEGATIVE — LOW-RANK HEBBIAN REJECTED** — capacity destroys (r1/2/4 = −1.35/−1648/−3172), every gain knob destructive, decay neutral, 1M compounds to −7381; **LoRA at the same 344K-param budget exceeds the 0.5 bar** (+1.52). Missing ingredient = **credit assignment**, not capacity/gain/decay. |
| **1.3** | Predictive Coding (re-scoped; protocol §11 deviation) | The last local-rule family with a credit-assignment story: **error-based predictive coding** (per-injection-site reconstruction error, surprise-gated) at the **matched 344K LoRA budget**. Can a no-backprop error signal adapt a frozen LM? [Report](docs/brain/07-e033-predictive-coding.md) | ❌ **NEGATIVE — INERT / LOCAL-RULE QUESTION CLOSED** — Δppl_PC = **+0.001 ± 0.003** (p=0.737, seeds +0.001/+0.004/−0.003): statistically null, ~500× below the 0.5 bar. The error-driven update is **stable** (source *improved* −0.012%, no forgetting — fixes E032's instability) but **inert** (plastic weights barely move; mean\|B\|≈0.0016). **Pre-registered kill criterion fired → pivot to the backprop-LoRA product path.** |

**Go/No-go gate (1.1) — LOCKED in [04-evaluation-protocol.md](docs/brain/04-evaluation-protocol.md):** at the 100K-token primary point — Δppl ≥ **0.5 ppl** on PubMed (practical bar; ≥~0.23 ppl is the detectable floor), p < 0.05 (paired, across ≥3 seeds), Δppl > random-plastic baseline, <1% source (WikiText-2) degradation, and surprise-modulated ≥ constant-M. Quick 10K run = mechanism go/no-go only (M≈const there).

---

## Brain Phase 2: Scaling Plasticity Capacity

**Goal:** Move from vector biases to richer plastic representations. Demonstrate that more capacity → better adaptation without forgetting.

| Step | What | Key Question | Status |
|:----:|:-----|:-------------|:------:|
| **2.1** | Low-Rank Plastic Matrices | LoRA-style BA^T updated via local (non-backprop) Hebbian rules. Does rank>1 help more than vector bias? | ❌ **ANSWERED NEGATIVE (E032 + E033) — LOCAL RULES REJECTED.** E032: local low-rank Hebbian is catastrophically unstable (rank-1 −1.35 … rank-4 −3172, 1M −7381). E033 (Step 1.3 re-scope): error-based predictive coding at the same matched budget is stable but **inert** (Δppl +0.001, p=0.737). **The Phase 2 scaling mechanism is backprop LoRA** (proven +1.52, the product path). |
| **2.2** | Ternary Plastic Weights | Convert plastic weights from float to {-1, 0, +1} using existing DQT/hysteresis infrastructure. Does ternary match float adaptation quality? | ⬜ |
| **2.3** | Consolidation Mechanism | Sleep-inspired memory transfer: important plastic changes move to long-term store with slower decay. Does this reduce forgetting across sequential domains? | ⬜ |

---

## Brain Phase 3: Continual Learning at Scale

**Goal:** Demonstrate that Brain Wrapper enables a single model to continually adapt across many domains — a capability no existing system has.

| Step | What | Key Question | Status |
|:----:|:-----|:-------------|:------:|
| **3.1** | Multi-Domain Sequential Adaptation | 5–10 diverse text domains in sequence. Does the model improve on ALL domains over frozen baseline? | ⬜ |
| **3.2** | Scaling Laws | How does adaptation improvement scale with model size (SmolLM2 135M → 360M → 1.7B)? With plastic capacity (0.01% → 5% of params)? | ⬜ |
| **3.3** | Beyond Language: Vision | Apply Brain Wrapper to ViT/DINOv2. Class-incremental learning on CIFAR-100. Is this a general principle, not just a language trick? | ⬜ |

---

## Current Focus (August 2026)

> **Brain Phase 1.3 (E033): ✅ COMPLETE — ❌ NEGATIVE / INERT — LOCAL-RULE
> QUESTION CLOSED (2026-08-14).**
> **Result (4 cells, 0 failures, pre-registered):** the final local-rule test
> — **error-based predictive coding** at the same matched 344K-param budget
> as the E032 LoRA baseline. **Δppl_PC = +0.001 ± 0.003 (p = 0.737, per-seed
> +0.001 / +0.004 / −0.003)** — statistically null, ~500× below the 0.5
> practical bar and below the 0.2-ppl "within noise" floor. The error-driven
> reconstruction-error update is **stable** (source *improved* −0.012%, zero
> forgetting — it removes E032's surprise-feedback/Hebbian-concentration
> instability) but **inert**: the plastic weights barely move (mean|A| ≈ init,
> mean|B| ≈ 0.0016) and the per-site linear-inverse reconstruction error
> carries no usable credit-assignment direction on a 1.7B residual
> transformer. **Pre-registered kill criterion fired (report §8/§12): the
> local-rule scientific question is CLOSED.** E031 (vector-bias Hebbian:
> +0.034, stable, sub-threshold), E032 (low-rank Hebbian: destructive
> −1.35), E033 (predictive coding: stable, inert +0.001) span every plausible
> local-rule axis; none reaches the 0.5-ppl bar at a matched budget.
> **Next (pre-registered consequence): PIVOT to the backprop-LoRA product
> path** — E032's proven +1.52 bound (344K params, source improved) becomes
> the Phase 2 adaptation/scaling mechanism; Phase 2.2 (ternary/DQT) is
> re-scoped to apply the existing DQT/hysteresis infrastructure to LoRA
> adapters. Protocol amendments for the 1.2 and 1.3 re-scopings are logged in
> [04-evaluation-protocol.md §11](docs/brain/04-evaluation-protocol.md);
> full report [07-e033-predictive-coding.md](docs/brain/07-e033-predictive-coding.md).

### Why This Direction

After 19 Hebbian experiments and a complete DQT training pipeline, we
concluded that (a) local Hebbian rules cannot train deep networks from
scratch (~88% MNIST ceiling, proven), and (b) DQT is excellent model
compression but is still backprop-based — it doesn't achieve brain-like
learning.

The breakthrough insight: **the brain is not trained from scratch.** It's
born pre-wired (evolution = pre-training) and learns through local
plasticity. We can replicate this: take any open-source pre-trained model
(the "born brain") and add local Hebbian/neuromodulated plasticity for
lifetime learning. No backprop through frozen layers. No forgetting.

### Infrastructure Status

> ✅ Phase 0–2.5 COMPLETE. DQT training, ternary weights, MoE, memory
> optimization, 1B-param ceiling — all ready to build the born networks
> that Brain Wrapper will use.
| dqt_cnn_cifar100 (CIFAR-100) | 614.9 KB | 2.40 MB | **4.00×** | 0.227 ms |

| Method | dqt_cnn | dqt_cnn_cifar100 |
|:-------|:-------:|:----------------:|
| DQT (measured) | 363.5 MB | 334.2 MB |
| STE (est.) | 1,635.6 MB | 1,504.1 MB |

TF Lite comparison is theoretical (no TF Lite installed); PH-Neuro numbers
are measured (CPU batch=1, ONNX runtime; GPU 1-epoch peak).

**Key deliverables:** `run_m1_5_benchmarks.py` (CLI runner), `run_m1_5_benchmarks.sh`
(orchestration), `results/phase1/m1_5_results/` (3 JSON), report `E024-m1-5-benchmarks.md`,
`docs/benchmarks.md` updated. Reproduce: `bash scripts/run_m1_5_benchmarks.sh`.

### M1.4 — CLOSED (GO) ✅

**Goal achieved:** Production-quality documentation. 5 files created/rewritten:
`README.md` (179 lines, <200 ✅), `docs/api.md` (12 public APIs, 880 lines),
`docs/quickstart.md` (4 runnable examples), `docs/benchmarks.md` (DQT vs STE vs TF Lite),
`README_EL.md` (Greek version). All examples verified runnable, all links checked.

### M1.3 — CLOSED (GO) ✅

**Goal achieved:** ONNX export pipeline for ALL DQT models. 3 models exported
and verified (ste_mlp, dqt_cnn, dqt_cnn_cifar100). torch ≡ ONNX output to
machine precision. All models <17 MB ONNX / <1 MB packed (2-bit).

| Μοντέλο | ONNX size | Packed (2-bit) | Verified |
|:--------|:---------:|:--------------:|:--------:|
| ste_mlp (MNIST) | 2.06 MB | 130.6 KB | ✅ |
| dqt_cnn (CIFAR-10) | 16.33 MB | 1.02 MB | ✅ |
| dqt_cnn_cifar100 (CIFAR-100) | 9.64 MB | 615 KB | ✅ |

**Key deliverables:** `export.py` (dqt_to_inference_model, export_to_onnx, verify_onnx),
CLI runner, 8/8 tests, Raspberry Pi deployment guide (export_guide.md),
TF32 precision bug found & fixed.

**Note:** demo models exported (1-2 epoch training). Re-export with trained
M1.1/M1.2 checkpoints for production artifacts (same sizes, 79%/54% accuracy).

### M1.2 — CLOSED (CONDITIONAL GO) 🟡

**Scientific goal achieved:** DQT generalizes from CIFAR-10 to CIFAR-100.
3-conv CNN (64→128→256), DQT **54.15%** vs STE baseline **38.2%** = **+15.95pp**.
Annealing @80% validated, flip noise eliminated (0.0005-0.0006).

**Accuracy gate missed:** Mean best 54.15% (gate: >55%), −0.85pp. Ceiling ~54%
for 3-conv CNN. 200 epochs did NOT help (53.65%, −0.50pp — confirmed architectural).

| Attempt | Config | Mean Best Acc | Result |
|:--------|:-------|:-------------:|:------:|
| E022 | 3-conv, 150ep, lr=0.01 | **54.15%** | 🟡 MARGINAL |
| E022.1 | 3-conv, 200ep, lr=0.01 | 53.65% | 🟡 MARGINAL (worse) |

**Key lessons:** (1) DQT >> STE on CIFAR-100 (+16pp), confirming M1.1 finding.
(2) Architectural ceiling, not epoch-limited — more epochs don't help.
(3) 4-conv (64→128→256→512) remains as future optimization.
(4) CosineAnnealing over 200ep lowers LR too slowly — longer stochastic phase
without higher peak.

**Deliverables:** `dqt_cnn_cifar100()` in `dqt_models.py`, runner, shell script,
8 integration tests, 6 result JSONs. See [E022](research/docs/experiments/).

### M1.1 — CLOSED (CONDITIONAL GO) ✅

**Scientific goal achieved:** First DQT convolutional layer (`TernaryDQTConv2d`)
validated end-to-end. DQT beats STE by +2.89pp on identical architecture.
Backward numerically exact vs PyTorch autograd. 16 unit + 6 integration tests.

**Accuracy gate missed:** Mean best 78.98% (gate: >80%), ceiling ~79% for this
2-conv CNN architecture. 4 attempts, 12 runs, spread only 1.53pp.

| Attempt | Config | Mean Best Acc | Result |
|:--------|:-------|:-------------:|:------:|
| E020 | 512-head, no anneal | 77.65% | 🔴 NO-GO |
| E020+ | 512-head, no anneal, 150ep | 78.36% | 🔴 NO-GO |
| E021 | 256-head, anneal@85%, p=15 | 78.42% | 🔴 NO-GO |
| E021.2 | 256-head, anneal@80%, p=25 | **78.98%** | 🔴 NO-GO (best) |
| E021.3 | 512-head, anneal@80%, p=25 | 78.80% | 🔴 NO-GO |

**Key lessons:** (1) Annealing stochastic→deterministic sign eliminates late-training
flip noise (0.18→0.0008). (2) 0% sparsity in deterministic phase — DQT loses
sparsity advantage when sign() is used. (3) Ceiling is architectural (2-conv CNN),
not tuning. (4) Larger CNN (3-conv layers) needed for M1.2.

**Deliverables:** `TernaryDQTConv2d` in `ste_dqt_conv.py`, `dqt_cnn()` in
`dqt_models.py`, runner, shell script, 22 tests, 12 result JSONs.
See [E020–E021.3](research/docs/experiments/).
