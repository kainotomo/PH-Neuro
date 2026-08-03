# PH-Neuro — Product Roadmap

> **Last updated:** 2026-08-03
> **Status:** Phase 1 — DQT Proof of Concept complete

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
| **M1.1** DQT CNN on CIFAR-10 | >80% accuracy | 🔴 Critical | ⬜ |
| **M1.2** DQT CNN on CIFAR-100 | >55% accuracy | 🟡 High | ⬜ |
| **M1.3** Model export (ONNX/C) | <100MB, runs on Raspberry Pi | 🟡 High | ⬜ |
| **M1.4** Production README + docs | Clear quickstart + API docs | 🟡 High | ⬜ |
| **M1.5** Memory benchmarks vs TF Lite | 4× smaller, 2× faster inference | 🟢 Medium | ⬜ |

**Go/No-go gate:** M1.1 must pass. If DQT fails on CIFAR-10 (<70%), pivot to Transformer-only path.

---

## Phase 2: Tiny Transformer (Sep–Nov 2026)

**Goal:** First ternary language model — the smallest useful LLM.

| Milestone | Target | Priority |
|:----------|:------:|:--------:|
| **M2.1** DQT Transformer 100M params | Perplexity <30 on TinyStories | 🔴 Critical |
| **M2.2** DQT Transformer 500M params | Perplexity <20 on WikiText-2 | 🟡 High |
| **M2.3** MoE DQT Transformer | 1B total, 200M active, <250MB on disk | 🟡 High |
| **M2.4** On-device inference demo | Token generation on smartphone | 🟡 High |
| **M2.5** Public demo + blog post | Hacker News / Reddit launch | 🟢 Medium |

**Go/No-go gate:** M2.1 must show that DQT Transformer trains stably. If perplexity <30, proceed to M2.3 (MoE scaling).

---

## Phase 3: MVP & First Customers (Dec 2026–Feb 2027)

**Goal:** Working product + paying customer.

| Milestone | Target | Priority |
|:----------|:------:|:--------:|
| **M3.1** SDK v0.1 | Python package: `pip install ph-neuro` | 🔴 Critical |
| **M3.2** 2 reference models | Vision classifier + Text chat | 🔴 Critical |
| **M3.3** Documentation site | Tutorials, API reference, benchmarks | 🟡 High |
| **M3.4** First paying customer | Pilot project or license | 🔴 Critical |
| **M3.5** Startup incorporation | Cyprus LLC, bank account, legal | 🟡 High |

---

## Phase 4: Scale (Mar–Aug 2027)

**Goal:** Funding, team, production scale.

| Milestone | Target |
|:----------|:------|
| **M4.1** Pre-seed funding | €50K–200K (EU grants, angels) |
| **M4.2** Hire 1–2 engineers | ML + Systems |
| **M4.3** 10B MoE model training | Cloud GPU cluster |
| **M4.4** Enterprise pilots | 3–5 paying customers |
| **M4.5** Publish results | arXiv + conference |

---

## Phase 5: Commercial Platform (2027+)

**Goal:** Sustainable business.

- Self-serve SDK for custom on-device models
- Enterprise tier with SLAs
- Target: 50+ customers, €1M+ ARR
- Exit: acquisition by edge AI platform or cloud provider

---

## Current Focus (August 2026)

Right now, the ONLY thing that matters:

> **M1.1: DQT CNN on CIFAR-10 >80% accuracy.**

Everything else depends on this.

### M1.1 progress

| Attempt | Config | Mean Best Acc | Result |
|:--------|:-------|:-------------:|:------:|
| E020 (original) | 8192→512 head, 100% stochastic | 77.65% | 🔴 NO-GO |
| E020 fallback | +150 epochs | 78.36% | 🔴 NO-GO |
| **E021 (RETRY)** | 8192→256 head + anneal→deterministic @85% | **78.42%** | 🔴 NO-GO |

**E021 analysis:** annealing removed the late-training flip jitter (flip rate
0.18 → 0.0006) and helped seeds 42/43 (+1.5/+2.5 pp), but seed 44 regressed
(−1.74 pp) and early-stopping cut the deterministic tail short for seeds 43/44.
Next fallbacks: anneal@80%, longer/disabled early stopping, lr=0.005, or
512-head + annealing only.
