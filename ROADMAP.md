# PH-Neuro — Product Roadmap

> **Last updated:** 2026-08-04
> **Status:** Phase 1 — M1.1 closed (CONDITIONAL GO), M1.2 next

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
| **M1.2** DQT CNN on CIFAR-100 | >55% accuracy | 🟡 High | ⬜ |
| **M1.3** Model export (ONNX/C) | <100MB, runs on Raspberry Pi | 🟡 High | ⬜ |
| **M1.4** Production README + docs | Clear quickstart + API docs | 🟡 High | ⬜ |
| **M1.5** Memory benchmarks vs TF Lite | 4× smaller, 2× faster inference | 🟢 Medium | ⬜ |

**Go/No-go gate:** M1.1 — accuracy gate 78.98% (missed 80% by 1.02pp), but **scientific goal achieved**: DQT Conv2d layer validated, DQT > STE by +2.89pp on identical architecture. Closed as CONDITIONAL GO. See [E020–E021.3](research/docs/RESEARCH_SUMMARY.md).

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

> **M1.2: DQT CNN on CIFAR-100 >55% accuracy.** Next milestone.

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
