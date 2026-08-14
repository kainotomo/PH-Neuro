# E033 cross-seed summary

| config | frozen tgt ppl | plastic tgt ppl | Δppl (mean±SD) | p | forgetting | block d |
|:-------|:--------------:|:---------------:|:---------------:|:--:|:----------:|:-------:|
| pc | 11.46 | [11.4562, 11.4536, 11.4597] | +0.001 ± 0.003 (p=0.737) | -0.012% | +0.031 |

**Verdict (pre-registered, 100K primary point):**

- **Δppl_PC = +0.001 ± 0.003 (p=0.737, per-seed [0.001, 0.0035, -0.0025])**
- **Δppl ≥ 0.5 practical bar?** ❌
- **Sign agreement with LoRA (Δppl > 0)?** ✅ (all 3 seeds positive: False)
- **Forgetting < 1%?** ✅ (mean -0.012%)
- **Ratio Δppl_PC / Δppl_LoRA = 0.000** (LoRA bound +1.520)

**Consequence (pre-registered):** PC failed the pre-registered criteria at matched budget (Δppl = +0.001, p = 0.737). The local-rule scientific question is CLOSED: the project pivots to the backprop-LoRA product path. No second PC variant, no hyperparameter rescue, no capacity escalation.
