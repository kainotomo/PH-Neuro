# E034 cross-seed summary

## Experiment 1 — single-domain (WikiText-2 → PubMed, 100K)

| method | Δppl (mean±SD, p) | per-seed | source forgetting | mean M | eff. mean lr |
|:-------|:------------------:|:--------:|:----------------:|:------:|:------------:|
| gated | +0.902±0.182 (p=0.013) | [0.6928, 1.0252, 0.9877] | -2.658% | 0.234 | 1.01e-04 |
| const_reduced control | +0.947±0.153 (p=0.009) | [0.7733, 1.0625, 1.0054] | -2.727% | 1.000 | 1.01e-04 |

**Exp. 1 verdict:**

- Gated Δppl = **+0.902 ± 0.182** (p=0.013, per-seed [0.6928, 1.0252, 0.9877])
- **Δppl ≥ 0.5 bar?** ✅
- **Source degradation < 1%?** ✅ (mean -2.658%)
- **Δppl_gated ≤ Δppl_plain** (+0.902 vs +1.520): ✅ (learns less by design)
- **M-trace:** mean M = 0.234, effective mean lr = 1.01e-04

## Experiment 2 — sequential two-domain (WikiText → PubMed → CNN/DailyMail)

| method | PubMed Δppl after seq. | CNN Δppl (adapt to d2) | source forgetting |
|:-------|:-----------------------:|:----------------------:|:-----------------:|
| plain | -0.334±0.183 (p=0.087) | +0.147±0.008 (p=0.001) | +4.855% |
| gated | +0.911±0.175 (p=0.012) | +0.116±0.019 (p=0.008) | -2.764% |

**Backward transfer on domain 1 (PubMed):** BT = pubmed_ppl_after_p2 − pubmed_ppl_after_p1 (+ = forgetting).

| method | BT mean | BT per seed |
|:-------|:-------:|:-----------:|
| plain | +1.8536 ± 0.2059 | [1.6445, 2.0562, 1.8601] |
| gated | -0.0090 ± 0.0104 | [-0.0177, -0.0118, 0.0026] |

**Selectivity claim (pre-registered: BT_gated < BT_plain):**
- BT_gated − BT_plain = **-1.8626** → BT_gated < BT_plain → ✅ **CLAIM HOLDS**
- Seed agreement: 3/3 (✅ ≥ 2/3)
