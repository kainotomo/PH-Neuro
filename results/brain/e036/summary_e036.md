# E036 cross-seed summary

| condition | D1 (PubMed) Δppl | D2 (CNN) Δppl | D3 (C4) Δppl | BT_D1 | BT_D2 |
|:----------|:-----------------:|:-------------:|:------------:|:-----:|:-----:|
| B1 (independent) | +0.892±0.206 (p=0.017) | +0.106±0.044 (p=0.052) | +0.178±0.034 (p=0.012) | — | — |
| B2 (continuing) | — | — | +0.187±0.017 (p=0.003) | +0.265±0.123 (p=0.065) | -0.039±0.010 (p=0.022) |
| C (consolidation, ST_D3) | — | — | +0.020±0.001 (p=0.002) | -0.001±0.001 (p=0.402) | -0.008±0.003 (p=0.039) |
| C (consolidation, LT_D3) | — | — | +0.020±0.001 (p=0.002) | -0.001±0.001 (p=0.402) | -0.008±0.003 (p=0.039) |

**Forward transfer (C ST_D3 vs B1, per-seed paired):**

- Δppl_C_D3 − Δppl_B1_D3 = **-0.158±0.034 (p=0.015)** (per-seed [-0.1208, -0.1869, -0.1675])
- ratio C/B1 on D3 Δppl = **0.112**

**Adaptation speed (steps to plateau, D3 50K probes):**

- C: {42: 367, 43: 357, 44: 367}
- B1: {42: 180, 43: 180, 44: 170}

**Storage:**

- B1 (3 adapters): **259200.0 B**
- B2 (1 adapter): **86400.0 B** (ratio 0.333)
- C deployed (LT): **86400.0 B** (ratio 0.333)
- C LT+deltas (bitmap): **241254.0 B** (ratio 0.931)
- C LT+deltas (int32-index, conservative): **525102.0 B** (ratio 2.026)

**Pre-registered gates (§7):**

- C D3 ≥ B1 D3 (forward transfer): ❌ (-0.158 mean paired Δ)
- C BT_D1 < 0.1: ✅ (mean -0.0005)
- C BT_D2 < 0.1: ✅ (mean -0.0077)
- C storage ≤ B1 (deployed): ✅ (ratio 0.333)
- C storage ≤ B1 (LT+deltas): ✅ (ratio 0.931)
