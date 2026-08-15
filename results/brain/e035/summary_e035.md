# E035 cross-seed summary

## Experiment 1 — single-domain (WikiText-2 → PubMed, 100K)

| variant | Δppl (mean±SD, p) | per-seed | source forgetting | s/step | storage |
|:--------|:------------------:|:--------:|:----------------:|:------:|:-------:|
| T-A-q (post-train q) | +0.618±0.150 (p=0.019) | [0.4453, 0.714, 0.6949] | -1.990% | 0.00 | 16× |
| T-A-qft (+calib) | +0.689±0.144 (p=0.014) | [0.523, 0.7808, 0.7641] | -2.239% | 0.00 | 16× |
| T-B (DQT) | +0.000±0.001 (p=0.992) | [-0.0006, 0.0011, -0.0004] | -0.007% | 0.42 | 16× |
| T-C (STE) | +0.892±0.206 (p=0.017) | [0.655, 1.0324, 0.9873] | -2.258% | 0.49 | 16× |

**Pre-registered bar (90% of float +0.902 = ≥ 0.81):**

- **ta_q**: Δppl = **+0.618 ± 0.150** (p=0.019) → ❌ ≥ 0.81 | forgetting ✅<1% | storage ✅ (15.9×, 86016 B packed / 86016 B disk) | 0.00 s/step
- **ta_qft**: Δppl = **+0.689 ± 0.144** (p=0.014) → ❌ ≥ 0.81 | forgetting ✅<1% | storage ✅ (15.9×, 86016 B packed / 86016 B disk) | 0.00 s/step
- **tb**: Δppl = **+0.000 ± 0.001** (p=0.992) → ❌ ≥ 0.81 | forgetting ✅<1% | storage ✅ (15.9×, 86016 B packed / 86016 B disk) | 0.42 s/step
- **tc**: Δppl = **+0.892 ± 0.206** (p=0.017) → ✅ ≥ 0.81 | forgetting ✅<1% | storage ✅ (15.9×, 86016 B packed / 86016 B disk) | 0.49 s/step

**Selection rule (§7.0) → best ternary variant: `tc`**


## Experiment 2 — sequential two-domain (WikiText → PubMed → CNN/DailyMail)

**Best variant: `tc`** — backward transfer on PubMed BT = **-0.0118 ± 0.0117** (per-seed [-0.0212, -0.0153, 0.0013]) → ✅ < 0.1

## Verdict

- Float gated baseline Δppl = **+0.902** (E034; 90% bar = **0.81**).
- **Any variant ≥ 90% bar?** ✅
- **Best variant (`tc`) ≥ 90%?** ✅
- **Source forgetting < 1% (best)?** ✅
- **Storage 16× confirmed?** ✅
- **Selectivity BT < 0.1 (best variant, two-domain)?** ✅