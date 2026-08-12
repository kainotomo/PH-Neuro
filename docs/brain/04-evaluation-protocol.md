# Step 0.5 — Evaluation Protocol

> **Status:** ⬜ Not Started
> **Goal:** Define exactly how we measure success. Lock the protocol before running any experiments to avoid post-hoc metric selection.

---

## Primary Metric

**Domain adaptation perplexity improvement:**
Δppl = ppl_frozen(target_domain) — ppl_plastic(target_domain)

Positive Δppl means the plastic weights improved performance on the target domain. The improvement must be statistically significant (p < 0.05, 3+ seeds).

---

## Secondary Metrics

### 1. Forgetting Resistance
Δppl_source = ppl_plastic(source_domain) — ppl_frozen(source_domain)

Should be ≤ 0 (plastic weights should never hurt source domain performance). Lenient threshold: <1% degradation. Strict threshold: <0.1% degradation.

### 2. Forward Transfer
After learning domain A, test on domain B (before any B training). Positive transfer means learning A helped B. Negative transfer means interference.

### 3. Backward Transfer
After learning domain B, re-test on domain A. Measures whether learning B caused forgetting of A.

### 4. Plasticity Efficiency
Δppl per plastic parameter. How much improvement per unit of plasticity capacity?

---

## Evaluation Domains

### Phase 1.1 — Single Domain Shift

| Domain | Corpus | Size (tokens) | Why |
|:-------|:-------|:-------------:|:----|
| **Source** | WikiText-2 (test set) | ~250K | Standard benchmark. GPT-2's training domain. |
| **Target** | PubMed abstracts | ~10K adaptation, ~50K test | Scientific/medical domain. Very different vocabulary and style from web text. Measurable shift. |

Alternative targets for follow-up:
- **Legal:** US Supreme Court opinions — formal, structured, domain-specific terminology
- **Code:** Python from The Stack — completely different syntactic structure
- **News:** Reuters or CNN articles — moderate shift from Wikipedia
- **Books:** Project Gutenberg — narrative style, longer-range dependencies

---

## Adaptation Data Sizes

Test adaptation at multiple data budgets to characterize the learning curve:

| Budget | Tokens | ~Pages of Text | Rationale |
|:-------|:------:|:-------------:|:----------|
| Micro | 1K | ~2 pages | Can plasticity help from a tiny sample? (Brain-like: humans learn from few examples) |
| Small | 10K | ~20 pages | Primary test point. Enough for statistical signal. |
| Medium | 100K | ~200 pages | Does adaptation plateau or continue improving? |
| Large | 1M | ~2,000 pages | Upper bound. Is there a saturation point? |

---

## Baselines

| Baseline | What | Expected Behavior |
|:---------|:-----|:------------------|
| **Frozen (zero plasticity)** | Evaluate GPT-2 out-of-the-box on target domain | Baseline ppl. Plasticity should beat this. |
| **Random plastic weights** | Initialize plastic weights randomly (fixed seed), no training | Controls for added capacity alone. Plasticity training should beat this. |
| **Plasticity with constant LR** | Hebbian updates with M=1 (no surprise modulation) | Tests whether surprise modulation matters. Surprise-modulated should beat or match this. |
| **Full fine-tuning (upper bound)** | Unfreeze GPT-2, train with backprop on target domain | What's the maximum possible adaptation? Plasticity won't match this, but the gap tells us the ceiling. |
| **LoRA fine-tuning (practical upper bound)** | LoRA with backprop, rank matching plastic capacity | Fair comparison: same parameter budget, different update rule. |
| **EWC** | Fine-tune with Elastic Weight Consolidation | Does local plasticity beat a classic continual-learning method? |

---

## Statistical Protocol

1. **Minimum 3 random seeds** per experiment (different weight initializations for plastic weights, different data order).
2. **Report mean ± standard deviation** for all metrics.
3. **Confidence intervals:** Bootstrap 95% CI for perplexity differences (to account for the fact that perplexity on held-out text has high variance).
4. **Significance testing:** Paired t-test comparing Δppl distributions across seeds.
5. **Effect size:** Cohen's d for Δppl. Small effect: d=0.2, medium: d=0.5, large: d=0.8.

---

## Perplexity Computation Details

- Use the model's own tokenizer (GPT-2 BPE tokenizer for GPT-2 experiments)
- Slide a context window of 1024 tokens (GPT-2's max context)
- Stride = 512 tokens (50% overlap for stable estimates)
- Compute token-level negative log-likelihood, aggregate to corpus-level perplexity
- Ignore the first token of each sequence (no context to condition on)
- Report both unweighted (per-token average) and weighted (per-sequence average, then mean across sequences)

---

## What Does Failure Look Like?

It's important to pre-register failure criteria to avoid post-hoc rationalization:

### Phase 1.1 is a failure if:
1. No statistically significant Δppl (plasticity doesn't help at all)
2. Δppl is positive but smaller than the random-weight baseline (plasticity training is worse than random)
3. Δppl is statistically significant but practically meaningless (e.g., 0.1 ppl improvement on a baseline of 30 ppl)

### Phase 1.1 is a success if:
1. Δppl > 0 with p < 0.05 AND
2. Δppl > random-weight baseline AND
3. Source domain ppl degradation < 1% AND
4. Surprise-modulated > constant LR (neuromodulation matters)

**Partial success (valuable result):** Δppl > 0 but surprise modulation doesn't matter — this would mean plasticity helps but our modulator design is wrong. That's still worth publishing.

**Informative failure:** Δppl ≈ 0 for all configurations — this would mean local Hebbian plasticity cannot adapt pre-trained transformers any better than random, confirming the negative result from our earlier Hebbian experiments extends to pre-trained backbones. Scientifically important. Publish as "Local Hebbian Plasticity Cannot Adapt Pre-Trained Transformers: Evidence from X Experiments."

---

## Experiment Tracking

Each experiment produces a results file:

```json
{
  "experiment": "e031_minimal_viable",
  "model": "gpt2",
  "plasticity": "vector_bias",
  "modulator": "prediction_error_ema",
  "target_domain": "pubmed",
  "adaptation_tokens": 10000,
  "seed": 42,
  "metrics": {
    "source_ppl_frozen": 29.40,
    "source_ppl_plastic": 29.38,
    "source_ppl_delta": -0.02,
    "target_ppl_frozen": 45.20,
    "target_ppl_plastic": 43.80,
    "target_ppl_delta": 1.40,
    "forgetting_pct": 0.07,
    "mean_surprise": 0.34,
    "plastic_weight_mean_magnitude": 0.012,
    "plastic_weight_sparsity": 0.87
  }
}
```

Results go in `results/brain/` with the naming convention:
`results/brain/e031/{model}_{target}_{modulator}_seed{seed}.json`

---

## Next Steps

- [ ] Set up evaluation harness (data loading, ppl computation, metric tracking)
- [ ] Establish baseline perplexities for all candidate models on all candidate domains
- [ ] Determine minimum sample size for statistical significance given ppl variance
- [ ] Create the results directory structure and JSON schema
