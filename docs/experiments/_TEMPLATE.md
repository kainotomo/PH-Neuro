# Experiment NNN: [Title]

- **Date:** YYYY-MM-DD
- **Git commit:** `abc1234`
- **Status:** [running | completed | failed | abandoned]
- **Phase:** [0 | 1 | 2 | 3 | 4]

---

## Hypothesis
_What are we testing? What do we expect to happen?_

[One sentence hypothesis.]

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | e.g., "3-layer CNN: Conv(3→64,3×3)→Conv(64→128,3×3)→FC(128→10)" |
| Total parameters | e.g., "312,456" |
| Ternary weights | e.g., "~1.2 MB (packed)" |
| Latent scores | e.g., "~0.6 MB (fp16)" |
| Weight init | e.g., "All zeros, latent scores ~ Uniform(-0.1, 0.1)" |
| θ_upper | e.g., "5.0" |
| θ_lower | e.g., "1.0" |
| Learning rate (hidden) | e.g., "0.001" |
| Learning rate (output) | e.g., "0.01" |
| Decay rate | e.g., "1e-5" |
| Hebbian variant | e.g., "Basic (Δw = lr × pre × post)" |
| Anti-Hebbian | e.g., "Yes, lr=-0.005 for wrong classes" |
| Batch size | e.g., "128" |
| Epochs | e.g., "50" |
| Training steps | e.g., "19,531" |
| Dataset | e.g., "CIFAR-10 (50K train, 10K test)" |
| Data augmentation | e.g., "RandomCrop(32, padding=4), RandomHorizontalFlip" |
| Hardware | e.g., "RTX 4060 8 GB" |
| Training time | e.g., "12 min" |
| Training throughput | e.g., "15,000 samples/sec" |
| Memory usage | e.g., "1.2 GB VRAM" |

---

## Results

### Main Metrics

| Metric | PH-Neuro (this run) | Baseline: Backprop | Baseline: Float Hebbian |
|--------|--------------------|--------------------|-------------------------|
| Accuracy (test) | XX.X% | XX.X% | XX.X% |
| Forgetting (if continual) | X.X% | XX.X% | — |
| Weight sparsity (% 0) | XX.X% | — | — |
| Weight flip rate (per step) | X.XX% | — | — |

### Learning Curves

```
[Accuracy vs epoch plot]
[Loss/perplexity vs epoch — if applicable]
[Weight distribution over time: stacked bar %+1, %0, %-1]
[Latent score distribution over time: histogram]
```

### Continual Learning (if applicable)

| Task Sequence | Accuracy After Each Phase |
|---------------|--------------------------|
| After Task 1 | T1: XX%, T2: —, T3: —, T4: —, T5: — |
| After Task 2 | T1: XX%, T2: XX%, T3: —, T4: —, T5: — |
| After Task 3 | T1: XX%, T2: XX%, T3: XX%, T4: —, T5: — |
| After Task 4 | T1: XX%, T2: XX%, T3: XX%, T4: XX%, T5: — |
| After Task 5 | T1: XX%, T2: XX%, T3: XX%, T4: XX%, T5: XX% |
| **Forgetting** | **X.X%** |

### Layer-wise Analysis

| Layer | Weight Sparsity | Mean Latent Score | Flip Rate | Activation Sparsity |
|-------|----------------|-------------------|-----------|---------------------|
| Conv1 | XX% | X.XX | X.XX% | XX% |
| Conv2 | XX% | X.XX | X.XX% | XX% |
| FC1 | XX% | X.XX | X.XX% | XX% |

### Sample Outputs (for generative models)

```
Prompt: "Once upon a time, there was a"
Generated: "[model output here]"
Quality rating: [1-5]
```

---

## Observations

### What worked well?
- 

### What failed or was surprising?
- 

### Comparison to hypothesis
- 

---

## Bugs & Issues

- [ ] **Bug**: [Description]
  - **Symptom**: [What went wrong]
  - **Cause**: [Root cause analysis]
  - **Fix**: [How it was resolved]
  - **Commit**: `abc1234`

---

## Ablation Notes

_Record any variations tried during this experiment._

| Variation | Result | Notes |
|-----------|--------|-------|
| Without decay | XX% acc | Worse — weights became saturated |
| θ_upper=3.0 | XX% acc | Better — faster learning but more oscillation |
| BCM rule instead of basic | XX% acc | Slightly better feature selectivity |

---

## Artifacts

- **Model checkpoint**: `checkpoints/exp-NNN-YYYYMMDD.pt`
- **TensorBoard logs**: `runs/exp-NNN/`
- **Weights & Biases run**: `[wandb link]`
- **Generated samples**: `samples/exp-NNN/`

---

## Next Steps

1. [Action item based on this experiment]
2. [New hypothesis to test]
3. [Parameter to tune]
