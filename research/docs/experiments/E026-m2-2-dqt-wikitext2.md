# Experiment E026: M2.2 — DQT Transformer 250M params on WikiText-2 (GO/NO-GO ppl<20)

- **Date:** 2026-08-06
- **Git commit:** `TBD`
- **Status:** ✅ COMPLETE — **🔴 NO-GO on the ppl<20 gate** (mean 479.9), but the
  scaling/stability goal (250M ternary fits 8 GB, trains stably, no NaN) was met.
- **Phase:** 2 (Tiny Transformer)

### Run log (interruption + recovery)

| Time (UTC) | Event |
|-----------|-------|
| 07:48 | Full run launched (detached) — seed 42, batch 4, ~7,005 steps |
| 07:48–08:01 | Seed 42 trained 1150 steps, loss 10.84 → **6.33**, ~1587 tok/s, GPU 4.8/8.0 GB (no NaN) |
| 08:01 | **Machine rebooted** (VS Code interruption) — process killed hard; checkpoint `ckpt_step1000.pt` + WikiText-2 cache survived on disk |
| 08:10 | Recovery: `research/scripts/run_m2_2_recover.sh` resumed seed 42 from step 1000 (auto-resume), then auto-runs seeds 43/44 |

Resume works cleanly: `--resume auto` picked `ckpt_step1000.pt`, restored both
optimizers (AdamW + SGD embedding), and continued from step 1000 of 7005
(anneal at step 5604). Only ~150 steps (~2.5 min) were lost.

---

## Hypothesis

Scaling the M2.1 DQT transformer (102M → 250M ternary params) keeps training
stable and reaches **mean validation perplexity < 20 on WikiText-2** across 3
seeds. This is the scaling test that validates DQT's 13 B/param training memory
claim at LLM scale — a ~250M ternary model that fits on an 8 GB consumer GPU.

GO: mean ppl < 20. MARGINAL: 20–25 (scaling test). NO-GO: >25 or OOM.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2-style decoder-only: `emb(50257→d) + L×[Attn(H heads, RoPE) + FFN(4d)] + RMSNorm + DQT LM Head` |
| **M2_2_CONFIG** | `d_model=1024, n_layers=16, n_heads=16, d_ff=4096, vocab=50257, seq_len=256` |
| Ternary weights | **252,789,760 (~252.8M int8)** — per block 12,582,912 × 16 + LM head 51,463,168 |
| Float embedding | 51,496,960 (50257×1024) — **trained with SGD, no AdamW moments** |
| Total parameters | **304,286,720 (~304M)** |
| Weight init | `weight_float ~ N(0, 0.1)`, ternary via `stochastic_round(init)` |
| Learning rate | 0.01 (DQT best) |
| Optimizer | AdamW (betas 0.9/0.95, wd 0.1) for all non-embedding params; **SGD for embedding** |
| LR schedule | linear warmup (100) → cosine to 10% |
| Gradient clipping | max_norm=1.0 |
| DQT rounding | stochastic_round after every step; anneal → deterministic sign at 80% |
| Batch size | **4** (verified — batch 8 exceeds the 7.5 GB memory gate) |
| Sequence length | 256 |
| Epochs | 3 |
| Dataset | WikiText-2 (`Salesforce/wikitext` → `wikitext-2-raw-v1`), GPT-2 BPE (tiktoken, vocab 50257) |
| Data | train 2,391,808 tok / val 247,040 / test 283,136; ~1,167 val batches@bs8 |
| Hardware | RTX 4060 8 GB (shared with gaming → pause/resume) |
| Seeds | 42, 43, 44 |

---

## 🔑 Memory budget — how 250M fits on 8 GB (measured, not estimated)

The M2.2 brief's naive estimate (250M × 13 B = 3.25 GB + 0.16 GB embed + 2–3 GB
activations + 0.5 GB context ≈ 6–7 GB) is **too optimistic**: it assumed the
embedding had no AdamW state AND didn't account for the logits/CE memory at
vocab 50257. Measured on the RTX 4060 with the raw config
(d=1024/L=16, batch 8, seq 256, grad checkpointing, full AdamW):

| Config | torch peak | torch reserved | nvidia-smi |
|--------|-----------:|---------------:|-----------:|
| baseline (batch 8, full AdamW, no expandable) | 6.89 GB | 8.12 GB | ~7.9 GB ❌ |
| + embed-SGD (no AdamW moments) | 6.51 GB | — | ~7.6 GB ❌ |
| + `expandable_segments:True` (batch 8) | 6.51 GB | 6.81 GB | ~7.4–7.6 GB ⚠️ |
| **batch 4 + embed-SGD + expandable + grad-ckpt (FINAL)** | **6.32–6.47 GB** | **6.46 GB** | **~7.2 GB ✅** |

Three memory levers (all in the runner/script, none change the optimization math):

1. **Gradient checkpointing** (`use_grad_checkpointing=True`, default on) —
   recompute each block's activations in backward. Cuts the 16-layer activation
   footprint from ~4 GB to ~0.4 GB. Blocks are pure (no in-place ops) so it's safe.
2. **Embedding trained with SGD, no AdamW moments** (`--embed-adamw` opts back in) —
   the brief's own "embedding χωρίς AdamW" budget. Saves 2 × 51.5M × 4 B = **0.41 GB**.
   Two optimizers (AdamW main + SGD embedding) are stepped in lockstep and both
   saved/restored in checkpoints.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (set in the shell script) —
   shrinks PyTorch's caching-allocator reserved pool from 8.12 GB → 6.81 GB
   (the allocator over-reserves ~1.6 GB beyond the active peak otherwise).
   Bonus: throughput improved ~1990 tok/s (batch 8).

**batch 8 with the real runner measures ~7.6 GB nvidia-smi — over the 7.5 GB hard
limit — so the M2.2 default is batch 4 (~7.2 GB, 1,525–1,600 tok/s).** This is a
memory-budget reduction, not a config reduction: the full 252.8M ternary config
is kept.

---

## Implementation

### New files

| File | Purpose |
|------|---------|
| `src/ph_neuro/training/wikitext2.py` | WikiText-2 loader: `get_wikitext2_data()` (train/val/test), GPT-2 BPE, concatenation + chunking to seq 256, disk cache (`data/wikitext2/`) |
| `src/ph_neuro/models/dqt_transformer.py` | + `M2_2_CONFIG`, + `use_grad_checkpointing` on `DQTTransformer`/`dqt_gpt2` |
| `src/ph_neuro/examples/run_m2_2_dqt_wikitext2.py` | Runner (adapted from M2.1): WikiText-2, M2_2_CONFIG, 2-optimizer setup, grad checkpointing, **SIGUSR1** pause, `--pause-file`, `status.json` progress, `--resume auto` |
| `scripts/run_m2_2_dqt_wikitext2.sh` | Root launcher |
| `research/scripts/run_m2_2_dqt_wikitext2.sh` | Orchestration: `full` (3 seeds) / `smoke` / `resume` / `status`, skip-if-exists, logs `logs/logs_m2_2/`, sets expandable_segments |
| `tests/integration/test_m2_2_wikitext2.py` | Loader + config + runner integration tests |
| `research/docs/experiments/E026-m2-2-dqt-wikitext2.md` | This report |

### Key reuse (100%)

- `TernaryDQTLinear3D` + `1/sqrt(d)` output scaling from M2.1 unchanged (the
  critical stabilization).
- M2.1's pause/resume infrastructure (checkpoint every N, `--resume auto`,
  best-ppl tracking, per-seed checkpoint dirs) kept identical, plus the M2.2
  brief's extras: **SIGUSR1** graceful pause (the gaming pause signal),
  **`--pause-file`** external pause control, and the **status.json** writer.
- Annealing (stochastic → deterministic sign at 80%), cosine+warmup, grad clip,
  RMSNorm/RoPE/GELU — all M2.1.

### Pause / resume (gaming co-use)

```
# pause before gaming:
kill -SIGUSR1 $(pgrep -f run_m2_2_dqt_wikitext2)
#   → saves checkpoint, prints the resume command, exits 130

# resume after gaming:
bash scripts/run_m2_2_dqt_wikitext2.sh resume 0.01 42

# what's running?
bash scripts/run_m2_2_dqt_wikitext2.sh status
```

---

## Results

### Smoke test (10 steps, real WikiText-2, M2_2_CONFIG, batch 4) — ✅ memory verified

`bash scripts/run_m2_2_dqt_wikitext2.sh smoke`

| Metric | Value |
|--------|------:|
| Train loss (step 2 → 10) | 10.865 → 10.839 (decreasing) |
| NaN | none |
| Torch peak GPU memory | **6,469 MB** |
| nvidia-smi during training | **~7.2 GB** (under 7.5 GB gate) |
| Throughput | ~1,500–1,600 tok/s |
| Verdict (memory) | ✅ fits 8 GB |

### Full run (3 seeds × 3 epochs) — ✅ COMPLETE, 🔴 NO-GO on ppl

Ran via the `recover` supervisor (detached, sequential seeds 42 → 43 → 44,
skip-if-exists, SIGUSR1 pause/resume, checkpoint-every 500). ~7,005 steps/seed at
batch 4. Per-seed results in
`m2_2_results/results_m2_2_dqt_wikitext2_lr0.01_seed{42,43,44}.json`.

| Metric | seed 42 | seed 43 | seed 44 | Mean |
|--------|--------:|--------:|--------:|-----:|
| Best val perplexity | 480.73 | **466.20** | 492.80 | **479.91** |
| Steps trained | 7,005 | 7,005 | 7,005 | 7,005 |
| Peak GPU memory (torch) | 6,127 MB | 5,912 MB | 6,481 MB | 6,173 MB |
| NaN / divergence | none | none | none | none |
| Final flip rate (deterministic tail) | 0.0005 | 0.0006 | 0.0554 | — |
| Verdict (<20) | 🔴 | 🔴 | 🔴 | **🔴 NO-GO** |

**🔴 NO-GO: mean validation perplexity 479.9, far above the <20 gate.**

### Why the ppl gate failed — the data budget, not the model

The milestone's <20 target is **not achievable with this data budget**. WikiText-2
is only ~2.4M training tokens; 3 epochs ≈ 7.2M tokens seen. Reference points:
GPT-2-small (124M) needs ~**1.5 B tokens** (WebText) to reach ~29 ppl on
WikiText-2; a ~250M model on **7.2M tokens** (~200× less) lands in the hundreds
of perplexity — exactly what we measured (~480). The training losses (5.8–6.2)
match the validation results; there is no pathology.

**What M2.2 DID validate (the actual scaling-test goal):**
- **Stability at 250M**: 3/3 seeds completed 7,005 steps with **no NaN and no
  divergence** — DQT's stochastic-rounding + annealing scales cleanly from
  102M (M2.1) to 253M ternary weights.
- **Memory**: full 250M config fits 8 GB (torch peak ≤ 6.5 GB, nvidia-smi ~7.2 GB
  at batch 4) using the three verified levers (grad checkpointing, embed-SGD,
  expandable_segments).
- **Deterministic tail**: post-anneal flip rates ~0.0005 (seed 44's 0.0554 is
  higher but still stable), matching M1.1/M2.1 behavior.
- **Infrastructure**: pause/resume survived two WSL restarts with ≤9 min total
  loss across the whole 3-seed run (idempotent `recover` supervisor).

**Options to actually reach ppl <20 (future):** (a) WikiText-103 (100× more
data) with the same 250M model — the realistic path; (b) many more epochs on
WikiText-2 (diminishing returns, severe overfitting risk); (c) a smaller vocab
(word-level ~33K) to cut the LM-head logits cost. The gate should be recalibrated
to the data budget before re-running.

---

## Known failure modes

- **Batch 8 OOM/borderline**: batch 8 measures ~7.6 GB nvidia-smi — over the gate
  and unsafe if the game holds memory. The runner defaults to batch 4; the script
  exposes `BATCH_SIZE=8` for users with a clean GPU who accept the risk.
- **Gaming contention**: pausing (SIGUSR1) is the intended workflow; do NOT run
  training concurrently with the game (M1.2 lesson: game at ~6 GB killed a seed).
- **`datasets` thread-abort at exit** (known cosmetic core-dump warning after the
  run finishes — harmless, matches TinyStories behavior).
- **Checkpoint size**: each checkpoint is ~3.9 GB (model + AdamW + SGD state for
  304M params). `--checkpoint-every 1000` → ~7 ckpts/seed ≈ 82 GB total for 3 seeds
  (~10% of the 822 GB free disk — acceptable, delete old ckpts after the run).
