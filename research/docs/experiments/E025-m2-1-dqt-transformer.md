# Experiment E025: M2.1 — DQT Transformer on TinyStories (GO/NO-GO ppl<30)

- **Date:** 2026-08-04 → 2026-08-06
- **Git commit:** `TBD`
- **Status:** ✅ **GO — mean val ppl 11.35 < 30 (3 seeds). COMPLETE.**
- **Phase:** 2 (Tiny Transformer)

---

## Hypothesis

A GPT-2-style decoder-only transformer whose linear projections use Direct
Quantized Training (DQT — int8 ternary weights + stochastic rounding + annealing)
can train **stably** on TinyStories and reach mean validation perplexity **< 30**
across 3 seeds. If it does → **GO** to MoE scaling (M2.3); if training is
unstable (NaN, divergence) or ppl ≥ 30 → **NO-GO**, pivot plan.

BitNet b1.58 (Microsoft, 2024) proved ternary transformers work with STE. This
milestone proves they also work with **DQT** (stochastic rounding, ~4.5× less
training memory than STE).

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2-style decoder-only: `emb(50257→d) + L×[Attn(H heads, RoPE) + FFN(4d)] + RMSNorm + DQT LM Head` |
| FULL config | `d_model=768, n_layers=9, n_heads=12, d_ff=3072, vocab=50257` |
| Total parameters | **140,910,336** (~141M) |
| Ternary weights | **102,298,368** (~102M int8, target ~100M) |
| Float parameters | 38,611,968 (token embedding 50257×768 + RMSNorm scales) |
| SMOKE config | `d_model=256, n_layers=4, n_heads=4, d_ff=1024` → 16,011,520 ternary / 28,879,616 total |
| Weight init | `weight_float ~ N(0, 0.1)` (M1.1-validated DQT init), ternary via `stochastic_round(init)` |
| Normalization | RMSNorm (float scale, no bias — NOT ternary) |
| Position encoding | RoPE, base 10000 (parameter-free) |
| Attention | causal, scaled dot-product, float softmax (never quantized) |
| Activation | GELU (float, never quantized) |
| Weight tying | **NO** — embedding is float, LM head is ternary → kept separate |
| Learning rate | 0.01 (DQT best, M1.1/M1.2-validated) |
| Optimizer | AdamW, betas=(0.9, 0.95), weight_decay=0.1 |
| LR schedule | linear warmup (100 steps) → cosine to 10% |
| Gradient clipping | max_norm=1.0 (critical for transformers) |
| DQT rounding | stochastic_round() after every optimizer.step(); **anneal → deterministic sign() at 80%** of steps |
| Batch size | 8 |
| Sequence length | 256 |
| Epochs | 3 (FULL), 2 (SMOKE) |
| Dataset | TinyStories (`roneneldan/TinyStories`), GPT-2 BPE tokenizer (tiktoken, vocab 50257) |
| Data (FULL) | max_samples=150,000 stories (~40M tokens cached), val = TinyStories validation split |
| Hardware | RTX 4060 8 GB |
| Seeds | 42, 43, 44 |

---

## Design decisions (M2.1 brief)

1. **Pre-norm** (`x = x + attn(RMSNorm(x))`), not post-norm — M1.1 showed DQT
   wants training stability; pre-norm is more stable.
2. **RMSNorm** (no mean subtraction, no bias) — simpler/faster than LayerNorm,
   float (never ternary).
3. **RoPE** instead of learned positional embeddings — parameter-free → fewer
   weights to quantize.
4. **GELU** (not ReLU) — standard for transformers, float.
5. **Embedding is float** `nn.Embedding` — it is a lookup table, not a matmul,
   so it is NOT quantized.
6. **No weight tying** — float embedding + ternary LM head can't share weights.
7. **Gradient clipping = 1.0** — stochastic rounding can inject spikes; clipping
   is the main stability lever for DQT transformers.

### Layer breakdown (FULL config)

| Component | Ternary weights |
|-----------|----------------:|
| Per block: Q/K/V/O projections (4 × 768×768) | 2,359,296 |
| Per block: FFN (768×3072 × 2) | 4,718,592 |
| Per block total | 7,077,888 |
| 9 blocks | 63,700,992 |
| LM Head (768×50257) | 38,597,376 |
| **Total ternary** | **102,298,368** |

> Note: the brief's alternate "640/8/10/2560" config actually gives ~71M ternary
> (not 100M); `768/9/12/3072` is the config that hits ~102M ternary, so it is
> used as the FULL default.

---

## Implementation

### New files

| File | Purpose |
|------|---------|
| `src/ph_neuro/layers/ste_dqt_transformer.py` | `TernaryDQTRMSNorm`, `TernaryDQTLinear3D`, `precompute_rotary_embeddings`, `apply_rotary_embeddings`, `TernaryDQTMultiheadAttention`, `TernaryDQTFeedForward`, `TernaryDQTTransformerBlock` |
| `src/ph_neuro/models/dqt_transformer.py` | `DQTTransformer`, `dqt_gpt2()`, `SMOKE_CONFIG`, `FULL_CONFIG`, `count_ternary_weights/parameters`, `model_summary()` |
| `src/ph_neuro/training/tinystories.py` | `make_gpt2_tokenizer()`, `tokenize_texts()`, `pack_sequences()`, `get_tinystories_data()` (HF download + disk cache + streaming), synthetic helpers for tests |
| `src/ph_neuro/examples/run_m2_1_dqt_transformer.py` | CLI runner: training loop, annealing, cosine+warmup scheduler, checkpointing, perplexity eval, JSON result |
| `scripts/run_m2_1_dqt_transformer.sh` | Root launcher |
| `research/scripts/run_m2_1_dqt_transformer.sh` | Orchestration: `full` (3 seeds) / `sweep` / `smoke`, skip-if-exists, logs to `logs/logs_m2_1/` |
| `tests/layers/test_ste_dqt_transformer.py` | 11 unit tests |
| `tests/integration/test_m2_1_transformer.py` | 12 integration tests |

### Key reuse

- `TernaryDQTLinear` + `stochastic_round` from `ste_dqt.py` unchanged — the
  transformer uses a shape-preserving wrapper `TernaryDQTLinear3D` (the base
  linear flattens >2D inputs to 2D per its M1.1 contract; the wrapper reshapes
  around it so `(B, T, C) → (B, T, O)`).
- `apply_dqt_rounding(model, use_stochastic)` and `ANNEAL_FRACTION=0.80`
  replicate the M1.1-RETRY annealing.

---

## 🔑 KEY FINDING: 1/sqrt(d) output scaling is REQUIRED for DQT transformers

The naive DQT transformer (validated CNN settings, no scaling) **diverges**:
after RMSNorm (unit-RMS ⇒ per-element std ≈ 1) the ternary matmuls (±1 weights,
~10% nonzero) amplify to std ≈ 5 per projection, and the attention + output
projection chain pushes each block output to std ≈ **20–30**. The residual
stream explodes 1 → 70 across 9 layers; logits grow std 4.4 → 7.3; loss rises
11 → 22. This happens even with smaller init (0.02), lower LR (0.003) or
residual scaling 0.5 — the amplification is ~20-30×, so those knobs can't fix it.

**Fix:** scale every DQT transformer projection output by `1/sqrt(in_features)`
(BitNet b1.58-style activation scaling). With it, the same config converges
(see smoke results below). Implemented as the default `scale` of
`TernaryDQTLinear3D` (transformer-only — M1.1/M1.2 use the unscaled base
`TernaryDQTLinear`, so vision results are unaffected).

Measured (d_model=256, L4, synthetic): without scaling first10 loss 11.47 →
last10 16.45 (diverge); with `1/sqrt(256)` first10 3.62 → last10 **0.04**
(converges, ppl → 1.06).

---

## Results

### Smoke test — synthetic (validates convergence, no download)

`bash scripts/run_m2_1_dqt_transformer.sh smoke` — SMOKE config (d256/L4/H4/ff1024,
vocab 64), 100 steps, pure stochastic rounding (anneal_fraction=1.0):

| Metric | Value |
|--------|------:|
| Train loss (epoch 1 → 2) | 3.70 → 1.01 |
| Best val perplexity | **1.06** (learns the synthetic LCG perfectly) |
| Peak GPU memory | 101 MB |
| Time | 2.4 s |
| Verdict (synthetic) | ✅ GO (ppl < 30) |

### Smoke test — real TinyStories (mini, 200 stories)

`--smoke --max-samples 200` — SMOKE config, vocab 50257, seq 128, 2 epochs, 76 steps:

| Metric | Value |
|--------|------:|
| Train loss (epoch 1 → 2) | 10.78 → 8.42 (decreasing — real text learning) |
| Val perplexity (epoch 1 → 2) | 37878 → 598 |
| Peak GPU memory | 1279 MB |
| Time | 5.2 s |

The downward trajectory confirms the DQT transformer learns real language
structure; 200 stories is far too little data to judge the final ppl.

### FULL config — GPU stability & throughput benchmark

d768/L9/H12/ff3072/vocab 50257 = **140.9M total, 102.3M ternary**, RTX 4060:

| Metric | Value |
|--------|------:|
| Peak GPU memory (batch 8, seq 256) | **4.83 GB** (fits 8 GB, ~60% util) |
| Throughput | **6230 tok/s** (~0.33 s/step) |
| Stability | finite, bounded logits, loss decreasing (no divergence) |
| Estimated FULL run (150K stories × 3 epochs) | ~3.5 h/seed → ~10.5 h for 3 seeds |

### FULL run (3 seeds × 3 epochs) — ✅ GO

Ran via `research/scripts/run_m2_1_supervisor.sh 42 43 44` (sequential, pause/resume
with SIGINT + `--resume auto` — see the pause/resume section). 150K stories,
lr=0.01, anneal@80%, batch 8, seq 256, **48,708 steps/seed**. Full val ppl
histories and per-epoch metrics are in the per-seed result JSONs
(`results/phase2/m2_1_results/results_m2_1_dqt_transformer_lr0.01_seed{42,43,44}.json`).

| Metric | seed 42 | seed 43 | seed 44 | Mean |
|--------|--------:|--------:|--------:|-----:|
| Best val perplexity | **11.47** | **11.32** | **11.27** | **11.35** |
| Final val perplexity | 11.47 | 11.32 | 11.27 | **11.35** |
| Steps trained | 48,708 | 48,708 | 48,708 | 48,708 |
| Time | ~2.0 h* | 4.80 h | 4.79 h | — |
| Verdict (<30) | ✅ | ✅ | ✅ | **GO ✅** |

\* seed 42 ran in two segments (an interrupted launch was resumed detached), so
its reported wall-clock covers only the second segment; steady-state is ~4.8 h/seed.

**Result: mean validation perplexity 11.35 — far below the <30 gate → GO ✅** to
M2.3 (MoE scaling). No NaN/divergence in any seed; all three trained stably and
the annealing tail (deterministic sign after step 38,966) worked as in M1.1.

---

## Known failure modes

1. **Annealing too early is catastrophic.** If the deterministic-sign switch
   (`sign(weight_float)`) fires while float buffers are near zero, it snaps
   ~90% of weights to ±1 in one step (measured flip rate 0.92), destroying the
   model (loss spike 6.8 → 18.5). Safe only when weights have converged (M1.1:
   switch at 80% of a converged run, post-switch flip ~0.0006). The FULL run
   keeps the 80% anneal; the integration training-loop test and smoke run use
   100% stochastic to avoid the premature-switch mode.
2. **Residual-stream explosion without 1/sqrt(d) scaling** (the key finding
   above) — this was the smoke-phase root cause; fixed by default scaling in
   `TernaryDQTLinear3D`.

---

## Pause / resume & monitoring (GPU shared with a game)

The runner supports **pause/resume** so a seed can be stopped (e.g. to free the
GPU for a game) and continued later:

- Every `--checkpoint-every` steps the runner saves `{output_dir}/checkpoints/seed{seed}/ckpt_step{N}.pt`
  (model + optimizer + scheduler + best-ppl state). Per-seed dirs prevent seeds
  from overwriting each other.
- Resume with `--resume auto` (latest checkpoint in the seed dir) or
  `--resume <path>`: `bash scripts/run_m2_1_dqt_transformer.sh resume 0.01 42`.
  Verified: train 6 steps → checkpoint → resume → continues to step 12 with
  loss still decreasing (ppl 63.76 → 57.74).
- Sequential multi-seed orchestration with pause-aware halting:
  `research/scripts/run_m2_1_supervisor.sh 42 43 44` — runs seeds one at a time
  (8 GB can't fit two 102M models), skips seeds with a result JSON, resumes
  (`--resume auto`) seeds with checkpoints, and **HALTS** (does not proceed) if
  a seed is paused. The current seed's PID is in `/tmp/m2_1_train.pid`.
- Resume of a mid-epoch checkpoint runs until the ORIGINAL step budget
  (`--max-steps` is set to `epochs × len(train_loader)` internally), so a
  paused+resumed seed still completes all its scheduled steps (this was a bug
  found and fixed during the FULL run).

**Monitoring** (while the run is going):
- Progress: `tail -f logs/logs_m2_1/results_m2_1_dqt_transformer_lr0.01_seed42.log`
  (per-epoch Train Loss / Val PPL / Flip / LR lines).
- GPU memory/contention: `nvidia-smi -l 5` (free VRAM in the `Memory-Usage` column).
- Final verdict: `results/phase2/m2_1_results/results_m2_1_dqt_transformer_lr0.01_seed42.json`.

**Gaming while training (8 GB RTX 4060):** the FULL run peaks at ~4.8 GB of the
8 GB. A game holding ~3+ GB risks CUDA OOM — in M1.2 a game holding ~6 GB killed
a seed silently (no traceback, log froze). Practical rules:
1. Check `nvidia-smi` free memory BEFORE starting (expect ≥ 3 GB free after the
   run allocates).
2. Lower the game's resolution/settings (target < 3 GB) or use the game's pause.
3. Keep `NUM_WORKERS=0` (already the script default) — the M1.2 OOM crash came
   from fork() workers under contention.
4. If the game needs the VRAM, **Ctrl+C the run** (it keeps the last checkpoint)
   and resume later with `resume` mode — no progress is lost beyond the last
   `--checkpoint-every` interval (2000 steps ≈ ~11 min).

---

## Verdict

- **Gate:** mean validation perplexity (3 seeds) < 30 → GO.
- **Result:** mean **11.35** (11.47 / 11.32 / 11.27) → **GO ✅**.
- DQT transformers train **stably** on TinyStories (no NaN/divergence across 3 seeds)
  with two required stabilizers: `1/sqrt(d)` projection scaling and the 80%
  anneal-to-deterministic-sign tail. Proceeds to **M2.3 (MoE scaling)**.
- Deliverables: layers, model factory, TinyStories loader, runner, supervisor +
  orchestration scripts, 24 tests (11 layer + 13 integration), report E025.
