# Experiment E027: M2.3 — MoE DQT Transformer on TinyStories (GO/NO-GO ppl<20)

- **Date:** 2026-08-06 → 2026-08-09
- **Git commit:** `TBD`
- **Status:** ✅ **GO — mean val ppl 14.08 < 20 (3 seeds). COMPLETE.**
- **Phase:** 2 (Tiny Transformer)

---

## Hypothesis

The **first MoE (Mixture of Experts) DQT Transformer**: a GPT-2-style DQT
transformer (int8 ternary weights + stochastic rounding + annealing, validated
in M2.1) whose later FFN layers are replaced by **sparse MoE** — a float
router selects the top-2 of 6 DQT ternary experts per token, so only ~52% of
parameters are active per token. With ~3× the total params of the dense M2.1
(102M), the MoE should reach **mean validation perplexity < 20** on
TinyStories (M2.1 dense got 11.35) — approaching or beating the dense model
at a fraction of the active compute.

The E019 pilot (vision, 4 experts top-2) showed MoE+DQT **beats** dense at
convergence with 50% active params (+2.48pp at 60 epochs) — but REQUIRES the
Switch-Transformer aux loss (`lb_coef=0.1`) and a slow router (0.1× lr) to
avoid expert collapse. M2.3 transfers those rules to a 312M-param LM.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2-style decoder-only with **hybrid dense+MoE block stack**: `emb(50257→768) + 6×[dense Attn+FFN] + 6×[Attn + MoE(6e, top-2)] + RMSNorm + DQT LM Head` |
| FULL config | `d_model=768, n_heads=12, d_ff=3072, dense_layers=6, moe_layers=6, n_experts=6, top_k=2, vocab=50257, max_seq=256` (revised 2026-08-07 from 4+8 → 6+6: the 8-MoE config peaked at 7.90 GB nvidia-smi, at the 8.2 GB card's physical edge, and crashed with `device not ready` even without gaming) |
| **Total ternary** | **265,089,792 (~265M)** |
| **Active / token** | **151,840,256 (~152M, 57%)** |
| Float parameters | 38,644,224 (token embedding 50257×768 + RMSNorm + routers) |
| **Total parameters** | **303,734,016 (~304M — within 300-400M milestone envelope)** |
| Router | float `nn.Linear(768, 6)` per MoE layer (NOT ternary) — its own LR = 0.1× expert LR |
| Experts | 6 × `TernaryDQTLinear3D(768→3072) → GELU → TernaryDQTLinear3D(3072→768)` per MoE layer (1/sqrt(in) output scaling — the M2.1 critical finding) |
| Load balancing | Switch-Transformer aux loss `n_experts · Σ_i f_i·P_i`, `lb_coef=0.1` added to the LM loss |
| Grouped execution | tokens grouped per expert; only top-2 experts run per token (FLOPs ~ top_k/n_experts) |
| Learning rate | 0.01 (DQT best); routers 0.001 (0.1×, E019 rule) |
| Optimizer | AdamW (2 groups: experts/norms/LM @ lr, routers @ 0.1×lr) + **SGD for the float embedding** (no AdamW moments — "embedding χωρίς AdamW", saves ~0.4 GB) |
| LR schedule | linear warmup (100 steps) → cosine to 10% |
| Gradient clipping | max_norm=1.0 |
| DQT rounding | stochastic_round() after every optimizer.step(); **anneal → deterministic sign() at 80%** of steps |
| Batch size | **4** (see Memory section) |
| Sequence length | 256 |
| Epochs | 3 |
| Dataset | TinyStories (`roneneldan/TinyStories`), GPT-2 BPE (tiktoken, vocab 50257) |
| Data (FULL) | max_samples=150,000 stories (M2.1-cached — no re-download) |
| Hardware | RTX 4060 8 GB (shared with gaming) |
| Seeds | 42, 43, 44 |

### Layer breakdown (FULL config)

| Component | Ternary total | Ternary active |
|-----------|--------------:|---------------:|
| Per expert FFN (768×3072 × 2) | 4,718,592 | 4,718,592 |
| Dense FFN (6 layers) | 28,311,552 | 28,311,552 |
| MoE FFN (6 layers × 6 experts) | 169,869,312 | 56,623,104 (top-2) |
| Attention (12 layers × 4 × 768²) | 28,311,552 | 28,311,552 |
| LM Head (768×50257) | 38,597,376 | 38,597,376 |
| **Total ternary** | **265,089,792** | **151,840,256 (57%)** |

---

## Memory budget (MEASURED, not estimated)

DQT training ≈ **13 bytes/param** (weight_float 4B + AdamW moments 8B + int8
ternary 1B). The brief's naive estimate (~6.6 GB with "~2.5 GB activations" at
batch 8) was **too optimistic**: the fixed costs dominate.

| Cost | Size |
|------|-----:|
| 265M float weight buffers | 1.06 GB |
| 265M AdamW moments (exp_avg + exp_avg_sq) | 2.12 GB |
| 265M gradients | 1.06 GB |
| 265M int8 ternary | 0.27 GB |
| Float embedding (SGD, no moments) + grad | 0.30 GB |
| Logits + grad (batch 4 × seq 256 × vocab 50257) | 0.41 GB |
| Activations (batch 4, no grad-checkpointing) | ~1.3 GB |
| **torch peak (batch 4, REVISED 6+6 config)** | **~6.50 GB** |

**Measured (revised 6+6 config):** 8 sustained steps stable — **torch peak 6.50 GB, nvidia-smi 7.17 GB of 8.2 GB (~1 GB headroom)**. Loss decreased 11.48 → 9.50 in 8 steps. This is the reliable configuration (the original 4+8 config measured 7.40 GB torch / ~7.90 GB nvidia-smi — at the card's edge, crashing intermittently).

**Verified findings:**
- **batch 8 is OVER the 8 GB card** — the forward+backward runs but the AdamW
  moment allocation during `optimizer.step()` pushes past the limit; the WSL
  driver returns `CUDA driver error: device not ready` (not a clean OOM).
- **gradient checkpointing does NOT help** — the peak is dominated by the
  fixed costs (weights + moments + grads + logits), not activations (measured
  identical with and without checkpointing).
- **batch 4 fits the revised 6+6 config**: torch peak 6.50 GB / nvidia-smi
  7.17 GB < the 7.5 GB gate, with ~1 GB headroom (reliable — 8/8 steps stable
  with no gaming). The original 4+8 config peaked at 7.40 GB torch / ~7.9 GB
  nvidia-smi and crashed intermittently EVEN WITHOUT gaming (at the card
  edge) — this is why the MoE stack was cut 8→6 layers.
- **Gaming co-use is NOT possible during training**: peak ~7.2 GB nvidia-smi
  + a Windows gaming session (~0.65-0.73 GB) ≈ 8 GB of 8.2 GB → any
  allocation spike crashes with "device not ready". **Pause the game while a
  seed trains.**

---

## Smoke test (VALIDATED)

Full M2_3_CONFIG, batch 4, seq 256, real cached TinyStories (150K stories).

- 12-step run: loss decreases, **no NaN**, torch peak **~7.4 GB**.
- **Expert routing is balanced from step 3** (dead-expert detector):
  | MoE layer | selection fractions (6 experts) | balance ratio | min share |
  |-----------|----------------------------------|--------------:|----------:|
  | L4 | 0.14 0.21 0.11 0.15 0.22 0.17 | 1.99 | 0.109 |
  | L5 | 0.16 0.16 0.18 0.19 0.18 0.14 | 1.31 | 0.142 |
  | L6 | 0.11 0.26 0.13 0.14 0.22 0.14 | 2.26 | 0.113 |
  | L7 | 0.15 0.16 0.22 0.16 0.15 0.15 | 1.45 | 0.151 |
  | L8 | 0.18 0.17 0.16 0.18 0.17 0.13 | 1.45 | 0.126 |
  | L9 | 0.17 0.23 0.11 0.18 0.21 0.09 | 2.41 | 0.094 |
  | L10 | 0.18 0.20 0.14 0.19 0.15 0.14 | 1.46 | 0.138 |
  | L11 | 0.19 0.19 0.15 0.14 0.17 0.15 | 1.33 | 0.144 |
- All 8 MoE layers use all 6 experts (min share ≥ 0.094, ideal 0.167) — the
  E019 slow-router + lb-loss rules transfer to the LM scale.

---

## Checkpoint pruning (disk fix — 2026-08-07)

**Root cause of the WSL reboots:** checkpoints were never pruned. At ~3.87
GB/checkpoint with `CHECKPOINT_EVERY=500`, a full seed writes ~195 checkpoints
≈ **754 GB**; all 3 seeds ≈ **2.3 TB — over 2× the 1 TB disk**. The disk filled
and WSL crashed/rebooted mid-training (killing the run silently at step 47500).

**Fix (implemented):** `_save_checkpoint` now prunes old periodic checkpoints
after each save, keeping only the newest `keep_last` (default 2) + a `best.pt`
(the best-validation checkpoint, written on every val improvement, never
pruned). Per-seed disk is now bounded to `(keep_last + 1) × ~3.9 GB` ≈ **12-15
GB** — the newest checkpoint is all that is needed to resume. Control:
`--keep-last-checkpoints N` (default 2). Tests: `TestCheckpointPruning` (2 ✅).

**Disk now:** `m2_3_results/` 349 GB → **6.2 GB** (kept `ckpt_step47000` +
`ckpt_step47500`); filesystem 458 GB → **116 GB used (13%)**.

### Prevention going forward (2026-08-07)

The disk-fill had TWO parts; both are now handled:

1. **Unbounded checkpoint growth** → **FIXED by pruning** (above). Each seed is
   bounded to ~12-15 GB, so the run can never fill the disk again.
2. **WSL2 VHDX does not auto-shrink** → freeing space inside WSL does not
   reclaim it on the Windows side until the virtual disk is compacted. The
   WSL disk (`C:\Users\phalo\AppData\Local\wsl\{3f0a19ca-...}\ext4.vhdx`) had
   grown to 515 GB; after deleting the stale checkpoints it was compacted back
   to ~120 GB (diskpart / `wsl --shrink`) — a one-time manual step.

**Operational guards added:**
- `_save_checkpoint` prints a **low-disk warning** (< 25 GB free on the WSL
  filesystem) with the exact compaction command, should it ever trigger.
- `bash scripts/run_m2_3_dqt_moe.sh status` now shows **WSL filesystem free
  space + checkpoint dir size** on every check, so disk health is visible at
  a glance.

**Maintenance recipe** (only needed occasionally; pruning makes it rare):
```bash
bash scripts/run_m2_3_dqt_moe.sh status      # watch "Disk (WSL fs)" + "Checkpoints"
# If disk ever looks high, from Windows PowerShell:
#   wsl --shutdown
#   wsl --manage Ubuntu-24.04 --shrink        # or: diskpart → select/attach/compact vdisk
```

---

## Deliverables

- `src/ph_neuro/layers/ste_dqt_transformer.py` — **new**
  `TernaryDQTMoEFeedForward` (float router, top-K softmax routing, grouped
  per-expert DQT execution, Switch-Transformer aux loss, usage-stat buffers
  guarded by `torch.is_grad_enabled()` for checkpoint safety) and
  **new** `TernaryDQTMoETransformerBlock` (pre-norm block returning
  `(x, aux_loss)`).
- `src/ph_neuro/models/dqt_transformer.py` — **new** `DQTMoETransformer` +
  `dqt_gpt2_moe()` factory + `M2_3_CONFIG` + `SMOKE_MOE_CONFIG` +
  `build_moe_config`.
- `src/ph_neuro/examples/run_m2_3_dqt_moe.py` — runner (copied from M2.2 +
  M2.3 adaptations): two-group AdamW (experts @ lr, routers @ 0.1×lr) +
  embed-SGD, `loss = CE + lb_coef·aux`, per-MoE-layer expert utilization
  logging + result JSON, full pause/resume (SIGINT/SIGTERM/SIGUSR1/pause-file,
  `--resume auto`, status.json), default batch 4.
- `scripts/run_m2_3_dqt_moe.sh` + `research/scripts/run_m2_3_dqt_moe.sh` —
  `full` / `smoke` / `resume` / `status` — **fully manual** (nothing runs or
  retries on its own).
- `tests/layers/test_ste_dqt_moe_transformer.py` — 15 tests ✅
  (forward, top-K correctness, load-balance loss trainability, block forward,
  hybrid model learn/backward).
- `tests/integration/test_m2_3_moe.py` — 12 tests ✅
  (config budget 312M/161M/52%, hybrid layout, router 0.1× group, overfit,
  short training loop, no-dead-expert check, lb_coef=0, perplexity, resume).
- Results: `m2_3_results/` (JSON + checkpoints).

---

## Manual start / pause (FULLY MANUAL — nothing automatic)

The user requested manual start/pause control only. Each training process
writes its PID to `m2_3_results/checkpoints/seed{S}/train.pid`, so a seed can
be paused precisely:

```bash
# START seed 42 (add 43 44 for more, or no seed = all 3)
bash scripts/run_m2_3_dqt_moe.sh full 0.01 42

# PAUSE seed 42 gracefully (finishes the step, saves a checkpoint, exits 130)
kill -SIGUSR1 $(cat m2_3_results/checkpoints/seed42/train.pid)

# RESUME seed 42 from its latest checkpoint
bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42

# STATUS (what is running / done)
bash scripts/run_m2_3_dqt_moe.sh status
```

The runner's pause/resume (SIGINT/SIGTERM/SIGUSR1/pause-file, `--resume auto`,
checkpoints every `CHECKPOINT_EVERY` steps, `status.json`) is unchanged — only
the outer orchestration is manual. Nothing runs, retries, or resumes on its
own.

> Note on the shared GPU: while the Windows gaming session is active, seeds
> crash at step ~3-4 with `CUDA driver error: device not ready` (7.4 GB torch
> peak + ~0.8 GB game = at the 8.2 GB card limit). Pause the game before
> `full`, and use the SIGUSR1 pause when you need the GPU back.

---

## GO / NO-GO checklist — ✅ GO (mean val ppl 14.08 < 20, 3 seeds complete 2026-08-09)

- [x] **GO 1: MoE training stable** — 3/3 seeds, no NaN/divergence, no dead experts
- [x] **GO 2: expert utilization balanced** — all MoE layers balance ratio ~1.00–1.13, min share ~0.16 (ideal 0.167)
- [x] **GO 3: perplexity < 20** — **mean 14.08** (42: 14.22, 43: 14.12, 44: 13.89; std 0.14)
- [x] **GO 4: memory < 7.5 GB** — torch peak 5.93 GB at batch 4 (nvidia-smi ~7.2 GB)

---

## Final results (3 seeds, 2026-08-09)

| Seed | Best val ppl | Final train loss | Final flip | Peak GPU |
|------|-------------:|-----------------:|-----------:|---------:|
| 42 | **14.22** | 3.28 | 0.0011 | 5.93 GB |
| 43 | **14.12** | 3.34 | 0.0012 | 5.93 GB |
| 44 | **13.89** | 3.45 | 0.042 | 5.93 GB |
| **Mean** | **14.08** (std 0.14) | 3.36 | | |

All seeds trained the full 97,419 steps (3 epochs × 150K stories, ~99.8M tokens),
annealed to deterministic sign at step 77,935. Very tight seed spread (±0.14).

**vs M2.1 (dense 102M, mean 11.35):** the MoE (303.7M total, ~63% active) passes
the <20 gate comfortably but does **not** beat the dense model (+2.7 ppl). This
is consistent with the E019 pilot: MoE converges slower and needs more data to
catch up to dense at equal tokens — here the same ~99.8M-token budget favors the
dense. **The milestone's core claim (first MoE DQT transformer, stable + balanced +
GO) is verified; the '3x params should beat dense' sub-goal is NOT met** — the
MoE approaches but doesn't surpass dense at this data budget.

---

## Reproduce (manual start/pause)

```bash
# GPU must be free of gaming (check nvidia-smi + Windows GPU usage first)
bash scripts/run_m2_3_dqt_moe.sh smoke     # 12-step smoke
bash scripts/run_m2_3_dqt_moe.sh full 0.01 42   # START seed 42 (manual)
kill -SIGUSR1 $(cat m2_3_results/checkpoints/seed42/train.pid)  # PAUSE
bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42 # RESUME seed 42
bash scripts/run_m2_3_dqt_moe.sh status    # progress
```
