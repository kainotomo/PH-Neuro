#!/usr/bin/env bash
# ── M2.3 — MoE DQT Transformer on TinyStories (GO/NO-GO ppl<20) ───
#
# First MoE (Mixture of Experts) DQT Transformer: ~312M ternary params total,
# ~161M active per token (52%). GPT-2-style DQT transformer with a hybrid
# block stack — layers 0-3 dense FFN, layers 4-11 MoE FFN (8 layers x 6 DQT
# experts, top-2 routing via a float router @ 0.1x lr). Trained on
# TinyStories. GO if the mean validation perplexity (3 seeds) is < 20.
#
# MEMORY (8 GB RTX 4060, VERIFIED by the M2.3 smoke test):
#   torch peak ~7.4 GB / nvidia-smi ~7.7-7.9 GB at batch 4 (fits 8 GB; the
#   fixed costs — weights + AdamW moments + grads + logits at vocab 50257 —
#   dominate, so gradient checkpointing does NOT help). batch 8 exceeds the
#   card under shared-GPU conditions (crashes with "device not ready"), so
#   batch 4 is the default. Levers retained from M2.2:
#     1. embedding trained with SGD (no AdamW moments — "χωρίς AdamW")
#     2. MoE routers at 0.1x lr (E019 slow-router rule)
#     3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   NOTE: this machine's GPU is SHARED with gaming (M1.2/M2.1 lesson). The
#   run fits 8 GB with ~0.8 GB headroom, so GAMING MUST BE PAUSED while a
#   seed is training, or the driver returns "device not ready". Check
#   `nvidia-smi` (and Windows-side usage) before launching.
#
# Usage:
#   bash scripts/run_m2_3_dqt_moe.sh                    # full run, 3 seeds
#   bash scripts/run_m2_3_dqt_moe.sh full 0.01 42 43 44 # custom lr/seeds
#   bash scripts/run_m2_3_dqt_moe.sh smoke              # 12-step GPU smoke
#   bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42     # resume a paused seed
#   bash scripts/run_m2_3_dqt_moe.sh status            # what is running?
#
# MANUAL start/pause (FULLY MANUAL — nothing runs or retries on its own):
#   START  : bash scripts/run_m2_3_dqt_moe.sh full 0.01 42
#   PAUSE  : kill -SIGUSR1 $(cat results/phase2/m2_3_results/checkpoints/seed42/train.pid)
#            → graceful checkpointed pause (saves ckpt, exits 130)
#   RESUME : bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42
#   STATUS : bash scripts/run_m2_3_dqt_moe.sh status
#
# Pause/resume (gaming co-use): each training process writes its PID to
# results/phase2/m2_3_results/checkpoints/seed{S}/train.pid. Sending SIGUSR1 to it makes the
# runner finish the current step, save a checkpoint and exit 130. Later
# `bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42` continues from the
# latest checkpoint.
#
# Output:
#   Results: results/phase2/m2_3_results/results_m2_3_dqt_moe_lr{lr}_seed{seed}.json
#   Checkpoints: results/phase2/m2_3_results/checkpoints/seed{seed}/ckpt_step{N}.pt (+ status.json)
#   Logs:    logs/logs_m2_3/run_lr{lr}_seed{seed}.log
#
# Runs whose result JSON already exists are SKIPPED, so re-running is safe.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Live line-by-line logs (B2 lesson: block-buffered stdout froze logs).
export PYTHONUNBUFFERED=1
# Shrink PyTorch's caching-allocator reserved pool on 8 GB (M2.2-verified).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── GPU wait gate (shared GPU with gaming) ─────────────────────────
# `--wait-gpu` blocks until GPU_WAIT_THRESHOLD GB are free via
# scripts/gpu_wait.py before launching. Default: ON for `full`/`sweep`
# runs, OFF for smoke/status/resume. `--no-wait-gpu` forces it off.
WAIT_GPU=""                            # "" = mode default, 1 = on, 0 = off
GPU_WAIT_THRESHOLD="${GPU_WAIT_THRESHOLD:-7.0}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-120}"
_POS_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --wait-gpu)   WAIT_GPU=1 ;;
        --no-wait-gpu) WAIT_GPU=0 ;;
        *) _POS_ARGS+=("$_arg") ;;
    esac
done
if [ ${#_POS_ARGS[@]} -gt 0 ]; then set -- "${_POS_ARGS[@]}"; else set --; fi
unset _POS_ARGS

# ── Resolve project root (works from anywhere) ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [ ! -x "$PROJECT_ROOT/.venv/bin/python" ] && [ "$PROJECT_ROOT" != "/" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "ERROR: could not locate project root (.venv/bin/python not found above $SCRIPT_DIR)" >&2
    exit 1
fi
cd "$PROJECT_ROOT"

RESULTS_DIR="results/phase2/m2_3_results"
SMOKE_RESULTS_DIR="results/phase2/m2_3_smoke_results"
LOG_DIR="logs/logs_m2_3"
mkdir -p "$RESULTS_DIR" "$SMOKE_RESULTS_DIR" "$LOG_DIR"

# ── Configuration ───────────────────────────────────────────────────

# M2_3_CONFIG (revised 2026-08-06: 6 dense + 6 MoE — ~265M ternary,
# ~152M active / token; the original 4+8 hit the 8 GB card's memory edge and
# crashed with "device not ready" even without gaming, so the MoE stack was
# cut 8→6 layers per the brief's memory-budget rule).
D_MODEL=768
N_HEADS=12
D_FF=3072
DENSE_LAYERS=6
MOE_LAYERS=6
N_EXPERTS=6
TOP_K=2

EPOCHS=3
# ── OPT-7 (Phase 2.5) — batch bumped 4 → 8 ────────────────────────
# Verified 2026-08-11 (M2.2 batch-8 smoke): 8-bit AdamW + bf16 + SDPA cut
# peak torch ~6.5 → ~5.2 GB at batch 8, nvidia-smi ~7.6 → ~6.3 GB — under
# the 7.5 GB limit. MoE has higher fixed cost than dense (routers + grouped
# experts) but the same savings apply; batch 8 is the default, drop to 4
# (BATCH_SIZE=4) if the M2.3 smoke ever exceeds ~7.5 GB under contention.
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN=256
LR=0.01
WEIGHT_DECAY=0.1
WARMUP_STEPS=100
GRAD_CLIP=1.0
ANNEAL_FRACTION=0.80
LB_COEF=0.1
ROUTER_LR_RATIO=0.1
NUM_WORKERS="${NUM_WORKERS:-0}"   # 0 safest under GPU contention (B2 lesson)
# TinyStories: full M2.1 run used 150000 stories (~25.8M tokens, 16236
# batches/epoch @ seq 256). Cached at data/tinystories/ — no re-download.
MAX_SAMPLES="${MAX_SAMPLES:-150000}"
# 500 steps @ batch 4 ≈ ~2-4 min of lost progress per WSL/gaming restart.
CHECKPOINT_EVERY=500

SEEDS=(42 43 44)
DEFAULT_LR=0.01

MODE="${1:-full}"

# ── Python interpreter ────────────────────────────────────────────
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -n "${PYTHON:-}" ] && command -v "$PYTHON" >/dev/null 2>&1; then
    :  # use $PYTHON as-is
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: no python interpreter found (tried .venv/bin/python, \$PYTHON, python)" >&2
    exit 1
fi

# ── Helper ─────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Block until the GPU is free before launching (shared GPU with gaming).
gpu_wait() {
    local default_on="$1"   # 1 = this mode waits by default
    local want="$default_on"
    if [ -n "$WAIT_GPU" ]; then
        want="$WAIT_GPU"
    fi
    if [ "$want" = "1" ]; then
        log "⏳ Waiting for GPU (need >=${GPU_WAIT_THRESHOLD} GB free, timeout ${GPU_WAIT_TIMEOUT} min)..."
        if ! "$PYTHON" scripts/gpu_wait.py --threshold "$GPU_WAIT_THRESHOLD" --timeout "$GPU_WAIT_TIMEOUT"; then
            log "❌ GPU not free in time — retry later (e.g. bash scripts/train.sh ...)"
            exit 1
        fi
    fi
}

print_config() {
    log "═══════════════════════════════════════════════════════════════"
    log "  M2.3 MoE DQT Transformer TinyStories  (GO/NO-GO: mean ppl < 20)"
    log "  Architecture: d_model=${D_MODEL} L=$((DENSE_LAYERS+MOE_LAYERS))"
    log "                 (dense ${DENSE_LAYERS} + MoE ${MOE_LAYERS}x${N_EXPERTS}e/top-${TOP_K})"
    log "                 H=${N_HEADS} ff=${D_FF} — ~312M ternary, ~161M active"
    log "  Optimizer: AdamW (lr=${LR}, wd=${WEIGHT_DECAY}, router ${ROUTER_LR_RATIO}x)"
    log "             + cosine warmup(${WARMUP_STEPS}) + embed SGD (no AdamW)"
    log "  Loss:      CE + ${LB_COEF} * Switch-Transformer aux (load balance)"
    log "  DQT: apply_dqt_rounding() after EVERY optimizer.step()"
    log "       stochastic_round() for first $(awk "BEGIN{printf \"%.0f\", ${ANNEAL_FRACTION}*100}")%, deterministic sign() after"
    log "  Epochs: ${EPOCHS}  Batch: ${BATCH_SIZE}  Seq: ${SEQ_LEN}  GradClip: ${GRAD_CLIP}"
    log "  Memory: embed-SGD + expandable_segments"
    log "          (measured torch ~7.4 GB / nvidia-smi ~7.7-7.9 GB at batch 4)"
    log "          ⚠️ GPU SHARED with gaming — pause the game while training"
    log "  Data: TinyStories (max_samples=${MAX_SAMPLES}), GPT-2 BPE"
    log "═══════════════════════════════════════════════════════════════"
}

run_one() {
    local lr="$1"
    local seed="$2"
    local mode_flags="$3"
    local out_json="$RESULTS_DIR/results_m2_3_dqt_moe_lr${lr}_seed${seed}.json"
    local log_file="$LOG_DIR/run_lr${lr}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "▶ MoE DQT Transformer  lr=${lr}  seed=${seed}  ${mode_flags}"
    # Per-seed PID file so the user can pause THIS seed precisely:
    #   kill -SIGUSR1 $(cat results/phase2/m2_3_results/checkpoints/seed{S}/train.pid)
    local seed_ckpt_dir="$RESULTS_DIR/checkpoints/seed${seed}"
    mkdir -p "$seed_ckpt_dir"
    # `if !` so a failed run does NOT abort the script (B2 lesson).
    if ! "$PYTHON" -m ph_neuro.examples.run_m2_3_dqt_moe \
        $mode_flags \
        --lr "$lr" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --weight-decay "$WEIGHT_DECAY" \
        --warmup-steps "$WARMUP_STEPS" \
        --grad-clip "$GRAD_CLIP" \
        --anneal-fraction "$ANNEAL_FRACTION" \
        --lb-coef "$LB_COEF" \
        --router-lr-ratio "$ROUTER_LR_RATIO" \
        --max-samples "$MAX_SAMPLES" \
        --num-workers "$NUM_WORKERS" \
        --pid-file "$seed_ckpt_dir/train.pid" \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1; then
        log "FAILED: $out_json (see $log_file)"
        FAILED_RUNS+=("$out_json")
        tail -18 "$log_file"
        echo ""
        return 1
    fi
    tail -18 "$log_file"
    echo ""
}

# Format a per-seed status line from its status.json (or result JSON).
status_seed() {
    local seed="$1"
    local lr="$2"
    local out_json="$RESULTS_DIR/results_m2_3_dqt_moe_lr${lr}_seed${seed}.json"
    local status_file="$RESULTS_DIR/checkpoints/seed${seed}/status.json"
    if [ -f "$out_json" ]; then
        local ppl
        ppl=$("$PYTHON" -c "import json;d=json.load(open('$out_json'));print(f\"{d['best_val_ppl']:.2f}\")")
        printf "  Seed %s: COMPLETED — val ppl %s\n" "$seed" "$ppl"
    elif [ -f "$status_file" ]; then
        "$PYTHON" - "$seed" "$status_file" "$lr" <<'PYEOF'
import json, sys
seed, path, lr = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
st = d.get("status", "RUNNING")
step, total = d.get("step", 0), d.get("total_steps", 0)
pct = 100.0 * step / total if total else 0.0
base = f"Seed {seed}: step {step}/{total} ({pct:.0f}%)"
if st == "COMPLETED":
    print(f"  Seed {seed}: COMPLETED — val ppl {d.get('ppl', float('nan')):.2f}")
elif st == "PAUSED":
    # lr here is the BASE lr passed to `status` (default 0.01), NOT the
    # annealed current lr stored in status.json — the correct resume value.
    print(f"{base}  PAUSED — resume with: bash scripts/run_m2_3_dqt_moe.sh resume {lr} {seed}")
else:
    print(f"{base}, loss {d.get('loss', float('nan')):.2f}, ppl {d.get('ppl', float('nan')):.1f}, "
          f"{d.get('tok_per_s', 0):.0f} tok/s, GPU {d.get('gpu_mem_gb', 0):.1f}/{d.get('gpu_total_gb', 0):.1f} GB")
PYEOF
    else
        printf "  Seed %s: NOT STARTED\n" "$seed"
    fi
}

# ── Main ────────────────────────────────────────────────────────────

print_config
echo ""

FAILED_RUNS=()

case "$MODE" in
    smoke)
        gpu_wait 0
        # 12-step GPU smoke — VERIFIES the memory budget (<7.5 GB torch),
        # no NaN, and that expert routing is balanced (no dead experts).
        # Writes to SMOKE_RESULTS_DIR so it never overwrites real results.
        log "SMOKE: 12 steps, M2_3_CONFIG, batch ${BATCH_SIZE}, real TinyStories"
        "$PYTHON" -m ph_neuro.examples.run_m2_3_dqt_moe \
            --max-steps 12 --seed 42 \
            --batch-size "$BATCH_SIZE" --seq-len "$SEQ_LEN" \
            --anneal-fraction "$ANNEAL_FRACTION" \
            --lb-coef "$LB_COEF" --router-lr-ratio "$ROUTER_LR_RATIO" \
            --max-samples "$MAX_SAMPLES" \
            --checkpoint-every 6 --progress-every 3 \
            --output-dir "$SMOKE_RESULTS_DIR" 2>&1 | tee "$LOG_DIR/smoke.log"
        ;;
    full)
        gpu_wait 1
        BEST_LR="${2:-$DEFAULT_LR}"
        SEED_ARGS=()
        for s in "${@:3}"; do
            SEED_ARGS+=("$s")
        done
        if [ ${#SEED_ARGS[@]} -eq 0 ]; then
            SEED_ARGS=("${SEEDS[@]}")
        fi
        log "Full run: seeds ${SEED_ARGS[*]} × lr=${BEST_LR} (M2_3_CONFIG ~312M ternary)"
        log "⚠️  IMPORTANT: the GPU is SHARED with gaming. Pause the game while"
        log "   a seed trains, or the run will crash with 'device not ready'."
        for seed in "${SEED_ARGS[@]}"; do
            run_one "$BEST_LR" "$seed" "--d-model $D_MODEL --n-heads $N_HEADS --d-ff $D_FF --dense-layers $DENSE_LAYERS --moe-layers $MOE_LAYERS --n-experts $N_EXPERTS --top-k $TOP_K --checkpoint-every $CHECKPOINT_EVERY"
            rc=$?
            if [ $rc -ne 0 ]; then
                # A deliberate SIGUSR1 pause writes status=PAUSED and exits
                # 130. HALT the batch so the NEXT seed does NOT start while
                # the user is away/gaming (manual-control rule). A genuine
                # crash (no PAUSED status) continues to the next seed.
                if grep -q '"status": "PAUSED"' "$RESULTS_DIR/checkpoints/seed${seed}/status.json" 2>/dev/null; then
                    log "⏸️  Seed $seed paused by request — halting. Remaining seeds not started."
                    exit 0
                fi
            fi
        done
        ;;
    resume)
        gpu_wait 0
        # Pause/resume: continue a seed from its latest checkpoint in
        # results/phase2/m2_3_results/checkpoints/seed{seed}/ (runner --resume auto).
        BEST_LR="${2:-$DEFAULT_LR}"
        SEED="${3:-42}"
        log "Resume: seed ${SEED} × lr=${BEST_LR} (auto-resume from latest checkpoint)"
        run_one "$BEST_LR" "$SEED" "--d-model $D_MODEL --n-heads $N_HEADS --d-ff $D_FF --dense-layers $DENSE_LAYERS --moe-layers $MOE_LAYERS --n-experts $N_EXPERTS --top-k $TOP_K --checkpoint-every $CHECKPOINT_EVERY --resume auto"
        ;;
    status)
        BEST_LR="${2:-$DEFAULT_LR}"
        log "M2.3 MoE DQT Transformer TinyStories — status (gate: mean val ppl < 20)"
        echo ""
        for seed in "${SEEDS[@]}"; do
            status_seed "$seed" "$BEST_LR"
        done
        echo ""
        # Disk visibility (disk-fill caused the WSL reboots — check it here).
        df -h /home 2>/dev/null | awk 'NR==2{printf "  Disk (WSL fs): %s used / %s free (%s of %s)\n", $3, $4, $5, $2}'
        # Checkpoint disk (should stay bounded ~12-15 GB/seed thanks to pruning).
        du -sh "$RESULTS_DIR" 2>/dev/null | awk '{printf "  Checkpoints:   %s total (pruned to latest 2 + best.pt)\n", $1}'
        echo ""
        log "Live progress: tail -f $LOG_DIR/run_lr${BEST_LR}_seed*.log"
        ;;
    *)
        echo "ERROR: unknown mode '$MODE' (expected 'full [lr] [seeds...]', 'smoke', 'resume [lr] [seed]' or 'status')" >&2
        exit 1
        ;;
esac

if [ "$MODE" != "status" ]; then
    log "All runs complete!"
    echo ""
    if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
        log "⚠️  ${#FAILED_RUNS[@]} run(s) FAILED:"
        for f in "${FAILED_RUNS[@]}"; do log "  - $f"; done
        log "   Resume failed seeds with: bash scripts/run_m2_3_dqt_moe.sh resume $DEFAULT_LR <seed>"
    else
        log "✅ No failed runs."
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Results in:  $RESULTS_DIR/"
    echo "  Logs in:     $LOG_DIR/"
    echo "  Gate:        mean val perplexity (3 seeds) < 20 → GO"
    echo "═══════════════════════════════════════════════════════════════"
fi
