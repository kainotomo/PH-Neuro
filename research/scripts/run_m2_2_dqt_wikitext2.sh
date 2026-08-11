#!/usr/bin/env bash
# ── M2.2 — DQT Transformer 250M params on WikiText-2 (GO/NO-GO ppl<20) ──
#
# Scaling test: 102M → 250M ternary params. The M2.1 DQT transformer
# (int8 ternary weights + stochastic rounding + annealing) is scaled to
# d_model=1024 / n_layers=16 / n_heads=16 / d_ff=4096 (~252.8M ternary +
# ~51.5M float embedding ≈ 304M total) and trained on WikiText-2. GO if
# the mean validation perplexity across 3 seeds is < 20.
#
# MEMORY (8 GB RTX 4060, VERIFIED by the E026 smoke test):
#   torch peak ~6.3-6.5 GB, nvidia-smi ~7.2 GB (batch 4). Three levers:
#     1. gradient checkpointing (recompute block activations in backward)
#     2. embedding trained with SGD (no AdamW moments — "χωρίς AdamW")
#     3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (shrinks the
#        caching allocator's reserved pool 8.1 → 6.8 GB)
#   batch 8 pushes nvidia-smi to ~7.6 GB (over the 7.5 GB hard limit), so
#   the default is batch 4 (override with BATCH_SIZE=8 at your own risk).
#
# Usage:
#   bash scripts/run_m2_2_dqt_wikitext2.sh                    # full run, 3 seeds
#   bash scripts/run_m2_2_dqt_wikitext2.sh full 0.01 42 43 44 # custom lr/seeds
#   bash scripts/run_m2_2_dqt_wikitext2.sh smoke              # 10-step GPU smoke
#   bash scripts/run_m2_2_dqt_wikitext2.sh resume 0.01 42     # pause/resume a seed
#   bash scripts/run_m2_2_dqt_wikitext2.sh status             # what is running?
#
# Pause/resume (gaming co-use): send SIGUSR1 to the training process
# (e.g. `kill -SIGUSR1 $(pgrep -f run_m2_2_dqt_wikitext2)`) — it saves a
# checkpoint and exits 130. Later `bash scripts/run_m2_2_dqt_wikitext2.sh
# resume 0.01 42` continues from the latest checkpoint.
#
# Output:
#   Results: m2_2_results/results_m2_2_dqt_wikitext2_lr{lr}_seed{seed}.json
#   Checkpoints: m2_2_results/checkpoints/seed{seed}/ckpt_step{N}.pt (+ status.json)
#   Logs:    logs/logs_m2_2/run_lr{lr}_seed{seed}.log
#
# Runs whose result JSON already exists are SKIPPED, so re-running is safe.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Live line-by-line logs (B2 lesson: block-buffered stdout froze logs).
export PYTHONUNBUFFERED=1
# Shrink PyTorch's caching-allocator reserved pool on 8 GB (measured
# 8.1 GB → 6.8 GB; keeps nvidia-smi ~7.2 GB at batch 4). E026-verified.
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

RESULTS_DIR="m2_2_results"
SMOKE_RESULTS_DIR="m2_2_smoke_results"
LOG_DIR="logs/logs_m2_2"
mkdir -p "$RESULTS_DIR" "$SMOKE_RESULTS_DIR" "$LOG_DIR"

# ── Configuration ───────────────────────────────────────────────────

# M2_2_CONFIG (~252.8M ternary weights)
D_MODEL=1024
N_LAYERS=16
N_HEADS=16
D_FF=4096

EPOCHS=3
# ── OPT-7 (Phase 2.5) — batch bumped 4 → 8 ────────────────────────
# Verified 2026-08-11: with 8-bit AdamW (OPT-2) + bf16/autocast (OPT-3) +
# SDPA (OPT-4) the 10-step smoke at batch 8 peaks at ~5.2 GB torch
# (nvidia-smi ~6.3 GB) — well under the 7.5 GB limit (was ~7.6 GB before
# the sprint). ~30% more throughput than the old batch-4 default.
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN=256
LR=0.01
WEIGHT_DECAY=0.1
WARMUP_STEPS=100
GRAD_CLIP=1.0
ANNEAL_FRACTION=0.80
NUM_WORKERS="${NUM_WORKERS:-0}"   # 0 safest under GPU contention (B2 lesson)
# 500 (not 1000): the host is WSL and has rebooted mid-run twice; a smaller
# cadence caps the loss per WSL restart at ~5 min (500 steps @ batch 4).
# ~3.9 GB/checkpoint → ~14 ckpts/seed ≈ 160 GB for 3 seeds (822 GB free).
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
    log "  M2.2 DQT Transformer WikiText-2  (GO/NO-GO: mean ppl < 20)"
    log "  Architecture: d_model=${D_MODEL} L=${N_LAYERS} H=${N_HEADS} ff=${D_FF}"
    log "                 emb(50257->${D_MODEL}) + ${N_LAYERS}x[Attn(${N_HEADS}h, RoPE)+FFN(${D_FF})] + RMSNorm + DQT LM Head"
    log "                 ~252.8M ternary + ~51.5M float embedding ≈ 304M total"
    log "  Optimizer: AdamW (lr=${LR}, wd=${WEIGHT_DECAY}) + cosine warmup(${WARMUP_STEPS})"
    log "             token embedding: SGD (no AdamW moments, saves ~0.4 GB)"
    log "  DQT: apply_dqt_rounding() after EVERY optimizer.step()"
    log "       stochastic_round() for first $(awk "BEGIN{printf \"%.0f\", ${ANNEAL_FRACTION}*100}")%, deterministic sign() after"
    log "  Epochs: ${EPOCHS}  Batch: ${BATCH_SIZE}  Seq: ${SEQ_LEN}  GradClip: ${GRAD_CLIP}"
    log "  Memory: grad-checkpointing + expandable_segments + embed-SGD"
    log "          (measured torch ~6.5 GB / nvidia-smi ~7.2 GB at batch 4)"
    log "  Data: WikiText-2 (Salesforce/wikitext wikitext-2-raw-v1), GPT-2 BPE"
    log "═══════════════════════════════════════════════════════════════"
}

run_one() {
    local lr="$1"
    local seed="$2"
    local mode_flags="$3"
    local out_json="$RESULTS_DIR/results_m2_2_dqt_wikitext2_lr${lr}_seed${seed}.json"
    local log_file="$LOG_DIR/run_lr${lr}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "▶ DQT Transformer 250M  lr=${lr}  seed=${seed}  ${mode_flags}"
    # `if !` so a failed run does NOT abort the script (B2 lesson).
    if ! "$PYTHON" -m ph_neuro.examples.run_m2_2_dqt_wikitext2 \
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
        --num-workers "$NUM_WORKERS" \
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
# Reads the lr from the result filename. Prints NOT STARTED if neither.
status_seed() {
    local seed="$1"
    local lr="$2"
    local out_json="$RESULTS_DIR/results_m2_2_dqt_wikitext2_lr${lr}_seed${seed}.json"
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
    print(f"{base}  PAUSED — resume with: bash scripts/run_m2_2_dqt_wikitext2.sh resume {lr} {seed}")
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
        # 10-step GPU smoke — VERIFIES the memory budget (<7.5 GB) and that
        # the loss decreases with no NaN, then evaluates on the val split.
        # Writes to SMOKE_RESULTS_DIR so it never overwrites real results.
        log "SMOKE: 10 steps, M2_2_CONFIG, batch ${BATCH_SIZE}, real WikiText-2"
        "$PYTHON" -m ph_neuro.examples.run_m2_2_dqt_wikitext2 \
            --max-steps 10 --seed 42 \
            --batch-size "$BATCH_SIZE" --seq-len "$SEQ_LEN" \
            --anneal-fraction "$ANNEAL_FRACTION" \
            --checkpoint-every 5 --progress-every 2 \
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
        log "Full run: seeds ${SEED_ARGS[*]} × lr=${BEST_LR} (M2_2_CONFIG ~252.8M ternary)"
        for seed in "${SEED_ARGS[@]}"; do
            run_one "$BEST_LR" "$seed" "--d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --d-ff $D_FF --checkpoint-every $CHECKPOINT_EVERY"
        done
        ;;
    resume)
        gpu_wait 0
        # Pause/resume: continue a seed from its latest checkpoint in
        # m2_2_results/checkpoints/seed{seed}/ (runner --resume auto).
        # The result JSON does not exist yet (run was interrupted), so the
        # skip-if-exists guard does not block us.
        BEST_LR="${2:-$DEFAULT_LR}"
        SEED="${3:-42}"
        log "Resume: seed ${SEED} × lr=${BEST_LR} (auto-resume from latest checkpoint)"
        run_one "$BEST_LR" "$SEED" "--d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --d-ff $D_FF --checkpoint-every $CHECKPOINT_EVERY --resume auto"
        ;;
    recover)
        # Idempotent supervisor (WSL-reboot safe): skip completed seeds,
        # resume interrupted ones from their latest checkpoint, run the
        # rest fresh. Safe to re-run after any WSL restart.
        log "Recover: delegating to the idempotent recovery supervisor"
        exec bash "$SCRIPT_DIR/run_m2_2_recover.sh" "${2:-$DEFAULT_LR}"
        ;;
    status)
        BEST_LR="${2:-$DEFAULT_LR}"
        log "M2.2 DQT Transformer WikiText-2 — status (gate: mean val ppl < 20)"
        echo ""
        for seed in "${SEEDS[@]}"; do
            status_seed "$seed" "$BEST_LR"
        done
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
