#!/usr/bin/env bash
# ── M2.1 supervisor: run multiple seeds sequentially with pause/resume ──
#
# Runs the DQT transformer runner for each seed in sequence (one at a time —
# the 8 GB GPU can't fit two 102M models). Skips seeds whose result JSON
# already exists, and resumes (--resume auto) seeds that have checkpoints.
#
# The currently-running seed's python PID is written to
# /tmp/m2_1_train.pid (via the runner's --pid-file), so an external
# controller can pause it with:
#     kill -INT "$(cat /tmp/m2_1_train.pid)"
# The runner saves a checkpoint and exits 130; the supervisor then HALTS
# (it does NOT proceed to the next seed). Re-launch the supervisor later to
# resume where it left off — it picks up via --resume auto.
#
# Launch detached so the whole thing survives terminal teardown:
#     setsid nohup bash research/scripts/run_m2_1_supervisor.sh 43 44 \
#         > logs/logs_m2_1/supervisor.log 2>&1 < /dev/null &
#
# Monitor: tail -f logs/logs_m2_1/seed{43,44}_full.log
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [ ! -x "$PROJECT_ROOT/.venv/bin/python" ] && [ "$PROJECT_ROOT" != "/" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python"
RESULTS_DIR="results/phase2/m2_1_results"
PID_FILE="${M2_1_PID_FILE:-/tmp/m2_1_train.pid}"
mkdir -p "logs/logs_m2_1"

# ── Config (matches the FULL run) ──────────────────────────────────
D_MODEL=768; N_LAYERS=9; N_HEADS=12; D_FF=3072
LR=0.01; EPOCHS=3; BATCH=8; SEQ=256; WD=0.1; WARMUP=100
GRAD_CLIP=1.0; ANNEAL=0.8; MAX_SAMPLES=150000; NW=0; CKPT_EVERY=2000

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 43 44)
fi

for SEED in "${SEEDS[@]}"; do
    RESULT="$RESULTS_DIR/results_m2_1_dqt_transformer_lr${LR}_seed${SEED}.json"
    if [ -f "$RESULT" ]; then
        echo "[supervisor] seed $SEED already done — skipping"
        continue
    fi
    RESUME_FLAG=""
    CKPT_DIR="$RESULTS_DIR/checkpoints/seed${SEED}"
    if [ -d "$CKPT_DIR" ] && ls "$CKPT_DIR"/ckpt_step*.pt >/dev/null 2>&1; then
        RESUME_FLAG="--resume auto"
    fi
    echo "[supervisor] $(date '+%H:%M:%S') running seed $SEED $RESUME_FLAG"
    # Foreground python (the supervisor itself is detached) so we wait for it.
    set +e
    nohup "$PYTHON" -m ph_neuro.examples.run_m2_1_dqt_transformer \
        --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" --d-ff "$D_FF" \
        --lr "$LR" --epochs "$EPOCHS" --batch-size "$BATCH" --seq-len "$SEQ" \
        --weight-decay "$WD" --warmup-steps "$WARMUP" --grad-clip "$GRAD_CLIP" \
        --anneal-fraction "$ANNEAL" --max-samples "$MAX_SAMPLES" --num-workers "$NW" \
        --checkpoint-every "$CKPT_EVERY" --pid-file "$PID_FILE" \
        --output-dir "$RESULTS_DIR" --seed "$SEED" $RESUME_FLAG \
        > "logs/logs_m2_1/seed${SEED}_full.log" 2>&1 < /dev/null
    status=$?
    set -e
    echo "[supervisor] $(date '+%H:%M:%S') seed $SEED exited with status $status"
    if [ "$status" -ne 0 ]; then
        echo "[supervisor] halting — seed $SEED did not complete (exit $status)."
        echo "[supervisor] re-run this supervisor later to resume (--resume auto)."
        exit "$status"
    fi
done

echo "[supervisor] all seeds complete: ${SEEDS[*]}"
