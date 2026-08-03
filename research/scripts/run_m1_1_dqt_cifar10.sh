#!/usr/bin/env bash
# ── M1.1-RETRY — DQT CNN on CIFAR-10 (GO/NO-GO >80%) ───────────────
#
# Milestone M1.1-RETRY: fix the original M1.1 NO-GO (mean 77.65%). Three
# changes, no new layers/runner:
#   1. Anneal stochastic rounding -> deterministic sign for the final 15%
#      of epochs (removes the late-training flip jitter).
#   2. Smaller FC head: 8192->256->10 instead of 8192->512->10 (halves
#      classifier flip noise).
#   3. Same 3 seeds (42/43/44), lr=0.01, 100 ep.
#
# Usage:
#   bash scripts/run_m1_1_dqt_cifar10.sh                    # 3 seeds, lr=0.01, 100 ep
#   bash scripts/run_m1_1_dqt_cifar10.sh sweep              # lr sweep (1 seed) to confirm lr
#   bash scripts/run_m1_1_dqt_cifar10.sh 42 43 44           # custom seeds
#
# Output:
#   JSON files: m1_1_retry_results/results_dqt_cifar10_lr{lr}_seed{seed}.json
#   Log files:  logs/m1_1_retry/  (gitignored)
#
# STE baseline (same architecture, for comparison) — run manually:
#   .venv/bin/python -m ph_neuro.examples.run_m1_1_dqt_cifar10 \
#       --lr 0.01 --epochs 100 --seed 42 --batch-size 128 \
#       --output-dir m1_1_retry_results
#
# Runs whose result JSON already exists are SKIPPED, so this script is
# safe to re-run.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve project root (works from anywhere) ─────────────────────
# Walk up from this script until we find the venv, so invocation works
# from the repo root, research/, or research/scripts/.
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

RESULTS_DIR="m1_1_retry_results"
LOG_DIR="logs/m1_1_retry"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── Configuration ──────────────────────────────────────────────────

EPOCHS=100
LR=0.01
BATCH_SIZE=128
WEIGHT_DECAY=1e-4
PATIENCE=15
SEEDS=(42 43 44)
SWEEP_LRS=(0.001 0.005 0.01 0.05)

MODE="${1:-full}"

# ── Python interpreter ────────────────────────────────────────────
# Prefer the project venv (per environment), then $PYTHON, then PATH.
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

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

print_config() {
    log "═══════════════════════════════════════════════════════════════"
    log "  M1.1-RETRY DQT CNN CIFAR-10  (GO/NO-GO >80%)"
    log "  Architecture: TernaryDQTConv2d(3->64)->Pool->TernaryDQTConv2d(64->128)"
    log "                ->Pool->TernaryDQTLinear(8192->256)->TernaryDQTLinear(256->10)"
    log "  Optimizer: AdamW (lr=${LR}, wd=${WEIGHT_DECAY}) + CosineAnnealingLR"
    log "  DQT: apply_dqt_rounding() after EVERY optimizer.step()"
    log "       stochastic_round() for first 85%, deterministic sign() for last 15%"
    log "  Epochs: ${EPOCHS}  Batch: ${BATCH_SIZE}  Patience: ${PATIENCE}"
    log "═══════════════════════════════════════════════════════════════"
}

run_one() {
    local lr="$1"
    local seed="$2"
    local lr_str
    lr_str=$(printf "%g" "$lr")
    local out_json="$RESULTS_DIR/results_dqt_cifar10_lr${lr_str}_seed${seed}.json"
    local log_file="$LOG_DIR/results_dqt_cifar10_lr${lr_str}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "▶ DQT CIFAR-10  lr=${lr}  seed=${seed}  epochs=${EPOCHS}  batch=${BATCH_SIZE}"
    "$PYTHON" -m ph_neuro.examples.run_m1_1_dqt_cifar10 \
        --lr "$lr" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --weight-decay "$WEIGHT_DECAY" \
        --patience "$PATIENCE" \
        --num-workers 2 \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1
    # Print the summary block (tail of log)
    tail -18 "$log_file"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────

print_config
echo ""

if [ "$MODE" = "sweep" ]; then
    log "LR sweep (1 seed = 42) to confirm the critical learning rate"
    for lr in "${SWEEP_LRS[@]}"; do
        run_one "$lr" 42
    done
else
    log "Full run: 3 seeds × lr=${LR}"
    for seed in "${SEEDS[@]}"; do
        run_one "$LR" "$seed"
    done
fi

log "All runs complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results in: $RESULTS_DIR/"
echo "  Logs in:    $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════════"
