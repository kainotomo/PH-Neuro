#!/usr/bin/env bash
# ── DQT + Hysteresis-STE Combined Experiment ────────────────────────
#
# Runs the combined DQT (stochastic rounding) + Hysteresis-STE experiment
# on MNIST. Uses DQT best hyperparameters (lr=0.01, epochs=60, batch=128,
# init_std=0.1) and L2 best thresholds (θ_u=0.3, θ_l=0.15).
#
# Usage:
#   bash scripts/run_dqt_hysteresis.sh              # main run (seed 42)
#   bash scripts/run_dqt_hysteresis.sh 42 43 44     # multiple seeds
#
# Output:
#   Results saved to dqt_hysteresis_results/
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve python: prefer project venv, else system
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

RESULTS_DIR="dqt_hysteresis_results"
mkdir -p "$RESULTS_DIR"

# Default seeds
if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(42)
fi

THETA_UPPER=0.3
THETA_LOWER=0.15
EPOCHS=60
LR=0.01
BATCH_SIZE=128
INIT_STD=0.1

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

for seed in "${SEEDS[@]}"; do
    logfile="$RESULTS_DIR/log_mnist_seed${seed}.log"
    log "▶ DQT+Hysteresis  seed=${seed}  θ_u=${THETA_UPPER}  θ_l=${THETA_LOWER}  lr=${LR}  epochs=${EPOCHS}"
    "$PY" -m ph_neuro.examples.run_dqt_hysteresis \
        --dataset mnist \
        --theta-upper "$THETA_UPPER" \
        --theta-lower "$THETA_LOWER" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --init-std "$INIT_STD" \
        --seed "$seed" \
        --num-workers 2 \
        --output-dir "$RESULTS_DIR" \
        > "$logfile" 2>&1
    # Print summary block (tail of log)
    tail -14 "$logfile"
    echo ""
done

log "All runs complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results in: $RESULTS_DIR/"
