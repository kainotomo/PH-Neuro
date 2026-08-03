#!/usr/bin/env bash
# ── MoE DQT Pilot Experiment (E019) ────────────────────────────────
#
# Runs the FIRST Mixture-of-Experts experiment with DQT ternary experts on
# MNIST. Trains a dense DQT baseline AND a MoE DQT model (N=4 experts,
# top-K=2) under an equal total-parameter budget, then writes a comparison
# JSON.
#
# Usage:
#   bash scripts/run_moe_dqt.sh                    # pilot: seed 42, 30 ep
#   bash scripts/run_moe_dqt.sh 42 43 44           # multiple seeds
#   bash scripts/run_moe_dqt.sh 42 --epochs 60     # pass extra args
#
# Output:
#   Results saved to moe_results/
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve python: prefer project venv, else system
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

RESULTS_DIR="moe_results"
mkdir -p "$RESULTS_DIR"

# Split positional args (seeds) from extra runner args
SEEDS=()
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        -*)
            EXTRA_ARGS+=("$arg")
            ;;
        *)
            SEEDS+=("$arg")
            ;;
    esac
done
if [ "${#SEEDS[@]}" -eq 0 ]; then
    SEEDS=(42)
fi

EPOCHS=30
LR=0.01
BATCH_SIZE=128
N_EXPERTS=4
TOP_K=2
EXPERT_WIDTH=128

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

for seed in "${SEEDS[@]}"; do
    logfile="$RESULTS_DIR/log_mnist_seed${seed}.log"
    log "▶ MoE DQT  seed=${seed}  N=${N_EXPERTS}  top-K=${TOP_K}  lr=${LR}  epochs=${EPOCHS}  batch=${BATCH_SIZE}"
    "$PY" -m ph_neuro.examples.run_moe_dqt \
        --dataset mnist \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --n-experts "$N_EXPERTS" \
        --top-k "$TOP_K" \
        --expert-width "$EXPERT_WIDTH" \
        --num-workers 2 \
        --output-dir "$RESULTS_DIR" \
        "${EXTRA_ARGS[@]}" \
        > "$logfile" 2>&1
    # Print summary block (tail of log)
    tail -24 "$logfile"
    echo ""
done

log "All runs complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results in: $RESULTS_DIR/"
