#!/usr/bin/env bash
# ── L8 Forgetting Baseline Sweep ────────────────────────────────────
#
# Runs the forgetting baseline experiment (control for Track B).
# Measures catastrophic forgetting of standard SGD training (ternary STE
# and FP16) on Split MNIST and Permuted MNIST — NO EWC, NO replay.
#
# Usage:
#   bash scripts/run_l8_forgetting.sh              # runs everything
#   bash scripts/run_l8_forgetting.sh split         # single protocol
#   bash scripts/run_l8_forgetting.sh split permuted # subset
#
# Output:
#   JSON files: l8_results/{protocol}_{weight_format}_seed{seed}.json
#   Aggregated: l8_results/aggregated_summary.txt
#
# Total runs: 2 protocols × 2 weight formats × 3 seeds = 12 runs
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="l8_results"
mkdir -p "$RESULTS_DIR"

# ── Configuration ──────────────────────────────────────────────────

PROTOCOLS=(split permuted)
WEIGHT_FORMATS=(ternary fp16)
EPOCHS_PER_TASK=10
LR=0.001
BATCH_SIZE=128
SEEDS=(42 43 44)

# Pick protocols from CLI args or default to all
if [ $# -gt 0 ]; then
    PROTOCOLS=("$@")
fi

# ── Helper ─────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_exp() {
    local protocol="$1"
    local weight_format="$2"
    local seed="$3"
    local n_tasks

    if [ "$protocol" = "permuted" ]; then
        n_tasks=10
    else
        n_tasks=5
    fi

    log "Running: protocol=$protocol, weight=$weight_format, seed=$seed"

    python -m ph_neuro.examples.run_l8_forgetting_baseline \
        --protocol "$protocol" \
        --weight-format "$weight_format" \
        --epochs-per-task "$EPOCHS_PER_TASK" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --n-tasks "$n_tasks" \
        --seed "$seed" \
        --output-dir "$RESULTS_DIR"

    log "Done: protocol=$protocol, weight=$weight_format, seed=$seed"
    echo ""
}

# ── Run all configurations ─────────────────────────────────────────

total=$(( ${#PROTOCOLS[@]} * ${#WEIGHT_FORMATS[@]} * ${#SEEDS[@]} ))
current=0

for protocol in "${PROTOCOLS[@]}"; do
    for weight_format in "${WEIGHT_FORMATS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            current=$((current + 1))
            log "[$current/$total] Starting..."
            run_exp "$protocol" "$weight_format" "$seed"
        done
    done
done

# ── Generate aggregated summary ────────────────────────────────────

log "All runs complete. Generating summary..."
python -m ph_neuro.examples.aggregate_l8_results \
    --results-dir "$RESULTS_DIR" \
    --output "$RESULTS_DIR/aggregated_summary.txt"

log "Summary written to $RESULTS_DIR/aggregated_summary.txt"
log "Done! All $total runs complete."
