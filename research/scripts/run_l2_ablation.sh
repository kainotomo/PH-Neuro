#!/usr/bin/env bash
# ── L2 Hysteresis-STE Ablation Sweep ────────────────────────────────
#
# Runs all Hysteresis-STE configurations across all MLP datasets.
# Compares against standard STE (control) as baseline.
#
# Usage:
#   bash scripts/run_l2_ablation.sh              # runs everything
#   bash scripts/run_l2_ablation.sh mnist         # single dataset
#   bash scripts/run_l2_ablation.sh mnist fashion # subset
#
# Output:
#   Results saved to l2_results/{dataset}_*.json
#
# Total runs: 4 theta_upper × 3 theta_lower × 3 datasets + 3 control = 39 runs
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="l2_results"
mkdir -p "$RESULTS_DIR"

# ── Configuration ──────────────────────────────────────────────────

THETA_UPPERS=(0.3 0.5 1.0 2.0)
THETA_LOWERS=(0.1 0.15 0.3)

# Pick datasets from CLI args or default to all
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(mnist fashion kmnist)
fi

EPOCHS=30
LR=0.001
SEED=42
BATCH_SIZE=128

# ── Helper ─────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_hyst() {
    local dataset="$1"
    local tu="$2"
    local tl="$3"
    local logfile="$RESULTS_DIR/log_${dataset}_th${tu}_tl${tl}.log"

    log "▶ Hysteresis-STE  dataset=${dataset}  θ_u=${tu}  θ_l=${tl}"
    python -m ph_neuro.examples.run_l2_hysteresis_ste \
        --dataset "$dataset" \
        --theta-upper "$tu" \
        --theta-lower "$tl" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$RESULTS_DIR" \
        > "$logfile" 2>&1

    # Print summary line from log
    tail -1 "$logfile"
    echo ""
}

run_control() {
    local dataset="$1"
    local logfile="$RESULTS_DIR/log_${dataset}_control.log"

    log "▶ Standard STE (control)  dataset=${dataset}"
    python -m ph_neuro.examples.run_l2_hysteresis_ste \
        --dataset "$dataset" \
        --control \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --seed "$SEED" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$RESULTS_DIR" \
        > "$logfile" 2>&1

    tail -1 "$logfile"
    echo ""
}

# ── Main sweep ─────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  L2 Hysteresis-STE Ablation Sweep"
echo "  Datasets: ${DATASETS[*]}"
echo "  Date:     $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for dataset in "${DATASETS[@]}"; do
    log "━━━ Dataset: ${dataset} ━━━"
    echo ""

    # Run control (standard STE) first
    run_control "$dataset"

    # Sweep Hysteresis-STE configurations
    for tu in "${THETA_UPPERS[@]}"; do
        for tl in "${THETA_LOWERS[@]}"; do
            # Skip invalid: theta_lower must be < theta_upper
            if (( $(echo "$tl >= $tu" | bc -l) )); then
                continue
            fi
            run_hyst "$dataset" "$tu" "$tl"
        done
    done
done

log "All runs complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results in: $RESULTS_DIR/"
echo "  Run aggregate: python -m ph_neuro.examples.aggregate_l2_results"
echo "═══════════════════════════════════════════════════════════════"
