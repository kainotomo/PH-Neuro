#!/usr/bin/env bash
# ── L7 Depth vs Width Scaling Sweep ─────────────────────────────────
#
# Runs all depth configurations (D=1..5) for both Ternary STE and FP16
# across the primary dataset (MNIST), with 3 seeds for statistical rigor.
#
# Usage:
#   bash scripts/run_l7_depth_vs_width.sh              # runs everything (30 runs)
#   bash scripts/run_l7_depth_vs_width.sh mnist         # single dataset
#   bash scripts/run_l7_depth_vs_width.sh mnist fashion # multiple datasets
#
# Output:
#   JSON files: l7_results/results_{dataset}_{format}_d{depth}_seed{seed}.json
#
# Total runs: 1 dataset × 5 depths × 2 formats × 3 seeds = 30 runs
# Estimated time: ~30 minutes on RTX 4060
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="l7_results"
mkdir -p "$RESULTS_DIR"

# ── Configuration ──────────────────────────────────────────────────

DEPTHS=(1 2 3 4 5)
WEIGHT_FORMATS=(ternary fp16)
EPOCHS=30
LR=0.001
BATCH_SIZE=128
SEEDS=(42 43 44)

# Pick datasets from CLI args or default to mnist
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(mnist)
fi

# ── Helper ─────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_exp() {
    local dataset="$1"
    local depth="$2"
    local weight_format="$3"
    local seed="$4"

    log "Running: dataset=$dataset, depth=$depth, format=$weight_format, seed=$seed"

    python -m ph_neuro.examples.run_l7_depth_vs_width \
        --dataset "$dataset" \
        --depth "$depth" \
        --weight-format "$weight_format" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --seed "$seed" \
        --output-dir "$RESULTS_DIR"

    log "Done: dataset=$dataset, depth=$depth, format=$weight_format, seed=$seed"
    echo ""
}

# ── Run all configurations ─────────────────────────────────────────

total=$(( ${#DATASETS[@]} * ${#DEPTHS[@]} * ${#WEIGHT_FORMATS[@]} * ${#SEEDS[@]} ))
current=0

for dataset in "${DATASETS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        for weight_format in "${WEIGHT_FORMATS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                current=$((current + 1))
                log "--- [$current/$total] ---"
                run_exp "$dataset" "$depth" "$weight_format" "$seed"
            done
        done
    done
done

# ── Aggregate ──────────────────────────────────────────────────────

log "All runs complete! Aggregating results..."
python -m ph_neuro.examples.aggregate_l7_results --results-dir "$RESULTS_DIR"

log "Done! Results in $RESULTS_DIR/"
