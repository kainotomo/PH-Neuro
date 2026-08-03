#!/usr/bin/env bash
# ── B3 Precision Comparison for Continual Learning ─────────────────
#
# Runs the B3 comparison (ternary vs INT8 vs INT4 vs FP16) on Split
# MNIST + Permuted MNIST. Ternary and FP16 were already measured by the
# L8 control (l8_results/, identical hyperparameters + seeds), so this
# script only runs the two NEW precisions: INT8 and INT4 QAT.
#
# Usage:
#   bash scripts/run_b3_precision.sh            # all 12 runs
#   bash scripts/run_b3_precision.sh int8       # INT8 only (6 runs)
#   bash scripts/run_b3_precision.sh int8 int4  # both (default)
#
# Output:
#   JSON files: b3_results/{protocol}_{weight_format}_seed{seed}.json
#   Log files:  logs/b3/{protocol}_{weight_format}_seed{seed}.log
#               (gitignored)
#
# Total runs: 2 CL protocols × 2 precisions × 3 seeds = 12
#
# Rules:
#   * PYTHONUNBUFFERED=1     → live line-by-line logs
#   * per-run log file       → logs/b3/*.log, console shows status lines
#   * skip existing JSONs    → resumable
#   * FAILED runs recorded, script continues, non-zero exit at end
#
# num_workers: default 2 (NOT the B2 rule of 0) — rationale:
#   * Empirically verified: nw=0 and nw=2 give BYTE-IDENTICAL results
#     for both split and permuted (checked on int8 seed42).
#   * The L8 ternary/fp16 baseline (which B3 compares against) ran with
#     the default nw=2, so nw=2 matches the baseline conditions exactly.
#   * Permuted MNIST permutes pixels in Python __getitem__; nw=0 makes
#     data loading the bottleneck (2.4× slower, ~2.8 h total vs ~75 min).
#   * The B2 nw=0 rule was a workaround for a silent OOM death on a
#     larger LoRA run; the tiny 535K-param QAT models here have no such
#     risk. Override with: NUM_WORKERS=0 bash scripts/run_b3_precision.sh
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="b3_results"
LOG_DIR="logs/b3"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── Configuration ──────────────────────────────────────────────────

EPOCHS_PER_TASK=10
LR=0.001
BATCH_SIZE=128
NUM_WORKERS="${NUM_WORKERS:-2}"
SEEDS=(42 43 44)
PROTOCOLS=(split permuted)
WEIGHT_FORMATS=(int8 int4)

# Pick weight formats from CLI args or default to all
if [ $# -gt 0 ]; then
    WEIGHT_FORMATS=("$@")
fi

# Stream log output line-by-line (live progress in logs/b3/*.log)
export PYTHONUNBUFFERED=1

# ── Python interpreter ────────────────────────────────────────────
# Prefer the project venv, then $PYTHON, then whatever is on PATH.
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

    local out_json="$RESULTS_DIR/${protocol}_${weight_format}_seed${seed}.json"
    local log_file="$LOG_DIR/${protocol}_${weight_format}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "Running: protocol=$protocol, weight=$weight_format, seed=$seed"
    log "Log: $log_file"

    if "$PYTHON" -m ph_neuro.examples.run_b3_precision_cl \
        --protocol "$protocol" \
        --weight-format "$weight_format" \
        --epochs-per-task "$EPOCHS_PER_TASK" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --n-tasks "$n_tasks" \
        --num-workers "$NUM_WORKERS" \
        --seed "$seed" \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1; then
        log "Done: protocol=$protocol, weight=$weight_format, seed=$seed"
    else
        log "FAILED (rc=$?): protocol=$protocol, weight=$weight_format, seed=$seed — see $log_file"
        return 1
    fi
}

# ── Run all configurations ─────────────────────────────────────────

log "B3: INT8 + INT4 continual learning (ternary/fp16 reused from L8)"
log "Precisions: ${WEIGHT_FORMATS[*]}  Protocols: ${PROTOCOLS[*]}  Seeds: ${SEEDS[*]}"

total=$(( ${#PROTOCOLS[@]} * ${#WEIGHT_FORMATS[@]} * ${#SEEDS[@]} ))
current=0
fails=0

for protocol in "${PROTOCOLS[@]}"; do
    for weight_format in "${WEIGHT_FORMATS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            current=$(( current + 1 ))
            log "[$current/$total]"
            if ! run_exp "$protocol" "$weight_format" "$seed"; then
                fails=$(( fails + 1 ))
            fi
        done
    done
done

# ── Completion check ────────────────────────────────────────────────

if [ "$fails" -gt 0 ]; then
    log "WARNING: $fails run(s) FAILED (see logs/b3/*.log)."
    exit 1
fi

n_json=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)
log "All $total runs complete. Result JSONs in $RESULTS_DIR: $n_json total."

# ── Generate aggregated summary ────────────────────────────────────

log "Generating aggregated summary..."
if "$PYTHON" -m ph_neuro.examples.aggregate_b3_results \
    --results-dir "$RESULTS_DIR" \
    --l8-dir l8_results \
    --output "$RESULTS_DIR/aggregated_summary.txt"; then
    log "Summary written to $RESULTS_DIR/aggregated_summary.txt"
else
    log "WARNING: aggregator failed (missing L8 baselines?)."
fi

log "Done! All $total runs complete."
