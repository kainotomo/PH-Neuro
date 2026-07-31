#!/usr/bin/env bash
# ── B1 EWC + Ternary STE Sweep ──────────────────────────────────────
#
# Runs the EWC continual learning experiment (Track B, first experiment).
# Compares against the L8 forgetting baseline (no EWC).
#
# Usage:
#   bash scripts/run_b1_ewc.sh sweep              # λ sweep on Split MNIST (seed=42)
#   bash scripts/run_b1_ewc.sh full 10.0          # best λ on Split + Permuted × 3 seeds
#
# Output:
#   JSON files: b1_results/{protocol}_ewc_lambda{L}_seed{seed}.json
#   Log files:  logs/b1/{protocol}_ewc_lambda{L}_seed{seed}.log  (gitignored)
#
# Full run output is redirected to the log file; the console only shows
# short status lines. Runs whose result JSON already exists are skipped.
#
# Total runs:
#   sweep: 5 (λ ∈ {0.1, 1.0, 10.0, 100.0, 1000.0} × split × seed 42)
#   full:  6 (best λ × {split, permuted} × seeds {42, 43, 44})
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="b1_results"
LOG_DIR="logs/b1"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── Configuration ──────────────────────────────────────────────────

EPOCHS_PER_TASK=10
LR=0.001
BATCH_SIZE=128
FISHER_SAMPLES=500
SEEDS=(42 43 44)
LAMBDAS=(0.1 1.0 10.0 100.0 1000.0)

MODE="${1:-sweep}"

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
    local lam="$2"
    local seed="$3"
    local n_tasks

    if [ "$protocol" = "permuted" ]; then
        n_tasks=10
    else
        n_tasks=5
    fi

    local lam_str
    lam_str=$(printf "%g" "$lam")
    local out_json="$RESULTS_DIR/${protocol}_ewc_lambda${lam_str}_seed${seed}.json"
    local log_file="$LOG_DIR/${protocol}_ewc_lambda${lam_str}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "Running: protocol=$protocol, lambda=$lam, seed=$seed"
    log "Log: $log_file"

    if "$PYTHON" -m ph_neuro.examples.run_b1_ewc \
        --protocol "$protocol" \
        --ewc-lambda "$lam" \
        --fisher-samples "$FISHER_SAMPLES" \
        --epochs-per-task "$EPOCHS_PER_TASK" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --n-tasks "$n_tasks" \
        --seed "$seed" \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1; then
        log "Done: protocol=$protocol, lambda=$lam, seed=$seed"
    else
        log "FAILED (rc=$?): protocol=$protocol, lambda=$lam, seed=$seed — see $log_file"
        return 1
    fi
    echo ""
}

# ── Modes ──────────────────────────────────────────────────────────

if [ "$MODE" = "sweep" ]; then
    log "Phase 1: λ sweep on Split MNIST (seed=42)"
    for lam in "${LAMBDAS[@]}"; do
        run_exp split "$lam" 42
    done
    log "Sweep complete. Inspect results, pick best λ, then run:"
    log "  bash scripts/run_b1_ewc.sh full <LAMBDA>"

elif [ "$MODE" = "full" ]; then
    LAMBDA="${2:?usage: bash scripts/run_b1_ewc.sh full LAMBDA}"

    log "Phase 2: full run with λ=$LAMBDA"
    total=$(( 2 * ${#SEEDS[@]} ))
    current=0
    for protocol in split permuted; do
        for seed in "${SEEDS[@]}"; do
            current=$(( current + 1 ))
            log "[$current/$total]"
            run_exp "$protocol" "$LAMBDA" "$seed"
        done
    done
    log "Full run complete."

else
    echo "Unknown mode: $MODE (use 'sweep' or 'full LAMBDA')" >&2
    exit 1
fi
