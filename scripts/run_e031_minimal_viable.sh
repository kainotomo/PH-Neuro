#!/usr/bin/env bash
# E031 — Minimal Viable Brain-Wrapper experiment orchestrator (Phase 1.1).
#
# Runs every (baseline × budget × seed) cell of the LOCKED Step 0.5 protocol
# with skip-if-exists (a finished cell is never re-run) and per-cell logging
# to logs/brain/e031/. The runner is the E031 single-cell script; this file
# only drives it.
#
# Usage:
#   bash scripts/run_e031_minimal_viable.sh            # all cells (16)
#   bash scripts/run_e031_minimal_viable.sh smoke      # micro 1K sanity, seed 42
#   bash scripts/run_e031_minimal_viable.sh surprise   # one baseline
#   bash scripts/run_e031_minimal_viable.sh constM     # one baseline
#
# Env overrides: RESULTS_DIR, LOG_DIR, GPU_POLICY (exit|wait|warn), SEEDS,
#   BUDGETS, EXTRA (extra args passed to the runner).
set -u

# Resolve repo root from this script's location (works from scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/brain/e031}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/brain/e031}"
GPU_POLICY="${GPU_POLICY:-exit}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"
M_SHORT="smolllm2_1p7b"
EXTRA="${EXTRA:-}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

RUNNER_ARGS="--model $MODEL --gpu-policy $GPU_POLICY --output-dir $RESULTS_DIR --log-dir $LOG_DIR $EXTRA"

fail_count=0
run_cell() {
    # run_cell BASELINE BUDGET SEED
    local baseline="$1" budget="$2" seed="$3"
    local tag
    if [[ "$baseline" == "constM" || "$baseline" == "surprise" ]]; then
        tag="$((budget / 1000))k"
    else
        tag="na"
    fi
    local out="$RESULTS_DIR/${M_SHORT}_pubmed_${tag}_${baseline}_seed${seed}.json"
    if [[ -f "$out" ]]; then
        echo "  [skip] $out already exists"
        return 0
    fi
    echo "  [run ] $baseline budget=$budget seed=$seed"
    if ! $PY -m ph_neuro.examples.run_e031_minimal_viable \
        --baseline "$baseline" --budget-tokens "$budget" --seed "$seed" \
        $RUNNER_ARGS; then
        echo "  [FAIL] $baseline budget=$budget seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

MODE="${1:-all}"
case "$MODE" in
    smoke)
        echo "== E031 smoke (micro 1K, seed 42, surprise + constM) =="
        run_cell constM    1000 42
        run_cell surprise  1000 42
        ;;
    frozen|random|constM|surprise)
        echo "== E031 baseline: $MODE =="
        if [[ "$MODE" == "frozen" ]]; then
            run_cell frozen 0 42
        elif [[ "$MODE" == "random" ]]; then
            for seed in ${SEEDS:-42 43 44}; do run_cell random 0 "$seed"; done
        else
            for budget in ${BUDGETS:-10000 100000}; do
                for seed in ${SEEDS:-42 43 44}; do run_cell "$MODE" "$budget" "$seed"; done
            done
        fi
        ;;
    all)
        echo "== E031 full protocol (16 cells) =="
        run_cell frozen 0 42
        for seed in ${SEEDS:-42 43 44}; do run_cell random 0 "$seed"; done
        for budget in ${BUDGETS:-10000 100000}; do
            for seed in ${SEEDS:-42 43 44}; do
                run_cell constM "$budget" "$seed"
                run_cell surprise "$budget" "$seed"
            done
        done
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "usage: $0 [all|smoke|frozen|random|constM|surprise]" >&2
        exit 2
        ;;
esac

echo "== E031 done ($MODE); failures: $fail_count =="
exit "$fail_count"
