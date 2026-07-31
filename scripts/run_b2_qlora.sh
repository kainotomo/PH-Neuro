#!/usr/bin/env bash
# ── B2 QLoRA + Frozen Ternary Backbone ─────────────────────────────
#
# Runs the QLoRA continual learning experiment (Track B, second
# experiment). Freezes a pre-trained ternary backbone and trains
# per-task LoRA adapters → zero forgetting by design.
#
# Usage:
#   bash scripts/run_b2_qlora.sh sweep              # rank sweep, 1 seed
#   bash scripts/run_b2_qlora.sh full 8             # best rank, 3 seeds
#
# Output:
#   JSON files: b2_results/{protocol}_{pretrain}_qlora_r{r}_seed{seed}.json
#   Log files:  logs/b2/{protocol}_{pretrain}_qlora_r{r}_seed{seed}.log
#               (gitignored)
#
# Full run output is redirected to the log file; the console only shows
# short status lines. Runs whose result JSON already exists are skipped.
#
# Total runs:
#   sweep: 16 (2 CL protocols × 2 pretrain modes × 4 ranks × seed 42)
#   full:  12 (best rank × 2 CL protocols × 2 pretrain modes × 3 seeds)
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="b2_results"
LOG_DIR="logs/b2"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── Configuration ──────────────────────────────────────────────────

EPOCHS_PRETRAIN=10
EPOCHS_PER_TASK=10
LR=0.001
BATCH_SIZE=128
SEEDS=(42 43 44)
LORA_RANKS=(2 4 8 16)
PROTOCOLS=(split permuted)
PRETRAINS=(full task1)

MODE="${1:-sweep}"

# Stream log output line-by-line (live progress in logs/b2/*.log)
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
    local pretrain="$2"
    local rank="$3"
    local seed="$4"
    local n_tasks

    if [ "$protocol" = "permuted" ]; then
        n_tasks=10
    else
        n_tasks=5
    fi

    local out_json="$RESULTS_DIR/${protocol}_${pretrain}_qlora_r${rank}_seed${seed}.json"
    local log_file="$LOG_DIR/${protocol}_${pretrain}_qlora_r${rank}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "Running: protocol=$protocol, pretrain=$pretrain, r=$rank, seed=$seed"
    log "Log: $log_file"

    if "$PYTHON" -m ph_neuro.examples.run_b2_qlora \
        --protocol "$protocol" \
        --pretrain "$pretrain" \
        --lora-r "$rank" \
        --epochs-pretrain "$EPOCHS_PRETRAIN" \
        --epochs-per-task "$EPOCHS_PER_TASK" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --n-tasks "$n_tasks" \
        --num-workers 0 \
        --seed "$seed" \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1; then
        log "Done: protocol=$protocol, pretrain=$pretrain, r=$rank, seed=$seed"
    else
        log "FAILED (rc=$?): protocol=$protocol, pretrain=$pretrain, r=$rank, seed=$seed — see $log_file"
        return 1
    fi
}

# ── Modes ──────────────────────────────────────────────────────────

if [ "$MODE" = "sweep" ]; then
    log "Phase 1: LoRA rank sweep (seed=42)"
    total=$(( ${#PROTOCOLS[@]} * ${#PRETRAINS[@]} * ${#LORA_RANKS[@]} ))
    current=0
    for protocol in "${PROTOCOLS[@]}"; do
        for pretrain in "${PRETRAINS[@]}"; do
            for rank in "${LORA_RANKS[@]}"; do
                current=$(( current + 1 ))
                log "[$current/$total]"
                run_exp "$protocol" "$pretrain" "$rank" 42
            done
        done
    done
    log "Sweep complete. Inspect results, pick best rank, then run:"
    log "  bash scripts/run_b2_qlora.sh full <RANK>"

elif [ "$MODE" = "full" ]; then
    RANK="${2:?usage: bash scripts/run_b2_qlora.sh full RANK}"

    log "Phase 2: full run with r=$RANK"
    total=$(( ${#PROTOCOLS[@]} * ${#PRETRAINS[@]} * ${#SEEDS[@]} ))
    current=0
    fails=0
    for protocol in "${PROTOCOLS[@]}"; do
        for pretrain in "${PRETRAINS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                current=$(( current + 1 ))
                log "[$current/$total]"
                if ! run_exp "$protocol" "$pretrain" "$RANK" "$seed"; then
                    fails=$(( fails + 1 ))
                fi
            done
        done
    done
    if [ "$fails" -gt 0 ]; then
        log "WARNING: $fails run(s) FAILED (see logs/b2/*.log)."
        exit 1
    fi
    log "Full run complete."

else
    echo "Unknown mode: $MODE (use 'sweep' or 'full RANK')" >&2
    exit 1
fi
