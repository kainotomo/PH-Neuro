#!/usr/bin/env bash
# ── M1.2-RETRY — DQT CNN on CIFAR-100 (GO/NO-GO >55%) ───────────────
#
# Milestone M1.2-RETRY: extend M1.2 (MARGINAL 54.15%, -0.85pp below gate)
# to 200 epochs. Best epochs (148-150) hit the 150-ep schedule ceiling, so
# a longer cosine + anneal@80% (deterministic tail 160-200) should push the
# 3-seed mean past 55%. Same 3-conv model `dqt_cnn_cifar100()`, lr=0.01,
# anneal@80% (validated M1.1-RETRY). Only runner defaults changed:
# --epochs 150->200, --patience 30->40 (> anneal_start_epoch = 160).
#
# STE baseline (E009/L1, CIFAR-100): 38.2%. Target: >55% (GO).
#
# Usage:
#   bash scripts/run_m1_2_dqt_cifar100.sh                    # full run, 3 seeds, lr=0.01, 200 ep
#   bash scripts/run_m1_2_dqt_cifar100.sh sweep              # LR sweep {0.01,0.005,0.001}, 1 seed
#   bash scripts/run_m1_2_dqt_cifar100.sh full 0.005         # full run with a chosen best LR
#   bash scripts/run_m1_2_dqt_cifar100.sh full 0.01 42 43 44 # custom seeds
#
# Output:
#   Sweep: m1_2_sweep_results/results_dqt_cifar100_lr{lr}_seed42.json
#   Full:  m1_2_retry_results/results_dqt_cifar100_lr{lr}_seed{seed}.json
#   Logs:  logs/logs_m1_2/
#
# Runs whose result JSON already exists are SKIPPED, so this script is
# safe to re-run.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Live line-by-line logs (B2 lesson: a silent mid-run OOM looked like a hang
# because Python's block-buffered stdout froze the log file).
export PYTHONUNBUFFERED=1

# ── GPU wait gate (shared GPU with gaming) ─────────────────────────
# `--wait-gpu` blocks until GPU_WAIT_THRESHOLD GB are free via
# scripts/gpu_wait.py before launching. Default: ON for `full`/`sweep`
# runs, OFF for smoke/status/resume. `--no-wait-gpu` forces it off.
WAIT_GPU=""                            # "" = mode default, 1 = on, 0 = off
GPU_WAIT_THRESHOLD="${GPU_WAIT_THRESHOLD:-7.0}"
GPU_WAIT_TIMEOUT="${GPU_WAIT_TIMEOUT:-120}"
_POS_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --wait-gpu)   WAIT_GPU=1 ;;
        --no-wait-gpu) WAIT_GPU=0 ;;
        *) _POS_ARGS+=("$_arg") ;;
    esac
done
if [ ${#_POS_ARGS[@]} -gt 0 ]; then set -- "${_POS_ARGS[@]}"; else set --; fi
unset _POS_ARGS

# ── Resolve project root (works from anywhere) ─────────────────────
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

FULL_RESULTS_DIR="m1_2_retry_results"
SWEEP_RESULTS_DIR="m1_2_sweep_results"
LOG_DIR="logs/logs_m1_2"
mkdir -p "$FULL_RESULTS_DIR" "$SWEEP_RESULTS_DIR" "$LOG_DIR"

# ── Configuration (M1.2-RETRY: 200 epochs, patience 40) ─────────────

EPOCHS=200
BATCH_SIZE=128
WEIGHT_DECAY=1e-4
PATIENCE=40          # > anneal_start_epoch = int(200 * 0.80) = 160
DEFAULT_LR=0.01      # used when no best-LR is passed to `full`
SEEDS=(42 43 44)
SWEEP_LRS=(0.01 0.005 0.001)
# DataLoader workers. Spec says 2; use 0 (env NUM_WORKERS=0) when the GPU is
# contended (e.g. a game) — avoids the fork() worker deadlock/OOM that killed
# seed 44 (B2 lesson: results identical, --num-workers 0).
NUM_WORKERS="${NUM_WORKERS:-2}"

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

# Block until the GPU is free before launching (shared GPU with gaming).
gpu_wait() {
    local default_on="$1"   # 1 = this mode waits by default
    local want="$default_on"
    if [ -n "$WAIT_GPU" ]; then
        want="$WAIT_GPU"
    fi
    if [ "$want" = "1" ]; then
        log "⏳ Waiting for GPU (need >=${GPU_WAIT_THRESHOLD} GB free, timeout ${GPU_WAIT_TIMEOUT} min)..."
        if ! "$PYTHON" scripts/gpu_wait.py --threshold "$GPU_WAIT_THRESHOLD" --timeout "$GPU_WAIT_TIMEOUT"; then
            log "❌ GPU not free in time — retry later (e.g. bash scripts/train.sh ...)"
            exit 1
        fi
    fi
}

print_config() {
    log "═══════════════════════════════════════════════════════════════"
    log "  M1.2-RETRY DQT CNN CIFAR-100  (GO/NO-GO >55%, 200 ep)"
    log "  Architecture: TernaryDQTConv2d(3->64)->Pool"
    log "                ->TernaryDQTConv2d(64->128)->Pool"
    log "                ->TernaryDQTConv2d(128->256)->Pool"
    log "                ->TernaryDQTLinear(4096->512)->TernaryDQTLinear(512->100)"
    log "  Optimizer: AdamW (lr=${DEFAULT_LR}, wd=${WEIGHT_DECAY}) + CosineAnnealingLR"
    log "  DQT: apply_dqt_rounding() after EVERY optimizer.step()"
    log "       stochastic_round() for first 80%, deterministic sign() for last 20%"
    log "  Epochs: ${EPOCHS}  Batch: ${BATCH_SIZE}  Patience: ${PATIENCE}"
    log "═══════════════════════════════════════════════════════════════"
}

run_one() {
    local lr="$1"
    local seed="$2"
    local results_dir="$3"
    local lr_str
    lr_str=$(printf "%g" "$lr")
    local out_json="$results_dir/results_dqt_cifar100_lr${lr_str}_seed${seed}.json"
    local log_file="$LOG_DIR/results_dqt_cifar100_lr${lr_str}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "▶ DQT CIFAR-100  lr=${lr}  seed=${seed}  epochs=${EPOCHS}  batch=${BATCH_SIZE}  workers=${NUM_WORKERS}"
    # `if !` so a failed run does NOT abort the whole script (B2 lesson: continue
    # past failed runs, report FAILED, other seeds still complete).
    if ! "$PYTHON" -m ph_neuro.examples.run_m1_2_dqt_cifar100 \
        --lr "$lr" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --weight-decay "$WEIGHT_DECAY" \
        --patience "$PATIENCE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$results_dir" \
        > "$log_file" 2>&1; then
        log "FAILED: $out_json (see $log_file)"
        FAILED_RUNS+=("$out_json")
        tail -18 "$log_file"
        echo ""
        return 1
    fi
    # Print the summary block (tail of log)
    tail -18 "$log_file"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────

print_config
echo ""

FAILED_RUNS=()

gpu_wait 1
if [ "$MODE" = "sweep" ]; then
    log "LR sweep (1 seed = 42) over ${SWEEP_LRS[*]} to pick the best LR"
    for lr in "${SWEEP_LRS[@]}"; do
        run_one "$lr" 42 "$SWEEP_RESULTS_DIR"
    done
elif [ "$MODE" = "full" ]; then
    BEST_LR="${2:-$DEFAULT_LR}"
    SEED_ARGS=()
    for s in "${@:3}"; do
        SEED_ARGS+=("$s")
    done
    if [ ${#SEED_ARGS[@]} -eq 0 ]; then
        SEED_ARGS=("${SEEDS[@]}")
    fi
    log "Full run: seeds ${SEED_ARGS[*]} × lr=${BEST_LR} (best from sweep)"
    for seed in "${SEED_ARGS[@]}"; do
        run_one "$BEST_LR" "$seed" "$FULL_RESULTS_DIR"
    done
else
    echo "ERROR: unknown mode '$MODE' (expected 'sweep' or 'full [lr] [seeds...]')" >&2
    exit 1
fi

log "All runs complete!"
echo ""
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log "⚠️  ${#FAILED_RUNS[@]} run(s) FAILED:"
    for f in "${FAILED_RUNS[@]}"; do log "  - $f"; done
else
    log "✅ No failed runs."
fi
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Sweep results in: $SWEEP_RESULTS_DIR/"
echo "  Full results in:  $FULL_RESULTS_DIR/"
echo "  Logs in:          $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════════"
