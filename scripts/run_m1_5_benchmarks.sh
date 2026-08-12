#!/usr/bin/env bash
# ── M1.5 — Memory benchmarks vs TF Lite (4× smaller, 2× faster) ──────
#
# Final Phase 1 milestone. Benchmarks the three PH-Neuro ternary models
# (ste_mlp, dqt_cnn, dqt_cnn_cifar100) on:
#
#   1. Model size      — measured packed 2-bit bytes + ONNX file size,
#                        vs *theoretical* TF Lite INT8/FP16 (same arch)
#   2. Inference speed — measured CPU latency (batch=1), torch fused +
#                        ONNX runtime; TF Lite INT8 is a theoretical 2×
#                        estimate (2-bit popcount vs 8-bit multiply-add)
#   3. Training memory — measured DQT peak CUDA memory (1 epoch) via
#                        torch.cuda.reset_peak_memory_stats(); STE memory
#                        is a theoretical 4.5× estimate (E017: ~2 vs ~9
#                        bytes/param)
#
# TF Lite is NOT installed — the comparison is theoretical + measured
# PyTorch/ONNX (per the milestone brief).
#
# Usage:
#   bash scripts/run_m1_5_benchmarks.sh                 # all 3 models
#   bash scripts/run_m1_5_benchmarks.sh dqt_cnn         # one model
#   bash scripts/run_m1_5_benchmarks.sh ste_mlp dqt_cnn # a subset
#
# Output:
#   results/phase1/m1_5_results/results_m1_5_{model}.json
#   logs/logs_m1_5/results_m1_5_{model}.log
#
# Runs whose result JSON already exists are SKIPPED (safe to re-run).
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Live line-by-line logs (B2 lesson: block-buffered stdout can look frozen).
export PYTHONUNBUFFERED=1

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

RESULTS_DIR="results/phase1/m1_5_results"
LOG_DIR="logs/logs_m1_5"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── Configuration ───────────────────────────────────────────────────

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(dqt_cnn dqt_cnn_cifar100 ste_mlp)
fi
BATCHES=100          # timed inference batches (after warmup)
WARMUP=10
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}"  # inference always CPU per brief
BATCH_SIZE=1

# ── Python interpreter ────────────────────────────────────────────
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

# ── Main ────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════════════════"
log "  M1.5 Memory benchmarks vs TF Lite (4× smaller, 2× faster)"
log "  Models: ${MODELS[*]}"
log "  Inference: device=${INFERENCE_DEVICE} batch=${BATCH_SIZE} batches=${BATCHES} (warmup ${WARMUP})"
log "  Training memory: measured DQT (CUDA) + theoretical STE (4.5×)"
log "  Results: ${RESULTS_DIR}/   Logs: ${LOG_DIR}/"
log "═══════════════════════════════════════════════════════════════"
echo ""

FAILED_RUNS=()

for model in "${MODELS[@]}"; do
    out_json="$RESULTS_DIR/results_m1_5_${model}.json"
    log_file="$LOG_DIR/results_m1_5_${model}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        continue
    fi

    log "▶ Benchmarking ${model}  (CPU inference + GPU training memory)"
    if ! "$PYTHON" -m ph_neuro.examples.run_m1_5_benchmarks \
        --model "$model" \
        --device "$INFERENCE_DEVICE" \
        --batches "$BATCHES" \
        --warmup "$WARMUP" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1; then
        log "FAILED: $out_json (see $log_file)"
        FAILED_RUNS+=("$out_json")
        tail -18 "$log_file"
        echo ""
        continue
    fi
    # Print the GO/NO-GO summary block (tail of log)
    tail -14 "$log_file"
    echo ""
done

log "All benchmarks complete!"
echo ""
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    log "⚠️  ${#FAILED_RUNS[@]} benchmark(s) FAILED:"
    for f in "${FAILED_RUNS[@]}"; do log "  - $f"; done
else
    log "✅ No failed runs."
fi
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results in: $RESULTS_DIR/"
echo "  Logs in:    $LOG_DIR/"
echo "  Report:     research/docs/experiments/E024-m1-5-benchmarks.md"
echo "═══════════════════════════════════════════════════════════════"
