#!/usr/bin/env bash
# ── M2.5 — Public demo + blog post: PH-Neuro launch ─────────────────
#
# End-to-end pipeline for the public launch: train the 3 M2.5 models,
# export them to ONNX + 2-bit packed ternary, and serve the 3-tab Gradio
# demo (text generation, image classification, benchmarks) on CPU.
#
# The three models:
#   📝 DQT Transformer 102M (TinyStories, ~2 h on RTX 4060)
#   🖼️ DQT CNN CIFAR-10 (~10 min)      → ~79% test accuracy
#   🖼️ DQT CNN CIFAR-100 (~20 min)     → ~54% test accuracy
#
# Usage:
#   bash scripts/run_m2_5_demo.sh                 # full: train + export + demo
#   bash scripts/run_m2_5_demo.sh train           # train any unfinished models
#   bash scripts/run_m2_5_demo.sh export          # export ONNX + packed (needs checkpoints)
#   bash scripts/run_m2_5_demo.sh demo            # run the Gradio demo only
#   bash scripts/run_m2_5_demo.sh demo --share    # pass extra args to the demo
#
# Transformer pause/resume (Option B, 150K samples):
#   ⏸ PAUSE : kill -INT $(cat logs/logs_m2_5/text_model.pid)
#   ▶ RESUME: bash scripts/run_m2_5_demo.sh resume
#   📈 STATUS: bash scripts/run_m2_5_demo.sh status
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

export PYTHONUNBUFFERED=1

# ── Resolve project root ────────────────────────────────────────────
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

PYTHON=".venv/bin/python"
RESULTS_DIR="results/phase2/m2_5"
LOG_DIR="logs/logs_m2_5"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

MODE="${1:-full}"
EXTRA_ARGS=("${@:2}")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Train step (idempotent: skips models that already have best.pt) ─
train_one() {
    local name="$1"; shift
    local best="$1"; shift
    if [ -f "$best" ]; then
        log "SKIP train $name — checkpoint exists: $best"
        return 0
    fi
    log "▶ Training $name"
    if ! "$PYTHON" "$@" > "$LOG_DIR/train_${name}.log" 2>&1; then
        log "FAILED training $name — see $LOG_DIR/train_${name}.log"
        tail -20 "$LOG_DIR/train_${name}.log"
        return 1
    fi
    tail -8 "$LOG_DIR/train_${name}.log"
    echo ""
}

train() {
    log "══ STEP 1 — Training the 3 M2.5 models (skips finished ones) ══"

    # 📝 DQT Transformer 102M (FULL_CONFIG: d768/L9/H12/ff3072)
    # Option B (quality): 150K TinyStories samples — matches M2.1 GO (ppl ~11.4),
    # ~2 h on RTX 4060. Detached-friendly: --pid-file for graceful pause.
    train_one text_model \
        "$RESULTS_DIR/text_model/checkpoints/seed42/best.pt" \
        -m ph_neuro.examples.run_m2_1_dqt_transformer \
        --d-model 768 --n-layers 9 --n-heads 12 --d-ff 3072 \
        --epochs 3 --lr 0.01 --seed 42 --batch-size 8 \
        --max-samples 150000 --num-workers 0 \
        --checkpoint-every 2000 \
        --pid-file "$LOG_DIR/text_model.pid" \
        --output-dir "$RESULTS_DIR/text_model"

    # 🖼️ DQT CNN CIFAR-10
    train_one vision_cifar10 \
        "$RESULTS_DIR/vision_cifar10/checkpoints/seed42/best.pt" \
        -m ph_neuro.examples.run_m1_1_dqt_cifar10 \
        --lr 0.01 --epochs 100 --seed 42 --patience 25 \
        --output-dir "$RESULTS_DIR/vision_cifar10"

    # 🖼️ DQT CNN CIFAR-100
    train_one vision_cifar100 \
        "$RESULTS_DIR/vision_cifar100/checkpoints/seed42/best.pt" \
        -m ph_neuro.examples.run_m1_2_dqt_cifar100 \
        --lr 0.01 --epochs 150 --seed 42 --patience 30 \
        --output-dir "$RESULTS_DIR/vision_cifar100"
}

# ── Export step (ONNX + 2-bit packed ternary) ──────────────────────
export_one() {
    local model="$1"; local ckpt="$2"; local out="$3"
    if [ -f "$out" ]; then
        log "SKIP export $model — $out exists"
        return 0
    fi
    log "▶ Exporting $model → $out"
    "$PYTHON" -m ph_neuro.examples.run_m1_3_export \
        --model "$model" --checkpoint "$ckpt" \
        --output "$out" --packed --verify \
        2>&1 | grep -v "torch.onnx\|FutureWarning\|copyreg\|Run decomp\|Optimize\|Translate\|Obtain model\|Compiling graph\|graph \("
    if [ ! -f "$out" ]; then
        log "FAILED export $model — no $out produced"
        return 1
    fi
    echo ""
}

export_all() {
    log "══ STEP 2 — Exporting the 3 models to ONNX + 2-bit packed ══"
    export_one dqt_gpt2 \
        "$RESULTS_DIR/text_model/checkpoints/seed42/best.pt" \
        "$RESULTS_DIR/text_model.onnx"
    export_one dqt_cnn \
        "$RESULTS_DIR/vision_cifar10/checkpoints/seed42/best.pt" \
        "$RESULTS_DIR/vision_cifar10.onnx"
    export_one dqt_cnn_cifar100 \
        "$RESULTS_DIR/vision_cifar100/checkpoints/seed42/best.pt" \
        "$RESULTS_DIR/vision_cifar100.onnx"
    log "  → packed .ternary companions written next to each .onnx"
}

# ── Run the Gradio demo ────────────────────────────────────────────
demo() {
    log "══ STEP 3 — Launching the Gradio demo (3 tabs, CPU) ══"
    "$PYTHON" scripts/run_m2_5_demo.py --onnx-dir "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
}

# ── Transformer pause/resume ───────────────────────────────────────
resume_text() {
    log "▶ Resuming DQT Transformer from the latest checkpoint (base lr 0.01)"
    "$PYTHON" -m ph_neuro.examples.run_m2_1_dqt_transformer \
        --d-model 768 --n-layers 9 --n-heads 12 --d-ff 3072 \
        --epochs 3 --lr 0.01 --seed 42 --batch-size 8 \
        --max-samples 150000 --num-workers 0 \
        --checkpoint-every 2000 \
        --pid-file "$LOG_DIR/text_model.pid" \
        --resume auto \
        --output-dir "$RESULTS_DIR/text_model"
}

status_text() {
    echo "── Transformer status ──"
    if [ -f "$LOG_DIR/text_model.pid" ]; then
        local pid; pid="$(cat "$LOG_DIR/text_model.pid")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "  RUNNING (pid $pid)"
        else
            echo "  pid $pid exists but process not running (may be paused/finished)"
        fi
    else
        echo "  no pid file — never started or log cleaned"
    fi
    echo "  checkpoints: $RESULTS_DIR/text_model/checkpoints/seed42/"
    echo "  log: $LOG_DIR/train_text_model.log"
    if [ -f "$LOG_DIR/train_text_model.log" ]; then
        echo "  last line:"; tail -2 "$LOG_DIR/train_text_model.log" | sed 's/^/    /'
    fi
}

case "$MODE" in
    full)   train; export_all; demo ;;
    train)  train ;;
    export) export_all ;;
    demo)   demo ;;
    resume) resume_text ;;
    status) status_text ;;
    *)
        echo "Unknown mode: $MODE (use full | train | export | demo | resume | status)" >&2
        exit 1
        ;;
esac
