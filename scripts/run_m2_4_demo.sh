#!/usr/bin/env bash
# ── M2.4 — On-device inference demo: DQT Transformer text generation ──
#
# End-to-end demo that a DQT Transformer can be trained on TinyStories and
# then generate text on CPU (smartphone simulation) — no GPU, no DQT
# custom autograd functions, just the frozen ternary weights running as
# standard layers (and as ONNX).
#
# Four steps (output is copy-paste ready for the README / E028 report):
#   1. Train  — DEMO_CONFIG DQT Transformer (2 epochs, ~1.6 h on RTX 4060).
#               SKIPPED automatically if best.pt already exists.
#   2. Export — ONNX (run_m1_3_export --model dqt_gpt2) + 2-bit packed file.
#   3. Generate — PyTorch CPU (torch.no_grad, inference model).
#   4. Generate — ONNX CPU (onnxruntime) + PyTorch vs ONNX speed comparison.
#
# DEMO_CONFIG:
#   d_model=512, n_layers=6, n_heads=8, d_ff=2048, vocab=50257 (GPT-2 BPE)
#   ~44.6M ternary weights + 25.7M float embedding ≈ 70.3M total params
#   TinyStories (reuses the existing M2.1 disk cache, max_samples=150000)
#
# Usage:
#   bash scripts/run_m2_4_demo.sh                        # full demo
#   bash scripts/run_m2_4_demo.sh "Lily was a small fox"  # custom prompt
#   bash scripts/run_m2_4_demo.sh generate               # skip train+export
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Live line-by-line logs (B2 lesson).
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

# ── Configuration ───────────────────────────────────────────────────
# DEMO_CONFIG (fast, demo-quality — NOT competitive ppl)
D_MODEL=512
N_LAYERS=6
N_HEADS=8
D_FF=2048
VOCAB_SIZE=50257

LR=0.01
EPOCHS=2
SEED=42
BATCH_SIZE=8
SEQ_LEN=256
MAX_SAMPLES=150000          # reuse the existing M2.1 TinyStories cache
NUM_WORKERS="${NUM_WORKERS:-0}"  # 0 safest under GPU contention
CHECKPOINT_EVERY=2000

RESULTS_DIR="m2_4_demo"
LOG_DIR="logs/logs_m2_4"
ONNX="models/dqt_transformer_demo.onnx"
PACKED="models/dqt_transformer_demo.ternary"

PROMPT="${2:-Once upon a time}"
MAX_TOKENS=100
TEMPERATURE=0.8
TOP_K=50

MODE="${1:-full}"

# ── Python interpreter ──────────────────────────────────────────────
PYTHON=".venv/bin/python"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "models"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

BEST_CKPT="$RESULTS_DIR/checkpoints/seed$SEED/best.pt"

# ── Step 1: Train (skip if best.pt exists) ─────────────────────────
train() {
    if [ -f "$BEST_CKPT" ]; then
        log "SKIP train — checkpoint already exists: $BEST_CKPT"
        return 0
    fi
    log "▶ STEP 1/4 — Training DQT Transformer (DEMO_CONFIG, ${EPOCHS} epochs)"
    log "  d_model=${D_MODEL} n_layers=${N_LAYERS} n_heads=${N_HEADS} d_ff=${D_FF} lr=${LR} seed=${SEED}"
    if ! "$PYTHON" -m ph_neuro.examples.run_m2_1_dqt_transformer \
        --d-model "$D_MODEL" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" --d-ff "$D_FF" \
        --lr "$LR" --epochs "$EPOCHS" --seed "$SEED" \
        --batch-size "$BATCH_SIZE" --seq-len "$SEQ_LEN" \
        --max-samples "$MAX_SAMPLES" --num-workers "$NUM_WORKERS" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --output-dir "$RESULTS_DIR" \
        > "$LOG_DIR/train_seed${SEED}.log" 2>&1; then
        log "FAILED training — see $LOG_DIR/train_seed${SEED}.log"
        tail -18 "$LOG_DIR/train_seed${SEED}.log"
        return 1
    fi
    tail -14 "$LOG_DIR/train_seed${SEED}.log"
    echo ""
}

# ── Step 2: Export ONNX (skip if exists) ───────────────────────────
export_onnx() {
    if [ -f "$ONNX" ]; then
        log "SKIP export — ONNX already exists: $ONNX"
        return 0
    fi
    log "▶ STEP 2/4 — Exporting ONNX + 2-bit packed ternary"
    # `|| true` keeps the noisy torch.onnx progress lines from failing the
    # script; the file-existence check below is the real failure signal.
    "$PYTHON" -m ph_neuro.examples.run_m1_3_export \
        --model dqt_gpt2 --checkpoint "$BEST_CKPT" \
        --output "$ONNX" --packed --verify \
        2>&1 | grep -v "torch.onnx\|FutureWarning\|copyreg\|Run decomp\|Optimize\|Translate\|Obtain model" || true
    if [ ! -f "$ONNX" ]; then
        log "FAILED export — no $ONNX produced"
        return 1
    fi
    echo ""
}

# ── Step 3: Generate (PyTorch CPU) ─────────────────────────────────
generate_torch() {
    log "▶ STEP 3/4 — Generation (PyTorch CPU — smartphone simulation)"
    "$PYTHON" -m ph_neuro.examples.generate_text \
        --checkpoint "$BEST_CKPT" \
        --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" --top-k "$TOP_K" \
        --seed "$SEED" --device cpu \
        --output-dir "$RESULTS_DIR"
    echo ""
}

# ── Step 4: Generate (ONNX CPU) + comparison ───────────────────────
generate_onnx() {
    log "▶ STEP 4/4 — Generation (ONNX CPU) + PyTorch vs ONNX comparison"
    "$PYTHON" -m ph_neuro.examples.generate_text \
        --checkpoint "$BEST_CKPT" \
        --onnx "$ONNX" --compare \
        --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" --top-k "$TOP_K" \
        --seed "$SEED" --device cpu \
        --output-dir "$RESULTS_DIR"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════════════"
log "  M2.4 — On-device inference demo (DQT Transformer text gen)"
log "  Config: d_model=${D_MODEL} L=${N_LAYERS} H=${N_HEADS} ff=${D_FF} (${VOCAB_SIZE} vocab)"
log "  Prompt: '${PROMPT}'  max_tokens=${MAX_TOKENS} t=${TEMPERATURE} top_k=${TOP_K}"
log "═══════════════════════════════════════════════════════════════"
echo ""

case "$MODE" in
    train)     train ;;
    export)    export_onnx ;;
    generate)  generate_torch; generate_onnx ;;
    full|*)
        train
        export_onnx
        generate_torch
        generate_onnx
        ;;
esac

log "M2.4 demo complete! Artifacts:"
log "  checkpoint : $BEST_CKPT"
log "  ONNX       : $ONNX"
log "  packed     : $PACKED"
log "  results    : $RESULTS_DIR/"
echo ""
