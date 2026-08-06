#!/usr/bin/env bash
# ── M2.1 — DQT Transformer on TinyStories (GO/NO-GO ppl<30) ─────────
#
# CRITICAL GO/NO-GO milestone. First DQT on a Transformer LM: GPT-2-style
# decoder-only model whose Q/K/V/O, FFN and LM-head projections use int8
# ternary DQT weights (stochastic rounding + annealing). If the mean
# validation perplexity across 3 seeds is < 30 → GO to MoE scaling (M2.3);
# otherwise the plan pivots.
#
# Architecture (FULL, ~102M ternary weights):
#   d_model=768, n_layers=9, n_heads=12, d_ff=3072, vocab=50257 (GPT-2)
#   emb(50257→768) + 9x[Attn(12h, RoPE) + FFN(3072)] + RMSNorm + DQT LM Head
#   = ~102M ternary + ~39M float embedding ≈ 141M total params
#
# Hyperparameters (DQT validated in M1.1/M1.2):
#   lr=0.01, AdamW betas=(0.9,0.95), wd=0.1, seq_len=256, batch=8,
#   warmup=100, grad_clip=1.0, anneal@80% (stochastic → deterministic sign),
#   3 epochs, 3 seeds.
#
# Usage:
#   bash scripts/run_m2_1_dqt_transformer.sh                    # full run, 3 seeds
#   bash scripts/run_m2_1_dqt_transformer.sh smoke              # tiny synthetic sanity
#   bash scripts/run_m2_1_dqt_transformer.sh sweep              # LR sweep {0.01,0.005,0.003}
#   bash scripts/run_m2_1_dqt_transformer.sh full 0.005         # full run, custom LR
#   bash scripts/run_m2_1_dqt_transformer.sh full 0.01 42 43 44 # custom seeds
#   bash scripts/run_m2_1_dqt_transformer.sh resume 0.01 42     # pause/resume a seed
#                                                                 (from latest checkpoint)
#
# Output:
#   Results: m2_1_results/results_m2_1_dqt_transformer_lr{lr}_seed{seed}.json
#   Checkpoints: m2_1_results/checkpoints/  (per-step, if --checkpoint-every)
#   Logs:    logs/logs_m2_1/
#
# Runs whose result JSON already exists are SKIPPED, so re-running is safe.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Live line-by-line logs (B2 lesson: block-buffered stdout froze logs).
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

RESULTS_DIR="m2_1_results"
SMOKE_RESULTS_DIR="m2_1_smoke_results"
LOG_DIR="logs/logs_m2_1"
mkdir -p "$RESULTS_DIR" "$SMOKE_RESULTS_DIR" "$LOG_DIR"

# ── Configuration ───────────────────────────────────────────────────

# FULL config (~102M ternary weights)
D_MODEL=768
N_LAYERS=9
N_HEADS=12
D_FF=3072
# SMOKE config (~16M ternary) — fast sanity that the DQT transformer converges
SMOKE_D_MODEL=256
SMOKE_N_LAYERS=4
SMOKE_N_HEADS=4
SMOKE_D_FF=1024

EPOCHS=3
SMOKE_EPOCHS=2
BATCH_SIZE=8
SEQ_LEN=256
LR=0.01
WEIGHT_DECAY=0.1
WARMUP_STEPS=100
GRAD_CLIP=1.0
ANNEAL_FRACTION=0.80
# Stories downloaded/tokenized (cache ~40M tokens for 3 epochs). Tune up for
# more data / longer runs. MAX_SAMPLES=0 uses the whole dataset.
MAX_SAMPLES=150000
NUM_WORKERS="${NUM_WORKERS:-0}"   # 0 safest under GPU contention (B2 lesson)
CHECKPOINT_EVERY=2000             # 0 disables checkpoints

SEEDS=(42 43 44)
SWEEP_LRS=(0.01 0.005 0.003)
DEFAULT_LR=0.01

MODE="${1:-full}"

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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

print_config() {
    log "═══════════════════════════════════════════════════════════════"
    log "  M2.1 DQT Transformer TinyStories  (GO/NO-GO: mean ppl < 30)"
    log "  Architecture: d_model=${D_MODEL} L=${N_LAYERS} H=${N_HEADS} ff=${D_FF}"
    log "                 emb(50257->${D_MODEL}) + ${N_LAYERS}x[Attn(${N_HEADS}h, RoPE)+FFN(${D_FF})] + RMSNorm + DQT LM Head"
    log "  Optimizer: AdamW (lr=${LR}, wd=${WEIGHT_DECAY}) + cosine warmup(${WARMUP_STEPS})"
    log "  DQT: apply_dqt_rounding() after EVERY optimizer.step()"
    log "       stochastic_round() for first $(awk "BEGIN{printf \"%.0f\", ${ANNEAL_FRACTION}*100}")%, deterministic sign() after"
    log "  Epochs: ${EPOCHS}  Batch: ${BATCH_SIZE}  Seq: ${SEQ_LEN}  GradClip: ${GRAD_CLIP}"
    log "  Data: TinyStories max_samples=${MAX_SAMPLES}  Workers: ${NUM_WORKERS}"
    log "═══════════════════════════════════════════════════════════════"
}

run_one() {
    local lr="$1"
    local seed="$2"
    local mode_flags="$3"
    local out_json="$RESULTS_DIR/results_m2_1_dqt_transformer_lr${lr}_seed${seed}.json"
    local log_file="$LOG_DIR/results_m2_1_dqt_transformer_lr${lr}_seed${seed}.log"

    if [ -f "$out_json" ]; then
        log "Skip (result exists): $out_json"
        return 0
    fi

    log "▶ DQT Transformer  lr=${lr}  seed=${seed}  ${mode_flags}"
    # `if !` so a failed run does NOT abort the script (B2 lesson).
    if ! "$PYTHON" -m ph_neuro.examples.run_m2_1_dqt_transformer \
        $mode_flags \
        --lr "$lr" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --weight-decay "$WEIGHT_DECAY" \
        --warmup-steps "$WARMUP_STEPS" \
        --grad-clip "$GRAD_CLIP" \
        --anneal-fraction "$ANNEAL_FRACTION" \
        --max-samples "$MAX_SAMPLES" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$RESULTS_DIR" \
        > "$log_file" 2>&1; then
        log "FAILED: $out_json (see $log_file)"
        FAILED_RUNS+=("$out_json")
        tail -18 "$log_file"
        echo ""
        return 1
    fi
    tail -18 "$log_file"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────

print_config
echo ""

FAILED_RUNS=()

case "$MODE" in
    smoke)
        # Tiny synthetic sanity — no TinyStories download, ~1-2 min on GPU.
        # Uses anneal-fraction 1.0 (pure stochastic rounding): with only ~100
        # steps the float buffers are still near-zero, so a premature
        # deterministic-sign switch would snap ~90% of weights (the known DQT
        # annealing failure mode). The anneal itself runs at full scale.
        log "SMOKE: tiny synthetic corpus (no download), ${SMOKE_EPOCHS} epochs, pure stochastic rounding"
        "$PYTHON" -m ph_neuro.examples.run_m2_1_dqt_transformer \
            --smoke --synthetic \
            --lr 0.01 --epochs "$SMOKE_EPOCHS" --seed 42 \
            --batch-size 8 --seq-len 128 --anneal-fraction 1.0 \
            --synthetic-batches 50 \
            --output-dir "$SMOKE_RESULTS_DIR" 2>&1 | tee "$LOG_DIR/smoke.log"
        ;;
    sweep)
        log "LR sweep (seed 42) over ${SWEEP_LRS[*]} to pick the best LR"
        for lr in "${SWEEP_LRS[@]}"; do
            run_one "$lr" 42 "--d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --d-ff $D_FF --checkpoint-every 0"
        done
        ;;
    full)
        BEST_LR="${2:-$DEFAULT_LR}"
        SEED_ARGS=()
        for s in "${@:3}"; do
            SEED_ARGS+=("$s")
        done
        if [ ${#SEED_ARGS[@]} -eq 0 ]; then
            SEED_ARGS=("${SEEDS[@]}")
        fi
        log "Full run: seeds ${SEED_ARGS[*]} × lr=${BEST_LR} (FULL config ~102M ternary)"
        for seed in "${SEED_ARGS[@]}"; do
            run_one "$BEST_LR" "$seed" "--d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --d-ff $D_FF --checkpoint-every $CHECKPOINT_EVERY"
        done
        ;;
    resume)
        # Pause/resume: continue a seed from its latest checkpoint in
        # m2_1_results/checkpoints/seed{seed}/ (runner --resume auto).
        # The result JSON does not exist yet (run was interrupted), so the
        # skip-if-exists guard does not block us.
        BEST_LR="${2:-$DEFAULT_LR}"
        SEED="${3:-42}"
        log "Resume: seed ${SEED} × lr=${BEST_LR} (auto-resume from latest checkpoint)"
        run_one "$BEST_LR" "$SEED" "--d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS --d-ff $D_FF --checkpoint-every $CHECKPOINT_EVERY --resume auto"
        ;;
    *)
        echo "ERROR: unknown mode '$MODE' (expected 'full [lr] [seeds...]', 'smoke', 'sweep' or 'resume [lr] [seed]')" >&2
        exit 1
        ;;
esac

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
echo "  Results in:  $RESULTS_DIR/"
echo "  Logs in:     $LOG_DIR/"
echo "  Gate:        mean val perplexity (3 seeds) < 30 → GO"
echo "═══════════════════════════════════════════════════════════════"
