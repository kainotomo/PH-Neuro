#!/usr/bin/env bash
# E035 — Ternary LoRA experiment orchestrator (Step 2.2).
#
# The on-device product test: does 2-bit ternary adaptation preserve >=90% of
# float gated-LoRA quality (+0.902 -> >= 0.81) at 16x smaller storage?
# Three ternarization approaches on the same 344K budget / gated-LoRA
# protocol as E034: T-A post-training quantization (CAT-Q style, reuses the
# E034 float checkpoints), T-B DQT-style training, T-C STE with latent scores.
# Skip-if-exists (a finished cell is never re-run), per-cell logging to
# logs/brain/e035/, frozen caches copied from E034.
#
# Usage:
#   bash scripts/run_e035_ternary_lora.sh smoke   # 10K mechanism gate + magnitude, all variants
#   bash scripts/run_e035_ternary_lora.sh single  # T-A-q, T-A-qft, T-B, T-C @ 100K, 3 seeds
#   bash scripts/run_e035_ternary_lora.sh two     # two-domain selectivity on the best variant
#   bash scripts/run_e035_ternary_lora.sh agg     # cross-seed verdict
#
# Env overrides: RESULTS_DIR, LOG_DIR, GPU_POLICY, SEEDS, BUDGET, EXTRA.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/brain/e035}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/brain/e035}"
GPU_POLICY="${GPU_POLICY:-exit}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"
M_SHORT="smolllm2_1p7b"
EXTRA="${EXTRA:-}"
SEEDS="${SEEDS:-42 43 44}"
BUDGET="${BUDGET:-100000}"
CALIB_STEPS="${CALIB_STEPS:-20}"

E034_CKPT_DIR="$ROOT/results/brain/e034/checkpoints"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$RESULTS_DIR/cache"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Reuse the seed-independent frozen baselines from E034 (identical model,
# eval corpora -> bit-identical ppl; computed once ever).
if [[ -f "$ROOT/results/brain/e034/cache/frozen_${M_SHORT}_wikitext2.json" ]]; then
    cp -n "$ROOT/results/brain/e034/cache/frozen_${M_SHORT}_wikitext2.json" "$RESULTS_DIR/cache/"
    cp -n "$ROOT/results/brain/e034/cache/frozen_${M_SHORT}_pubmed.json" "$RESULTS_DIR/cache/"
    cp -n "$ROOT/results/brain/e034/cache/frozen_${M_SHORT}_cnn_dailymail.json" "$RESULTS_DIR/cache/"
    echo "== frozen wiki/pubmed/cnn eval caches copied from E034 =="
else
    echo "!! E034 frozen cache not found — E035 runner will compute it (first cell only)" >&2
fi

fail_count=0

run_cell() {
    # run_cell TERNARY PHASES TAG SEED [FLOAT_CKPT] [CALIB]
    local ternary="$1" phases="$2" tag="$3" seed="$4" float_ckpt="${5:-}" calib="${6:-0}"
    local budget_tag=$((BUDGET / 1000))k
    local out
    if [[ "$phases" == "2" ]]; then
        out="$RESULTS_DIR/${M_SHORT}_pubmed_cnn_${budget_tag}_${tag}_seed${seed}.json"
    else
        out="$RESULTS_DIR/${M_SHORT}_pubmed_${budget_tag}_${tag}_seed${seed}.json"
    fi
    if [[ -f "$out" ]]; then
        echo "  [skip] $out"
        return 0
    fi
    echo "  [run ] (ternary=$ternary, ${phases}p) $tag budget=$BUDGET seed=$seed calib=$calib"
    local args=""
    if [[ -n "$float_ckpt" ]]; then
        args="$args --float-ckpt $float_ckpt"
    fi
    if [[ "$calib" != "0" ]]; then
        args="$args --calib-steps $calib"
    fi
    if ! $PY -m ph_neuro.examples.run_e035_lora \
        --model "$MODEL" --ternary "$ternary" --phases "$phases" --tag "$tag" \
        --budget-tokens "$BUDGET" --lr 1e-3 --seed "$seed" --gpu-policy "$GPU_POLICY" \
        --output-dir "$RESULTS_DIR" --log-dir "$LOG_DIR" \
        --frozen-cache-dir "$RESULTS_DIR/cache" $args $EXTRA; then
        echo "  [FAIL] $tag seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

e034_float_ckpt() {
    # E034 float gated-LoRA checkpoint for a seed and phase count.
    local phases="$1" seed="$2"
    if [[ "$phases" == "2" ]]; then
        echo "$E034_CKPT_DIR/surprise_2p_gated2_budget${BUDGET}_seed${seed}/brain_ckpt_step296.pt"
    else
        echo "$E034_CKPT_DIR/surprise_1p_gated_budget${BUDGET}_seed${seed}/brain_ckpt_step198.pt"
    fi
}

best_variant() {
    # Read the best ternary variant from the last aggregation ("" if none).
    local f="$RESULTS_DIR/summary_e035.json"
    if [[ ! -f "$f" ]]; then
        echo ""
        return 1
    fi
    $PY -c "
import json
v = json.load(open('$f')).get('best_variant')
print(v if v else '')
"
}

MODE="${1:-agg}"
case "$MODE" in
    smoke)
        echo "== E035 smoke: 10K mechanism gate, all variants, seed 42 =="
        local_budget="$BUDGET"
        BUDGET="${SMOKE_BUDGET:-10000}"
        run_cell "float" 1 "float_smoke" 42
        # T-A smoke: quantize a real E034 float ckpt? No — use a tiny float train
        # would need 10K float training; instead smoke T-A from the E034 ckpt at
        # full budget is not a smoke. Here: smoke the TRAINED variants only.
        run_cell "tb" 1 "tb_smoke" 42
        run_cell "tc" 1 "tc_smoke" 42
        BUDGET="$local_budget"
        ;;
    single)
        echo "== E035 single-domain @ ${BUDGET} tokens, 3 seeds =="
        for s in $SEEDS; do
            # T-A: reuse the E034 float gated checkpoint (no float re-training).
            run_cell "ta" 1 "ta_q" "$s" "$(e034_float_ckpt 1 "$s")" 0
            run_cell "ta" 1 "ta_qft" "$s" "$(e034_float_ckpt 1 "$s")" "$CALIB_STEPS"
            run_cell "tb" 1 "tb" "$s"
            run_cell "tc" 1 "tc" "$s"
        done
        ;;
    two)
        echo "== E035 two-domain: best variant (from agg), 3 seeds =="
        bv=$(best_variant)
        if [[ -z "$bv" ]]; then
            echo "  [SKIP] no best variant yet (run \`agg\` first)" >&2
            exit 2
        fi
        echo "  best variant: $bv"
        for s in $SEEDS; do
            case "$bv" in
                ta_q)
                    run_cell "ta" 2 "ta_q2" "$s" "$(e034_float_ckpt 2 "$s")" 0 ;;
                ta_qft)
                    run_cell "ta" 2 "ta_qft2" "$s" "$(e034_float_ckpt 2 "$s")" "$CALIB_STEPS" ;;
                tb)
                    run_cell "tb" 2 "tb2" "$s" ;;
                tc)
                    run_cell "tc" 2 "tc2" "$s" ;;
                *)
                    echo "  [SKIP] unknown best variant '$bv'" >&2
                    exit 2 ;;
            esac
        done
        ;;
    agg)
        echo "== E035 cross-seed aggregation + verdict =="
        $PY -m ph_neuro.examples.aggregate_e035 \
            --results-dir "$RESULTS_DIR" --e034-dir "$ROOT/results/brain/e034"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "usage: $0 [smoke|single|two|agg]" >&2
        exit 2
        ;;
esac

echo "== E035 $MODE done; failures: $fail_count =="
exit "$fail_count"
