#!/usr/bin/env bash
# E034 — Surprise-Gated LoRA experiment orchestrator (Step 2.1).
#
# Drives the value-add test of the surprise gate on top of backprop LoRA with
# skip-if-exists (a finished cell is never re-run) and per-cell logging to
# logs/brain/e034/. Frozen eval baselines are reused from the E031/E032 cache
# (wiki + pubmed are seed-independent and bit-identical); the CNN/DailyMail
# frozen baseline is computed once here (verify-ds mode).
#
# Usage:
#   bash scripts/run_e034_surprise_gated_lora.sh verify-ds  # CNN/DailyMail frozen cache (once)
#   bash scripts/run_e034_surprise_gated_lora.sh smoke      # 10K mechanism gate, seed 42
#   bash scripts/run_e034_surprise_gated_lora.sh single     # gated single-domain, 3 seeds @ 100K
#   bash scripts/run_e034_surprise_gated_lora.sh two        # plain + gated two-domain, 3 seeds @ 100K
#   bash scripts/run_e034_surprise_gated_lora.sh control    # const_reduced control (needs `single`)
#   bash scripts/run_e034_surprise_gated_lora.sh agg        # cross-seed verdict
#
# Env overrides: RESULTS_DIR, LOG_DIR, GPU_POLICY, SEEDS, BUDGET, EXTRA.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/brain/e034}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/brain/e034}"
GPU_POLICY="${GPU_POLICY:-exit}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"
M_SHORT="smolllm2_1p7b"
EXTRA="${EXTRA:-}"
SEEDS="${SEEDS:-42 43 44}"
BUDGET="${BUDGET:-100000}"
LR="${LR:-1e-3}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$RESULTS_DIR/cache"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Reuse the seed-independent frozen baselines measured in E031/E032 (identical
# model, identical eval corpora → identical ppl; only computed once ever).
if [[ -f "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_wikitext2.json" ]]; then
    cp -n "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_wikitext2.json" "$RESULTS_DIR/cache/"
    cp -n "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_pubmed.json" "$RESULTS_DIR/cache/"
    echo "== frozen wiki/pubmed eval caches copied from E031 =="
else
    echo "!! E031 frozen cache not found — E034 runner will compute it (first cell only)" >&2
fi

fail_count=0

run_cell() {
    # run_cell METHOD PHASES TAG SEED [CONST_SCALE]
    local method="$1" phases="$2" tag="$3" seed="$4" const_scale="${5:-}"
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
    echo "  [run ] ($method, ${phases}p) $tag budget=$BUDGET seed=$seed${const_scale:+ const=$const_scale}"
    local args=""
    if [[ -n "$const_scale" ]]; then
        args="--const-scale $const_scale"
    fi
    if ! $PY -m ph_neuro.examples.run_e034_lora \
        --model "$MODEL" --method "$method" --phases "$phases" --tag "$tag" \
        --budget-tokens "$BUDGET" --lr "$LR" --seed "$seed" --gpu-policy "$GPU_POLICY" \
        --output-dir "$RESULTS_DIR" --log-dir "$LOG_DIR" \
        --frozen-cache-dir "$RESULTS_DIR/cache" $args $EXTRA; then
        echo "  [FAIL] $tag seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

mean_m_adapt_for_seed() {
    # Print the gated single-domain cell's mean M over adapt steps for a seed.
    local seed="$1"
    local budget_tag=$((BUDGET / 1000))k
    local f="$RESULTS_DIR/${M_SHORT}_pubmed_${budget_tag}_gated_seed${seed}.json"
    if [[ ! -f "$f" ]]; then
        echo ""
        return 1
    fi
    $PY -c "
import json
r = json.load(open('$f'))
m = r['metrics'].get('mean_M_adapt')
print(m if m is not None else '')
"
}

MODE="${1:-agg}"
case "$MODE" in
    verify-ds)
        echo "== E034 verify-ds: verify second domain + compute CNN/DailyMail frozen cache =="
        if [[ -f "$RESULTS_DIR/cache/e034_second_domain_verify.json" ]]; then
            echo "  [skip] CNN/DailyMail verify already present"
            cat "$RESULTS_DIR/cache/e034_second_domain_verify.json"
        else
            $PY research/scripts/verify_e034_second_domain.py \
                --model "$MODEL" --cache-dir "$RESULTS_DIR/cache" \
                --min-free-gb 6.0 \
                || fail_count=$((fail_count + 1))
        fi
        ;;
    smoke)
        echo "== E034 smoke: 10K mechanism gate, gated, seed 42 =="
        # Local budget override — the smoke gate is 10K, NOT the 100K primary.
        local_budget="$BUDGET"
        BUDGET="${SMOKE_BUDGET:-10000}"
        run_cell "surprise" 1 "smoke_gated" 42
        BUDGET="$local_budget"
        ;;
    single)
        echo "== E034 single-domain: gated LoRA @ ${BUDGET} tokens, 3 seeds =="
        for s in $SEEDS; do
            run_cell "surprise" 1 "gated" "$s"
        done
        ;;
    two)
        echo "== E034 two-domain: plain + gated @ ${BUDGET} tokens/phase, 3 seeds =="
        for s in $SEEDS; do
            run_cell "plain" 2 "plain2" "$s"
            run_cell "surprise" 2 "gated2" "$s"
        done
        ;;
    control)
        echo "== E034 control: const_reduced lr (η·mean_M_adapt), 3 seeds =="
        for s in $SEEDS; do
            mm=$(mean_m_adapt_for_seed "$s")
            if [[ -z "$mm" ]]; then
                echo "  [SKIP] no gated single-domain result for seed $s (run \`single\` first)"
                fail_count=$((fail_count + 1))
                continue
            fi
            run_cell "const_reduced" 1 "constred" "$s" "$mm"
        done
        ;;
    agg)
        echo "== E034 cross-seed aggregation + verdict =="
        $PY -m ph_neuro.examples.aggregate_e034 \
            --results-dir "$RESULTS_DIR" --e032-dir "$ROOT/results/brain/e032"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "usage: $0 [verify-ds|smoke|single|two|control|agg]" >&2
        exit 2
        ;;
esac

echo "== E034 $MODE done; failures: $fail_count =="
exit "$fail_count"
