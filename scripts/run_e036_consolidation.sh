#!/usr/bin/env bash
# E036 — Consolidation Mechanism orchestrator (Step 2.3).
#
# Sleep-like consolidation on the T-C ternary STE gated LoRA stack. The
# 3-domain sequence is WikiText-2 (warmup) → PubMed (D1) → CNN/DailyMail (D2)
# → C4 (D3, Common Crawl web, odc-by). Three conditions, same protocol
# (SmolLM2-1.7B, 100K/domain, 3 seeds 42/43/44):
#   B1 — independent per-domain adapters, no consolidation (3 sub-cells/seed)
#   B2 — one continuing adapter (interference floor)
#   C  — long-term store + per-domain short-term (top-10% transfer, add rule)
# Skip-if-exists (a finished cell is never re-run), per-cell logging to
# logs/brain/e036/, frozen caches copied from E034/E035 + C4 computed.
#
# Usage:
#   bash scripts/run_e036_consolidation.sh smoke   # 10K mechanism gate
#   bash scripts/run_e036_consolidation.sh b1      # independent per-domain, 3 seeds
#   bash scripts/run_e036_consolidation.sh b2      # continuing adapter, 3 seeds
#   bash scripts/run_e036_consolidation.sh c       # consolidation, 3 seeds
#   bash scripts/run_e036_consolidation.sh all     # b1 + b2 + c
#   bash scripts/run_e036_consolidation.sh agg     # cross-seed verdict
#
# Env overrides: RESULTS_DIR, LOG_DIR, GPU_POLICY, SEEDS, BUDGET, TAG, EXTRA.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/brain/e036}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/brain/e036}"
GPU_POLICY="${GPU_POLICY:-exit}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"
M_SHORT="smolllm2_1p7b"
TAG="${TAG:-e036}"
EXTRA="${EXTRA:-}"
SEEDS="${SEEDS:-42 43 44}"
BUDGET="${BUDGET:-100000}"
CONSOLIDATE_K="${CONSOLIDATE_K:-0.10}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$RESULTS_DIR/cache"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Reuse the seed-independent frozen baselines from E034/E035 (bit-identical
# model + eval corpora); C4's frozen ppl (13.568) was computed once already.
for d in wikitext2 pubmed cnn_dailymail; do
    if [[ -f "$ROOT/results/brain/e034/cache/frozen_${M_SHORT}_${d}.json" ]]; then
        cp -n "$ROOT/results/brain/e034/cache/frozen_${M_SHORT}_${d}.json" "$RESULTS_DIR/cache/"
    fi
done
if [[ -f "$ROOT/results/brain/e035/cache/frozen_${M_SHORT}_wikitext2.json" ]]; then
    cp -n "$ROOT/results/brain/e035/cache/frozen_${M_SHORT}_wikitext2.json" "$RESULTS_DIR/cache/"
fi
echo "== frozen wiki/pubmed/cnn/c4 eval caches ready in $RESULTS_DIR/cache =="

fail_count=0

run_cell() {
    # run_cell CONDITION B1_DOMAIN SEED [TAG_SUFFIX]
    local condition="$1" b1_domain="$2" seed="$3" suffix="${4:-}"
    local budget_tag=$((BUDGET / 1000))k
    local out
    if [[ "$condition" == "b1" ]]; then
        out="$RESULTS_DIR/${M_SHORT}_b1_${b1_domain}_${budget_tag}_${TAG}${suffix}_seed${seed}.json"
    else
        out="$RESULTS_DIR/${M_SHORT}_${condition}_${budget_tag}_${TAG}${suffix}_seed${seed}.json"
    fi
    if [[ -f "$out" ]]; then
        echo "  [skip] $out"
        return 0
    fi
    echo "  [run ] condition=$condition b1_domain=${b1_domain:-—} seed=$seed budget=$BUDGET"
    local b1arg=""
    if [[ -n "$b1_domain" ]]; then
        b1arg="--b1-domain $b1_domain"
    fi
    local cache_dir="$RESULTS_DIR/cache"
    local extra="$EXTRA"
    if [[ -n "${SMOKE_CACHE:-}" ]]; then
        cache_dir="$SMOKE_CACHE"
        extra="$extra --eval-max-tokens ${SMOKE_EVAL_TOKENS:-50000}"
    fi
    if ! $PY -m ph_neuro.examples.run_e036_consolidation \
        --model "$MODEL" --condition "$condition" $b1arg --tag "$TAG" \
        --budget-tokens "$BUDGET" --lr 1e-3 --consolidate-k "$CONSOLIDATE_K" \
        --seed "$seed" --gpu-policy "$GPU_POLICY" \
        --output-dir "$RESULTS_DIR" --log-dir "$LOG_DIR" \
        --frozen-cache-dir "$cache_dir" $extra; then
        echo "  [FAIL] condition=$condition b1_domain=${b1_domain:-—} seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

MODE="${1:-agg}"
case "$MODE" in
    smoke)
        echo "== E036 smoke: 10K mechanism gate, seed 42, 50K evals (separate cache) =="
        local_budget="$BUDGET"
        local_tag="$TAG"
        BUDGET="${SMOKE_BUDGET:-10000}"
        TAG="${TAG}_smoke"
        export SMOKE_CACHE="$RESULTS_DIR/cache_smoke"
        export SMOKE_EVAL_TOKENS="${SMOKE_EVAL_TOKENS:-50000}"
        mkdir -p "$SMOKE_CACHE"
        run_cell "b1" "pubmed" 42 ""
        run_cell "b1" "c4" 42 ""
        run_cell "b2" "" 42 ""
        run_cell "c" "" 42 ""
        BUDGET="$local_budget"
        TAG="$local_tag"
        ;;
    b1)
        echo "== E036 B1: independent per-domain adapters, 3 seeds =="
        for s in $SEEDS; do
            run_cell "b1" "pubmed" "$s"
            run_cell "b1" "cnn" "$s"
            run_cell "b1" "c4" "$s"
        done
        ;;
    b2)
        echo "== E036 B2: continuing adapter (interference floor), 3 seeds =="
        for s in $SEEDS; do
            run_cell "b2" "" "$s"
        done
        ;;
    c)
        echo "== E036 C: consolidation (LT + ST), 3 seeds =="
        for s in $SEEDS; do
            run_cell "c" "" "$s"
        done
        ;;
    all)
        echo "== E036 all conditions, 3 seeds =="
        bash "$0" b1
        bash "$0" b2
        bash "$0" c
        ;;
    agg)
        echo "== E036 cross-seed aggregation + verdict =="
        $PY -m ph_neuro.examples.aggregate_e036 --results-dir "$RESULTS_DIR" --tag "$TAG"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "usage: $0 [smoke|b1|b2|c|all|agg]" >&2
        exit 2
        ;;
esac

echo "== E036 $MODE done; failures: $fail_count =="
exit "$fail_count"
