#!/usr/bin/env bash
# E033 — Predictive Coding experiment orchestrator (Phase 1.3, re-scoped).
#
# Drives the LAST local-rule experiment with skip-if-exists (a finished cell
# is never re-run) and per-cell logging to logs/brain/e033/. Frozen eval
# baselines are reused from the E031/E032 cache (same seed-independent
# measurements) via a copy into results/brain/e033/cache.
#
# One shot per formulation (pre-registered): the primary cell is the
# per-injection-site reconstruction-error PC (PC-ERR) at matched LoRA budget
# (rank 1, 344,064 params), 100K primary point, 3 seeds. No second PC variant,
# no hyperparameter rescue, no capacity escalation. Additional formulations
# are permitted ONLY if the primary shows Δppl > 0 with p < 0.05.
#
# Usage:
#   bash scripts/run_e033_predictive_coding.sh smoke    # 1-cell mechanism gate
#   bash scripts/run_e033_predictive_coding.sh primary  # 3 seeds @ 100K (the run)
#   bash scripts/run_e033_predictive_coding.sh agg      # cross-seed verdict
#
# Env overrides: RESULTS_DIR, LOG_DIR, GPU_POLICY, SEEDS, EXTRA, BUDGET.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/brain/e033}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/brain/e033}"
GPU_POLICY="${GPU_POLICY:-exit}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-1.7B}"
M_SHORT="smolllm2_1p7b"
EXTRA="${EXTRA:-}"
SEEDS="${SEEDS:-42 43 44}"
BUDGET="${BUDGET:-100000}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$RESULTS_DIR/cache"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

# Reuse the seed-independent frozen baselines measured in E031/E032 (identical
# model, identical eval corpora → identical ppl; only computed once ever).
if [[ -f "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_wikitext2.json" ]]; then
    cp -n "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_wikitext2.json" "$RESULTS_DIR/cache/"
    cp -n "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_pubmed.json" "$RESULTS_DIR/cache/"
    echo "== frozen eval caches copied from E031 =="
else
    echo "!! E031 frozen cache not found — E033 runner will compute it (first cell only)" >&2
fi

fail_count=0

run_cell_pc() {
    # run_cell_pc TAG RANK BUDGET SEED
    local tag="$1" rank="$2" budget="$3" seed="$4"
    local out="$RESULTS_DIR/${M_SHORT}_pubmed_$((budget / 1000))k_${tag}_seed${seed}.json"
    if [[ -f "$out" ]]; then
        echo "  [skip] $out"
        return 0
    fi
    echo "  [run ] (pc) $tag rank=$rank budget=$budget seed=$seed"
    if ! $PY -m ph_neuro.examples.run_e033_predictive_coding \
        --model "$MODEL" --tag "$tag" --plasticity predictive_coding --rank "$rank" \
        --budget-tokens "$budget" --seed "$seed" --gpu-policy "$GPU_POLICY" \
        --output-dir "$RESULTS_DIR" --log-dir "$LOG_DIR" \
        --frozen-cache-dir "$RESULTS_DIR/cache" $EXTRA; then
        echo "  [FAIL] $tag seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

MODE="${1:-primary}"
case "$MODE" in
    smoke)
        echo "== E033 smoke: 10K mechanism gate, seed 42 =="
        run_cell_pc "pc" 1 10000 42
        ;;
    primary)
        echo "== E033 primary: PC-ERR rank-1 (matched LoRA budget) @ ${BUDGET} tokens, 3 seeds =="
        for s in $SEEDS; do
            run_cell_pc "pc" 1 "$BUDGET" "$s"
        done
        ;;
    agg)
        echo "== E033 cross-seed aggregation + verdict =="
        $PY -m ph_neuro.examples.aggregate_e033 \
            --results-dir "$RESULTS_DIR" \
            --e032-summary "$ROOT/results/brain/e032/summary_e032.json"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "usage: $0 [smoke|primary|agg]" >&2
        exit 2
        ;;
esac

echo "== E033 $MODE done; failures: $fail_count =="
exit "$fail_count"
