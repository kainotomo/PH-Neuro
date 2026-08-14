#!/usr/bin/env bash
# E032 — Capacity & Gain experiment orchestrator (Phase 1.2).
#
# Drives the staged protocol with skip-if-exists (a finished cell is never
# re-run) and per-cell logging to logs/brain/e032/. Frozen eval baselines are
# reused from the E031 cache (same seed-independent measurements) via a copy
# into results/brain/e032/cache.
#
# Usage (stages run in order; each consumes the previous stage's winner):
#   bash scripts/run_e032_capacity_gain.sh rank                 # A: rank sweep
#   bash scripts/run_e032_capacity_gain.sh gain_eta RANK        # B stage 1 (η)
#   bash scripts/run_e032_capacity_gain.sh gain_sk RANK ETA     # B stage 2 (s₀,k)
#   bash scripts/run_e032_capacity_gain.sh gain_mmax RANK ETA S0 K   # B stage 3
#   bash scripts/run_e032_capacity_gain.sh decay RANK ETA S0 K MMAX   # C
#   bash scripts/run_e032_capacity_gain.sh lora RANK            # D: LoRA
#   bash scripts/run_e032_capacity_gain.sh anneal TAG           # E: 1M
#
# Env overrides: RESULTS_DIR, LOG_DIR, GPU_POLICY, SEEDS, EXTRA.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/brain/e032}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/brain/e032}"
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

# Reuse the seed-independent frozen baselines measured in E031 (identical
# model, identical eval corpora → identical ppl; only computed once ever).
if [[ -f "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_wikitext2.json" ]]; then
    cp -n "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_wikitext2.json" "$RESULTS_DIR/cache/"
    cp -n "$ROOT/results/brain/e031/cache/frozen_${M_SHORT}_pubmed.json" "$RESULTS_DIR/cache/"
    echo "== frozen eval caches copied from E031 =="
else
    echo "!! E031 frozen cache not found — E032 runner will compute it (first cell only)" >&2
fi

fail_count=0

run_cell_local() {
    # run_cell_local TAG PLASTICITY RANK BUDGET LR DECAY S0 K MMAX SEED
    local tag="$1" plasticity="$2" rank="$3" budget="$4" lr="$5" decay="$6"
    local s0="$7" k="$8" mmax="$9" seed="${10}"
    local out="$RESULTS_DIR/${M_SHORT}_pubmed_$((budget / 1000))k_${tag}_seed${seed}.json"
    if [[ -f "$out" ]]; then
        echo "  [skip] $out"
        return 0
    fi
    echo "  [run ] $tag rank=$rank budget=$budget lr=$lr decay=$decay s0=$s0 k=$k Mmax=$mmax seed=$seed"
    if ! $PY -m ph_neuro.examples.run_e032_capacity_gain \
        --model "$MODEL" --tag "$tag" --plasticity "$plasticity" --rank "$rank" \
        --budget-tokens "$budget" --lr "$lr" --decay "$decay" --s0 "$s0" --k "$k" \
        --m-max "$mmax" --seed "$seed" --gpu-policy "$GPU_POLICY" \
        --output-dir "$RESULTS_DIR" --log-dir "$LOG_DIR" \
        --frozen-cache-dir "$RESULTS_DIR/cache" $EXTRA; then
        echo "  [FAIL] $tag seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

run_cell_lora() {
    # run_cell_lora TAG RANK BUDGET LR SEED
    local tag="$1" rank="$2" budget="$3" lr="$4" seed="$5"
    local out="$RESULTS_DIR/${M_SHORT}_pubmed_$((budget / 1000))k_${tag}_seed${seed}.json"
    if [[ -f "$out" ]]; then
        echo "  [skip] $out"
        return 0
    fi
    echo "  [run ] (lora) $tag rank=$rank budget=$budget lr=$lr seed=$seed"
    if ! $PY -m ph_neuro.examples.run_e032_lora \
        --model "$MODEL" --tag "$tag" --rank "$rank" \
        --budget-tokens "$budget" --lr "$lr" --seed "$seed" --gpu-policy "$GPU_POLICY" \
        --output-dir "$RESULTS_DIR" --log-dir "$LOG_DIR" \
        --frozen-cache-dir "$RESULTS_DIR/cache" $EXTRA; then
        echo "  [FAIL] $tag seed=$seed"
        fail_count=$((fail_count + 1))
    fi
}

MODE="${1:-rank}"
case "$MODE" in
    rank)
        echo "== A: rank sweep r∈{1,2,4} @ E031 defaults =="
        for r in 1 2 4; do
            for s in $SEEDS; do
                run_cell_local "lrr$r" low_rank "$r" "$BUDGET" 1e-3 0.0 0.05 60 1.0 "$s"
            done
        done
        ;;
    gain_eta)
        RANK="${2:-4}"
        echo "== B1: η sweep at rank=$RANK (s0=0.05 k=60 Mmax=1.0) =="
        # η=1e-3 is the lrr$RANK cell (already run in stage A)
        for eta in 3e-3 1e-2; do
            case "$eta" in
                3e-3) tag="eta3e3" ;;
                1e-2) tag="eta1e2" ;;
            esac
            for s in $SEEDS; do
                run_cell_local "$tag" low_rank "$RANK" "$BUDGET" "$eta" 0.0 0.05 60 1.0 "$s"
            done
        done
        ;;
    gain_sk)
        RANK="${2:-4}"; ETA="${3:-3e-3}"
        echo "== B2: s₀/k sweep at rank=$RANK η=$ETA (Mmax=1.0) =="
        # (s0=0.05,k=60) is the stage-B1 winner cell (already run)
        for cfg in "0.02 30 s002k30" "0.02 60 s002k60" "0.05 30 s005k30"; do
            read -r s0 k tag <<< "$cfg"
            for s in $SEEDS; do
                run_cell_local "$tag" low_rank "$RANK" "$BUDGET" "$ETA" 0.0 "$s0" "$k" 1.0 "$s"
            done
        done
        ;;
    gain_mmax)
        RANK="${2:-4}"; ETA="${3:-3e-3}"; S0="${4:-0.02}"; K="${5:-60}"
        echo "== B3: M_max sweep at rank=$RANK η=$ETA s0=$S0 k=$K =="
        # Mmax=1.0 is the stage-B2 winner cell (already run)
        for s in $SEEDS; do
            run_cell_local "mmax2" low_rank "$RANK" "$BUDGET" "$ETA" 0.0 "$S0" "$K" 2.0 "$s"
        done
        ;;
    decay)
        RANK="${2:-4}"; ETA="${3:-3e-3}"; S0="${4:-0.02}"; K="${5:-60}"; MMAX="${6:-2.0}"
        echo "== C: decay ablation at rank=$RANK η=$ETA s0=$S0 k=$K Mmax=$MMAX =="
        # λ=0.0 is the best-config cell (already run)
        for dec in 1e-5 1e-4; do
            case "$dec" in
                1e-5) tag="decay1e5" ;;
                1e-4) tag="decay1e4" ;;
            esac
            for s in $SEEDS; do
                run_cell_local "$tag" low_rank "$RANK" "$BUDGET" "$ETA" "$dec" "$S0" "$K" "$MMAX" "$s"
            done
        done
        ;;
    lora)
        RANK="${2:-4}"
        echo "== D: LoRA backprop at rank=$RANK (matched budget) =="
        for lr in 1e-4 3e-4 1e-3; do
            case "$lr" in
                1e-4) tag="lora_lr1e4" ;;
                3e-4) tag="lora_lr3e4" ;;
                1e-3) tag="lora_lr1e3" ;;
            esac
            for s in $SEEDS; do
                run_cell_lora "$tag" "$RANK" "$BUDGET" "$lr" "$s"
            done
        done
        ;;
    anneal)
        TAG="${2:-}"
        if [[ -z "$TAG" ]]; then
            echo "usage: $0 anneal TAG (the best local config tag)" >&2
            exit 2
        fi
        echo "== E: 1M anneal at best local config '$TAG' =="
        for s in $SEEDS; do
            local_out="$RESULTS_DIR/${M_SHORT}_pubmed_100k_${TAG}_seed${s}.json"
            if [[ ! -f "$local_out" ]]; then
                echo "  [SKIP] $TAG has no 100K cell — cannot derive 1M config" >&2
                continue
            fi
            cfg=$(python3 - "$local_out" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
mod = r["modulator"]
print(f"{r['rank']} {r['lr']} {r['decay_rate']} {mod['s0']} {mod['k']} {mod['M_max']}")
PYEOF
)
            read -r rank lr decay s0 k mmax <<< "$cfg"
            run_cell_local "$TAG" low_rank "$rank" 1000000 "$lr" "$decay" "$s0" "$k" "$mmax" "$s"
        done
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "usage: $0 [rank|gain_eta|gain_sk|gain_mmax|decay|lora|anneal]" >&2
        exit 2
        ;;
esac

echo "== E032 $MODE done; failures: $fail_count =="
exit "$fail_count"
