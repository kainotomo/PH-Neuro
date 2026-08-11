#!/usr/bin/env bash
# ── Unified training status checker ─────────────────────────────────
#
# Show the status of all (or one) DQT training milestone(s). Where the
# underlying research/scripts/run_*.sh has a `status` mode (m2_2, m2_3)
# we delegate to it; otherwise we show a heuristic summary from the
# launcher's pid/log files and the result JSONs.
#
# Usage:
#   bash scripts/status.sh          # show ALL trainings
#   bash scripts/status.sh m2_2     # show M2.2 only
# ────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [ ! -d "$PROJECT_ROOT/research/scripts" ] && [ "$PROJECT_ROOT" != "/" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -d "$PROJECT_ROOT/research/scripts" ]; then
    echo "ERROR: could not locate project root (research/scripts not found above $SCRIPT_DIR)" >&2
    exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Milestone → canonical runner (research/scripts/run_*.sh).
declare -A RUNNER=(
    [m1_1]="research/scripts/run_m1_1_dqt_cifar10.sh"
    [m1_2]="research/scripts/run_m1_2_dqt_cifar100.sh"
    [m2_1]="research/scripts/run_m2_1_dqt_transformer.sh"
    [m2_2]="research/scripts/run_m2_2_dqt_wikitext2.sh"
    [m2_3]="research/scripts/run_m2_3_dqt_moe.sh"
)

# Milestone → human label + result dir (for the heuristic summary).
declare -A LABEL=(
    [m1_1]="M1.1 DQT CNN CIFAR-10 (gate >80%)"
    [m1_2]="M1.2 DQT CNN CIFAR-100 (gate >55%)"
    [m2_1]="M2.1 DQT Transformer TinyStories (gate ppl<30)"
    [m2_2]="M2.2 DQT Transformer WikiText-2 (gate ppl<20)"
    [m2_3]="M2.3 MoE DQT Transformer TinyStories (gate ppl<20)"
)
declare -A RESULTS_DIR=(
    [m1_1]="m1_1_retry_results"
    [m1_2]="m1_2_retry_results"
    [m2_1]="m2_1_results"
    [m2_2]="m2_2_results"
    [m2_3]="m2_3_results"
)

# Milestones whose runner has a rich `status` mode (per-seed detail).
HAS_STATUS_MODE="m2_2 m2_3"

status_one() {
    local id="$1"
    local log_file="$PROJECT_ROOT/logs/train_${id}.log"
    local pid_file="$PROJECT_ROOT/logs/train_${id}.pid"
    local running=""
    local pid=""

    if [ -f "$pid_file" ]; then
        pid="$(cat "$pid_file")"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            running=1
        fi
    fi

    # Milestones with a `status` mode: delegate for the per-seed detail.
    if echo "$HAS_STATUS_MODE" | grep -qw "$id"; then
        echo ""
        log "── ${LABEL[$id]} ──"
        if [ -n "$running" ]; then
            log "  🟢 RUNNING (pid $pid) — live tail: tail -f $log_file"
        else
            log "  ⚪ not running"
        fi
        (cd "$PROJECT_ROOT" && bash "${RUNNER[$id]}" status) || log "  (status mode unavailable)"
        return
    fi

    # Heuristic summary for milestones without a `status` mode.
    echo ""
    log "── ${LABEL[$id]} ──"
    if [ -n "$running" ]; then
        log "  🟢 RUNNING (pid $pid)"
        if [ -f "$log_file" ]; then
            log "  last log: $(tail -n 3 "$log_file" | tr '\n' ' ')"
        fi
    else
        local n_results
        n_results=$(find "$PROJECT_ROOT/${RESULTS_DIR[$id]}" -name 'results_*.json' 2>/dev/null | wc -l)
        if [ "$n_results" -gt 0 ]; then
            log "  ✅ COMPLETED ($n_results result JSON(s) in ${RESULTS_DIR[$id]}/)"
        elif [ -f "$log_file" ]; then
            log "  ⏸️  PAUSED/stopped (log exists; re-launch with: bash scripts/train.sh $id resume)"
        else
            log "  ⚪ NOT STARTED"
        fi
    fi
}

if [ "$#" -gt 0 ]; then
    id="$1"
    if [ -z "${LABEL[$id]:-}" ]; then
        echo "ERROR: unknown milestone '$id' (expected one of: m1_1, m1_2, m2_1, m2_2, m2_3)" >&2
        exit 1
    fi
    status_one "$id"
    echo ""
else
    echo "═══════════════════════════════════════════════════════════════"
    log "Training status — $(date '+%Y-%m-%d %H:%M:%S')"
    for id in m1_1 m1_2 m2_1 m2_2 m2_3; do
        status_one "$id"
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
fi
