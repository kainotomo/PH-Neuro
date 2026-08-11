#!/usr/bin/env bash
# ── Unified training launcher ───────────────────────────────────────
#
# Single entry point for all DQT training milestones. Delegates to the
# canonical research/scripts/run_*.sh scripts, launches in the
# background (the chat session stays free), and logs to logs/train_{id}.log.
#
# Usage:
#   bash scripts/train.sh m2_2           # launch M2.2 in background (waits for GPU)
#   bash scripts/train.sh m2_2 --no-wait # launch immediately (GPU is free now)
#   bash scripts/train.sh m2_2 smoke     # quick GPU smoke (no GPU wait)
#   bash scripts/train.sh m2_2 resume    # resume M2.2 from latest checkpoint
#   bash scripts/train.sh m2_2 full 0.01 42 43 44 --no-wait
#
# Supported milestones: m1_1, m1_2, m2_1, m2_2, m2_3
# Check progress with:   bash scripts/status.sh [id]
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

TRAIN_ID="${1:-}"
if [ -z "$TRAIN_ID" ]; then
    echo "ERROR: no milestone given (expected one of: m1_1, m1_2, m2_1, m2_2, m2_3)" >&2
    exit 1
fi
shift

case "$TRAIN_ID" in
    m1_1) RUNNER="research/scripts/run_m1_1_dqt_cifar10.sh" ;;
    m1_2) RUNNER="research/scripts/run_m1_2_dqt_cifar100.sh" ;;
    m2_1) RUNNER="research/scripts/run_m2_1_dqt_transformer.sh" ;;
    m2_2) RUNNER="research/scripts/run_m2_2_dqt_wikitext2.sh" ;;
    m2_3) RUNNER="research/scripts/run_m2_3_dqt_moe.sh" ;;
    *)
        echo "ERROR: unknown milestone '$TRAIN_ID' (expected m1_1, m1_2, m2_1, m2_2, m2_3)" >&2
        exit 1
        ;;
esac

# Translate the friendly --no-wait/--wait flags to the underlying script's
# --no-wait-gpu/--wait-gpu (shared GPU gate added in Phase 2.5).
PASS_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --no-wait) PASS_ARGS+=("--no-wait-gpu") ;;
        --wait)    PASS_ARGS+=("--wait-gpu") ;;
        *)         PASS_ARGS+=("$_arg") ;;
    esac
done

mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/train_${TRAIN_ID}.log"
PID_FILE="$PROJECT_ROOT/logs/train_${TRAIN_ID}.pid"

# Refuse to double-launch a milestone that is already running.
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "❌ $TRAIN_ID already running (pid $OLD_PID, log: $LOG_FILE)."
        echo "   Check: bash scripts/status.sh $TRAIN_ID"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

echo "▶ Launching $TRAIN_ID in the background (log: $LOG_FILE)"
if [ ${#PASS_ARGS[@]} -gt 0 ]; then
    echo "  args: ${PASS_ARGS[*]}"
fi

# Fully detach (new session) so the process survives this shell exiting.
setsid nohup bash "$PROJECT_ROOT/$RUNNER" "${PASS_ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "  pid: $PID"
echo "  status: bash scripts/status.sh $TRAIN_ID"
echo "  live tail: tail -f $LOG_FILE"
