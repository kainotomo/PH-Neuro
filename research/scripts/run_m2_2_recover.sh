#!/usr/bin/env bash
# ── M2.2 idempotent recovery / supervisor (WSL-reboot safe) ─────────
#
# The host runs inside WSL, which has restarted twice and killed the run
# (seed 42 @ step 1150, then seed 43 @ step 3400). This script is an
# IDEMPOTENT supervisor: for each seed it
#   - SKIPs      if the result JSON exists (already completed),
#   - RESUMEs    from the latest checkpoint if one exists (interrupted),
#   - RUNS fresh otherwise.
# Re-running it after ANY interruption picks up exactly where things left
# off. Safe to call repeatedly.
#
# Usage:
#   bash scripts/run_m2_2_dqt_wikitext2.sh recover [lr]   # (via launcher)
#   bash research/scripts/run_m2_2_recover.sh [lr]        # (direct)
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
while [ ! -x "$PROJECT_ROOT/.venv/bin/python" ] && [ "$PROJECT_ROOT" != "/" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
cd "$PROJECT_ROOT"

LR="${1:-0.01}"
SEEDS=(42 43 44)
RESULTS_DIR="results/phase2/m2_2_results"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "M2.2 recovery (lr=${LR}) — skip completed / resume interrupted / run fresh"

for seed in "${SEEDS[@]}"; do
    out_json="$RESULTS_DIR/results_m2_2_dqt_wikitext2_lr${LR}_seed${seed}.json"
    ckpt_dir="$RESULTS_DIR/checkpoints/seed${seed}"

    if [ -f "$out_json" ]; then
        log "Seed $seed: COMPLETED — skipping"
        continue
    fi

    if ls "$ckpt_dir"/ckpt_step*.pt >/dev/null 2>&1; then
        log "Seed $seed: RESUME from latest checkpoint"
        # `if !` so a failed run does NOT abort the supervisor loop.
        if ! bash "$PROJECT_ROOT/scripts/run_m2_2_dqt_wikitext2.sh" resume "$LR" "$seed"; then
            rc=$?
            # A deliberate SIGUSR1 pause writes status=PAUSED and exits 130 —
            # HALT the supervisor so it does not start the next seed (the
            # M2.1 supervisor rule). A genuine crash has no PAUSED status.
            if grep -q '"status": "PAUSED"' "$ckpt_dir/status.json" 2>/dev/null; then
                log "⏸️  Seed $seed paused by request (exit $rc) — halting supervisor"
                exit 0
            fi
            log "⚠️  seed $seed resume failed (exit $rc) — continuing"
        fi
    else
        log "Seed $seed: FRESH run"
        if ! bash "$PROJECT_ROOT/scripts/run_m2_2_dqt_wikitext2.sh" full "$LR" "$seed"; then
            rc=$?
            if grep -q '"status": "PAUSED"' "$ckpt_dir/status.json" 2>/dev/null; then
                log "⏸️  Seed $seed paused by request (exit $rc) — halting supervisor"
                exit 0
            fi
            log "⚠️  seed $seed run failed (exit $rc) — continuing"
        fi
    fi
done

log "M2.2 recovery pass complete. Re-run this script after any WSL reboot to continue."
