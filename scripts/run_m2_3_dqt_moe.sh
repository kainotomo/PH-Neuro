#!/usr/bin/env bash
# ── M2.3 — MoE DQT Transformer on TinyStories (GO/NO-GO ppl<20) ─────
#
# Thin launcher for the canonical script at research/scripts/. Keeps the
# documented entry point `bash scripts/run_m2_3_dqt_moe.sh` working while
# the implementation lives with the other experiment scripts.
#
# Milestone M2.3: first MoE DQT Transformer — ~312M ternary params total,
# ~161M active per token (52%). Hybrid dense+MoE block stack trained on
# TinyStories. GO if the mean validation perplexity (3 seeds) is < 20.
#
# Usage:
#   bash scripts/run_m2_3_dqt_moe.sh                    # full run, 3 seeds
#   bash scripts/run_m2_3_dqt_moe.sh full 0.01 42       # start seed 42
#   bash scripts/run_m2_3_dqt_moe.sh smoke              # 12-step GPU smoke
#   bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42     # resume a paused seed
#   bash scripts/run_m2_3_dqt_moe.sh status             # what is running?
#
# MANUAL only — nothing runs or retries on its own:
#   START  : bash scripts/run_m2_3_dqt_moe.sh full 0.01 42
#   PAUSE  : kill -SIGUSR1 $(cat results/phase2/m2_3_results/checkpoints/seed42/train.pid)
#   RESUME : bash scripts/run_m2_3_dqt_moe.sh resume 0.01 42
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../research/scripts/run_m2_3_dqt_moe.sh" "$@"
