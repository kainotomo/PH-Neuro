#!/usr/bin/env bash
# ── M2.1 — DQT Transformer on TinyStories (GO/NO-GO ppl<30) ─────────
#
# Thin launcher for the canonical script at research/scripts/. Keeps the
# documented entry point `bash scripts/run_m2_1_dqt_transformer.sh` working
# while the implementation lives with the other experiment scripts.
#
# Milestone M2.1: first DQT on a Transformer LM. GPT-2-style decoder-only
# model with int8 ternary DQT projections (~102M ternary weights), trained
# on TinyStories. GO if mean validation perplexity (3 seeds) < 30.
#
# Usage:
#   bash scripts/run_m2_1_dqt_transformer.sh                    # full run, 3 seeds
#   bash scripts/run_m2_1_dqt_transformer.sh smoke              # tiny synthetic sanity
#   bash scripts/run_m2_1_dqt_transformer.sh sweep              # LR sweep (1 seed)
#   bash scripts/run_m2_1_dqt_transformer.sh full 0.005         # full run, chosen best LR
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../research/scripts/run_m2_1_dqt_transformer.sh" "$@"
