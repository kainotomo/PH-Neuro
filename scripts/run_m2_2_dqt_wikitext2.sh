#!/usr/bin/env bash
# ── M2.2 — DQT Transformer 250M on WikiText-2 (GO/NO-GO ppl<20) ─────
#
# Thin launcher for the canonical script at research/scripts/. Keeps the
# documented entry point `bash scripts/run_m2_2_dqt_wikitext2.sh` working
# while the implementation lives with the other experiment scripts.
#
# Milestone M2.2: scaling test 102M → 250M ternary params. GPT-2-style
# decoder-only DQT transformer (~252.8M ternary weights) trained on
# WikiText-2. GO if mean validation perplexity (3 seeds) < 20.
#
# Usage:
#   bash scripts/run_m2_2_dqt_wikitext2.sh                    # full run, 3 seeds
#   bash scripts/run_m2_2_dqt_wikitext2.sh smoke              # 10-step GPU smoke
#   bash scripts/run_m2_2_dqt_wikitext2.sh resume 0.01 42     # pause/resume a seed
#   bash scripts/run_m2_2_dqt_wikitext2.sh status             # what is running?
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../research/scripts/run_m2_2_dqt_wikitext2.sh" "$@"
