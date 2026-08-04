#!/usr/bin/env bash
# ── M1.2-RETRY — DQT CNN on CIFAR-100 (GO/NO-GO >55%) ───────────────
#
# Thin launcher for the canonical script at research/scripts/. Keeps the
# documented entry point `bash scripts/run_m1_2_dqt_cifar100.sh` working
# while the implementation lives with the other experiment scripts.
#
# Milestone M1.2-RETRY: larger 3-conv DQT CNN (Conv 3→64→128→256 → FC
# 4096→512→100, ~2.5M ternary weights) on CIFAR-100, lr=0.01, 200 ep,
# anneal@80%, patience=40. STE baseline (E009/L1): 38.2%. Target: >55%.
# (M1.2 was MARGINAL 54.15% — the 200-ep extension targets the 55% gate.)
#
# Usage:
#   bash scripts/run_m1_2_dqt_cifar100.sh                    # full run, 3 seeds, lr=0.01, 200 ep
#   bash scripts/run_m1_2_dqt_cifar100.sh sweep              # LR sweep (1 seed)
#   bash scripts/run_m1_2_dqt_cifar100.sh full 0.005         # full run, chosen best LR
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../research/scripts/run_m1_2_dqt_cifar100.sh" "$@"
