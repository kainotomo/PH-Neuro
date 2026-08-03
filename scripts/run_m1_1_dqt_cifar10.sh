#!/usr/bin/env bash
# ── M1.1 — DQT CNN on CIFAR-10 (GO/NO-GO >80%) ─────────────────────
#
# Thin launcher for the canonical script at research/scripts/. Keeps the
# documented entry point `bash scripts/run_m1_1_dqt_cifar10.sh` working
# while the implementation lives with the other experiment scripts.
#
# Usage:
#   bash scripts/run_m1_1_dqt_cifar10.sh                    # 3 seeds, lr=0.01
#   bash scripts/run_m1_1_dqt_cifar10.sh sweep              # lr sweep (seed 42)
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../research/scripts/run_m1_1_dqt_cifar10.sh" "$@"
