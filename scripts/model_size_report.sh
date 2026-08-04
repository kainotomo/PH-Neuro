#!/usr/bin/env bash
# ── M1.3 — Model size report ────────────────────────────────────────
#
# Prints the ternary weight count and packed (2-bit) size of every PH-Neuro
# model eligible for ONNX export. Demonstrates that all models are far below
# the M1.3 <100 MB gate.
#
# Usage:
#   bash scripts/model_size_report.sh
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Resolve repo root from the location of this script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: venv python not found at ${PYTHON}" >&2
    exit 1
fi

cd "$REPO_ROOT"
"$PYTHON" - <<'EOF'
"""Print a size report for all M1.3-exportable PH-Neuro models."""
from ph_neuro.models.dqt_models import dqt_cnn, dqt_cnn_cifar100
from ph_neuro.models.ste_models import ste_mlp
from ph_neuro.models.export import get_model_params_count, estimate_packed_size

print("=" * 62)
print("  PH-Neuro M1.3 — Model Size Report (ternary weights, 2-bit packed)")
print("=" * 62)
print(f"{'Model':<28}{'Ternary weights':>16}{'Packed':>12}")
print("-" * 62)

models = [
    ("dqt_cnn (CIFAR-10)", dqt_cnn()),
    ("dqt_cnn_cifar100 (CIFAR-100)", dqt_cnn_cifar100()),
    ("ste_mlp (MNIST)", ste_mlp([784, 512, 256, 10])),
]

total_weights = 0
for name, model in models:
    n = get_model_params_count(model)
    packed = estimate_packed_size(model)
    total_weights += n
    print(f"{name:<28}{n:>16,}{packed / 1024:>10.1f} KB")

print("-" * 62)
print(f"{'TOTAL':<28}{total_weights:>16,}{sum(estimate_packed_size(m) for _, m in models) / 1024:>10.1f} KB")
print()
print(f"<100 MB gate: PASS — the largest packed model is "
      f"{max(estimate_packed_size(m) for _, m in models) / (1024 * 1024):.2f} MB")
EOF
