"""Integration tests for B3: Precision Comparison for Continual Learning.

Tests:
    1. Model builders for all 4 precisions (ternary/int8/int4/fp16)
    2. Weight-statistics helpers for all 4 precisions
    3. End-to-end runs: INT8/INT4 on Split + Permuted MNIST (smoke)
    4. JSON output contains all required keys (L8-compatible schema)
    5. Determinism: same seed → same results
    6. Forgetting metrics computed correctly
    7. Aggregator: loads B3 + L8 runs and builds the comparison table

These tests run the real MNIST training pipeline, so they are marked
``slow`` (skipped by ``pytest -m "not slow"``).
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch
import torch.nn as nn

from ph_neuro.examples.aggregate_b3_results import (
    build_comparison_table,
    load_b3_results,
    load_l8_results,
    merge_results,
)
from ph_neuro.examples.run_b3_precision_cl import (
    _compute_float_weight_stats,
    _compute_quant_weight_stats,
    _compute_weight_stats,
    build_model,
)

slow = pytest.mark.slow

# ── Test helpers ────────────────────────────────────────────────────


def _run_experiment(
    output_dir: str,
    protocol: str,
    weight_format: str,
    n_tasks: int = 2,
    epochs_per_task: int = 1,
    seed: int = 42,
) -> str:
    """Run the B3 experiment via CLI and return the output file path.

    ``num_workers=0`` avoids DataLoader ``fork()`` from a multi-threaded
    pytest process (which can corrupt the heap on some platforms).
    """
    sys.argv = [
        "run_b3_precision_cl",
        "--protocol", protocol,
        "--weight-format", weight_format,
        "--epochs-per-task", str(epochs_per_task),
        "--n-tasks", str(n_tasks),
        "--num-workers", "0",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--device", "cpu",
    ]
    # Import inside the helper so sys.argv mutation does not affect the
    # module import of other test files.
    from ph_neuro.examples.run_b3_precision_cl import main

    main()

    expected = f"{protocol}_{weight_format}_seed{seed}.json"
    output_path = os.path.join(output_dir, expected)
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    return output_path


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def int8_split_run(tmp_path_factory):
    """One shared INT8/split run reused across many tests."""
    out = str(tmp_path_factory.mktemp("b3_int8_split"))
    return _run_experiment(out, "split", "int8")


@pytest.fixture
def temp_dir(tmp_path):
    """Per-test temporary directory (one per test invocation)."""
    return str(tmp_path)


# ── Test 1: Model builders ─────────────────────────────────────────


class TestModelBuilders:
    """Verify all four precision builders produce usable MLPs."""

    @pytest.mark.parametrize("weight_format", ["ternary", "fp16", "int8", "int4"])
    def test_forward_backward(self, weight_format):
        """Each precision model runs a forward + backward pass."""
        model = build_model(weight_format, torch.device("cpu"))
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        out = model(x)
        assert tuple(out.shape) == (4, 10), f"{weight_format}: bad output shape {out.shape}"
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        # All parameters should receive gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, f"{weight_format}: no gradients computed"

    def test_parameter_counts_match(self):
        """All four precisions use the same architecture (~535K params)."""
        counts = {
            wf: sum(p.numel() for p in build_model(wf, torch.device("cpu")).parameters())
            for wf in ("ternary", "fp16", "int8", "int4")
        }
        ref = counts["fp16"]
        for wf, c in counts.items():
            # Same MLP width → parameter count within ~0.2% (ternary omits
            # per-layer bias only when BatchNorm is present).
            assert abs(c - ref) / ref < 0.002, f"{wf}: {c} vs fp16 {ref}"

    def test_qat_layers_quantize(self):
        """INT8/INT4 models use the fake-quantize path (num_bits set)."""
        for wf, bits in (("int8", 8), ("int4", 4)):
            model = build_model(wf, torch.device("cpu"))
            qlayers = [m for m in model.modules() if getattr(m, "num_bits", None) == bits]
            assert qlayers, f"{wf}: no quantized layers found"


# ── Test 2: Weight statistics ───────────────────────────────────────


class TestWeightStats:
    """Verify per-format weight-statistics helpers."""

    def test_ternary_distribution_sums_to_100(self):
        stats = _compute_weight_stats(build_model("ternary", torch.device("cpu")), "ternary")
        total = stats["weight_pos_pct"] + stats["weight_neg_pct"] + stats["weight_zero_pct"]
        assert abs(total - 100.0) < 1.0, f"ternary dist doesn't sum to 100%: {total}"
        assert stats["n_parameters"] > 0

    def test_quant_distribution_sums_to_100(self):
        for wf, bits in (("int8", 8), ("int4", 4)):
            stats = _compute_weight_stats(build_model(wf, torch.device("cpu")), wf)
            total = stats["weight_pos_pct"] + stats["weight_neg_pct"] + stats["weight_zero_pct"]
            assert abs(total - 100.0) < 1.0, f"{wf} dist doesn't sum to 100%: {total}"
            assert stats["n_parameters"] > 0
            assert 0.0 <= stats["weight_sparsity_pct"] <= 100.0

    def test_quant_helper_matches_stats(self):
        """_compute_quant_weight_stats == _compute_weight_stats for QAT."""
        model = build_model("int8", torch.device("cpu"))
        a = _compute_weight_stats(model, "int8")
        b = _compute_quant_weight_stats(model, 8)
        assert a["weight_zero_pct"] == b["weight_zero_pct"]
        assert a["n_parameters"] == b["n_parameters"]

    def test_float_stats(self):
        stats = _compute_weight_stats(build_model("fp16", torch.device("cpu")), "fp16")
        assert stats["n_parameters"] > 0
        assert stats["weight_mean_abs"] > 0.0


# ── Test 3: End-to-end runs (smoke) ────────────────────────────────


class TestEndToEnd:
    """Run the full experiment pipeline (smoke tests)."""

    @slow
    def test_int8_split_mnist(self, int8_split_run):
        """INT8 + Split MNIST — no crash, valid output."""
        data = _load_json(int8_split_run)
        assert data["experiment"] == "B3 Precision Comparison"
        assert data["protocol"] == "split"
        assert data["weight_format"] == "int8"
        assert len(data["accuracy_matrix"]) == 5  # split always defines 5 tasks

    @slow
    def test_int4_split_mnist(self, temp_dir):
        """INT4 + Split MNIST — no crash."""
        path = _run_experiment(temp_dir, "split", "int4")
        data = _load_json(path)
        assert data["weight_format"] == "int4"
        assert "average_forgetting" in data["metrics"]
        assert "average_accuracy" in data["metrics"]

    @slow
    def test_int8_permuted_mnist(self, temp_dir):
        """INT8 + Permuted MNIST (2 tasks) — no crash."""
        path = _run_experiment(temp_dir, "permuted", "int8", n_tasks=2)
        data = _load_json(path)
        assert data["protocol"] == "permuted"
        assert data["n_tasks"] == 2

    @slow
    def test_int4_permuted_mnist(self, temp_dir):
        """INT4 + Permuted MNIST (2 tasks) — no crash."""
        path = _run_experiment(temp_dir, "permuted", "int4", n_tasks=2)
        data = _load_json(path)
        assert data["weight_format"] == "int4"
        assert data["n_tasks"] == 2


# ── Test 4: JSON output schema ─────────────────────────────────────


class TestJsonSchema:
    """Verify the result JSON matches the L8-compatible schema."""

    @slow
    def test_required_keys(self, int8_split_run):
        data = _load_json(int8_split_run)
        required = [
            "experiment", "protocol", "weight_format", "seed", "device",
            "epochs_per_task", "batch_size", "learning_rate", "weight_decay",
            "n_tasks", "n_parameters", "total_training_time_seconds",
            "accuracy_matrix", "metrics", "weight_snapshots",
        ]
        for key in required:
            assert key in data, f"missing key: {key}"
        assert set(data["metrics"]) >= {
            "average_accuracy", "average_forgetting",
            "per_task_accuracy", "per_task_forgetting",
        }

    @slow
    def test_accuracy_matrix_shapes(self, int8_split_run):
        data = _load_json(int8_split_run)
        n = data["n_tasks"]
        assert len(data["accuracy_matrix"]) == n
        for i, row in enumerate(data["accuracy_matrix"]):
            assert len(row) == i + 1  # upper-triangular

    @slow
    def test_weight_snapshots_present(self, int8_split_run):
        data = _load_json(int8_split_run)
        snaps = data["weight_snapshots"]
        assert "-1" in snaps  # initial snapshot
        assert all(f"{i}" in snaps for i in range(data["n_tasks"]))


# ── Test 5: Determinism ────────────────────────────────────────────


class TestDeterminism:
    """Same seed → same results (num_workers=0)."""

    @slow
    def test_same_seed_reproduces(self, temp_dir):
        out1 = os.path.join(temp_dir, "run1")
        out2 = os.path.join(temp_dir, "run2")
        os.makedirs(out1)
        os.makedirs(out2)
        p1 = _run_experiment(out1, "split", "int8", seed=42)
        p2 = _run_experiment(out2, "split", "int8", seed=42)
        d1, d2 = _load_json(p1), _load_json(p2)
        assert d1["metrics"]["average_forgetting"] == d2["metrics"]["average_forgetting"]
        assert d1["metrics"]["average_accuracy"] == d2["metrics"]["average_accuracy"]
        assert d1["accuracy_matrix"] == d2["accuracy_matrix"]


# ── Test 6: Forgetting metric sanity ───────────────────────────────


class TestForgettingSanity:
    """Forgetting = peak−final per task, all within [0, 1]."""

    @slow
    def test_forgetting_in_range(self, int8_split_run):
        data = _load_json(int8_split_run)
        for fgt in data["metrics"]["per_task_forgetting"]:
            assert -1e-6 <= fgt <= 1.0 + 1e-6


# ── Test 7: Aggregator ─────────────────────────────────────────────


class TestAggregator:
    """Aggregator loads B3 + L8 and builds the comparison table."""

    @slow
    def test_merge_and_table(self, int8_split_run, tmp_path):
        # Write a tiny L8-style ternary run so the merge has two sources.
        l8_dir = tmp_path / "l8"
        l8_dir.mkdir()
        def _fake_l8_run(wf, acc, fgt):
            return {
                "experiment": "L8 Forgetting Baseline",
                "protocol": "split",
                "weight_format": wf,
                "seed": 42,
                "n_tasks": 5,
                "metrics": {
                    "average_accuracy": acc,
                    "average_forgetting": fgt,
                    "per_task_accuracy": [0.7] * 5,
                    "per_task_forgetting": [0.3] * 5,
                },
            }

        with open(l8_dir / "split_ternary_seed42.json", "w") as f:
            json.dump(_fake_l8_run("ternary", 0.6, 0.4), f)
        with open(l8_dir / "split_fp16_seed42.json", "w") as f:
            json.dump(_fake_l8_run("fp16", 0.62, 0.37), f)

        b3 = load_b3_results(os.path.dirname(int8_split_run))
        l8 = load_l8_results(str(l8_dir))
        merged = merge_results(b3, l8)

        assert ("split", "ternary", 42) in merged
        assert ("split", "int8", 42) in merged

        table = build_comparison_table(merged)
        assert "Ternary (STE)" in table
        assert "INT8 (QAT)" in table
        assert "Δ vs FP16" in table
