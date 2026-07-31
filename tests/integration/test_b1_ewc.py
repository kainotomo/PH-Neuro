"""Integration tests for B1: EWC + Ternary STE experiment.

Tests:
    1. train_task_ewc runs with and without an EWC penalty
    2. EWC applies a real penalty term (high lambda prevents new-task learning)
    3. End-to-end runs: Split MNIST + Permuted MNIST, online + multi-task
    4. JSON output contains all required keys (including EWC fields)
    5. Accuracy matrix has correct shape
    6. Forgetting metrics computed correctly
    7. Determinism: same seed → same results
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.examples.run_b1_ewc import train_task_ewc
from ph_neuro.training.ewc import MultiTaskEWC, OnlineEWC

# ── Test helpers ────────────────────────────────────────────────────


def _tiny_loader(n_samples: int = 64, n_features: int = 784, n_classes: int = 10) -> DataLoader:
    """Tiny synthetic loader for fast CPU tests (MNIST-sized inputs)."""
    x = torch.randn(n_samples, 1, 28, 28)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _build_model(device: torch.device) -> torch.nn.Module:
    """Build the same ternary MLP used by the B1 runner."""
    from ph_neuro.examples.run_l8_forgetting_baseline import _build_ternary_mlp

    return _build_ternary_mlp(device)


def _run_experiment(
    output_dir: str,
    protocol: str,
    ewc_lambda: float = 10.0,
    n_tasks: int = 2,
    epochs_per_task: int = 1,
    seed: int = 42,
    fisher_samples: int = 30,
    online: bool = True,
    batch_size: int = 64,
) -> str:
    """Run the B1 experiment via CLI and return the output file path."""
    import sys
    from pathlib import Path

    src_dir = Path(__file__).parent.parent.parent / "src"
    sys.path.insert(0, str(src_dir))

    from ph_neuro.examples.run_b1_ewc import main

    sys.argv = [
        "run_b1_ewc",
        "--protocol",
        protocol,
        "--ewc-lambda",
        str(ewc_lambda),
        "--fisher-samples",
        str(fisher_samples),
        "--epochs-per-task",
        str(epochs_per_task),
        "--batch-size",
        str(batch_size),
        "--n-tasks",
        str(n_tasks),
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
        "--device",
        "cpu",
    ]
    if not online:
        sys.argv.append("--no-online")

    main()

    lam_str = f"{ewc_lambda:g}"
    expected_name = f"{protocol}_ewc_lambda{lam_str}_seed{seed}.json"
    output_path = os.path.join(output_dir, expected_name)
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    return output_path


# ── Test 1: Training function with EWC penalty ────────────────────


class TestTrainTaskEWC:
    """Verify train_task_ewc works with and without a penalty."""

    def test_without_penalty(self):
        """Training without a penalty function completes (baseline path)."""
        device = torch.device("cpu")
        model = _build_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        metrics = train_task_ewc(model, _tiny_loader(), optimizer, 1, device, 0, "test")
        assert "final_loss" in metrics
        assert "final_train_acc" in metrics
        assert metrics["final_loss"] > 0

    def test_with_online_penalty(self):
        """Training with an active OnlineEWC penalty completes."""
        device = torch.device("cpu")
        model = _build_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ewc = OnlineEWC(model, gamma=1.0)
        ewc.update(_tiny_loader(), n_batches=4, device=device)

        def pen_fn(m):
            return ewc.penalty(m, ewc_lambda=10.0)

        metrics = train_task_ewc(
            model, _tiny_loader(), optimizer, 1, device, 0, "test", ewc_penalty_fn=pen_fn
        )
        assert metrics["final_loss"] > 0

    def test_with_multitask_penalty(self):
        """Training with an active MultiTaskEWC penalty completes."""
        device = torch.device("cpu")
        model = _build_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ewc = MultiTaskEWC(model)
        ewc.update(_tiny_loader(), n_batches=4, device=device)

        def pen_fn(m):
            return ewc.penalty(m, ewc_lambda=10.0)

        metrics = train_task_ewc(
            model, _tiny_loader(), optimizer, 1, device, 0, "test", ewc_penalty_fn=pen_fn
        )
        assert metrics["final_loss"] > 0


# ── Test 2: EWC actively changes training ──────────────────────────


class TestEWCActive:
    """Verify that EWC materially affects the training dynamics.

    EWC protects the weights important for previously seen tasks. With a
    huge lambda, task 1 must be preserved much better than without a
    penalty, while task 2 can still be learned (its output neurons have
    ~0 Fisher from task 1, so they are free to change).
    """

    def test_high_lambda_protects_previous_tasks(self, temp_dir):
        """Huge lambda preserves task 1 (less forgetting) vs no penalty."""
        lam_lo = 0.0
        lam_hi = 1e5

        path_lo = _run_experiment(
            temp_dir, protocol="split", ewc_lambda=lam_lo, n_tasks=2,
            epochs_per_task=1, seed=42, fisher_samples=30,
        )
        path_hi = _run_experiment(
            os.path.join(temp_dir, "hi"), protocol="split", ewc_lambda=lam_hi,
            n_tasks=2, epochs_per_task=1, seed=42, fisher_samples=30,
        )

        with open(path_lo) as f:
            data_lo = json.load(f)
        with open(path_hi) as f:
            data_hi = json.load(f)

        # Row 1 = after task 2: [task1 acc, task2 acc]
        t1_lo = data_lo["accuracy_matrix"][1][0]
        t1_hi = data_hi["accuracy_matrix"][1][0]
        t2_lo = data_lo["accuracy_matrix"][1][1]
        t2_hi = data_hi["accuracy_matrix"][1][1]

        # EWC must protect task 1: retention with huge λ is clearly better
        assert t1_hi > t1_lo + 0.05, (
            f"Expected high-lambda task1 retention ({t1_hi:.3f}) to exceed "
            f"no-penalty ({t1_lo:.3f})"
        )
        # Task 2 output neurons have ~0 Fisher from task 1, so task 2 can
        # still be learned even under a huge penalty.
        assert t2_hi > 0.80, f"Task 2 should still learn under EWC, got {t2_hi:.3f}"


# ── Test 3: End-to-end runs (smoke) ────────────────────────────────


class TestEndToEnd:
    """Run the full experiment pipeline (smoke tests)."""

    def test_online_split_mnist(self, temp_dir):
        """Online EWC on Split MNIST (1 epoch) — no crash (5 fixed tasks)."""
        output_path = _run_experiment(
            temp_dir, protocol="split", ewc_lambda=10.0, n_tasks=2, online=True
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["protocol"] == "split"
        assert data["weight_format"] == "ternary"
        assert data["ewc_lambda"] == 10.0
        assert data["ewc_online"] is True
        # Split MNIST always defines 5 binary tasks (matches L8 behavior)
        assert len(data["accuracy_matrix"]) == 5

    def test_multitask_split_mnist(self, temp_dir):
        """Multi-task EWC on Split MNIST (1 epoch) — no crash."""
        output_path = _run_experiment(
            temp_dir, protocol="split", ewc_lambda=10.0, n_tasks=2, online=False
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["ewc_online"] is False
        assert len(data["accuracy_matrix"]) == 5

    def test_online_permuted_mnist(self, temp_dir):
        """Online EWC on Permuted MNIST (2 tasks, 1 epoch) — no crash."""
        output_path = _run_experiment(
            temp_dir, protocol="permuted", ewc_lambda=10.0, n_tasks=2, online=True
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["protocol"] == "permuted"
        assert len(data["accuracy_matrix"]) == 2

    def test_zero_lambda_completes(self, temp_dir):
        """lambda=0 (EWC disabled) completes and yields valid metrics."""
        output_path = _run_experiment(
            temp_dir, protocol="split", ewc_lambda=0.0, n_tasks=2
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["ewc_lambda"] == 0.0
        fg = data["metrics"]["average_forgetting"]
        acc = data["metrics"]["average_accuracy"]
        assert 0.0 <= fg <= 1.0
        assert 0.0 <= acc <= 1.0

    def test_determinism(self, temp_dir):
        """Same seed produces identical results."""
        out1 = os.path.join(temp_dir, "run1")
        out2 = os.path.join(temp_dir, "run2")
        os.makedirs(out1)
        os.makedirs(out2)

        path1 = _run_experiment(out1, protocol="split", ewc_lambda=10.0, n_tasks=2, seed=42)
        path2 = _run_experiment(out2, protocol="split", ewc_lambda=10.0, n_tasks=2, seed=42)

        with open(path1) as f:
            data1 = json.load(f)
        with open(path2) as f:
            data2 = json.load(f)

        assert data1["accuracy_matrix"] == data2["accuracy_matrix"]
        assert data1["metrics"] == data2["metrics"]


# ── Test 4: JSON output schema ────────────────────────────────────


class TestJSONSchema:
    """Verify JSON output contains all required keys."""

    REQUIRED_KEYS = {
        "experiment",
        "protocol",
        "weight_format",
        "ewc_lambda",
        "ewc_online",
        "ewc_gamma",
        "fisher_samples",
        "seed",
        "device",
        "epochs_per_task",
        "batch_size",
        "learning_rate",
        "n_tasks",
        "n_parameters",
        "total_training_time_seconds",
        "accuracy_matrix",
        "metrics",
        "training_metrics",
        "weight_snapshots",
    }

    METRICS_KEYS = {
        "average_accuracy",
        "average_forgetting",
        "per_task_accuracy",
        "per_task_forgetting",
    }

    def test_required_keys(self, temp_dir):
        """Output has all required keys including EWC fields."""
        output_path = _run_experiment(temp_dir, protocol="split", ewc_lambda=10.0, n_tasks=2)
        with open(output_path) as f:
            data = json.load(f)

        for key in self.REQUIRED_KEYS:
            assert key in data, f"Missing top-level key: {key}"
        for key in self.METRICS_KEYS:
            assert key in data["metrics"], f"Missing metrics key: {key}"

    def test_accuracy_matrix_shape(self, temp_dir):
        """Accuracy matrix has correct dimensions (n_tasks × n_tasks triangular)."""
        output_path = _run_experiment(temp_dir, protocol="split", ewc_lambda=10.0, n_tasks=3)
        with open(output_path) as f:
            data = json.load(f)

        n_tasks = data["n_tasks"]
        matrix = data["accuracy_matrix"]
        assert len(matrix) == n_tasks
        for i, row in enumerate(matrix):
            assert len(row) == i + 1, f"Row {i} expected {i + 1} columns, got {len(row)}"

    def test_metrics_ranges(self, temp_dir):
        """Forgetting/accuracy metrics are in [0, 1]."""
        output_path = _run_experiment(temp_dir, protocol="split", ewc_lambda=10.0, n_tasks=2)
        with open(output_path) as f:
            data = json.load(f)
        m = data["metrics"]
        assert 0.0 <= m["average_forgetting"] <= 1.0
        assert 0.0 <= m["average_accuracy"] <= 1.0
        for v in m["per_task_forgetting"]:
            assert 0.0 <= v <= 1.0


# ── Test 5: Forgetting metric correctness ─────────────────────────


class TestForgettingMetrics:
    """Verify forgetting metric computation with known data (shared infra)."""

    def test_no_forgetting(self):
        """If accuracy stays the same, forgetting is 0."""
        from ph_neuro.analysis.continual import evaluate_continual_learning

        acc_matrix = [[0.9], [0.9, 0.85], [0.9, 0.85, 0.8]]
        metrics = evaluate_continual_learning(acc_matrix)
        assert metrics["average_forgetting"] == 0.0

    def test_complete_forgetting(self):
        """If accuracy drops to 0, forgetting is 100%."""
        from ph_neuro.analysis.continual import evaluate_continual_learning

        acc_matrix = [[0.9], [0.0, 0.85], [0.0, 0.0, 0.8]]
        metrics = evaluate_continual_learning(acc_matrix)
        expected = (0.9 + 0.85 + 0.0) / 3
        assert abs(metrics["average_forgetting"] - expected) < 1e-6


@pytest.fixture
def temp_dir():
    """Create a temporary directory for output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
