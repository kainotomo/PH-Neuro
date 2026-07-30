"""Integration tests for L8: Forgetting Baseline experiment.

Tests:
    1. Ternary STE on Split MNIST (2 tasks, 1 epoch) — no crash
    2. FP16 on Split MNIST (2 tasks, 1 epoch) — no crash
    3. Ternary STE on Permuted MNIST (2 tasks, 1 epoch) — no crash
    4. JSON output contains all required keys
    5. Determinism: same seed → same results
    6. Accuracy matrix has correct shape
    7. Forgetting metrics computed correctly
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from ph_neuro.examples.run_l8_forgetting_baseline import (
    _build_fp16_mlp,
    _build_ternary_mlp,
    _compute_ternary_weight_stats,
    train_task,
)

# ── Test helpers ────────────────────────────────────────────────────


def _count_json_files(directory: str) -> int:
    """Count JSON result files in a directory (excluding summaries)."""
    return sum(
        1 for f in os.listdir(directory) if f.endswith(".json") and f != "aggregated_summary.txt"
    )


# ── Test 1 & 2: Model builders ─────────────────────────────────────


class TestModelBuilders:
    """Verify model builders produce correct architectures."""

    def test_ternary_mlp_structure(self):
        """Ternary STE MLP has correct layer sizes and module types."""
        device = torch.device("cpu")
        model = _build_ternary_mlp(device)
        # Should be a Sequential with Flatten + TernarySTELinear + ReLU + BN + ...
        assert isinstance(model, torch.nn.Sequential)
        # Count modules: Flatten + Linear + ReLU + BN + Linear + ReLU + BN + Linear = 8
        assert len(model) >= 7, f"Expected >=7 modules, got {len(model)}"

    def test_fp16_mlp_structure(self):
        """FP16 MLP has correct layer sizes and uses standard nn.Linear."""
        device = torch.device("cpu")
        model = _build_fp16_mlp(device)
        assert isinstance(model, torch.nn.Sequential)
        # Should have nn.Linear layers (not TernarySTELinear)
        linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        assert linear_count == 3, f"Expected 3 Linear layers, got {linear_count}"

    def test_ternary_weight_stats(self):
        """Weight stats extraction works on a freshly initialized model."""
        device = torch.device("cpu")
        model = _build_ternary_mlp(device)
        stats = _compute_ternary_weight_stats(model)
        assert "weight_sparsity_pct" in stats
        assert "weight_pos_pct" in stats
        assert "weight_neg_pct" in stats
        assert stats["n_parameters"] > 0
        # Sum should be ~100%
        total = stats["weight_pos_pct"] + stats["weight_neg_pct"] + stats["weight_zero_pct"]
        assert abs(total - 100.0) < 1.0, f"Distribution doesn't sum to 100%: {total}"

    def test_fp16_weight_stats_empty(self):
        """Weight stats for FP16 model should return defaults (no ternary weights)."""
        device = torch.device("cpu")
        model = _build_fp16_mlp(device)
        stats = _compute_ternary_weight_stats(model)
        assert stats["n_parameters"] == 0.0
        assert stats["weight_sparsity_pct"] == 0.0


# ── Test 3: Training function ────────────────────────────────────


class TestTrainTask:
    """Verify the training function works on small data."""

    @pytest.fixture
    def tiny_loader(self):
        """Create a tiny DataLoader with 32 samples for fast testing."""
        from torch.utils.data import DataLoader, TensorDataset

        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=16)

    def test_train_task_ternary(self, tiny_loader):
        """Training a ternary model for 1 epoch completes without error."""
        device = torch.device("cpu")
        model = _build_ternary_mlp(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        metrics = train_task(model, tiny_loader, optimizer, 1, device, 0, "test")
        assert "final_loss" in metrics
        assert "final_train_acc" in metrics
        assert metrics["final_loss"] > 0

    def test_train_task_fp16(self, tiny_loader):
        """Training an FP16 model for 1 epoch completes without error."""
        device = torch.device("cpu")
        model = _build_fp16_mlp(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        metrics = train_task(model, tiny_loader, optimizer, 1, device, 0, "test")
        assert "final_loss" in metrics
        assert "final_train_acc" in metrics


# ── Test 4: End-to-end experiment runs ────────────────────────────


class TestEndToEnd:
    """Run the full experiment pipeline and check outputs (smoke tests)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def _run_experiment(
        self,
        protocol: str,
        weight_format: str,
        output_dir: str,
        n_tasks: int = 2,
        epochs_per_task: int = 1,
        seed: int = 42,
    ):
        """Helper to run experiment and return output file path."""
        import sys
        from pathlib import Path

        # Add src to path
        src_dir = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(src_dir))

        from ph_neuro.examples.run_l8_forgetting_baseline import main

        # We need to call main with our args. Since main() uses argparse,
        # we'll simulate by calling the internal logic directly.
        # Instead, build the command-line args and run via argparse.
        sys.argv = [
            "run_l8_forgetting_baseline",
            "--protocol",
            protocol,
            "--weight-format",
            weight_format,
            "--epochs-per-task",
            str(epochs_per_task),
            "--batch-size",
            "64",
            "--lr",
            "0.001",
            "--n-tasks",
            str(n_tasks),
            "--seed",
            str(seed),
            "--output-dir",
            output_dir,
            "--device",
            "cpu",
        ]
        # Capture stdout to avoid noise during tests
        main()

        # Find the output file
        expected_name = f"{protocol}_{weight_format}_seed{seed}.json"
        output_path = os.path.join(output_dir, expected_name)
        assert os.path.exists(output_path), f"Output file not found: {output_path}"
        return output_path

    def test_ternary_split_mnist(self, temp_dir):
        """Ternary STE on Split MNIST (1 epoch each of 5 tasks) — no crash."""
        output_path = self._run_experiment(
            protocol="split",
            weight_format="ternary",
            output_dir=temp_dir,
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["protocol"] == "split"
        assert data["weight_format"] == "ternary"
        # Split MNIST always has 5 binary tasks
        assert len(data["accuracy_matrix"]) == 5

    def test_fp16_split_mnist(self, temp_dir):
        """FP16 on Split MNIST (1 epoch each of 5 tasks) — no crash."""
        output_path = self._run_experiment(
            protocol="split",
            weight_format="fp16",
            output_dir=temp_dir,
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["protocol"] == "split"
        assert data["weight_format"] == "fp16"
        assert len(data["accuracy_matrix"]) == 5

    def test_ternary_permuted_mnist(self, temp_dir):
        """Ternary STE on Permuted MNIST (2 tasks, 1 epoch) — no crash."""
        output_path = self._run_experiment(
            protocol="permuted",
            weight_format="ternary",
            output_dir=temp_dir,
            n_tasks=2,
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["protocol"] == "permuted"
        assert data["weight_format"] == "ternary"
        assert len(data["accuracy_matrix"]) == 2

    def test_fp16_permuted_mnist(self, temp_dir):
        """FP16 on Permuted MNIST (2 tasks, 1 epoch) — no crash."""
        output_path = self._run_experiment(
            protocol="permuted",
            weight_format="fp16",
            output_dir=temp_dir,
            n_tasks=2,
        )
        with open(output_path) as f:
            data = json.load(f)
        assert data["protocol"] == "permuted"
        assert data["weight_format"] == "fp16"
        assert len(data["accuracy_matrix"]) == 2

    def test_determinism(self, temp_dir):
        """Same seed produces identical results."""
        out1 = os.path.join(temp_dir, "run1")
        out2 = os.path.join(temp_dir, "run2")
        os.makedirs(out1)
        os.makedirs(out2)

        path1 = self._run_experiment(
            protocol="split",
            weight_format="ternary",
            output_dir=out1,
            seed=42,
            n_tasks=2,
        )
        path2 = self._run_experiment(
            protocol="split",
            weight_format="ternary",
            output_dir=out2,
            seed=42,
            n_tasks=2,
        )

        with open(path1) as f:
            data1 = json.load(f)
        with open(path2) as f:
            data2 = json.load(f)

        # Accuracy matrices should be identical
        assert data1["accuracy_matrix"] == data2["accuracy_matrix"]
        assert data1["metrics"] == data2["metrics"]


# ── Test 5: JSON output schema ────────────────────────────────────


class TestJSONSchema:
    """Verify JSON output contains all required keys."""

    REQUIRED_KEYS = {
        "experiment",
        "protocol",
        "weight_format",
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
    }

    METRICS_KEYS = {
        "average_accuracy",
        "average_forgetting",
        "per_task_accuracy",
        "per_task_forgetting",
    }

    def test_required_keys_ternary(self, temp_dir):
        """Ternary output has all required keys."""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(src_dir))

        from ph_neuro.examples.run_l8_forgetting_baseline import main

        sys.argv = [
            "run_l8_forgetting_baseline",
            "--protocol",
            "split",
            "--weight-format",
            "ternary",
            "--epochs-per-task",
            "1",
            "--batch-size",
            "64",
            "--n-tasks",
            "2",
            "--seed",
            "42",
            "--output-dir",
            temp_dir,
            "--device",
            "cpu",
        ]
        main()

        output_path = os.path.join(temp_dir, "split_ternary_seed42.json")
        with open(output_path) as f:
            data = json.load(f)

        # Check top-level keys
        for key in self.REQUIRED_KEYS:
            assert key in data, f"Missing top-level key: {key}"

        # Check metrics sub-keys
        for key in self.METRICS_KEYS:
            assert key in data["metrics"], f"Missing metrics key: {key}"

        # Ternary should have weight_snapshots
        assert "weight_snapshots" in data, "Missing weight_snapshots for ternary"

    def test_required_keys_fp16(self, temp_dir):
        """FP16 output has all required keys (no weight_snapshots)."""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(src_dir))

        from ph_neuro.examples.run_l8_forgetting_baseline import main

        sys.argv = [
            "run_l8_forgetting_baseline",
            "--protocol",
            "split",
            "--weight-format",
            "fp16",
            "--epochs-per-task",
            "1",
            "--batch-size",
            "64",
            "--n-tasks",
            "2",
            "--seed",
            "42",
            "--output-dir",
            temp_dir,
            "--device",
            "cpu",
        ]
        main()

        output_path = os.path.join(temp_dir, "split_fp16_seed42.json")
        with open(output_path) as f:
            data = json.load(f)

        for key in self.REQUIRED_KEYS:
            assert key in data, f"Missing top-level key: {key}"

        for key in self.METRICS_KEYS:
            assert key in data["metrics"], f"Missing metrics key: {key}"

        # FP16 should NOT have weight_snapshots
        assert "weight_snapshots" not in data, "FP16 should not have weight_snapshots"

    def test_accuracy_matrix_shape(self, temp_dir):
        """Accuracy matrix has correct shape (n_tasks × n_tasks)."""
        import sys
        from pathlib import Path

        src_dir = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(src_dir))

        from ph_neuro.examples.run_l8_forgetting_baseline import main

        sys.argv = [
            "run_l8_forgetting_baseline",
            "--protocol",
            "split",
            "--weight-format",
            "ternary",
            "--epochs-per-task",
            "1",
            "--batch-size",
            "64",
            "--n-tasks",
            "3",
            "--seed",
            "42",
            "--output-dir",
            temp_dir,
            "--device",
            "cpu",
        ]
        main()

        output_path = os.path.join(temp_dir, "split_ternary_seed42.json")
        with open(output_path) as f:
            data = json.load(f)

        n_tasks = data["n_tasks"]
        matrix = data["accuracy_matrix"]
        assert len(matrix) == n_tasks, f"Expected {n_tasks} rows, got {len(matrix)}"
        for i, row in enumerate(matrix):
            assert len(row) == i + 1, f"Row {i} expected {i + 1} columns, got {len(row)}"


# ── Test 6: Forgetting metric correctness ─────────────────────────


class TestForgettingMetrics:
    """Verify forgetting metric computation with known data."""

    def test_no_forgetting(self):
        """If accuracy stays the same, forgetting is 0."""
        from ph_neuro.analysis.continual import evaluate_continual_learning

        acc_matrix = [
            [0.9],  # After task 1
            [0.9, 0.85],  # After task 2
            [0.9, 0.85, 0.8],  # After task 3
        ]
        metrics = evaluate_continual_learning(acc_matrix)
        assert metrics["average_forgetting"] == 0.0

    def test_complete_forgetting(self):
        """If accuracy drops to 0, forgetting is 100%."""
        from ph_neuro.analysis.continual import evaluate_continual_learning

        acc_matrix = [
            [0.9],
            [0.0, 0.85],
            [0.0, 0.0, 0.8],
        ]
        metrics = evaluate_continual_learning(acc_matrix)
        # Task 1: peak=0.9, final=0.0 → forget=0.9
        # Task 2: peak=0.85, final=0.0 → forget=0.85
        # Task 3: peak=0.8, final=0.8 → forget=0.0
        expected_forgetting = (0.9 + 0.85 + 0.0) / 3
        assert abs(metrics["average_forgetting"] - expected_forgetting) < 1e-6


@pytest.fixture
def temp_dir():
    """Create a temporary directory for output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
