"""Integration tests for Phase 1.3 — continual learning experiments.

Tests:
    1. Split MNIST task generation (data shapes, label remapping)
    2. Permuted MNIST task generation (permutation consistency)
    3. Continual learning experiment loop (no crash on synthetic data)
    4. Forgetting metrics computation (known answer test)
    5. Weight stability analysis functions
    6. Hebbian vs backprop interface consistency
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from ph_neuro.analysis.continual import (
    analyze_hysteresis_protection,
    analyze_weight_sparsity,
    compute_forgetting_metric,
    compute_per_class_weight_stability,
    compute_weight_overlap,
    evaluate_continual_learning,
)
from ph_neuro.training.continual import (
    ContinualTask,
    create_permuted_mnist_tasks,
    create_split_mnist_tasks,
    make_backprop_predict_fn,
    make_hebbian_predict_fn,
    run_continual_experiment,
)
from ph_neuro.training.data import (
    _make_permutation,
    get_binary_mnist_loaders,
    get_mnist_full_test_loader,
    get_permuted_mnist_loaders,
)


# ── Helpers ────────────────────────────────────────────────────────


def _synthetic_loader(n_samples: int = 32, n_features: int = 10) -> DataLoader:
    """Create a tiny synthetic DataLoader for fast tests."""
    x = torch.randn(n_samples, 1, 1, n_features)
    y = torch.randint(0, 2, (n_samples,))
    return DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=8,
    )


class _DummyModel(torch.nn.Module):
    """Minimal model stub for testing experiment infrastructure."""

    def __init__(self, n_features: int = 10, n_classes: int = 2):
        super().__init__()
        self._weights = torch.nn.Parameter(torch.randn(n_classes, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return x_flat @ self._weights.T


# ── Test 1: Binary MNIST task data ─────────────────────────────────


class TestBinaryMNISTData:
    """Verify binary task loaders produce correct shapes and labels."""

    def test_binary_loader_shapes(self):
        """Binary MNIST loader yields correct shapes with labels 0/1."""
        train_loader, test_loader = get_binary_mnist_loaders(
            class_a=3, class_b=7, batch_size=32
        )
        for x, y in train_loader:
            assert x.shape[0] <= 32
            assert y.shape[0] == x.shape[0]
            assert y.dtype == torch.long
            assert y.min() >= 0
            assert y.max() <= 1
            break

    def test_binary_filter_correct_classes(self):
        """Binary loader only contains the two requested classes."""
        train_loader, _ = get_binary_mnist_loaders(class_a=0, class_b=1, batch_size=128)
        # Check a few batches
        for x, y in train_loader:
            assert y.min() >= 0
            assert y.max() <= 1
            break
        # Verify dataset is non-empty
        assert len(train_loader.dataset) > 0

    def test_full_test_loader(self):
        """Full test loader returns all 10 classes."""
        loader = get_mnist_full_test_loader(batch_size=128)
        seen = set()
        for _, y in loader:
            seen.update(y.tolist())
        assert len(seen) == 10, f"Expected 10 classes, got {len(seen)}"


# ── Test 2: Permuted MNIST data ───────────────────────────────────


class TestPermutedMNISTData:
    """Verify permuted MNIST loaders produce consistent permutations."""

    def test_permutation_is_deterministic(self):
        """Same seed produces the same permutation."""
        p1 = _make_permutation(seed=42)
        p2 = _make_permutation(seed=42)
        assert torch.equal(p1, p2), "Permutations should be identical for same seed"

    def test_different_seeds_different_permutations(self):
        """Different seeds produce different permutations."""
        p1 = _make_permutation(seed=0)
        p2 = _make_permutation(seed=1)
        assert not torch.equal(p1, p2), "Permutations should differ for different seeds"

    def test_permuted_loader_shapes(self):
        """Permuted MNIST loader yields correct shapes."""
        train_loader, _ = get_permuted_mnist_loaders(perm_seed=0, batch_size=32)
        for x, y in train_loader:
            assert x.shape == (32, 1, 28, 28), f"Unexpected shape: {x.shape}"
            assert y.min() >= 0
            assert y.max() <= 9
            break


# ── Test 3: Task generation ────────────────────────────────────────


class TestTaskGeneration:
    """Verify task sequence generators."""

    def test_split_mnist_5_tasks(self):
        """Split MNIST generates exactly 5 tasks."""
        tasks = create_split_mnist_tasks(batch_size=128)
        assert len(tasks) == 5
        expected_names = ["0 vs 1", "2 vs 3", "4 vs 5", "6 vs 7", "8 vs 9"]
        for task, expected in zip(tasks, expected_names):
            assert task.name == expected
            assert task.n_classes == 2

    def test_split_mnist_task_test_loaders(self):
        """Each split MNIST task has test loaders for all previous tasks."""
        tasks = create_split_mnist_tasks(batch_size=128)
        for task_idx, task in enumerate(tasks):
            # Should have test loaders for all tasks up to this one
            for prev_idx in range(task_idx + 1):
                assert prev_idx in task.test_loaders, (
                    f"Task {task_idx} missing test loader for task {prev_idx}"
                )
            # Plus the global test loader at key -1
            assert -1 in task.test_loaders, "Missing global test loader"

    def test_permuted_mnist_n_tasks(self):
        """Permuted MNIST generates the requested number of tasks."""
        tasks = create_permuted_mnist_tasks(n_tasks=3, batch_size=32)
        assert len(tasks) == 3
        for task in tasks:
            assert task.n_classes == 10


# ── Test 4: Continual experiment loop ──────────────────────────────


class TestExperimentLoop:
    """Verify the experiment loop runs without errors."""

    def _make_dummy_tasks(self, n_tasks: int = 2) -> list[ContinualTask]:
        """Create synthetic tasks for fast testing."""
        tasks = []
        for i in range(n_tasks):
            train_loader = _synthetic_loader()
            test_loaders = {j: _synthetic_loader() for j in range(i + 1)}
            tasks.append(
                ContinualTask(
                    name=f"Task {i}",
                    train_loader=train_loader,
                    test_loaders=test_loaders,
                    n_classes=2,
                    task_id=i,
                )
            )
        return tasks

    def test_experiment_loop_completes(self):
        """Experiment loop runs without errors on synthetic data."""
        tasks = self._make_dummy_tasks(n_tasks=2)
        model = _DummyModel(n_features=10, n_classes=2)
        predict_fn = make_backprop_predict_fn()

        def train_fn(model, task, task_idx):
            return {"loss": 0.5, "acc": 0.75}

        results = run_continual_experiment(
            model=model,
            tasks=tasks,
            train_fn=train_fn,
            predict_fn=predict_fn,
        )

        assert "accuracy_matrix" in results
        assert "metrics" in results
        assert len(results["accuracy_matrix"]) == 2
        assert results["n_tasks"] == 2

    def test_experiment_loop_with_weight_recording(self):
        """Experiment loop supports weight snapshot recording."""
        tasks = self._make_dummy_tasks(n_tasks=2)
        model = _DummyModel(n_features=10, n_classes=2)
        predict_fn = make_backprop_predict_fn()

        def train_fn(model, task, task_idx):
            return {"loss": 0.5}

        def record_fn(model, task_idx):
            return {"sparsity": 0.5}

        results = run_continual_experiment(
            model=model,
            tasks=tasks,
            train_fn=train_fn,
            predict_fn=predict_fn,
            record_weight_fn=record_fn,
        )

        assert len(results["weight_snapshots"]) == 2

    def test_hebbian_predict_fn(self):
        """Hebbian predict function works with SupervisedHebbianClassifier-style models."""
        model = _DummyModel(n_features=10, n_classes=2)
        predict_fn = make_hebbian_predict_fn(epsilon=0.1)
        x = torch.randn(4, 1, 1, 10)
        preds = predict_fn(model, x)
        assert preds.shape == (4,)

    def test_backprop_predict_fn(self):
        """Backprop predict function works with raw nn.Module models."""
        model = _DummyModel(n_features=10, n_classes=2)
        predict_fn = make_backprop_predict_fn()
        x = torch.randn(4, 1, 1, 10)
        preds = predict_fn(model, x)
        assert preds.shape == (4,)


# ── Test 5: Forgetting metrics ─────────────────────────────────────


class TestForgettingMetrics:
    """Verify forgetting metric computations with known answers."""

    def test_no_forgetting(self):
        """Perfect retention yields 0 forgetting."""
        acc_matrix = [
            [0.9],
            [0.9, 0.8],
        ]
        metrics = evaluate_continual_learning(acc_matrix)
        assert metrics["average_forgetting"] == 0.0
        assert metrics["average_accuracy"] == pytest.approx(0.85)

    def test_complete_forgetting(self):
        """Complete forgetting of task 0 yields expected metrics."""
        acc_matrix = [
            [0.9],
            [0.0, 0.8],
        ]
        metrics = evaluate_continual_learning(acc_matrix)
        assert metrics["average_forgetting"] == pytest.approx(0.45)  # (0.9 - 0.0) / 2
        assert metrics["average_accuracy"] == pytest.approx(0.4)

    def test_single_task(self):
        """Single task yields 0 forgetting and correct accuracy."""
        acc_matrix = [[0.85]]
        metrics = evaluate_continual_learning(acc_matrix)
        assert metrics["average_forgetting"] == 0.0
        assert metrics["average_accuracy"] == 0.85

    def test_forgetting_metric_function(self):
        """compute_forgetting_metric returns correct values."""
        assert compute_forgetting_metric(0.9, 0.9) == 0.0
        assert compute_forgetting_metric(0.9, 0.0) == pytest.approx(1.0)
        assert compute_forgetting_metric(0.9, 0.45) == pytest.approx(0.5)
        assert compute_forgetting_metric(0.0, 0.0) == 0.0  # edge case


# ── Test 6: Weight stability analysis ──────────────────────────────


class TestWeightStabilityAnalysis:
    """Verify weight stability analysis functions."""

    def test_weight_overlap_identical(self):
        """Identical weights yield perfect overlap."""
        w = torch.tensor([[1, 0, -1], [0, 1, 0]], dtype=torch.int8)
        result = compute_weight_overlap(w, w)
        assert result["agreement_rate"] == 1.0
        assert result["flip_rate"] == 0.0
        assert result["jaccard_similarity"] == 1.0

    def test_weight_overlap_completely_different(self):
        """Completely different non-zero sets yield 0 Jaccard."""
        w1 = torch.tensor([[1, 0, 0]], dtype=torch.int8)
        w2 = torch.tensor([[0, 0, 1]], dtype=torch.int8)
        result = compute_weight_overlap(w1, w2)
        assert result["jaccard_similarity"] == 0.0

    def test_per_class_stability(self):
        """Per-class stability tracks changes across tasks."""
        snapshots = {
            0: torch.tensor([[1, 0, -1], [0, 1, 0]], dtype=torch.int8),
            1: torch.tensor([[1, 0, -1], [1, 0, 0]], dtype=torch.int8),
        }
        result = compute_per_class_weight_stability(snapshots)
        assert "per_neuron_flip_rate" in result
        assert len(result["per_neuron_flip_rate"]) == 2
        # Neuron 0: identical -> 0 flip
        assert result["per_neuron_flip_rate"][0][0] == 0.0
        # Neuron 1: 2/3 changed
        assert result["per_neuron_flip_rate"][1][0] == pytest.approx(2 / 3)

    def test_hysteresis_analysis(self):
        """Hysteresis analysis classifies weights correctly."""
        scores = torch.tensor([-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0], dtype=torch.float16)
        result = analyze_hysteresis_protection(scores, theta_upper=3.0, theta_lower=1.0)
        # -6.0, 6.0: above upper (2 weights)
        assert result["pct_above_upper"] == pytest.approx(100.0 * 2 / 7)
        # -0.5, 0.0, 0.5: below lower (3 weights)
        assert result["pct_below_lower"] == pytest.approx(100.0 * 3 / 7)
        # -2.0, 2.0: in gap (2 weights)
        assert result["pct_in_hysteresis_gap"] == pytest.approx(100.0 * 2 / 7)

    def test_weight_sparsity(self):
        """Weight sparsity analysis computes correctly."""
        w = torch.tensor([[1, 0, 0, -1, 0], [0, 1, 1, 0, -1]], dtype=torch.int8)
        result = analyze_weight_sparsity(w)
        assert result["global_sparsity"] == 50.0  # 5/10 zeros
        assert result["pos_pct"] == 30.0  # 3/10
        assert result["neg_pct"] == 20.0  # 2/10
        assert "per_neuron" in result



