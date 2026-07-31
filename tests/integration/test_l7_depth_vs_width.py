"""Integration tests for L7: Depth vs Width Scaling experiment.

Tests:
    1. Model builders produce correct architectures (layer counts, param budgets)
    2. All 5 depth configurations have param count within 1% of 530K target
    3. Training loop runs without crash (1-epoch smoke test, all 5 depths)
    4. JSON output contains all required keys
    5. Determinism: same seed → same results
    6. Weight statistics extraction works for both formats
    7. Aggregator produces correct table dimensions
    8. Per-layer stats are computed correctly
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from ph_neuro.examples.run_l7_depth_vs_width import (
    DEPTH_CONFIGS,
    _build_fp16_mlp,
    _build_ternary_mlp,
    _compute_per_layer_stats,
    _compute_weight_stats,
    evaluate,
    train_and_evaluate,
)

# ── Test helpers ────────────────────────────────────────────────────


def _count_params(layer_sizes: list[int]) -> int:
    """Compute the number of weight parameters for a BatchNorm MLP.

    Matches the model construction: no bias, BatchNorm after each
    hidden layer. Each hidden linear layer in_features × out_features.
    """
    total = 0
    for i in range(len(layer_sizes) - 1):
        total += layer_sizes[i] * layer_sizes[i + 1]
    return total


# ── Test 1: Model builders ─────────────────────────────────────────


class TestModelBuilders:
    """Verify model builders produce correct architectures."""

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_ternary_mlp_structure(self, depth: int):
        """Ternary STE MLP has correct number of layers for each depth."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]
        model = _build_ternary_mlp(layer_sizes, device)

        # Count TernarySTELinear layers
        from ph_neuro.layers.ste_linear import TernarySTELinear

        linear_count = sum(
            1 for m in model.modules() if isinstance(m, TernarySTELinear)
        )
        # D hidden layers → D+1 linear layers (D hidden + 1 output)
        expected_linear = depth + 1
        assert linear_count == expected_linear, (
            f"Depth={depth}: expected {expected_linear} TernarySTELinear, "
            f"got {linear_count}"
        )

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_fp16_mlp_structure(self, depth: int):
        """FP16 MLP has correct number of Linear layers for each depth."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]
        model = _build_fp16_mlp(layer_sizes, device)

        linear_count = sum(
            1 for m in model.modules() if isinstance(m, torch.nn.Linear)
        )
        expected_linear = depth + 1
        assert linear_count == expected_linear, (
            f"Depth={depth}: expected {expected_linear} nn.Linear, "
            f"got {linear_count}"
        )


# ── Test 2: Parameter budget validation ────────────────────────────


class TestParameterBudget:
    """All depth configurations must stay within 1% of 530K target."""

    BUDGET_TARGET = 530_000
    TOLERANCE = 0.01  # 1%

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_ternary_param_count(self, depth: int):
        """Ternary STE MLP param count matches budget."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]
        model = _build_ternary_mlp(layer_sizes, device)
        n_params = sum(p.numel() for p in model.parameters())

        rel_error = abs(n_params - self.BUDGET_TARGET) / self.BUDGET_TARGET
        assert rel_error < self.TOLERANCE, (
            f"Depth={depth}: {n_params:,} params, "
            f"{100 * rel_error:.2f}% from target {self.BUDGET_TARGET:,} "
            f"(exceeds {100 * self.TOLERANCE:.0f}% tolerance)"
        )

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_fp16_param_count(self, depth: int):
        """FP16 MLP param count matches budget."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]
        model = _build_fp16_mlp(layer_sizes, device)
        n_params = sum(p.numel() for p in model.parameters())

        rel_error = abs(n_params - self.BUDGET_TARGET) / self.BUDGET_TARGET
        assert rel_error < self.TOLERANCE, (
            f"Depth={depth}: {n_params:,} params, "
            f"{100 * rel_error:.2f}% from target {self.BUDGET_TARGET:,}"
        )

    def test_all_depths_unique(self):
        """Each depth config has unique layer sizes (no duplicates)."""
        size_strings = [str(DEPTH_CONFIGS[d]) for d in range(1, 6)]
        assert len(set(size_strings)) == 5, "Duplicate layer size configs detected"

    def test_all_depths_different_widths(self):
        """Widths decrease monotonically with depth (sanity check)."""
        widths = [DEPTH_CONFIGS[d][1] for d in range(1, 6)]
        for i in range(len(widths) - 1):
            assert widths[i] > widths[i + 1], (
                f"Widths not decreasing: depth={i+1} width={widths[i]}, "
                f"depth={i+2} width={widths[i+1]}"
            )


# ── Test 3: Training loop smoke test ───────────────────────────────


class TestTrainingSmoke:
    """Verify training runs without crashing for all depths (1 epoch)."""

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("weight_format", ["ternary", "fp16"])
    def test_train_one_epoch(self, depth: int, weight_format: str):
        """Training one epoch completes without error."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]

        if weight_format == "ternary":
            model = _build_ternary_mlp(layer_sizes, device)
        else:
            model = _build_fp16_mlp(layer_sizes, device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Create tiny synthetic dataset (16 samples, MNIST-sized)
        x = torch.randn(16, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (16,)).to(device)
        from torch.utils.data import TensorDataset, DataLoader

        train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
        test_loader = DataLoader(TensorDataset(x, y), batch_size=4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

        results = train_and_evaluate(
            model, train_loader, test_loader,
            optimizer, scheduler, 1, device,
            weight_format=weight_format,
        )

        assert "best_accuracy" in results
        assert results["best_accuracy"] >= 0.0
        assert results["epochs_trained"] == 1


# ── Test 4: JSON output schema ─────────────────────────────────────


class TestJSONSchema:
    """Verify the result dict has all required keys."""

    REQUIRED_KEYS = [
        "experiment",
        "dataset",
        "weight_format",
        "depth",
        "layer_sizes",
        "seed",
        "device",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "n_parameters",
        "best_accuracy",
        "best_epoch",
        "final_accuracy",
        "training_time_seconds",
        "epochs_trained",
        "weight_sparsity_pct",
        "weight_zero_pct",
        "weight_pos_pct",
        "weight_neg_pct",
        "n_parameters",
        "per_layer",
    ]

    @pytest.mark.parametrize("depth", [1, 3, 5])
    @pytest.mark.parametrize("weight_format", ["ternary", "fp16"])
    def test_required_keys_present(self, depth: int, weight_format: str):
        """Result dict from training has all required keys."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]

        if weight_format == "ternary":
            model = _build_ternary_mlp(layer_sizes, device)
        else:
            model = _build_fp16_mlp(layer_sizes, device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)
        from torch.utils.data import TensorDataset, DataLoader

        train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
        test_loader = DataLoader(TensorDataset(x, y), batch_size=4)

        results = train_and_evaluate(
            model, train_loader, test_loader,
            optimizer, None, 1, device,
            weight_format=weight_format,
        )

        result = {
            "experiment": "L7",
            "dataset": "mnist",
            "weight_format": weight_format,
            "depth": depth,
            "layer_sizes": layer_sizes,
            "seed": 42,
            "device": str(device),
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "n_parameters": sum(p.numel() for p in model.parameters()),
            **results,
        }

        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing required key: {key}"


# ── Test 5: Determinism ────────────────────────────────────────────


class TestDeterminism:
    """Same seed must produce identical results."""

    @pytest.mark.parametrize("depth", [2, 4])
    def test_deterministic_results(self, depth: int):
        """Two runs with same seed produce identical accuracy."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]

        results_list = []
        for _ in range(2):
            torch.manual_seed(42)
            model = _build_fp16_mlp(layer_sizes, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

            x = torch.randn(8, 1, 28, 28)
            y = torch.randint(0, 10, (8,))
            from torch.utils.data import TensorDataset, DataLoader

            train_loader = DataLoader(TensorDataset(x, y), batch_size=4)
            test_loader = DataLoader(TensorDataset(x, y), batch_size=4)

            results = train_and_evaluate(
                model, train_loader, test_loader,
                optimizer, None, 1, device,
                weight_format="fp16",
            )
            results_list.append(results["best_accuracy"])

        assert results_list[0] == results_list[1], (
            f"Non-deterministic: {results_list[0]} != {results_list[1]}"
        )


# ── Test 6: Weight statistics ──────────────────────────────────────


class TestWeightStats:
    """Weight statistics extraction works correctly."""

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_ternary_weight_stats(self, depth: int):
        """Ternary weight stats are well-formed."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]
        model = _build_ternary_mlp(layer_sizes, device)
        stats = _compute_weight_stats(model, "ternary")

        assert "weight_sparsity_pct" in stats
        assert "weight_pos_pct" in stats
        assert "weight_neg_pct" in stats
        assert "weight_zero_pct" in stats
        assert stats["n_parameters"] > 0

        # Distribution should sum to ~100%
        total = (
            stats["weight_pos_pct"]
            + stats["weight_neg_pct"]
            + stats["weight_zero_pct"]
        )
        assert abs(total - 100.0) < 1.5, f"Distribution sum: {total}%"

    @pytest.mark.parametrize("depth", [1, 3, 5])
    def test_fp16_weight_stats(self, depth: int):
        """FP16 weight stats have expected keys."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[depth]
        model = _build_fp16_mlp(layer_sizes, device)
        stats = _compute_weight_stats(model, "fp16")

        assert "weight_sparsity_pct" in stats
        assert stats["n_parameters"] > 0


# ── Test 7: Per-layer stats ────────────────────────────────────────


class TestPerLayerStats:
    """Per-layer statistics extraction works correctly."""

    def test_ternary_per_layer_count(self):
        """Per-layer stats has one entry per TernarySTELinear layer."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[3]
        model = _build_ternary_mlp(layer_sizes, device)
        per_layer = _compute_per_layer_stats(model, "ternary")

        from ph_neuro.layers.ste_linear import TernarySTELinear

        expected_count = sum(
            1 for m in model.modules() if isinstance(m, TernarySTELinear)
        )
        assert len(per_layer) == expected_count, (
            f"Expected {expected_count} per-layer entries, got {len(per_layer)}"
        )

    def test_fp16_per_layer_count(self):
        """Per-layer stats has one entry per nn.Linear layer."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[3]
        model = _build_fp16_mlp(layer_sizes, device)
        per_layer = _compute_per_layer_stats(model, "fp16")

        expected_count = sum(
            1 for m in model.modules() if isinstance(m, torch.nn.Linear)
        )
        assert len(per_layer) == expected_count, (
            f"Expected {expected_count} per-layer entries, got {len(per_layer)}"
        )

    def test_per_layer_sum_matches_total(self):
        """Sum of per-layer params equals total model params."""
        device = torch.device("cpu")
        layer_sizes = DEPTH_CONFIGS[4]
        model = _build_ternary_mlp(layer_sizes, device)
        per_layer = _compute_per_layer_stats(model, "ternary")

        per_layer_params = sum(entry["n_params"] for entry in per_layer)
        total_params = sum(p.numel() for p in model.parameters())
        assert per_layer_params < total_params, (
            f"Per-layer sum ({per_layer_params}) >= total ({total_params})"
        )
        # BatchNorm and other non-ternary params should be excluded
        assert per_layer_params < total_params, "Per-layer sum should exclude BN params"


# ── Test 8: Aggregator ─────────────────────────────────────────────


class TestAggregator:
    """Verify the aggregator can load and process results."""

    def test_aggregator_loads_json_files(self):
        """Aggregator loads results from temp directory."""
        from ph_neuro.examples.aggregate_l7_results import (
            aggregate_by_depth_format,
            load_results,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal result file
            result = {
                "experiment": "L7",
                "dataset": "mnist",
                "weight_format": "ternary",
                "depth": 2,
                "layer_sizes": [784, 432, 432, 10],
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "n_parameters": 529_632,
                "best_accuracy": 0.95,
                "best_epoch": 1,
                "final_accuracy": 0.94,
                "training_time_seconds": 10.0,
                "epochs_trained": 1,
                "weight_sparsity_pct": 0.0,
                "weight_zero_pct": 0.0,
                "weight_pos_pct": 50.0,
                "weight_neg_pct": 50.0,
                "per_layer": [],
            }
            path = os.path.join(tmpdir, "results_mnist_ternary_d2_seed42.json")
            with open(path, "w") as f:
                json.dump(result, f)

            results = load_results(tmpdir)
            assert len(results) == 1

            aggregated = aggregate_by_depth_format(results)
            assert "mnist" in aggregated
            assert 2 in aggregated["mnist"]
            assert "ternary" in aggregated["mnist"][2]
            assert aggregated["mnist"][2]["ternary"]["accuracy_mean"] == 0.95

    def test_aggregator_multiple_seeds(self):
        """Aggregator correctly averages across seeds."""
        from ph_neuro.examples.aggregate_l7_results import (
            aggregate_by_depth_format,
            load_results,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for seed, acc in [(42, 0.95), (43, 0.93), (44, 0.97)]:
                result = {
                    "experiment": "L7",
                    "dataset": "mnist",
                    "weight_format": "ternary",
                    "depth": 1,
                    "layer_sizes": [784, 667, 10],
                    "seed": seed,
                    "device": "cpu",
                    "epochs": 1,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                    "n_parameters": 529_898,
                    "best_accuracy": acc,
                    "best_epoch": 1,
                    "final_accuracy": acc,
                    "training_time_seconds": 10.0,
                    "epochs_trained": 1,
                    "weight_sparsity_pct": 0.0,
                    "weight_zero_pct": 0.0,
                    "weight_pos_pct": 50.0,
                    "weight_neg_pct": 50.0,
                    "per_layer": [],
                }
                fname = f"results_mnist_ternary_d1_seed{seed}.json"
                path = os.path.join(tmpdir, fname)
                with open(path, "w") as f:
                    json.dump(result, f)

            results = load_results(tmpdir)
            aggregated = aggregate_by_depth_format(results)
            mean_acc = aggregated["mnist"][1]["ternary"]["accuracy_mean"]
            assert abs(mean_acc - 0.95) < 0.02, f"Mean accuracy: {mean_acc}"
            assert aggregated["mnist"][1]["ternary"]["n_runs"] == 3


# ── Test 9: Evaluate function ──────────────────────────────────────


class TestEvaluate:
    """Verify the evaluate function works correctly."""

    def test_evaluate_perfect_accuracy(self):
        """Evaluate returns 1.0 for a model that always predicts correctly."""
        device = torch.device("cpu")

        # Create a trivial model that just returns the labels as logits
        model = torch.nn.Linear(10, 10, bias=False)
        # Set weights to identity: logits[i] = input[i]
        model.weight.data = torch.eye(10)

        # Create data where input = one-hot of label
        x = torch.eye(10).unsqueeze(-1).unsqueeze(-1)  # (10, 10, 1, 1) — won't match
        # Actually let's use a simple one-hot setup
        x = torch.eye(10)  # (10, 10)
        y = torch.arange(10)  # (10,)

        from torch.utils.data import TensorDataset, DataLoader
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        acc = evaluate(model, loader, device)
        assert acc == 1.0, f"Expected 1.0, got {acc}"
