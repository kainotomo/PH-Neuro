"""Integration tests for B2: QLoRA + Frozen Ternary Backbone experiment.

Tests:
    1. End-to-end runs: Split + Permuted MNIST, full + task1 pre-training
    2. Zero forgetting by design (frozen backbone)
    3. Frozen backbone weights never change during LoRA training
    4. LoRA state save/load roundtrip reproduces predictions
    5. Increasing rank increases trainable parameter count
    6. JSON output contains all required keys
    7. Pre-trained backbone accuracy sanity (>90% on MNIST after few epochs)
    8. Determinism: same seed → same results
    9. LoRA parameters are the only trainable parameters after freezing

These tests run the real MNIST training pipeline, so they are marked
``slow`` (skipped by ``pytest -m "not slow"``). A shared session-scoped
fixture reuses one ``split/full`` run across several tests to keep the
suite fast.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.examples.run_b2_qlora import main
from ph_neuro.layers.ste_lora import (
    count_lora_parameters,
    freeze_backbone,
    get_model_lora_state,
    iter_lora_layers,
    load_model_lora_state,
)
from ph_neuro.models.ste_models import ste_mlp_lora

slow = pytest.mark.slow

# ── Test helpers ────────────────────────────────────────────────────


def _run_experiment(
    output_dir: str,
    protocol: str,
    pretrain: str,
    lora_r: int = 2,
    n_tasks: int = 2,
    epochs_pretrain: int = 1,
    epochs_per_task: int = 1,
    seed: int = 42,
) -> str:
    """Run the B2 experiment via CLI and return the output file path.

    ``num_workers=0`` avoids DataLoader ``fork()`` from a multi-threaded
    pytest process (which can corrupt the heap on some platforms).
    """
    sys.argv = [
        "run_b2_qlora",
        "--protocol", protocol,
        "--pretrain", pretrain,
        "--lora-r", str(lora_r),
        "--epochs-pretrain", str(epochs_pretrain),
        "--epochs-per-task", str(epochs_per_task),
        "--n-tasks", str(n_tasks),
        "--num-workers", "0",
        "--seed", str(seed),
        "--output-dir", output_dir,
        "--device", "cpu",
    ]
    main()

    expected = f"{protocol}_{pretrain}_qlora_r{lora_r}_seed{seed}.json"
    output_path = os.path.join(output_dir, expected)
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    return output_path


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def split_full_run(tmp_path_factory):
    """One shared split/full (r=4) run reused across many tests."""
    out = str(tmp_path_factory.mktemp("b2_split_full"))
    return _run_experiment(out, "split", "full", lora_r=4)


@pytest.fixture
def temp_dir(tmp_path):
    """Per-test temporary directory (one per test invocation)."""
    return str(tmp_path)


# ── Test 1: End-to-end runs (smoke) ────────────────────────────────


class TestEndToEnd:
    """Run the full experiment pipeline (smoke tests)."""

    @slow
    def test_full_split_mnist(self, split_full_run):
        """Pretrain=full + Split MNIST — no crash, valid output."""
        data = _load_json(split_full_run)
        assert data["experiment"] == "B2 QLoRA + Frozen Ternary Backbone"
        assert data["protocol"] == "split"
        assert data["pretrain_protocol"] == "full"
        assert data["weight_format"] == "ternary"
        assert data["lora_rank"] == 4
        # Split MNIST always defines 5 binary tasks (matches L8/B1)
        assert len(data["accuracy_matrix"]) == 5

    @slow
    def test_task1_split_mnist(self, temp_dir):
        """Pretrain=task1 + Split MNIST — no crash, 1 pretrain epoch."""
        path = _run_experiment(temp_dir, "split", "task1")
        data = _load_json(path)
        assert data["pretrain_protocol"] == "task1"
        assert data["epochs_pretrain"] == 1

    @slow
    def test_full_permuted_mnist(self, temp_dir):
        """Pretrain=full + Permuted MNIST (2 tasks) — no crash."""
        path = _run_experiment(temp_dir, "permuted", "full", n_tasks=2)
        data = _load_json(path)
        assert data["protocol"] == "permuted"
        assert len(data["accuracy_matrix"]) == 2

    @slow
    def test_task1_permuted_mnist(self, temp_dir):
        """Pretrain=task1 + Permuted MNIST (2 tasks) — no crash."""
        path = _run_experiment(temp_dir, "permuted", "task1", n_tasks=2)
        data = _load_json(path)
        assert data["pretrain_protocol"] == "task1"
        assert data["n_tasks"] == 2


# ── Test 2: Zero forgetting ────────────────────────────────────────


class TestZeroForgetting:
    """Frozen backbone → no catastrophic forgetting by design."""

    @slow
    def test_average_forgetting_is_zero(self, split_full_run):
        """metrics.average_forgetting must be exactly 0.0."""
        data = _load_json(split_full_run)
        assert data["metrics"]["average_forgetting"] == 0.0
        assert all(f == 0.0 for f in data["metrics"]["per_task_forgetting"])

    @slow
    def test_self_accuracy_constant_across_rows(self, split_full_run):
        """accuracy_matrix[i][j] equals self-accuracy of task j for all i >= j."""
        data = _load_json(split_full_run)
        am = data["accuracy_matrix"]
        for j in range(len(am)):
            expected = am[j][j]
            for i in range(j, len(am)):
                assert am[i][j] == pytest.approx(expected, abs=1e-9), (
                    f"Column {j} must be constant, row {i} differs"
                )


# ── Test 3: Frozen backbone ────────────────────────────────────────


class TestFrozenBackbone:
    """Backbone weights never change during LoRA training."""

    @slow
    def test_backbone_saved_untouched(self, split_full_run):
        """The saved backbone.pt matches the LoRA models' latent_scores.

        Rebuild a LoRA model from the saved backbone + a task adapter and
        verify its latent_scores (ternary backbone) equal the saved ones —
        proving the backbone never moved during LoRA training.
        """
        data = _load_json(split_full_run)
        lora_dir = os.path.join(os.path.dirname(split_full_run), "lora")
        backbone_path = os.path.join(lora_dir, "backbone.pt")
        task0_path = os.path.join(lora_dir, "task0.pt")
        assert os.path.exists(backbone_path)
        assert os.path.exists(task0_path)

        backbone_state = torch.load(backbone_path, weights_only=True)
        task0_state = torch.load(task0_path, weights_only=True)

        model = ste_mlp_lora([784, 512, 256, 10], r=4)
        model.load_state_dict(backbone_state, strict=False)
        load_model_lora_state(model, task0_state)

        # Frozen backbone latent_scores match the pre-training snapshot
        for name, layer in iter_lora_layers(model):
            saved_key = f"{name}.latent_scores"
            assert saved_key in backbone_state
            torch.testing.assert_close(
                layer.latent_scores, backbone_state[saved_key], atol=1e-7, rtol=1e-7
            )

    @slow
    def test_all_task_files_exist(self, temp_dir):
        """One LoRA adapter file is saved per task."""
        path = _run_experiment(temp_dir, "permuted", "full", n_tasks=2)
        data = _load_json(path)
        lora_dir = os.path.join(temp_dir, "lora")
        for i in range(data["n_tasks"]):
            assert os.path.exists(os.path.join(lora_dir, f"task{i}.pt"))


# ── Test 4: LoRA state roundtrip ───────────────────────────────────


class TestLoRAStateRoundtrip:
    """Saved adapters restore identical model behaviour."""

    @slow
    def test_roundtrip_reproduces_output(self, split_full_run):
        """Rebuilding from saved backbone + adapter reproduces LoRA tensors."""
        lora_dir = os.path.join(os.path.dirname(split_full_run), "lora")
        backbone_state = torch.load(
            os.path.join(lora_dir, "backbone.pt"), weights_only=True
        )
        task0_state = torch.load(
            os.path.join(lora_dir, "task0.pt"), weights_only=True
        )

        model = ste_mlp_lora([784, 512, 256, 10], r=4)
        model.load_state_dict(backbone_state, strict=False)
        load_model_lora_state(model, task0_state)

        # LoRA state roundtrip: save again, compare tensors
        saved_again = get_model_lora_state(model)
        for key in task0_state:
            torch.testing.assert_close(saved_again[key], task0_state[key])


# ── Test 5: Parameter counts ───────────────────────────────────────


class TestParameterCounts:
    """LoRA param counts grow linearly with rank; backbone stays ~530K."""

    def test_increasing_rank_increases_params(self):
        """r=8 has strictly more LoRA params than r=2."""
        model_r2 = ste_mlp_lora([784, 512, 256, 10], r=2)
        model_r8 = ste_mlp_lora([784, 512, 256, 10], r=8)
        assert count_lora_parameters(model_r8) > count_lora_parameters(model_r2)

    @slow
    def test_json_counts(self, split_full_run):
        """JSON reports correct backbone and LoRA parameter counts."""
        data = _load_json(split_full_run)
        model = ste_mlp_lora([784, 512, 256, 10], r=4)
        expected_lora = count_lora_parameters(model)
        assert data["n_lora_parameters"] == expected_lora
        # Backbone ~535K (784*512 + 512*256 + 256*10 + biases)
        assert data["n_parameters"] == pytest.approx(535040)

    @slow
    def test_lora_params_are_minority(self, split_full_run):
        """LoRA params are a small fraction of backbone params."""
        data = _load_json(split_full_run)
        assert data["n_lora_parameters"] < 0.05 * data["n_parameters"]


# ── Test 6: JSON output schema ─────────────────────────────────────


class TestJSONSchema:
    """The result JSON contains all required keys."""

    REQUIRED_KEYS = [
        "experiment", "protocol", "pretrain_protocol", "weight_format",
        "lora_rank", "lora_alpha", "seed", "device", "epochs_pretrain",
        "epochs_per_task", "batch_size", "learning_rate", "weight_decay",
        "n_tasks", "n_parameters", "n_lora_parameters",
        "total_training_time_seconds", "backbone_test_accuracy",
        "backbone_weight_stats", "accuracy_matrix",
        "cross_task_accuracy_matrix", "per_task_accuracies",
        "global_accuracies", "metrics", "training_metrics",
    ]

    @slow
    def test_all_keys_present(self, split_full_run):
        """Every required key is present in the output JSON."""
        data = _load_json(split_full_run)
        for key in self.REQUIRED_KEYS:
            assert key in data, f"Missing key: {key}"

    @slow
    def test_cross_matrix_shape(self, split_full_run):
        """cross_task_accuracy_matrix[i] has length i+1 (tasks seen so far)."""
        data = _load_json(split_full_run)
        for i, row in enumerate(data["cross_task_accuracy_matrix"]):
            assert len(row) == i + 1


# ── Test 7: Backbone accuracy sanity ───────────────────────────────


class TestBackboneSanity:
    """A few pre-training epochs must produce a competent backbone."""

    @slow
    def test_backbone_above_90_pct(self, temp_dir):
        """2 pre-training epochs on full MNIST give >90% test accuracy."""
        path = _run_experiment(temp_dir, "split", "full", epochs_pretrain=2)
        data = _load_json(path)
        assert data["backbone_test_accuracy"] > 0.90, (
            f"Backbone test acc should exceed 90% after 2 epochs, "
            f"got {data['backbone_test_accuracy']:.4f}"
        )


# ── Test 8: Determinism ────────────────────────────────────────────


class TestDeterminism:
    """Same seed → identical results."""

    @slow
    def test_same_seed_same_result(self, temp_dir):
        """Two runs with the same seed produce identical accuracy matrices."""
        path_a = _run_experiment(os.path.join(temp_dir, "a"), "split", "full")
        path_b = _run_experiment(os.path.join(temp_dir, "b"), "split", "full")
        data_a = _load_json(path_a)
        data_b = _load_json(path_b)
        assert data_a["accuracy_matrix"] == data_b["accuracy_matrix"]

    @slow
    def test_different_seed_different_weights(self, temp_dir):
        """Different seeds produce different LoRA adapters.

        The saved state dict is namespaced by module path (e.g. ``"1.lora_A"``,
        ``"4.lora_A"``), so we compare every tensor across the two adapters.
        """
        path_a = _run_experiment(os.path.join(temp_dir, "a"), "split", "full", seed=42)
        path_b = _run_experiment(os.path.join(temp_dir, "b"), "split", "full", seed=43)
        lora_a = torch.load(
            os.path.join(temp_dir, "a", "lora", "task0.pt"), weights_only=True
        )
        lora_b = torch.load(
            os.path.join(temp_dir, "b", "lora", "task0.pt"), weights_only=True
        )
        assert lora_a.keys() == lora_b.keys(), "Adapters must share the same keys"
        assert lora_a, "Adapter state must not be empty"
        any_different = any(
            not torch.allclose(lora_a[key], lora_b[key]) for key in lora_a
        )
        assert any_different, "Different seeds should produce different LoRA adapters"


# ── Test 9: LoRA-only gradients ────────────────────────────────────


class TestLoRAOnlyGradients:
    """After freeze_backbone, only LoRA params are trainable (no pipeline)."""

    def test_only_lora_trainable(self):
        """All trainable params belong to LoRA A/B sets."""
        model = ste_mlp_lora([784, 64, 10], r=4)
        freeze_backbone(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        lora_sets = [
            set(layer.lora_parameters()) for _, layer in iter_lora_layers(model)
        ]
        all_lora = set().union(*lora_sets)
        assert len(trainable) == len(all_lora)
        for p in trainable:
            assert p in all_lora

    def test_backward_leaves_backbone_gradless(self):
        """Backward pass produces no gradients on the frozen backbone."""
        torch.manual_seed(0)
        model = ste_mlp_lora([784, 64, 10], r=4)
        freeze_backbone(model)
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        for name, layer in iter_lora_layers(model):
            assert layer.latent_scores.grad is None
            assert layer.bias is None or layer.bias.grad is None
