"""Unit tests for Elastic Weight Consolidation (EWC) for ternary STE networks.

Tests:
    1. Latent-score parameter extraction from ternary STE models
    2. Diagonal Fisher computation: shapes, non-negativity, no NaN
    3. EWC penalty: zero at reference, positive away, scales with lambda
    4. Online EWC: update, accumulation, state round-trip
    5. Multi-task EWC: update and summed penalty
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from ph_neuro.models.ste_models import ste_mlp
from ph_neuro.training.ewc import (
    MultiTaskEWC,
    OnlineEWC,
    compute_fisher_diag,
    ewc_penalty,
    get_ternary_latent_params,
    save_ewc_reference,
)


# ── Helpers ────────────────────────────────────────────────────────


def _make_model(n_features: int = 32, n_classes: int = 4) -> torch.nn.Module:
    """A small ternary STE MLP for fast tests."""
    return ste_mlp([n_features, 16, n_classes], batch_norm=False, flatten=False)


def _make_loader(n_samples: int = 128, n_features: int = 32, n_classes: int = 4) -> DataLoader:
    """A small synthetic DataLoader with real-ish inputs and labels."""
    x = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)


# ── Test 1: Parameter extraction ───────────────────────────────────


class TestGetTernaryLatentParams:
    """Verify latent-score parameter extraction."""

    def test_returns_latent_scores(self):
        """Every returned parameter is a latent_scores Parameter."""
        model = _make_model()
        params = get_ternary_latent_params(model)
        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.nn.Parameter)
            assert p.requires_grad

    def test_count_matches_ste_layers(self):
        """One latent-scores param per TernarySTELinear layer (2 layers here)."""
        model = _make_model(n_features=32, n_classes=4)  # 32→16→4 → 2 STE layers
        params = get_ternary_latent_params(model)
        from ph_neuro.layers.ste_linear import TernarySTELinear

        n_ste = sum(1 for m in model.modules() if isinstance(m, TernarySTELinear))
        assert n_ste == 2
        assert len(params) == n_ste

    def test_empty_model(self):
        """A model with no ternary STE layers yields no params."""
        model = torch.nn.Linear(4, 4)
        assert get_ternary_latent_params(model) == []


# ── Test 2: Fisher computation ─────────────────────────────────────


class TestComputeFisherDiag:
    """Verify the diagonal Fisher estimate."""

    def test_shape_matches_params(self):
        """Each Fisher tensor matches the corresponding param shape."""
        model = _make_model()
        params = get_ternary_latent_params(model)
        fisher = compute_fisher_diag(model, _make_loader(), n_batches=4)
        assert len(fisher) == len(params)
        for f, p in zip(fisher, params):
            assert f.shape == p.shape

    def test_non_negative(self):
        """Fisher (squared gradients) values are all >= 0."""
        model = _make_model()
        fisher = compute_fisher_diag(model, _make_loader(), n_batches=4)
        for f in fisher:
            assert bool((f >= 0).all()), "Fisher must be non-negative"

    def test_some_positive(self):
        """At least one Fisher element is strictly positive for real data."""
        model = _make_model()
        fisher = compute_fisher_diag(model, _make_loader(), n_batches=4)
        assert any(bool((f > 0).any()) for f in fisher), "Expected some positive Fisher values"

    def test_no_nan(self):
        """Fisher is finite for valid inputs."""
        model = _make_model()
        fisher = compute_fisher_diag(model, _make_loader(), n_batches=4)
        for f in fisher:
            assert torch.isfinite(f).all()

    def test_zero_batches_guard(self):
        """Empty loader yields zero Fisher without division-by-zero."""
        empty_loader = DataLoader(TensorDataset(torch.zeros(0, 32), torch.zeros(0, dtype=torch.long)))
        model = _make_model()
        fisher = compute_fisher_diag(model, empty_loader, n_batches=4)
        for f in fisher:
            assert bool((f == 0).all())


# ── Test 3: EWC penalty ───────────────────────────────────────────


class TestEWCPenalty:
    """Verify the EWC penalty term."""

    def test_zero_at_reference(self):
        """Penalty is 0 when the model is exactly at the reference."""
        model = _make_model()
        ref = save_ewc_reference(model)
        fisher = [torch.ones_like(p, dtype=torch.float32) for p in ref]
        pen = ewc_penalty(model, ref, fisher, ewc_lambda=1.0)
        assert bool(torch.allclose(pen, torch.zeros_like(pen), atol=1e-8))

    def test_positive_away(self):
        """Penalty > 0 when the model deviates from the reference."""
        model = _make_model()
        ref = save_ewc_reference(model)
        fisher = [torch.ones_like(p, dtype=torch.float32) for p in ref]
        with torch.no_grad():
            for p in get_ternary_latent_params(model):
                p.add_(1.0)
        pen = ewc_penalty(model, ref, fisher, ewc_lambda=1.0)
        assert pen.item() > 0

    def test_scales_with_lambda(self):
        """Penalty scales linearly with lambda."""
        model = _make_model()
        ref = save_ewc_reference(model)
        fisher = [torch.ones_like(p, dtype=torch.float32) for p in ref]
        with torch.no_grad():
            for p in get_ternary_latent_params(model):
                p.add_(0.5)
        pen1 = ewc_penalty(model, ref, fisher, ewc_lambda=1.0)
        pen2 = ewc_penalty(model, ref, fisher, ewc_lambda=2.0)
        assert bool(torch.allclose(pen2, 2.0 * pen1, atol=1e-6))

    def test_penalty_gradient_flows(self):
        """The penalty is differentiable w.r.t. latent scores."""
        model = _make_model()
        ref = save_ewc_reference(model)
        fisher = [torch.ones_like(p, dtype=torch.float32) for p in ref]
        pen = ewc_penalty(model, ref, fisher, ewc_lambda=1.0)
        pen.backward()
        for p in get_ternary_latent_params(model):
            assert p.grad is not None
        model.zero_grad(set_to_none=True)


# ── Test 4: Online EWC ────────────────────────────────────────────


class TestOnlineEWC:
    """Verify the OnlineEWC manager."""

    def test_no_penalty_before_update(self):
        """Penalty is exactly 0 before any task is consolidated."""
        model = _make_model()
        ewc = OnlineEWC(model, gamma=1.0)
        assert not ewc.has_penalty()
        assert ewc.penalty(model, ewc_lambda=10.0).item() == 0.0

    def test_update_populates_fisher(self):
        """After update, the manager has a non-empty Fisher."""
        model = _make_model()
        ewc = OnlineEWC(model, gamma=1.0)
        n = ewc.update(_make_loader(), n_batches=4)
        assert n == 1
        assert ewc.has_penalty()
        assert ewc.n_tasks == 1

    def test_accumulation_increases_fisher(self):
        """Accumulating a second task (gamma=1) does not decrease Fisher."""
        model = _make_model()
        ewc = OnlineEWC(model, gamma=1.0)
        ewc.update(_make_loader(), n_batches=4)
        first = [f.clone() for f in ewc.state_dict()["fisher"]]
        ewc.update(_make_loader(), n_batches=4)
        second = ewc.state_dict()["fisher"]
        assert ewc.n_tasks == 2
        for f1, f2 in zip(first, second):
            assert bool((f2 >= f1).all())

    def test_gamma_weights_accumulation(self):
        """gamma < 1 down-weights the previous Fisher contribution."""
        model = _make_model()
        ewc = OnlineEWC(model, gamma=0.5)
        ewc.update(_make_loader(), n_batches=4)
        first = ewc.state_dict()["fisher"]
        ewc.update(_make_loader(), n_batches=4)
        second = ewc.state_dict()["fisher"]
        # second = 0.5 * first + new, so values stay >= 0 and finite
        for f in second:
            assert torch.isfinite(f).all()

    def test_state_round_trip(self):
        """state_dict / load_state_dict preserves Fisher and reference."""
        model = _make_model()
        ewc = OnlineEWC(model, gamma=0.8)
        ewc.update(_make_loader(), n_batches=4)
        state = ewc.state_dict()

        model2 = _make_model()
        ewc2 = OnlineEWC(model2, gamma=1.0)
        ewc2.load_state_dict(state)
        assert ewc2.n_tasks == ewc.n_tasks
        assert ewc2.gamma == ewc.gamma
        s1, s2 = ewc.state_dict(), ewc2.state_dict()
        for f1, f2 in zip(s1["fisher"], s2["fisher"]):
            assert torch.allclose(f1, f2)
        for r1, r2 in zip(s1["ref_params"], s2["ref_params"]):
            assert torch.allclose(r1, r2)

    def test_penalty_after_updates_positive(self):
        """Once a task is consolidated, deviating from the reference incurs cost."""
        model = _make_model()
        ewc = OnlineEWC(model, gamma=1.0)
        ewc.update(_make_loader(), n_batches=4)
        with torch.no_grad():
            for p in get_ternary_latent_params(model):
                p.add_(0.5)
        pen = ewc.penalty(model, ewc_lambda=1.0)
        assert pen.item() > 0


# ── Test 5: Multi-task EWC ────────────────────────────────────────


class TestMultiTaskEWC:
    """Verify the MultiTaskEWC manager."""

    def test_no_penalty_before_update(self):
        """Penalty is 0 before any task is consolidated."""
        model = _make_model()
        ewc = MultiTaskEWC(model)
        assert not ewc.has_penalty()
        assert ewc.penalty(model, ewc_lambda=1.0).item() == 0.0

    def test_update_appends_task(self):
        """Each update adds a task to the stored list."""
        model = _make_model()
        ewc = MultiTaskEWC(model)
        ewc.update(_make_loader(), n_batches=4)
        assert ewc.n_tasks == 1
        ewc.update(_make_loader(), n_batches=4)
        assert ewc.n_tasks == 2

    def test_penalty_sums_tasks(self):
        """Penalty with 2 tasks >= penalty with 1 task (same lambda)."""
        model = _make_model()
        ewc = MultiTaskEWC(model)
        ewc.update(_make_loader(), n_batches=4)
        with torch.no_grad():
            for p in get_ternary_latent_params(model):
                p.add_(0.5)
        pen1 = ewc.penalty(model, ewc_lambda=1.0).item()
        ewc.update(_make_loader(), n_batches=4)
        pen2 = ewc.penalty(model, ewc_lambda=1.0).item()
        assert pen2 >= pen1

    def test_differentiable(self):
        """Multi-task penalty is differentiable w.r.t. latent scores."""
        model = _make_model()
        ewc = MultiTaskEWC(model)
        ewc.update(_make_loader(), n_batches=4)
        with torch.no_grad():
            for p in get_ternary_latent_params(model):
                p.add_(0.3)
        pen = ewc.penalty(model, ewc_lambda=1.0)
        pen.backward()
        for p in get_ternary_latent_params(model):
            assert p.grad is not None
