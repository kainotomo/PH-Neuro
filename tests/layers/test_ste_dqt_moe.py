"""Unit tests for the TernaryDQTMoELayer (E019 pilot MoE)."""

from __future__ import annotations

import pytest
import torch

from ph_neuro.layers.ste_dqt import TernaryDQTLinear, stochastic_round
from ph_neuro.layers.ste_dqt_moe import TernaryDQTMoELayer


@pytest.fixture
def moe_layer() -> TernaryDQTMoELayer:
    return TernaryDQTMoELayer(
        in_features=32, expert_width=16, n_experts=4, top_k=2,
    )


# ── Construction / validation ───────────────────────────────────────


def test_construction(moe_layer: TernaryDQTMoELayer) -> None:
    assert moe_layer.n_experts == 4
    assert moe_layer.top_k == 2
    assert len(moe_layer.experts) == 4
    assert all(isinstance(e, TernaryDQTLinear) for e in moe_layer.experts)
    # Router is float linear 32 → 4
    assert moe_layer.router.in_features == 32
    assert moe_layer.router.out_features == 4


def test_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        TernaryDQTMoELayer(32, 16, n_experts=4, top_k=5)
    with pytest.raises(ValueError):
        TernaryDQTMoELayer(32, 16, n_experts=4, top_k=0)


# ── Forward / routing ───────────────────────────────────────────────


def test_forward_shape_and_dtype(moe_layer: TernaryDQTMoELayer) -> None:
    x = torch.randn(8, 32)
    out = moe_layer(x)
    assert out.shape == (8, 16)
    assert torch.isfinite(out).all()


def test_only_top_k_experts_run(moe_layer: TernaryDQTMoELayer) -> None:
    """Each sample's output should equal the weighted sum of exactly K experts."""
    x = torch.randn(16, 32)
    combined, logits, indices, weights = moe_layer(x, return_routing=True)

    # Manually compute the same result by running ALL experts and masking
    all_out = torch.stack([e(x) for e in moe_layer.experts], dim=-1)  # (B, W, N)
    probs = torch.softmax(logits, dim=-1)
    w = probs.gather(1, indices)  # (B, K)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

    manual = torch.zeros_like(combined)
    for k in range(moe_layer.top_k):
        manual += all_out[torch.arange(x.size(0)), :, indices[:, k]] * w[:, k].unsqueeze(-1)

    assert torch.allclose(combined, manual, atol=1e-5)


def test_gradient_flows_to_router_and_experts(moe_layer: TernaryDQTMoELayer) -> None:
    x = torch.randn(8, 32)
    out = moe_layer(x)
    loss = out.sum()
    loss.backward()

    assert moe_layer.router.weight.grad is not None
    assert torch.isfinite(moe_layer.router.weight.grad).all()
    for expert in moe_layer.experts:
        assert expert.weight_float.grad is not None
        assert torch.isfinite(expert.weight_float.grad).all()


# ── Load balancing tracking ─────────────────────────────────────────


def test_selection_fractions_sum_to_one(moe_layer: TernaryDQTMoELayer) -> None:
    torch.manual_seed(0)
    for _ in range(5):
        x = torch.randn(32, 32)
        moe_layer(x)
    fracs = moe_layer.selection_fractions()
    assert fracs.shape == (4,)
    assert torch.allclose(fracs.sum(), torch.tensor(1.0), atol=1e-5)


def test_coverage_fractions_bounded(moe_layer: TernaryDQTMoELayer) -> None:
    torch.manual_seed(1)
    for _ in range(5):
        moe_layer(torch.randn(16, 32))
    cov = moe_layer.coverage_fractions()
    # Each expert is active on at most top_k/n_experts = 0.5 of samples (ideal)
    assert ((cov >= 0) & (cov <= 1)).all()
    assert torch.allclose(cov.sum(), torch.tensor(float(moe_layer.top_k)), atol=1e-4)


def test_reset_usage_stats(moe_layer: TernaryDQTMoELayer) -> None:
    moe_layer(torch.randn(16, 32))
    moe_layer.reset_usage_stats()
    assert moe_layer.n_selections.item() == 0
    assert moe_layer.n_samples.item() == 0
    assert (moe_layer.selection_counts == 0).all()


def test_uniform_router_gives_uniform_selection() -> None:
    """A near-uniform router should produce balanced selection shares."""
    layer = TernaryDQTMoELayer(32, 16, n_experts=4, top_k=2)
    torch.manual_seed(3)
    for _ in range(200):
        layer(torch.randn(64, 32))
    fracs = layer.selection_fractions()
    # With random inputs and a small random router, shares stay within 0.1 of ideal
    assert torch.allclose(fracs, torch.full((4,), 0.25), atol=0.10)


# ── Aux load balancing loss ─────────────────────────────────────────


def test_aux_loss_requires_grad_and_is_positive(moe_layer: TernaryDQTMoELayer) -> None:
    x = torch.randn(16, 32)
    moe_layer(x)
    aux = moe_layer.aux_load_balance_loss()
    assert aux.requires_grad
    assert aux.item() >= 1.0
    aux.backward()
    assert moe_layer.router.weight.grad is not None


def test_aux_loss_lower_for_balanced_than_collapsed() -> None:
    """Uniform routing → aux loss near 1.0; collapsed routing → higher."""
    torch.manual_seed(0)
    balanced = TernaryDQTMoELayer(32, 16, n_experts=4, top_k=2)
    for _ in range(50):
        balanced(torch.randn(64, 32))
    aux_balanced = balanced.aux_load_balance_loss().item()

    # Force collapse: all inputs select the same expert(s)
    collapsed = TernaryDQTMoELayer(32, 16, n_experts=4, top_k=2)
    with torch.no_grad():
        collapsed.router.weight.zero_()
        collapsed.router.weight[0, :] = 10.0  # always pick experts 0, 1
        collapsed.router.weight[1, :] = 5.0
    for _ in range(50):
        collapsed(torch.randn(64, 32))
    aux_collapsed = collapsed.aux_load_balance_loss().item()

    assert aux_balanced < aux_collapsed


# ── DQT interaction ─────────────────────────────────────────────────


def test_expert_stochastic_rounding_works(moe_layer: TernaryDQTMoELayer) -> None:
    """apply_stochastic_rounding keeps expert weights ternary int8."""
    for expert in moe_layer.experts:
        stats = expert.apply_stochastic_rounding()
        assert set(expert.weight_ternary.unique().tolist()).issubset({-1, 0, 1})
        assert expert.weight_ternary.dtype == torch.int8
        assert 0.0 <= stats["flip_rate"] <= 1.0


def test_weight_stats(moe_layer: TernaryDQTMoELayer) -> None:
    stats = moe_layer.get_weight_stats()
    assert {"pos_pct", "neg_pct", "zero_pct"} == set(stats.keys())
    total = stats["pos_pct"] + stats["neg_pct"] + stats["zero_pct"]
    assert abs(total - 100.0) < 1e-6


def test_count_parameters(moe_layer: TernaryDQTMoELayer) -> None:
    counts = moe_layer.count_parameters()
    assert counts["router"] == 32 * 4
    assert counts["experts"] == 4 * (32 * 16)
    assert counts["total"] == counts["router"] + counts["experts"]
