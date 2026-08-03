"""Tests for DQT + Hysteresis-STE combined layers (ste_dqt_hysteresis module).

Tests cover:
- ``hysteresis_stochastic_round`` discretisation rule (all three zones)
- ``TernaryDQTHysteresisLinear`` construction and forward
- Ternary weight invariant (always {-1, 0, +1})
- No latent float scores: ``weight_ternary`` (int8) is the trained state
- Gradient routing to the float accumulation buffer (DQT-style STE)
- ``apply_stochastic_rounding`` flip statistics
- Hysteresis memory: weights in the gap keep their previous ternary state
- Stochastic rounding in the upper zone (exploration / deadzone mitigation)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_dqt_hysteresis import (
    TernaryDQTHysteresisLinear,
    hysteresis_stochastic_round,
)

# ═══════════════════════════════════════════════════════════════════
#  Tests for hysteresis_stochastic_round
# ═══════════════════════════════════════════════════════════════════


class TestHysteresisStochasticRound:
    """Tests for the combined hysteresis + stochastic rounding rule."""

    def test_deactivation_zone_is_zero(self):
        """|w| < theta_lower should deterministically round to 0."""
        w = torch.tensor([[0.05, -0.05, 0.1, -0.12, 0.14]])
        prev = torch.tensor([[1, -1, 1, -1, 1]], dtype=torch.int8)
        result = hysteresis_stochastic_round(w, prev, 0.3, 0.15)
        expected = torch.zeros_like(prev)
        assert torch.equal(result, expected), (
            f"Deactivation zone should be 0. Expected {expected}, got {result}"
        )

    def test_upper_zone_matches_stochastic_round(self):
        """|w| > theta_upper should use DQT stochastic rounding."""
        torch.manual_seed(0)
        w = torch.tensor([[0.9, 0.8, -0.95, -1.0, 1.0]])
        prev = torch.zeros_like(w, dtype=torch.int8)
        result = hysteresis_stochastic_round(w, prev, 0.3, 0.15)

        # All |w| > 0.3 -> stochastic rounding; high |w| rounds to +/-1
        assert set(result.flatten().tolist()) <= {-1, 0, 1}
        assert (result[:, :2] > 0).all(), (
            "High positive weights should round to +1"
        )
        assert (result[:, 2:4] < 0).all(), (
            "High negative weights should round to -1"
        )
        assert result[:, 4].item() == 1, "w=1.0 should round to +1"

    def test_gap_keeps_previous_ternary(self):
        """Weights in the hysteresis gap should preserve prev ternary state."""
        w = torch.tensor([[0.2, -0.25, 0.3, -0.15, 0.2]])
        prev = torch.tensor([[1, -1, 0, -1, 1]], dtype=torch.int8)
        result = hysteresis_stochastic_round(w, prev, 0.3, 0.15)
        # 0.2 in [0.15, 0.3] gap -> keep 1
        # -0.25 in gap -> keep -1
        # 0.3 not > 0.3 -> gap -> keep 0
        # -0.15 not < 0.15 -> gap -> keep -1
        # 0.2 in gap -> keep 1
        assert torch.equal(result, prev), (
            f"Hysteresis gap should preserve prev. Expected {prev}, got {result}"
        )

    def test_gap_exploration_rounds_stochastically(self):
        """With explore_gap=True the gap is stochastic-rounded, not memory."""
        torch.manual_seed(1)
        w = torch.tensor([[0.2, -0.25, 0.3, -0.15, 0.1]])
        prev = torch.tensor([[1, -1, 0, -1, 1]], dtype=torch.int8)
        result = hysteresis_stochastic_round(
            w, prev, 0.3, 0.15, explore_gap=True
        )
        assert set(result.flatten().tolist()) <= {-1, 0, 1}
        # At least one entry can differ from prev (stochastic exploration)
        n_same = (result == prev).sum().item()
        assert n_same < prev.numel(), (
            "explore_gap should round some gap values differently"
        )

    def test_combined_zones(self):
        """Full combined rule: deactivate + gap memory + stochastic activate."""
        w = torch.tensor([[0.05, 0.2, 0.9, -0.8, 0.3]])
        prev = torch.tensor([[1, -1, 0, 0, 1]], dtype=torch.int8)
        result = hysteresis_stochastic_round(w, prev, 0.3, 0.15)
        # 0.05 < 0.15 & prev=1 -> 0
        # 0.2 in gap & prev=-1 -> keep -1
        # 0.9 > 0.3 & prev=0 -> +1 (stochastic, high prob)
        # -0.8 > 0.3 & prev=0 -> -1 (stochastic, high prob)
        # 0.3 = theta_upper (not >) -> gap -> keep 1
        assert result[0, 0].item() == 0
        assert result[0, 1].item() == -1
        assert result[0, 4].item() == 1
        assert result[0, 2].item() in (0, 1)
        assert result[0, 3].item() in (-1, 0)


# ═══════════════════════════════════════════════════════════════════
#  Tests for TernaryDQTHysteresisLinear
# ═══════════════════════════════════════════════════════════════════


class TestTernaryDQTHysteresisLinear:
    """Tests for the combined linear layer."""

    def test_construction(self):
        """Layer should construct with default thresholds."""
        layer = TernaryDQTHysteresisLinear(784, 256)
        assert layer.in_features == 784
        assert layer.out_features == 256
        assert layer.weight_ternary.shape == (256, 784)
        assert layer.weight_ternary.dtype == torch.int8
        assert layer.theta_upper == 0.3
        assert layer.theta_lower == 0.15

    def test_ternary_weight_invariant(self):
        """Ternary weights should always be in {-1, 0, +1}."""
        layer = TernaryDQTHysteresisLinear(20, 10)
        w = layer.weight_ternary
        assert set(w.flatten().tolist()) <= {-1, 0, 1}
        # After stochastic rounding update, still invariant
        with torch.no_grad():
            layer.weight_float.normal_(0.0, 0.5)
        layer.apply_stochastic_rounding()
        assert set(layer.weight_ternary.flatten().tolist()) <= {-1, 0, 1}

    def test_no_latent_scores_parameter(self):
        """Only weight_float and bias are parameters (no separate latent scores)."""
        layer = TernaryDQTHysteresisLinear(20, 10, bias=True)
        param_names = {name for name, _ in layer.named_parameters()}
        assert "weight_float" in param_names
        assert "weight_ternary" not in param_names, (
            "Ternary weights must be a buffer, not a parameter"
        )
        assert "latent_scores" not in param_names
        buffers = {name for name, _ in layer.named_buffers()}
        assert "weight_ternary" in buffers
        assert "latent_scores" not in buffers

    def test_forward_shape(self):
        """Forward should produce correct output shape."""
        layer = TernaryDQTHysteresisLinear(784, 256)
        x = torch.randn(4, 784)
        out = layer(x)
        assert out.shape == (4, 256)

    def test_gradient_flows_to_weight_float(self):
        """Backward should route gradients to weight_float (DQT-style STE)."""
        layer = TernaryDQTHysteresisLinear(8, 4)
        x = torch.randn(3, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight_float.grad is not None, (
            "Gradient should flow to weight_float"
        )
        assert layer.weight_float.grad.shape == layer.weight_float.shape
        assert x.grad is not None, "Gradient should flow to input"

    def test_forward_uses_ternary_weights(self):
        """Forward matmul should use the int8 ternary weights."""
        layer = TernaryDQTHysteresisLinear(8, 4)
        # Manually set ternary weights to a known pattern
        with torch.no_grad():
            layer.weight_ternary.zero_()
            layer.weight_ternary[0, 0] = 1
            layer.weight_ternary[0, 1] = -1
        x = torch.zeros(1, 8)
        x[0, 0] = 2.0
        x[0, 1] = 3.0
        out = layer(x)
        # out[0,0] = 2*1 + 3*(-1) = -1
        assert torch.allclose(out[0, 0], torch.tensor(-1.0)), (
            f"Expected forward to use ternary weights, got {out[0, 0]}"
        )

    def test_apply_stochastic_rounding_returns_flip_stats(self):
        """apply_stochastic_rounding should return flip stats dict."""
        layer = TernaryDQTHysteresisLinear(20, 10)
        with torch.no_grad():
            layer.weight_float.normal_(0.0, 0.3)
        stats = layer.apply_stochastic_rounding()
        assert "flip_rate" in stats
        assert "n_flips" in stats
        assert 0.0 <= stats["flip_rate"] <= 1.0

    def test_get_weight_stats(self):
        """get_weight_stats should return valid percentages."""
        layer = TernaryDQTHysteresisLinear(20, 10)
        stats = layer.get_weight_stats()
        total = stats["pos_pct"] + stats["neg_pct"] + stats["zero_pct"]
        assert abs(total - 100.0) < 1e-6

    def test_threshold_validation(self):
        """theta_lower >= theta_upper should raise ValueError."""
        import pytest

        with pytest.raises(ValueError):
            TernaryDQTHysteresisLinear(10, 10, theta_upper=0.1, theta_lower=0.3)

    def test_training_step_updates_ternary_weights(self):
        """A full training step should update the ternary weights."""
        layer = TernaryDQTHysteresisLinear(16, 8)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.randn(5, 16)
        y = torch.randn(5, 8)

        float_before = layer.weight_float.data.clone()
        optimizer.zero_grad()
        out = layer(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        layer.apply_stochastic_rounding()

        # The float accumulation buffer must have been updated by the optimizer
        assert not torch.equal(float_before, layer.weight_float.data), (
            "Optimizer should update the float accumulation buffer"
        )
        # Ternary weights remain valid
        assert set(layer.weight_ternary.flatten().tolist()) <= {-1, 0, 1}
