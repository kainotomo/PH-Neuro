"""Unit tests for SurpriseModulator — EMA, sigmoid, float32 precision.

Covers the locked Step 0.3 defaults (α=0.99, s₀=0.05, k=60, M_max=1.0) and
the constant-M ablation mode.
"""

from __future__ import annotations

import math

import pytest
import torch

from ph_neuro.brain.modulator import SurpriseModulator

ALPHA, S0, K, M_MAX = 0.99, 0.05, 60.0, 1.0


def sigmoid(s: float) -> float:
    return M_MAX / (1.0 + math.exp(-K * (s - S0)))


class TestEMA:
    def test_first_update_sets_ema_to_loss(self):
        m = SurpriseModulator()
        s, M = m.update(1.0)  # noqa: N806 - spec notation
        assert m.initialized is True
        assert m.ema_loss is not None
        assert float(m.ema_loss) == pytest.approx(1.0)
        # s = (L - L_hat)/L_hat with L_hat == L → 0
        assert s == pytest.approx(0.0)

    def test_ema_converges_to_constant_loss(self):
        m = SurpriseModulator()
        for _ in range(1000):
            m.update(1.5)
        assert float(m.ema_loss) == pytest.approx(1.5, abs=1e-6)

    def test_ema_tracks_step_change(self):
        m = SurpriseModulator()
        for _ in range(500):
            m.update(1.0)
        baseline = float(m.ema_loss)
        m.update(1.0)
        assert float(m.ema_loss) == pytest.approx(baseline, abs=1e-6)
        # a jump moves the EMA by (1-alpha) of the gap
        m.update(2.0)
        moved = float(m.ema_loss)
        assert moved > baseline
        assert moved == pytest.approx(
            baseline + (1 - ALPHA) * (2.0 - baseline), rel=1e-3
        )

    def test_reset_clears_state(self):
        m = SurpriseModulator()
        m.update(1.0)
        m.reset()
        assert m.initialized is False
        assert m.ema_loss is None


class TestSurpriseSigmoid:
    def test_s_at_zero_gives_baseline_m(self):
        # s = 0 → M = sigmoid(-k·s0) ≈ 0.047 (small, "no learning")
        m = SurpriseModulator()
        s, M = m.update(1.0)  # noqa: N806 - first step sets L_hat = L → s = 0
        assert pytest.approx(sigmoid(0.0), abs=1e-6) == M

    def test_midpoint_s0_gives_half_max(self):
        # EMA is updated BEFORE s is computed (locked ordering), so feed the
        # L that yields s exactly = s0 given the EMA update: with L_hat→1.0,
        # L = alpha·(1+s0) / (alpha − s0·(1−alpha)).
        m = SurpriseModulator()
        for _ in range(100):
            m.update(1.0)
        l_target = ALPHA * (1.0 + S0) / (ALPHA - S0 * (1.0 - ALPHA))
        s, M = m.update(l_target)  # noqa: N806 - spec notation
        assert s == pytest.approx(S0, abs=1e-3)
        assert pytest.approx(0.5, abs=1e-3) == M

    def test_strong_surprise_saturates(self):
        m = SurpriseModulator()
        for _ in range(100):
            m.update(1.0)
        L_hat = float(m.ema_loss)  # noqa: N806
        _, M = m.update(L_hat * (1.0 + 0.2))  # noqa: N806
        assert pytest.approx(1.0, abs=1e-3) == M  # noqa: N806

    def test_bounded(self):
        m = SurpriseModulator()
        for _ in range(100):
            m.update(1.0)
        L_hat = float(m.ema_loss)  # noqa: N806
        for mult in (0.5, 0.9, 1.0, 1.1, 1.5, 2.0):
            _, M = m.update(L_hat * mult)  # noqa: N806
            assert 0.0 <= M <= 1.0 + 1e-9  # noqa: N806

    def test_in_domain_m_is_low(self):
        """Stationary loss → s ≈ 0 → M near the floor (protects source)."""
        m = SurpriseModulator()
        m.update(1.0)
        ms = [m.update(1.0 + 0.02 * math.sin(i))[1] for i in range(200)]
        assert float(sum(ms) / len(ms)) < 0.3


class TestPrecision:
    def test_float32_end_to_end(self):
        m = SurpriseModulator()
        for _ in range(100):
            m.update(1.0)
        L_hat = float(m.ema_loss)  # noqa: N806
        # Feed a bf16 loss (as the frozen model would produce) — the
        # modulator must upcast to float32 so M does not underflow to 0.
        loss_bf16 = torch.tensor(L_hat * (1.0 + S0), dtype=torch.bfloat16)
        s, M = m.update(loss_bf16)  # noqa: N806
        # bf16 input must NOT underflow M to 0 — it stays a usable signal.
        assert M > 0.3  # noqa: N806
        assert 0.0 < M < 0.7  # noqa: N806
        assert isinstance(s, float) and isinstance(M, float)

    def test_ema_stored_float32(self):
        m = SurpriseModulator()
        m.update(1.0)
        assert m.ema_loss.dtype == torch.float32


class TestConstantMode:
    def test_constant_returns_fixed_m(self):
        m = SurpriseModulator(mode="constant", constant_M=1.0)
        for _ in range(5):
            s, M = m.update(3.7)  # noqa: N806 - spec notation
            assert s == 0.0
            assert M == 1.0

    def test_constant_other_value(self):
        m = SurpriseModulator(mode="constant", constant_M=0.5)
        assert m.update(1.0)[1] == 0.5

    def test_constant_does_not_touch_ema(self):
        m = SurpriseModulator(mode="constant")
        m.update(1.0)
        assert m.initialized is False
        assert m.ema_loss is None


class TestConfig:
    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError):
            SurpriseModulator.from_config({"mode": "surprise_ema", "bogus": 1})

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError):
            SurpriseModulator(mode="nope")

    def test_from_config_defaults(self):
        m = SurpriseModulator.from_config(None)
        assert m.mode == "surprise_ema"
        assert (m.alpha, m.s0, m.k, m.M_max) == (ALPHA, S0, K, M_MAX)

    def test_state_roundtrip(self):
        m = SurpriseModulator()
        for _ in range(10):
            m.update(1.0)
        state = m.state_dict()
        m2 = SurpriseModulator()
        m2.load_state_dict(state)
        assert m2.initialized is True
        assert float(m2.ema_loss) == pytest.approx(float(m.ema_loss))
        # identical next update
        assert m2.update(1.0)[1] == pytest.approx(m.update(1.0)[1])
