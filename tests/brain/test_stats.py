"""E031 stats regression tests — verify units + the protocol §6 statistics.

`BrainWrapper.evaluate()` returns **per-window sum NLL** (nats over the
window's tokens). `block_paired_stats` must convert to per-token means for
the paired t/Cohen's d and use token-weighted aggregation for ppl — mixing
those up overflows or biases the numbers (regression: E031 smoke
OverflowError from ``sum_nll × tokens``).
"""

from __future__ import annotations

import math

import pytest

from ph_neuro.brain.stats import block_paired_stats, cross_seed_summary


def _synthetic_blocks(n: int, mean_nll: float, sd: float, seed: int = 0, tok: int = 256):
    """Per-window sum-NLL lists: mean_nll per token, window length ``tok``."""
    import random

    rng = random.Random(seed)
    vals = [rng.gauss(mean_nll, sd) for _ in range(n)]
    sums = [v * tok for v in vals]
    tokens = [tok] * n
    return sums, tokens


class TestUnits:
    def test_no_overflow_with_realistic_sums(self):
        # mean ~2.5 nats/token (ppl ~12), 500 windows — the E031 regime.
        # 0.05 nats/token improvement ≈ 0.6 ppl on ppl 12 (d≈0.17) → strong.
        sums, tokens = _synthetic_blocks(500, 2.5, 0.3, tok=256)
        plastic_sums = [s - 0.05 * t for s, t in zip(sums, tokens)]  # plastic better
        out = block_paired_stats(sums, plastic_sums, tokens)
        assert "error" not in out
        assert math.isfinite(out["delta_ppl"])
        assert out["delta_ppl"] > 0  # plastic better → positive Δppl
        assert out["paired_p"] < 0.05
        assert 0 < out["delta_ppl_ci95"][0] < out["delta_ppl_ci95"][1]

    def test_aggregate_ppl_matches_manual(self):
        sums, tokens = _synthetic_blocks(100, 2.4, 0.25, tok=256)
        plastic_sums = list(sums)
        out = block_paired_stats(sums, plastic_sums, tokens)
        expected = math.exp(sum(sums) / sum(tokens))
        assert out["ppl_frozen"] == pytest.approx(expected, rel=1e-12)
        assert out["ppl_plastic"] == pytest.approx(expected, rel=1e-12)
        assert out["delta_ppl"] == pytest.approx(0.0, abs=1e-12)

    def test_identical_blocks_give_zero_effect(self):
        sums, tokens = _synthetic_blocks(50, 2.2, 0.4, tok=512)
        out = block_paired_stats(sums, sums, tokens)
        assert out["paired_t"] == 0.0
        assert out["paired_p"] == 1.0
        assert out["cohens_d"] == 0.0
        assert out["delta_ppl"] == pytest.approx(0.0, abs=1e-12)

    def test_variable_window_lengths_handled(self):
        sums = [10.0 * 500, 20.0 * 300, 30.0 * 200]  # different lengths
        tokens = [500, 300, 200]
        plastic_sums = [s - 1.0 * t for s, t in zip(sums, tokens)]
        out = block_paired_stats(sums, plastic_sums, tokens)
        assert math.isfinite(out["delta_ppl"])
        assert out["delta_ppl"] > 0

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            block_paired_stats([1.0, 2.0], [1.0], [2, 2])

    def test_mismatched_token_length_raises(self):
        with pytest.raises(ValueError):
            block_paired_stats([1.0, 2.0], [1.0, 2.0], [2])


class TestCrossSeed:
    def test_positive_delta_gives_low_p(self):
        per_seed = [
            {"target_ppl_delta": 0.55},
            {"target_ppl_delta": 0.60},
            {"target_ppl_delta": 0.50},
        ]
        out = cross_seed_summary(per_seed, "target_ppl_delta")
        assert out["n"] == 3
        assert out["mean"] == pytest.approx(0.55, abs=1e-9)
        assert out["p"] < 0.05
        assert out["cohens_d"] > 1.0

    def test_zero_delta_gives_p_one(self):
        per_seed = [{"target_ppl_delta": 0.0}] * 3
        out = cross_seed_summary(per_seed, "target_ppl_delta")
        assert out["p"] == 1.0
        assert out["t"] == 0.0
