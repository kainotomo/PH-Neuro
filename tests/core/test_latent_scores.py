"""Tests for LatentScoreTensor."""

from __future__ import annotations

import torch

from ph_neuro.core.latent_scores import LatentScoreTensor


class TestLatentScoreTensor:
    """Suite of tests for LatentScoreTensor."""

    def test_create(self):
        """Creating a LatentScoreTensor should produce correct shape and dtype."""
        t = LatentScoreTensor((10, 20))
        assert t.scores.shape == (10, 20)
        assert t.scores.dtype == torch.float16

    def test_random_init(self):
        """Initial scores should be small random values near zero."""
        t = LatentScoreTensor((100, 100), init_std=0.1)
        mean = t.scores.mean().item()
        assert -0.5 < mean < 0.5, f"Mean {mean} is too far from zero"

    def test_get_ternary_below_threshold(self):
        """Scores below theta_upper should yield zero weights."""
        t = LatentScoreTensor((3, 3))
        t.scores = torch.tensor(
            [[0.5, -0.3, 0.1], [0.0, -0.5, 0.2], [0.3, -0.1, 0.4]],
            dtype=torch.float16,
        )
        ternary = t.get_ternary(theta_upper=5.0, theta_lower=1.0)
        assert torch.all(ternary == 0), "All weights should be zero"

    def test_get_ternary_above_threshold(self):
        """Scores above theta_upper should activate."""
        t = LatentScoreTensor((2, 2))
        t.scores = torch.tensor(
            [[6.0, -7.0], [2.0, -3.0]],
            dtype=torch.float16,
        )
        ternary = t.get_ternary(theta_upper=5.0, theta_lower=1.0)
        assert ternary[0, 0] == 1, "Positive score above threshold should activate"
        assert ternary[0, 1] == -1, "Negative score below -threshold should activate"
        assert ternary[1, 0] == 0, "Score below threshold should not activate"
        assert ternary[1, 1] == 0, "Score below threshold should not activate"

    def test_apply_hebbian_correlated(self):
        """Correlated pre/post should increase the score."""
        t = LatentScoreTensor((2, 2))
        t.scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[1, 1]], dtype=torch.int8)
        post = torch.tensor([[1, 0]], dtype=torch.int8)
        old = t.scores[0, 0].clone()
        t.apply_hebbian(pre, post, lr=0.1)
        assert t.scores[0, 0] > old, "Correlated pair should strengthen"

    def test_apply_hebbian_anticorrelated(self):
        """Anti-correlated pre/post should decrease the score."""
        t = LatentScoreTensor((2, 2))
        t.scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[1, -1]], dtype=torch.int8)
        post = torch.tensor([[-1, 1]], dtype=torch.int8)
        old = t.scores[0, 0].clone()
        t.apply_hebbian(pre, post, lr=0.1)
        assert t.scores[0, 0] < old, "Anti-correlated pair should weaken"

    def test_apply_decay(self):
        """Decay should move scores toward zero."""
        t = LatentScoreTensor((2, 2))
        t.scores = torch.full((2, 2), 10.0, dtype=torch.float16)
        t.apply_decay(decay_rate=0.1)
        assert torch.all(t.scores < 10.0), "Decay should reduce scores"
        assert torch.all(t.scores > 0), "Scores should stay positive"

    def test_clone(self):
        """Clone should create an independent copy."""
        t = LatentScoreTensor((3, 3))
        t.scores[0, 0] = 5.0
        c = t.clone()
        assert c.scores[0, 0] == 5.0
        c.scores[0, 0] = -5.0
        assert t.scores[0, 0] == 5.0  # Original unchanged

    # --- Hysteresis tests for get_ternary(current=...) ---

    def test_hysteresis_activation_threshold(self):
        """Score above theta_upper activates a zero weight; score in the hysteresis gap does not."""
        t = LatentScoreTensor((3, 3))
        t.scores = torch.zeros((3, 3), dtype=torch.float16)
        current = torch.zeros((3, 3), dtype=torch.int8)

        # Push one score well above theta_upper
        t.scores[0, 0] = 6.0  # theta_upper = 5.0
        ternary = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=current)
        assert ternary[0, 0] == 1, "Score above upper threshold should activate"

        # Now lower score to the hysteresis gap (between theta_lower and theta_upper)
        t.scores[0, 0] = 3.0  # < 5.0, > 1.0
        ternary = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=ternary)
        assert ternary[0, 0] == 1, (
            "Score in hysteresis gap should keep existing weight active"
        )

    def test_hysteresis_deactivation_threshold(self):
        """Active weight deactivates only when score falls below theta_lower, not before."""
        t = LatentScoreTensor((3, 3))
        t.scores = torch.full((3, 3), 6.0, dtype=torch.float16)
        current = torch.full((3, 3), 1, dtype=torch.int8)  # all active

        # Drop scores to the hysteresis gap (between theta_lower and theta_upper)
        t.scores[:] = 3.0  # < 5.0, > 1.0
        ternary = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=current)
        assert torch.all(ternary == 1), (
            "Score in hysteresis gap should keep weight active"
        )

        # Drop scores below theta_lower
        t.scores[:] = 0.5  # < 1.0
        ternary = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=ternary)
        assert torch.all(ternary == 0), (
            "Score below lower threshold should deactivate"
        )

    def test_hysteresis_no_oscillation(self):
        """Constant input should not cause weight flips after convergence.

        Simulate: score oscillates near the upper threshold. The weight
        should activate once and stay active, not flip back to 0.
        """
        t = LatentScoreTensor((1, 1))
        t.scores = torch.zeros((1, 1), dtype=torch.float16)
        current = torch.zeros((1, 1), dtype=torch.int8)

        # Step 1: score crosses upper threshold → activate
        t.scores[0, 0] = 6.0
        w = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=current)
        assert w[0, 0] == 1, "Should activate on first crossing"

        # Steps 2-10: score oscillates near but above theta_lower
        for val in [4.5, 3.0, 4.8, 2.5, 4.2, 3.5, 4.0, 2.0, 3.8]:
            t.scores[0, 0] = val
            w = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=w)
            assert w[0, 0] == 1, (
                f"Weight should stay active when score={val} (gap: 1.0-5.0)"
            )

        # Step 11: score finally drops below theta_lower → deactivate
        t.scores[0, 0] = 0.5
        w = t.get_ternary(theta_upper=5.0, theta_lower=1.0, current=w)
        assert w[0, 0] == 0, "Should deactivate when score < theta_lower"
