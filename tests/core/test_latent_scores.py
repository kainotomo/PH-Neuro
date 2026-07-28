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
