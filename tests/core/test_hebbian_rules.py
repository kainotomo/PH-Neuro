"""Tests for Hebbian plasticity rules."""

from __future__ import annotations

import torch

from ph_neuro.core.hebbian_rules import (
    anti_hebbian_update,
    bcm_update,
    hebbian_update,
    oja_update,
)


class TestHebbianRules:
    """Suite of tests for Hebbian update rules."""

    def test_hebbian_correlated_increase(self):
        """Basic Hebbian: correlated pre/post should increase scores."""
        scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[1, 1]], dtype=torch.int8)
        post = torch.tensor([[1, 0]], dtype=torch.int8)
        result = hebbian_update(scores, pre, post, lr=0.1)
        assert result[0, 0] > 0, "Correlated pair should increase"
        assert result[1, 1] == 0, "Uncorrelated pair should stay zero"

    def test_hebbian_anticorrelated_decrease(self):
        """Basic Hebbian: anti-correlated pre/post should decrease scores."""
        scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[1, -1]], dtype=torch.int8)
        post = torch.tensor([[-1, 1]], dtype=torch.int8)
        result = hebbian_update(scores, pre, post, lr=0.1)
        # pre[0]=+1, post[0]=-1 → anti-correlated → decrease
        # pre[1]=-1, post[1]=+1 → anti-correlated → decrease
        assert result[0, 0] < 0, "Anti-correlated should decrease"
        assert result[1, 1] < 0, "Anti-correlated should decrease"

    def test_hebbian_silent_no_update(self):
        """If either pre or post is 0, no update should occur."""
        scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[0, 1]], dtype=torch.int8)
        post = torch.tensor([[1, 0]], dtype=torch.int8)
        result = hebbian_update(scores, pre, post, lr=0.1)
        assert result[0, 0] == 0, "Silent pre → no update"
        assert result[1, 1] == 0, "Silent post → no update"

    def test_anti_hebbian_inverts_sign(self):
        """Anti-Hebbian should invert the sign of the update."""
        scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[1, 1]], dtype=torch.int8)
        post = torch.tensor([[1, 1]], dtype=torch.int8)
        basic = hebbian_update(scores.clone(), pre, post, lr=0.1)
        anti = anti_hebbian_update(scores.clone(), pre, post, lr=0.1)
        assert torch.allclose(basic, -anti), "Anti-Hebbian should invert"

    def test_oja_normalization(self):
        """Oja's rule should produce smaller scores than basic Hebbian."""
        scores = torch.ones((2, 2), dtype=torch.float16) * 5.0
        pre = torch.tensor([[1, 1]], dtype=torch.int8)
        post = torch.tensor([[1, 1]], dtype=torch.int8)
        basic = hebbian_update(scores.clone(), pre, post, lr=0.1)
        oja = oja_update(scores.clone(), pre, post, lr=0.1)
        assert oja[0, 0] < basic[0, 0], "Oja should normalize"

    def test_bcm_modulation(self):
        """BCM rule: theta_m should modulate the update direction."""
        scores = torch.zeros((2, 2), dtype=torch.float16)
        pre = torch.tensor([[1, 1]], dtype=torch.int8)
        post = torch.tensor([[1, 1]], dtype=torch.int8)

        # post=+1, theta_m=+2 → modulated = +1 * (+1 - 2) = -1 → LTD
        bcm_high = bcm_update(scores.clone(), pre, post, lr=0.1, theta_m=2.0)
        assert bcm_high[0, 0] < 0, "BCM with high theta_m should depress"

        # post=+1, theta_m=-2 → modulated = +1 * (+1 - (-2)) = +3 → LTP
        bcm_low = bcm_update(scores.clone(), pre, post, lr=0.1, theta_m=-2.0)
        assert bcm_low[0, 0] > 0, "BCM with low theta_m should potentiate"
