"""Tests for TernaryHebbianConv2d placeholder."""

from __future__ import annotations

import pytest

from ph_neuro.layers.conv import TernaryHebbianConv2d


class TestTernaryHebbianConv2d:
    """Suite of tests — currently all raise NotImplementedError."""

    def test_create(self):
        """Creating the layer should work."""
        layer = TernaryHebbianConv2d(3, 64, kernel_size=3)
        assert layer is not None

    def test_forward_not_implemented(self):
        """Forward should raise NotImplementedError in Phase 0."""
        layer = TernaryHebbianConv2d(3, 64, kernel_size=3)
        with pytest.raises(NotImplementedError):
            import torch

            layer(torch.randn(1, 3, 32, 32))
