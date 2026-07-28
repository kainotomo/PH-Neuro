"""Tests for the ternary activation function."""

from __future__ import annotations

import torch
import pytest

from ph_neuro.core.activation import ternary_sign


class TestTernarySign:
    """Suite of tests for :func:`ternary_sign`."""

    def test_positive_values_become_one(self):
        """Positive values should map to +1."""
        x = torch.tensor([3.0, 0.5, 100.0, 1e-6])
        result = ternary_sign(x)
        assert torch.all(result == 1), f"Expected all +1, got {result}"

    def test_negative_values_become_negative_one(self):
        """Negative values should map to -1."""
        x = torch.tensor([-3.0, -0.5, -100.0, -1e-6])
        result = ternary_sign(x)
        assert torch.all(result == -1), f"Expected all -1, got {result}"

    def test_exact_zero_stays_zero(self):
        """Exact zero should remain 0."""
        x = torch.tensor([0.0])
        result = ternary_sign(x)
        assert result[0] == 0

    def test_mixed_values(self):
        """Mixed positive, negative, and zero values should map correctly."""
        x = torch.tensor([2.0, 0.0, -3.0, 0.5, -0.5, 0.0])
        result = ternary_sign(x)
        expected = torch.tensor([1, 0, -1, 1, -1, 0], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_epsilon_small_values_map_to_zero(self):
        """Values within epsilon of zero should become 0."""
        x = torch.tensor([2.0, 0.1, -0.05, -3.0])
        result = ternary_sign(x, epsilon=0.5)
        expected = torch.tensor([1, 0, 0, -1], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_epsilon_boundary(self):
        """Values exactly at ±epsilon should map to ±1 (boundary inclusive)."""
        x = torch.tensor([0.5, -0.5])
        result = ternary_sign(x, epsilon=0.5)
        expected = torch.tensor([1, -1], dtype=torch.int8)
        assert torch.equal(result, expected), f"Boundary values should be ±1, got {result}"

    def test_epsilon_zero_is_default_behavior(self):
        """epsilon=0 should give standard sign behavior."""
        x = torch.tensor([1.0, -1.0, 0.0, 0.5, -0.5])
        result = ternary_sign(x, epsilon=0.0)
        expected = ternary_sign(x)  # default epsilon=0
        assert torch.equal(result, expected)

    def test_output_dtype_is_int8(self):
        """Output should always be int8."""
        x = torch.tensor([1.0, 0.0, -1.0])
        result = ternary_sign(x)
        assert result.dtype == torch.int8

    def test_preserves_shape(self):
        """Input shape should be preserved in the output."""
        shapes = [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]
        for shape in shapes:
            x = torch.randn(shape)
            result = ternary_sign(x)
            assert result.shape == shape, f"Shape mismatch for {shape}: got {result.shape}"

    def test_large_values(self):
        """Very large positive and negative values should still map correctly."""
        x = torch.tensor([1e10, -1e10])
        result = ternary_sign(x)
        expected = torch.tensor([1, -1], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_no_inplace_modification(self):
        """Input tensor should not be modified in-place."""
        x = torch.tensor([0.3, -0.3])
        x_copy = x.clone()
        _ = ternary_sign(x, epsilon=0.5)
        assert torch.equal(x, x_copy), "Input tensor was modified in-place"

    def test_device_agnostic(self):
        """Should work on CPU (and CUDA if available)."""
        x = torch.tensor([1.0, 0.0, -1.0])
        result = ternary_sign(x)
        assert result.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Should work on CUDA device when available."""
        x = torch.tensor([1.0, 0.0, -1.0], device="cuda")
        result = ternary_sign(x)
        assert result.device.type == "cuda"
        assert torch.equal(result.cpu(), torch.tensor([1, 0, -1], dtype=torch.int8))

    def test_raises_on_negative_epsilon(self):
        """Negative epsilon should raise ValueError."""
        x = torch.tensor([1.0, -1.0])
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            ternary_sign(x, epsilon=-0.1)

    def test_empty_tensor(self):
        """Empty tensor should produce an empty output."""
        x = torch.tensor([], dtype=torch.float32)
        result = ternary_sign(x)
        assert result.shape == (0,)
        assert result.dtype == torch.int8
