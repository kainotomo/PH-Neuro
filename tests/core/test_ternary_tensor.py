"""Tests for TernaryTensor."""

from __future__ import annotations

import torch

from ph_neuro.core.ternary_tensor import TernaryTensor


class TestTernaryTensor:
    """Suite of tests for TernaryTensor storage and conversion."""

    def test_create_naive(self):
        """Creating a naive TernaryTensor should produce the right shape."""
        t = TernaryTensor((10, 20), packed=False)
        assert t.shape == (10, 20)
        assert t.packed is False
        assert t.data.shape == (10, 20)
        assert t.data.dtype == torch.int8

    def test_create_packed(self):
        """Creating a packed TernaryTensor."""
        t = TernaryTensor((10, 20), packed=True)
        assert t.shape == (10, 20)
        assert t.packed is True

    def test_initial_weights_are_zero(self):
        """All ternary weights should start at zero."""
        t = TernaryTensor((5, 5))
        assert torch.all(t.unpack() == 0)

    def test_unpack_naive(self):
        """Unpacking a naive tensor should return the same data."""
        t = TernaryTensor((4, 4), packed=False)
        t._data = torch.tensor(
            [[1, 0, -1, 1], [0, -1, 1, 0], [1, 1, 0, -1], [-1, 0, 1, 0]],
            dtype=torch.int8,
        )
        unpacked = t.unpack()
        assert torch.equal(unpacked, t._data)

    def test_to_dense(self):
        """to_dense should return a float32 copy of the weights."""
        t = TernaryTensor((3, 4), packed=False)
        t._data = torch.tensor(
            [[1, 0, -1, 1], [0, -1, 1, 0], [1, 0, 0, -1]],
            dtype=torch.int8,
        )
        dense = t.to_dense()
        assert dense.dtype == torch.float32
        assert torch.equal(dense, t._data.float())

    def test_pack_static(self, small_ternary_tensor):
        """Static pack method should create a packed TernaryTensor."""
        packed = TernaryTensor.pack(small_ternary_tensor)
        assert packed.packed is True
        assert packed.shape == small_ternary_tensor.shape

    def test_pack_unpack_roundtrip(self, small_ternary_tensor):
        """Pack then unpack should recover the original tensor."""
        packed = TernaryTensor.pack(small_ternary_tensor)
        unpacked = packed.unpack()
        assert torch.equal(unpacked, small_ternary_tensor)

    def test_clone(self):
        """Clone should create an independent copy."""
        t = TernaryTensor((5, 5), packed=False)
        t._data[0, 0] = 1
        c = t.clone()
        assert torch.equal(c._data, t._data)
        c._data[0, 0] = -1
        assert t._data[0, 0] == 1  # original unchanged

    def test_repr(self):
        """String representation should be informative."""
        t = TernaryTensor((3, 4))
        r = repr(t)
        assert "TernaryTensor" in r
        assert "(3, 4)" in r
