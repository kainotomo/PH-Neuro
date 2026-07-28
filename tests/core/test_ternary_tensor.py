"""Tests for TernaryTensor."""

from __future__ import annotations

import pytest
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

    def test_to_dense_device_consistency(self):
        """to_dense should return a tensor on the same device as the data."""
        t = TernaryTensor((3, 4), packed=False)
        t._data[0, 0] = 1
        dense = t.to_dense()
        assert dense.device == t._data.device

    def test_pack_noncontiguous(self):
        """Packing a non-contiguous tensor should work correctly."""
        w = torch.tensor(
            [[1, 0, -1, 1], [0, -1, 1, 0], [-1, 1, 0, -1]],
            dtype=torch.int8,
        )
        transposed = w.T  # non-contiguous
        packed = TernaryTensor.pack(transposed)
        unpacked = packed.unpack()
        assert torch.equal(unpacked, transposed)

    def test_unpack_packed_shape(self):
        """Unpacking a packed tensor should yield the original shape."""
        w = torch.tensor([1, 0, -1, 1, 0, -1], dtype=torch.int8)
        packed = TernaryTensor.pack(w)
        unpacked = packed.unpack()
        assert unpacked.shape == w.shape

    def test_unpack_returns_int8(self):
        """Unpack should always return an int8 tensor."""
        t = TernaryTensor((5, 5), packed=False)
        t._data[0, 0] = 1
        unpacked = t.unpack()
        assert unpacked.dtype == torch.int8

        t_packed = TernaryTensor.pack(
            torch.tensor([[1, 0, -1], [0, 1, -1]], dtype=torch.int8)
        )
        unpacked_packed = t_packed.unpack()
        assert unpacked_packed.dtype == torch.int8

    def test_data_property_reflects_internal_state(self):
        """The data property should match the internal storage."""
        t = TernaryTensor((4, 4), packed=False)
        t._data[0, 0] = 1
        t._data[1, 1] = -1
        assert t.data[0, 0] == 1
        assert t.data[1, 1] == -1
        assert t.data is t._data

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_on_cuda(self):
        """TernaryTensor should be creatable on a CUDA device."""
        t = TernaryTensor((10, 20), packed=False, device="cuda")
        assert t.data.device.type == "cuda"
        assert t.data.dtype == torch.int8

        t_packed = TernaryTensor((10, 20), packed=True, device="cuda")
        assert t_packed.data.device.type == "cuda"
