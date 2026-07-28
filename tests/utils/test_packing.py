"""Tests for weight packing utilities."""

from __future__ import annotations

import torch

from ph_neuro.utils.packing import pack_ternary, unpack_ternary


class TestPacking:
    """Suite of tests for pack_ternary and unpack_ternary."""

    def test_pack_basic(self):
        """Pack a simple tensor and check size."""
        w = torch.tensor([1, 0, -1, 1], dtype=torch.int8)
        packed = pack_ternary(w)
        # 4 weights → 1 byte
        assert packed.shape == (1,), f"Expected 1 byte, got {packed.shape}"

    def test_pack_roundtrip(self):
        """Pack → unpack should recover the original tensor."""
        w = torch.tensor([1, 0, -1, 1, 0, -1, 1, 0], dtype=torch.int8)
        packed = pack_ternary(w)
        unpacked = unpack_ternary(packed, w.shape)
        assert torch.equal(unpacked, w)

    def test_pack_roundtrip_large(self):
        """Roundtrip on a larger random tensor."""
        w = torch.randint(0, 3, (100,), dtype=torch.int8) - 1  # {-1, 0, +1}
        packed = pack_ternary(w)
        unpacked = unpack_ternary(packed, w.shape)
        assert torch.equal(unpacked, w)

    def test_pack_2d_roundtrip(self):
        """Roundtrip on a 2D tensor."""
        w = torch.tensor(
            [[1, 0, -1, 1], [0, -1, 1, 0], [-1, 1, 0, -1]],
            dtype=torch.int8,
        )
        packed = pack_ternary(w)
        unpacked = unpack_ternary(packed, w.shape)
        assert torch.equal(unpacked, w)

    def test_pack_invalid_values(self):
        """Packing values outside {-1, 0, +1} should raise ValueError."""
        w = torch.tensor([2, 0, -1], dtype=torch.int8)
        try:
            pack_ternary(w)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_unpack_shape(self):
        """Unpacked tensor should have the requested shape."""
        packed = pack_ternary(torch.tensor([1, 0, -1, 1, 0, -1, 1, 0], dtype=torch.int8))
        unpacked = unpack_ternary(packed, (2, 4))
        assert unpacked.shape == (2, 4)
