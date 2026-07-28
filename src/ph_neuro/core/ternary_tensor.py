"""Ternary weight representation.

Stores weights as {-1, 0, +1} in two modes:
- **Naive (Phases 0-2)**: 1 byte per weight (int8) — simple and debuggable.
- **Packed (Phases 3+)** : 4 weights per byte (2-bit encoding) — memory efficient.

The public API is identical regardless of internal storage mode.
"""

from __future__ import annotations

import torch


class TernaryTensor:
    """Storage for ternary weights {-1, 0, +1}.

    Args:
        shape: Tensor shape (typically ``(out_features, in_features)``).
        packed: If ``True``, use 2-bit packed storage (4 weights/byte).
            If ``False`` (default), use naive int8 storage (1 weight/byte).
        device: Torch device for the underlying tensor.

    Attributes:
        data: Underlying tensor — int8 in naive mode, or bit-packed in packed mode.
        shape: Shape of the logical ternary tensor.
        packed: Whether packed storage is active.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        packed: bool = False,
        device: torch.device | str | None = None,
    ):
        self._shape = shape
        self._packed = packed
        if packed:
            # Packed: 4 weights per int8 byte, ceil(prod(shape) / 4) elements
            n_bytes = (self._numel() + 3) // 4
            self._data: torch.Tensor = torch.zeros(n_bytes, dtype=torch.int8, device=device)
        else:
            # Naive: 1 weight per int8 byte
            self._data: torch.Tensor = torch.zeros(shape, dtype=torch.int8, device=device)

    @property
    def data(self) -> torch.Tensor:
        """Underlying storage tensor."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def packed(self) -> bool:
        return self._packed

    def _numel(self) -> int:
        out = 1
        for s in self._shape:
            out *= s
        return out

    def unpack(self) -> torch.Tensor:
        """Convert to a dense {-1, 0, +1} int8 tensor.

        Returns:
            A tensor of shape ``self.shape`` with values in {-1, 0, +1}.
        """
        if not self._packed:
            return self._data.clone()

        from ph_neuro.utils.packing import unpack_ternary

        return unpack_ternary(self._data, self._shape)

    def to_dense(self) -> torch.Tensor:
        """Return a float tensor for use in matrix multiplication.

        Returns:
            A float32 tensor of shape ``self.shape``.
        """
        return self.unpack().to(torch.float32)

    @staticmethod
    def pack(weights: torch.Tensor) -> TernaryTensor:
        """Pack a dense {-1, 0, +1} tensor into 2-bit packed storage.

        Args:
            weights: int8 tensor with values in {-1, 0, +1}.

        Returns:
            A ``TernaryTensor`` in packed mode.
        """
        from ph_neuro.utils.packing import pack_ternary

        packed_data = pack_ternary(weights)
        result = TernaryTensor.__new__(TernaryTensor)
        result._shape = tuple(weights.shape)
        result._packed = True
        result._data = packed_data
        return result

    def clone(self) -> TernaryTensor:
        """Return a deep copy of this tensor."""
        t = TernaryTensor(self._shape, packed=self._packed, device=self._data.device)
        t._data = self._data.clone()
        return t

    def __repr__(self) -> str:
        mode = "packed" if self._packed else "naive"
        return f"TernaryTensor(shape={self._shape}, mode={mode}, dtype=int8)"
