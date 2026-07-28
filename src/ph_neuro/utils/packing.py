"""Weight packing utilities for efficient ternary storage.

Converts between naive int8 representation (1 byte per weight) and
packed 2-bit representation (4 weights per byte).

2-bit encoding:
    - ``00`` = 0
    - ``01`` = +1
    - ``10`` = -1
    - ``11`` = unused
"""

from __future__ import annotations

import torch

# Bit patterns for 2-bit encoding
_BITS_ZERO = 0b00  # value 0
_BITS_POS = 0b01  # value +1
_BITS_NEG = 0b10  # value -1


def pack_ternary(weights: torch.Tensor) -> torch.Tensor:
    """Pack a dense {-1, 0, +1} tensor into 2-bit packed storage.

    Args:
        weights: int8 tensor with values in {-1, 0, +1}.

    Returns:
        int8 tensor with ``ceil(numel / 4)`` elements, each byte
        storing 4 ternary weights.

    Raises:
        ValueError: If ``weights`` contains values outside {-1, 0, +1}.
    """
    if not torch.all((weights >= -1) & (weights <= 1)):
        raise ValueError("Weights must contain only values in {-1, 0, +1}")

    flat = weights.flatten().to(torch.int8)
    n = flat.shape[0]
    n_packed = (n + 3) // 4
    packed = torch.zeros(n_packed, dtype=torch.int8, device=weights.device)

    # We need -1 → 10, 0 → 00, +1 → 01
    mapped = torch.where(
        flat == -1,
        torch.tensor(2, dtype=torch.int8),
        torch.where(
            flat == 1, torch.tensor(1, dtype=torch.int8), torch.tensor(0, dtype=torch.int8)
        ),
    )

    for i in range(n_packed):
        byte_val = 0
        for j in range(4):
            idx = i * 4 + j
            if idx < n:
                byte_val |= int(mapped[idx].item()) << (j * 2)
        # Convert to int8 (signed); values >127 wrap to negative
        packed[i] = byte_val if byte_val < 128 else byte_val - 256

    return packed


def unpack_ternary(
    packed: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Unpack a 2-bit packed tensor to a dense {-1, 0, +1} tensor.

    Args:
        packed: int8 tensor in packed format (4 weights per byte).
        shape: Desired output shape. The number of elements must
            match ``packed.numel() * 4`` or less.

    Returns:
        int8 tensor of ``shape`` with values in {-1, 0, +1}.
    """
    n_elements = 1
    for s in shape:
        n_elements *= s

    flat = torch.zeros(n_elements, dtype=torch.int8, device=packed.device)

    for i in range(len(flat)):
        byte_idx = i // 4
        bit_offset = (i % 4) * 2
        bits = (int(packed[byte_idx].item()) >> bit_offset) & 0b11

        # Map 2-bit back to {-1, 0, +1}
        if bits == _BITS_ZERO:
            val = 0
        elif bits == _BITS_POS:
            val = 1
        elif bits == _BITS_NEG:
            val = -1
        else:
            val = 0  # Unused bits → 0

        flat[i] = val

    return flat.reshape(shape)
