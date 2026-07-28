"""Ternary activation function.

Maps any tensor to {-1, 0, +1} using a sign-based quantization.
Supports an optional epsilon dead-zone to suppress small values.
"""

from __future__ import annotations

import torch


def ternary_sign(x: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """Map activations to {-1, 0, +1}.

    With ``epsilon=0`` (default), ``torch.sign`` behavior applies: positive
    values become +1, negative values become -1, and exact zeros stay 0.

    With ``epsilon > 0``, values in the interval ``(-epsilon, +epsilon)``
    are mapped to 0 before the sign is taken, creating a dead-zone that
    suppresses small-magnitude activations.

    Args:
        x: Input tensor of any shape and dtype.
        epsilon: Values in ``(-epsilon, +epsilon)`` map to 0.
            Must be non-negative.

    Returns:
        int8 tensor with values in {-1, 0, +1}, same shape as ``x``.

    Raises:
        ValueError: If ``epsilon`` is negative.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    if epsilon > 0:
        x = torch.where(torch.abs(x) < epsilon, torch.zeros_like(x), x)
    return torch.sign(x).to(torch.int8)
