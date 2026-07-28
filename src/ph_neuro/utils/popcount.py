"""Popcount-based matrix multiplication for ternary {-1, 0, +1} tensors.

Ternary MatMul reduces to ``popcount(x AND w_pos) - popcount(x AND w_neg)``
where positive and negative weights are separated into binary masks.

This module provides a reference PyTorch implementation. A fused CUDA
kernel will be developed in Phase 4 for production use.
"""

from __future__ import annotations

import torch


def popcount_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Popcount-based matrix multiplication for ternary tensors.

    Computes ``x @ w.T`` where both inputs are in {-1, 0, +1} using
    bitwise operations.

    The algorithm:
    1. Separate positive and negative weights into binary masks
    2. ``out = popcount(x=+1 AND w=+1) + popcount(x=-1 AND w=-1)
            - popcount(x=+1 AND w=-1) - popcount(x=-1 AND w=+1)``

    Args:
        x: Input tensor, shape ``(batch, in_features)``, values in {-1, 0, +1}.
        w: Weight tensor, shape ``(out_features, in_features)``, values in {-1, 0, +1}.

    Returns:
        Output tensor, shape ``(batch, out_features)``.

    Note:
        This is a reference implementation using float ops for correctness.
        A fast bitwise version will replace it in Phase 4.
    """
    return x.float() @ w.float().T
