"""Visualization utilities for ternary Hebbian networks.

Provides functions to inspect weight distributions, latent score
evolution, and learned features.
"""

from __future__ import annotations

from typing import Any

import torch


def plot_weight_histogram(
    weights: torch.Tensor,
    title: str = "Ternary Weight Distribution",
) -> dict[str, Any]:
    """Compute histogram data for ternary weight distribution.

    Args:
        weights: int8 tensor with values in {-1, 0, +1}.
        title: Plot title.

    Returns:
        Dictionary with count statistics for each ternary value.
    """
    n_neg = (weights == -1).sum().item()
    n_zero = (weights == 0).sum().item()
    n_pos = (weights == 1).sum().item()
    total = n_neg + n_zero + n_pos

    return {
        "title": title,
        "-1": {"count": n_neg, "pct": 100.0 * n_neg / total if total else 0},
        "0": {"count": n_zero, "pct": 100.0 * n_zero / total if total else 0},
        "+1": {"count": n_pos, "pct": 100.0 * n_pos / total if total else 0},
        "total": total,
    }


def visualize_filters(
    weights: torch.Tensor,
    n_cols: int = 8,
) -> list[dict[str, Any]]:
    """Prepare filter visualization data for a conv layer.

    Args:
        weights: Ternary weight tensor for a conv layer.
        n_cols: Number of columns for display grid.

    Returns:
        List of filter data dicts (placeholder — full impl in Phase 1).
    """
    return []


def plot_latent_score_distribution(
    scores: torch.Tensor,
    title: str = "Latent Score Distribution",
) -> dict[str, Any]:
    """Compute distribution statistics for latent scores.

    Args:
        scores: Float tensor of latent scores.
        title: Plot title.

    Returns:
        Dictionary with distribution statistics.
    """
    return {
        "title": title,
        "mean": scores.mean().item(),
        "std": scores.std().item(),
        "min": scores.min().item(),
        "max": scores.max().item(),
        "n_above_upper": (scores.abs() > 5.0).sum().item(),
        "n_below_lower": (scores.abs() < 1.0).sum().item(),
    }
