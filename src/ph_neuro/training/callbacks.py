"""Callback utilities for Hebbian training.

Provides logging, checkpointing, and metric tracking callbacks
for use with :class:`~ph_neuro.training.trainer.HebbianTrainer` and
:class:`~ph_neuro.training.supervised.SupervisedHebbianClassifier`.
"""

from __future__ import annotations

from typing import Any

import torch

from ph_neuro.layers.linear import TernaryHebbianLinear


class MetricsTracker:
    """Collects and reports training metrics.

    Args:
        metrics: List of metric names to track.
    """

    def __init__(self, metrics: list[str] | None = None):
        self._metrics: dict[str, list[float]] = {m: [] for m in (metrics or [])}

    def log(self, epoch: int, epoch_metrics: dict[str, float]) -> None:
        """Log metrics for one epoch.

        Args:
            epoch: Current epoch number.
            epoch_metrics: Metric values for this epoch.
        """
        for key, value in epoch_metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all tracked metrics."""
        return {k: {"values": v, "mean": sum(v) / len(v)} for k, v in self._metrics.items()}


class WeightDistributionCallback:
    """Logs ternary weight distribution and flip rate after each step.

    Attaches to the model and tracks:
    - Percentage of weights at +1, -1, and 0
    - Average flip rate per epoch (fraction of weights that change per step)

    Args:
        model: A module containing ``TernaryHebbianLinear`` layers.
    """

    def __init__(self, model: torch.nn.Module):
        self._layers: list[TernaryHebbianLinear] = []
        for module in model.modules():
            if isinstance(module, TernaryHebbianLinear):
                self._layers.append(module)
        self._prev_weights: dict[int, torch.Tensor] = {}
        self._step_flip_rates: list[float] = []

    def on_step_begin(self) -> None:
        """Snapshot weights before refresh (called before each step)."""
        self._prev_weights.clear()
        for i, layer in enumerate(self._layers):
            self._prev_weights[i] = layer.weight.unpack().clone()

    def on_step_end(self) -> dict[str, float]:
        """Compute metrics after refresh.

        Returns:
            Dict with pos_pct, neg_pct, zero_pct, and flip_rate.
        """
        pos = 0.0
        neg = 0.0
        zero = 0.0
        total = 0
        total_flips = 0

        for i, layer in enumerate(self._layers):
            w = layer.weight.unpack()
            total += w.numel()
            pos += (w == 1).sum().item()
            neg += (w == -1).sum().item()
            zero += (w == 0).sum().item()

            if i in self._prev_weights:
                total_flips += (self._prev_weights[i] != w).sum().item()

        t = max(total, 1)
        flip_rate = total_flips / t

        self._step_flip_rates.append(flip_rate)

        return {
            "pos_pct": 100.0 * pos / t,
            "neg_pct": 100.0 * neg / t,
            "zero_pct": 100.0 * zero / t,
            "flip_rate": flip_rate,
        }

    def get_avg_flip_rate(self) -> float:
        """Average flip rate across all recorded steps."""
        if not self._step_flip_rates:
            return 0.0
        return sum(self._step_flip_rates) / len(self._step_flip_rates)
