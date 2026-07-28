"""Callback utilities for Hebbian training.

Provides logging, checkpointing, and metric tracking callbacks
for use with :class:`~ph_neuro.training.trainer.HebbianTrainer`.
"""

from __future__ import annotations

from typing import Any


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
