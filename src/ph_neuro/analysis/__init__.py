"""Analysis and visualization tools for Hebbian networks."""

from ph_neuro.analysis.continual import (
    compute_forgetting_metric,
    evaluate_continual_learning,
)
from ph_neuro.analysis.visualization import (
    plot_latent_score_distribution,
    plot_weight_histogram,
    visualize_filters,
)

__all__ = [
    "plot_weight_histogram",
    "visualize_filters",
    "plot_latent_score_distribution",
    "evaluate_continual_learning",
    "compute_forgetting_metric",
]
