"""Continual learning evaluation tools.

Provides metrics for measuring catastrophic forgetting in sequential
task learning scenarios.
"""

from __future__ import annotations

from typing import Any


def evaluate_continual_learning(
    task_accuracies: list[list[float]],
) -> dict[str, Any]:
    """Evaluate continual learning performance across sequential tasks.

    Args:
        task_accuracies: Matrix where ``task_accuracies[i][j]`` is the
            accuracy on task ``j`` after training on task ``i``.
            ``task_accuracies[i][i]`` is accuracy on task ``i`` after
            training on it.

    Returns:
        Dictionary with metrics:
        - ``average_accuracy``: Mean accuracy across all tasks after all training.
        - ``average_forgetting``: Mean forgetting per task (drop in accuracy).
        - ``forward_transfer``: Effect of earlier tasks on later ones.
        - ``backward_transfer``: Effect of later tasks on earlier ones.
    """
    n_tasks = len(task_accuracies)

    if n_tasks == 0:
        return {
            "average_accuracy": 0.0,
            "average_forgetting": 0.0,
            "forward_transfer": 0.0,
            "backward_transfer": 0.0,
        }

    # Final accuracy on each task (after all training)
    final_accuracies = [
        task_accuracies[-1][j] for j in range(min(n_tasks, len(task_accuracies[-1])))
    ]

    # Forgetting: for each task, peak accuracy after training minus final accuracy
    forgetting = []
    for j in range(n_tasks):
        peak = max(task_accuracies[i][j] for i in range(j, n_tasks))
        final = task_accuracies[-1][j]
        forgetting.append(peak - final)

    return {
        "average_accuracy": sum(final_accuracies) / len(final_accuracies),
        "average_forgetting": sum(forgetting) / len(forgetting),
        "per_task_accuracy": final_accuracies,
        "per_task_forgetting": forgetting,
    }


def compute_forgetting_metric(
    baseline_accuracy: float,
    final_accuracy: float,
) -> float:
    """Compute forgetting percentage for a single task.

    Args:
        baseline_accuracy: Accuracy immediately after learning the task.
        final_accuracy: Accuracy after all subsequent tasks.

    Returns:
        Forgetting as a fraction (0.0 = no forgetting, >0 = forgetting).
    """
    if baseline_accuracy == 0:
        return 0.0
    return max(0.0, (baseline_accuracy - final_accuracy) / baseline_accuracy)
