"""Continual learning evaluation tools.

Provides metrics for measuring catastrophic forgetting in sequential
task learning scenarios, plus weight stability and hysteresis analysis
for understanding WHY Hebbian learning resists forgetting.
"""

from __future__ import annotations

from typing import Any

import torch


# ── Core forgetting metrics ────────────────────────────────────────


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


# ── Weight stability analysis ──────────────────────────────────────


def compute_weight_overlap(
    weights_before: torch.Tensor,
    weights_after: torch.Tensor,
) -> dict[str, float]:
    """Compute Jaccard-style overlap between two ternary weight tensors.

    Measures how much of the weight configuration is preserved.

    Args:
        weights_before: Ternary weights from an earlier state, shape ``(out, in)``.
        weights_after: Ternary weights from a later state, same shape.

    Returns:
        Dict with:
        - ``jaccard_similarity``: Fraction of non-zero weights that agree.
          1.0 = identical non-zero sets, 0.0 = completely different.
        - ``flip_rate``: Fraction of all weights that changed value.
        - ``agreement_rate``: Fraction of all weights that are identical.
        - ``n_changed``: Absolute count of changed weights.
    """
    assert weights_before.shape == weights_after.shape, "Shape mismatch"
    total = weights_before.numel()

    changed = (weights_before != weights_after).sum().item()
    agreement = (weights_before == weights_after).sum().item()

    # Jaccard: intersection / union of non-zero sets
    non_zero_before = weights_before != 0
    non_zero_after = weights_after != 0
    intersection = (non_zero_before & non_zero_after).sum().item()
    union = (non_zero_before | non_zero_after).sum().item()
    jaccard = intersection / max(union, 1)

    return {
        "jaccard_similarity": jaccard,
        "flip_rate": changed / max(total, 1),
        "agreement_rate": agreement / max(total, 1),
        "n_changed": changed,
    }


def compute_per_class_weight_stability(
    weight_snapshots: dict[int, torch.Tensor],
) -> dict[str, list[float]]:
    """Track how each output neuron's weights change across tasks.

    Args:
        weight_snapshots: Dict mapping ``task_idx`` to weight tensor
            of shape ``(out_features, in_features)``. One entry per
            task, recorded after training on that task.

    Returns:
        Dict with:
        - ``per_neuron_flip_rate``: ``list[list[float]]`` where
          ``[neuron][task]`` = flip rate of that neuron from task
          ``task`` to ``task + 1``.
        - ``per_neuron_agreement``: Same structure, agreement rate.
    """
    task_indices = sorted(weight_snapshots.keys())
    n_neurons = weight_snapshots[task_indices[0]].shape[0]

    per_neuron_flip: list[list[float]] = [[] for _ in range(n_neurons)]
    per_neuron_agree: list[list[float]] = [[] for _ in range(n_neurons)]

    for i in range(len(task_indices) - 1):
        before = weight_snapshots[task_indices[i]]
        after = weight_snapshots[task_indices[i + 1]]

        for neuron in range(n_neurons):
            w_b = before[neuron]
            w_a = after[neuron]
            total = w_b.numel()
            changed = (w_b != w_a).sum().item()
            agreed = (w_b == w_a).sum().item()
            per_neuron_flip[neuron].append(changed / max(total, 1))
            per_neuron_agree[neuron].append(agreed / max(total, 1))

    return {
        "per_neuron_flip_rate": per_neuron_flip,
        "per_neuron_agreement": per_neuron_agree,
    }


def analyze_hysteresis_protection(
    latent_scores: torch.Tensor,
    theta_upper: float,
    theta_lower: float,
) -> dict[str, float]:
    """Analyze how hysteresis protects weights from flipping.

    Classifies each weight into one of three zones based on its latent
    score and the hysteresis thresholds:

    - **Active-protected** (weight = ±1, |score| between θ_l and θ_u):
      Needs significant counter-evidence to deactivate.
    - **Inactive-potential** (weight = 0, |score| between θ_l and θ_u):
      Was never activated or was deactivated; could reactivate.
    - **Frozen-active** (weight = ±1, |score| > θ_u): Strongly clamped.
    - **Frozen-inactive** (weight = 0, |score| < θ_l): Deeply dormant.

    Args:
        latent_scores: fp16 latent score tensor, shape ``(out, in)``.
        theta_upper: Hysteresis upper threshold.
        theta_lower: Hysteresis lower threshold.

    Returns:
        Dict with percentage of weights in each zone.
    """
    scores = latent_scores.float()
    total = scores.numel()

    score_abs = scores.abs()

    # Estimate where weights would be (ternary sign at current scores)
    # Active if |score| > theta_upper, inactive if |score| < theta_lower,
    # ambiguous if between
    n_above_upper = (score_abs > theta_upper).sum().item()
    n_below_lower = (score_abs < theta_lower).sum().item()
    n_in_gap = total - n_above_upper - n_below_lower

    # Protected: in the hysteresis gap, won't flip without significant change
    gap_pct = 100.0 * n_in_gap / max(total, 1)

    return {
        "pct_above_upper": 100.0 * n_above_upper / max(total, 1),
        "pct_below_lower": 100.0 * n_below_lower / max(total, 1),
        "pct_in_hysteresis_gap": gap_pct,
        "theta_upper": theta_upper,
        "theta_lower": theta_lower,
    }


def analyze_weight_sparsity(
    weights: torch.Tensor,
) -> dict[str, Any]:
    """Analyze sparsity patterns in ternary weights.

    Args:
        weights: Ternary weight tensor, shape ``(out, in)``.

    Returns:
        Dict with sparsity stats per output neuron and globally.
    """
    total = weights.numel()
    n_zero = (weights == 0).sum().item()
    n_pos = (weights == 1).sum().item()
    n_neg = (weights == -1).sum().item()

    per_neuron = {}
    for i in range(weights.shape[0]):
        n = weights[i]
        nt = n.numel()
        per_neuron[f"neuron_{i}"] = {
            "sparsity": 100.0 * (n == 0).sum().item() / max(nt, 1),
            "pos_pct": 100.0 * (n == 1).sum().item() / max(nt, 1),
            "neg_pct": 100.0 * (n == -1).sum().item() / max(nt, 1),
        }

    return {
        "global_sparsity": 100.0 * n_zero / max(total, 1),
        "pos_pct": 100.0 * n_pos / max(total, 1),
        "neg_pct": 100.0 * n_neg / max(total, 1),
        "per_neuron": per_neuron,
    }


def format_analysis_report(
    forgetting_metrics: dict[str, Any],
    weight_overlap: dict[str, float] | None = None,
    hysteresis: dict[str, float] | None = None,
    sparsity: dict[str, Any] | None = None,
) -> str:
    """Format a comprehensive analysis report.

    Args:
        forgetting_metrics: Output from ``evaluate_continual_learning``.
        weight_overlap: Output from ``compute_weight_overlap``.
        hysteresis: Output from ``analyze_hysteresis_protection``.
        sparsity: Output from ``analyze_weight_sparsity``.

    Returns:
        Formatted string report.
    """
    lines = [
        "## Continual Learning Analysis Report",
        "",
        f"Average accuracy: {100 * forgetting_metrics['average_accuracy']:.2f}%",
        f"Average forgetting: {100 * forgetting_metrics['average_forgetting']:.2f}%",
        "",
        "### Per-Task Metrics",
        "| Task | Final Accuracy | Forgetting |",
        "|------|---------------|------------|",
    ]

    for j, (acc, fgt) in enumerate(
        zip(
            forgetting_metrics["per_task_accuracy"],
            forgetting_metrics["per_task_forgetting"],
        )
    ):
        lines.append(f"| {j + 1} | {100 * acc:.2f}% | {100 * fgt:.2f}% |")

    if hysteresis is not None:
        lines.extend(
            [
                "",
                "### Hysteresis Protection",
                f"Weights in hysteresis gap (protected): {hysteresis['pct_in_hysteresis_gap']:.1f}%",
                f"Weights above θ_upper (strongly active): {hysteresis['pct_above_upper']:.1f}%",
                f"Weights below θ_lower (deeply dormant): {hysteresis['pct_below_lower']:.1f}%",
                f"θ_upper = {hysteresis['theta_upper']}, θ_lower = {hysteresis['theta_lower']}",
            ]
        )

    if weight_overlap is not None:
        lines.extend(
            [
                "",
                "### Weight Stability (Last Task Change)",
                f"Jaccard similarity: {weight_overlap['jaccard_similarity']:.4f}",
                f"Flip rate: {100 * weight_overlap['flip_rate']:.2f}%",
                f"Agreement rate: {100 * weight_overlap['agreement_rate']:.2f}%",
            ]
        )

    if sparsity is not None:
        lines.extend(
            [
                "",
                "### Weight Sparsity",
                f"Global sparsity (% zero): {sparsity['global_sparsity']:.1f}%",
                f"+1 weights: {sparsity['pos_pct']:.1f}%",
                f"-1 weights: {sparsity['neg_pct']:.1f}%",
            ]
        )

    return "\n".join(lines)
