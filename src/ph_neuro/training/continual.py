"""Continual learning task infrastructure.

Defines task abstractions and experiment orchestration for measuring
catastrophic forgetting in Hebbian vs backprop networks.

Supports:
- **Split MNIST**: 5 binary tasks (0 vs 1, 2 vs 3, ..., 8 vs 9)
- **Permuted MNIST**: 5 tasks with different random pixel permutations
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from ph_neuro.analysis.continual import evaluate_continual_learning
from ph_neuro.training.data import (
    get_binary_mnist_loaders,
    get_mnist_full_test_loader,
    get_permuted_mnist_loaders,
)


# ── Task data structures ───────────────────────────────────────────


@dataclass
class ContinualTask:
    """A single task in a continual learning sequence.

    Args:
        name: Human-readable task name (e.g. ``"0 vs 1"``).
        train_loader: DataLoader for training on this task.
        test_loaders: Dict mapping task_id -> DataLoader for evaluation.
            ``test_loaders[0]`` is the test set for the first task, etc.
        n_classes: Number of output classes for this task.
        task_id: Index of this task in the sequence.
    """

    name: str
    train_loader: DataLoader
    test_loaders: dict[int, DataLoader]
    n_classes: int
    task_id: int = 0


# ── Task sequence generators ───────────────────────────────────────


def create_split_mnist_tasks(
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
) -> list[ContinualTask]:
    """Create the 5-task Split MNIST sequence.

    Returns:
        List of 5 ``ContinualTask`` objects, one per binary pair.
        Each task has ``test_loaders`` containing all previous tasks'
        test sets, enabling evaluation on all seen tasks.
    """
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks: list[ContinualTask] = []

    # Build all binary loaders first
    all_binary_loaders: list[tuple[DataLoader, DataLoader]] = []
    for a, b in pairs:
        all_binary_loaders.append(
            get_binary_mnist_loaders(a, b, batch_size=batch_size, root=root, num_workers=num_workers)
        )

    # Full 10-class test loader for global accuracy
    full_test_loader = get_mnist_full_test_loader(
        batch_size=batch_size, root=root, num_workers=num_workers
    )

    for task_idx, ((train_loader, _), (a, b)) in enumerate(zip(all_binary_loaders, pairs)):
        # Build test loaders for all tasks up to this one
        test_loaders: dict[int, DataLoader] = {}
        for prev_idx in range(task_idx + 1):
            test_loaders[prev_idx] = all_binary_loaders[prev_idx][1]

        # Attach full 10-class test loader as special key -1
        test_loaders[-1] = full_test_loader

        task = ContinualTask(
            name=f"{a} vs {b}",
            train_loader=train_loader,
            test_loaders=test_loaders,
            n_classes=2,
            task_id=task_idx,
        )
        tasks.append(task)

    return tasks


def create_permuted_mnist_tasks(
    n_tasks: int = 5,
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
    seeds: list[int] | None = None,
) -> list[ContinualTask]:
    """Create a Permuted MNIST task sequence.

    Each task uses all 10 MNIST digits but with a different random
    pixel permutation. The permutation seed determines the mapping.

    Args:
        n_tasks: Number of tasks to generate.
        batch_size: Batch size for all loaders.
        root: Root directory for dataset storage.
        num_workers: Number of data loading workers.
        seeds: Optional list of seeds (one per task). If ``None``,
            defaults to ``list(range(n_tasks))``.

    Returns:
        List of ``ContinualTask`` objects.
    """
    if seeds is None:
        seeds = list(range(n_tasks))

    tasks: list[ContinualTask] = []

    for task_idx in range(n_tasks):
        seed = seeds[task_idx]
        train_loader, test_loader = get_permuted_mnist_loaders(
            perm_seed=seed, batch_size=batch_size, root=root, num_workers=num_workers
        )

        # Build test loaders for all tasks up to this one
        test_loaders: dict[int, DataLoader] = {}
        for prev_idx in range(task_idx + 1):
            prev_seed = seeds[prev_idx]
            _, prev_test = get_permuted_mnist_loaders(
                perm_seed=prev_seed, batch_size=batch_size, root=root, num_workers=num_workers
            )
            test_loaders[prev_idx] = prev_test

        task = ContinualTask(
            name=f"Permute seed={seed}",
            train_loader=train_loader,
            test_loaders=test_loaders,
            n_classes=10,
            task_id=task_idx,
        )
        tasks.append(task)

    return tasks


# ── Evaluation ─────────────────────────────────────────────────────


def evaluate_on_task(
    model: Any,
    loader: DataLoader,
    predict_fn: Callable[[Any, torch.Tensor], torch.Tensor],
) -> float:
    """Evaluate accuracy on a single task's test loader.

    Args:
        model: The model (Hebbian or backprop).
        loader: DataLoader for the task's test set.
        predict_fn: Callable ``predict_fn(model, x) -> predictions``.

    Returns:
        Accuracy as a fraction in [0.0, 1.0].
    """
    correct = 0
    total = 0
    for x, y in loader:
        pred = predict_fn(model, x)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def evaluate_all_tasks(
    model: Any,
    tasks: list[ContinualTask],
    current_task_idx: int,
    predict_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    global_test_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Evaluate model on all tasks up to ``current_task_idx``.

    Args:
        model: The model.
        tasks: All task definitions.
        current_task_idx: Index of the most recently trained task
            (inclusive). Evaluation runs on tasks 0..current_task_idx.
        predict_fn: ``predict_fn(model, x) -> predictions``.
        global_test_loader: Optional 10-class loader for global accuracy
            (used in split MNIST).

    Returns:
        Dict with:
        - ``per_task_accuracy``: ``{task_id: accuracy}`` for each task.
        - ``global_accuracy``: Accuracy on ``global_test_loader`` (if provided).
    """
    results: dict[str, Any] = {
        "per_task_accuracy": {},
    }

    for task_id in range(current_task_idx + 1):
        loader = tasks[task_id].test_loaders.get(task_id)
        if loader is not None:
            acc = evaluate_on_task(model, loader, predict_fn)
            results["per_task_accuracy"][task_id] = acc

    if global_test_loader is not None:
        results["global_accuracy"] = evaluate_on_task(model, global_test_loader, predict_fn)

    return results


# ── Main experiment loop ───────────────────────────────────────────


def run_continual_experiment(
    model: Any,
    tasks: list[ContinualTask],
    train_fn: Callable[[Any, ContinualTask, int], dict[str, Any]],
    predict_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    global_test_loader: DataLoader | None = None,
    record_weight_fn: Callable[[Any, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a full continual learning experiment.

    For each task in sequence:
        1. Train the model on the current task
        2. Evaluate on all tasks seen so far
        3. Record weight statistics (optional)

    Args:
        model: The model (Hebbian or backprop).
        tasks: Sequence of ``ContinualTask`` objects.
        train_fn: ``train_fn(model, task, task_idx) -> metrics_dict``.
            Called once per task. Returns training metrics.
        predict_fn: ``predict_fn(model, x) -> predictions``.
        global_test_loader: Optional 10-class loader for global accuracy
            on split MNIST.
        record_weight_fn: Optional callback to record per-task weight
            snapshots. Signature ``record_weight_fn(model, task_idx)``.
            Return value is stored in ``weight_snapshots[task_idx]``.

    Returns:
        Dict with:
        - ``accuracy_matrix``: ``list[list[float]]`` where
          ``accuracy_matrix[i][j]`` = accuracy on task ``j`` after
          training on task ``i``. Conveniently formatted for
          :func:`~ph_neuro.analysis.continual.evaluate_continual_learning`.
        - ``per_task_experiment_acc``: Same data as ``accuracy_matrix``
          but keyed by task ID for readability.
        - ``global_accuracies``: ``list[float]`` of global 10-class
          accuracy after each task (if ``global_test_loader`` provided).
        - ``metrics``: ``evaluate_continual_learning()`` output.
        - ``weight_snapshots``: ``dict[int, Any]`` of weight stats per
          task (if ``record_weight_fn`` provided).
        - ``training_metrics``: ``list[dict]`` of per-task training metrics.
        - ``n_tasks``: Total number of tasks.
    """
    n_tasks = len(tasks)
    accuracy_matrix: list[list[float]] = []
    global_accuracies: list[float] = []
    weight_snapshots: dict[int, Any] = {}
    training_metrics: list[dict[str, Any]] = []

    for task_idx, task in enumerate(tasks):
        # ── Train on current task ───────────────────────────────
        task_metrics = train_fn(model, task, task_idx)
        training_metrics.append(task_metrics)

        # ── Record weight snapshot ──────────────────────────────
        if record_weight_fn is not None:
            weight_snapshots[task_idx] = record_weight_fn(model, task_idx)

        # ── Evaluate on all tasks up to this one ────────────────
        row: list[float] = []
        for eval_task_idx in range(task_idx + 1):
            loader = tasks[eval_task_idx].test_loaders.get(eval_task_idx)
            if loader is not None:
                acc = evaluate_on_task(model, loader, predict_fn)
            else:
                acc = 0.0
            row.append(acc)
        accuracy_matrix.append(row)

        # Global accuracy
        if global_test_loader is not None:
            ga = evaluate_on_task(model, global_test_loader, predict_fn)
            global_accuracies.append(ga)

    # Compute forgetting metrics
    metrics = evaluate_continual_learning(accuracy_matrix)

    return {
        "accuracy_matrix": accuracy_matrix,
        "per_task_accuracies": {
            f"after_task_{i}": row for i, row in enumerate(accuracy_matrix)
        },
        "global_accuracies": global_accuracies,
        "metrics": metrics,
        "weight_snapshots": weight_snapshots,
        "training_metrics": training_metrics,
        "n_tasks": n_tasks,
    }


# ── Predict function factories ─────────────────────────────────────


def make_hebbian_predict_fn(
    epsilon: float = 0.1,
) -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """Create a predict function for Hebbian classifiers.

    The returned function queries ``model.predict(x, epsilon=epsilon)``
    when available, falling back to ``model(x)`` for raw forward passes.

    Args:
        epsilon: Dead-zone width for ``ternary_sign``.

    Returns:
        Callable ``(model, x) -> predictions``.
    """

    def _predict(model: Any, x: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "predict") and callable(model.predict):
            return model.predict(x, epsilon=epsilon)
        # Raw model forward pass
        out = model(x)
        return out.argmax(dim=1)

    return _predict


def make_backprop_predict_fn() -> Callable[[Any, torch.Tensor], torch.Tensor]:
    """Create a predict function for backprop models.

    Uses argmax on the model's output logits.

    Returns:
        Callable ``(model, x) -> predictions``.
    """

    def _predict(model: Any, x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        return out.argmax(dim=1)

    return _predict


# ── Forgetting comparison utilities ────────────────────────────────


def format_comparison_table(
    hebbian_metrics: dict[str, Any],
    backprop_metrics: dict[str, Any],
) -> str:
    """Format a side-by-side comparison table of Hebbian vs backprop.

    Args:
        hebbian_metrics: Output of ``run_continual_experiment`` for Hebbian.
        backprop_metrics: Output of ``run_continual_experiment`` for backprop.

    Returns:
        Formatted markdown table.
    """
    h = hebbian_metrics
    b = backprop_metrics

    lines = [
        "| Metric | PH-Neuro (Ternary Hebbian) | Backprop (SGD) |",
        "|--------|--------------------------|----------------|",
        f"| Average accuracy | {100 * h['metrics']['average_accuracy']:.2f}% | {100 * b['metrics']['average_accuracy']:.2f}% |",
        f"| Average forgetting | {100 * h['metrics']['average_forgetting']:.2f}% | {100 * b['metrics']['average_forgetting']:.2f}% |",
    ]

    # Per-task forgetting
    h_forget = h["metrics"]["per_task_forgetting"]
    b_forget = b["metrics"]["per_task_forgetting"]
    lines.append("")
    lines.append("### Per-task Forgetting")
    lines.append("")
    lines.append("| Task | Hebbian | Backprop |")
    lines.append("|------|---------|----------|")
    for i, (hf, bf) in enumerate(zip(h_forget, b_forget)):
        lines.append(f"| Task {i + 1} | {100 * hf:.2f}% | {100 * bf:.2f}% |")

    return "\n".join(lines)
