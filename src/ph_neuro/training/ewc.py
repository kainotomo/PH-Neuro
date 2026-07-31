"""Elastic Weight Consolidation (EWC) for ternary STE networks.

EWC is a continual-learning method (Kirkpatrick et al., 2017) that reduces
catastrophic forgetting by penalizing movement of parameters that were
**important** for previously learned tasks.

For ternary STE networks the key observation is that the discrete ternary
weights {-1, 0, +1} are a deterministic function of the float **latent
scores** (``TernarySTELinear.latent_scores``) via ``sign()``. The latent
scores are the differentiable parameters that backpropagation actually
updates, so EWC must operate on the latent scores, not on the discrete
ternary weights.

The diagonal Fisher Information Matrix (FIM) is estimated as the average
squared gradient of the negative log-likelihood with respect to each latent
score, evaluated over the task's training data. Because gradients flow
through the Straight-Through Estimator (``_STESign``), the Fisher captures
how sensitive each latent score is to the task.

Two flavours are provided:

- :class:`OnlineEWC` — accumulates a single Fisher across tasks
  (``F_acc = gamma * F_prev + F_new``) and keeps one reference snapshot
  (Schwarz et al., 2018). Memory cost: O(2 * |theta|), independent of the
  number of tasks.
- :class:`MultiTaskEWC` — stores a separate (Fisher, reference) pair per
  task. Memory cost: O(2 * T * |theta|). Provided as an ablation.

References:
    Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting in
    neural networks". PNAS 114(13): 3521-3526.
    Schwarz, J. et al. (2018). "Progress & compress: A scalable framework
    for continual learning". ICML 2018.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader


# ── Parameter extraction ────────────────────────────────────────────


def get_ternary_latent_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect the float ``latent_scores`` parameters of ternary STE layers.

    EWC operates on latent scores because they are the differentiable
    parameters trained by STE backprop — the ternary weights are derived via
    ``sign()`` and are not directly differentiable.

    Only parameters that require gradients are returned, so frozen layers
    (e.g. a frozen backbone in QLoRA-style experiments) are skipped.

    Args:
        model: A model containing ternary STE layers (e.g. an
            ``nn.Sequential`` built by :func:`ph_neuro.models.ste_models.ste_mlp`).

    Returns:
        List of ``latent_scores`` ``nn.Parameter`` objects, in module order.
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if hasattr(module, "latent_scores"):
            latent = getattr(module, "latent_scores")
            if isinstance(latent, nn.Parameter) and latent.requires_grad:
                params.append(latent)
    return params


# ── Fisher Information ─────────────────────────────────────────────


def compute_fisher_diag(
    model: nn.Module,
    dataloader: DataLoader,
    n_batches: int = 500,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """Estimate the diagonal Fisher Information on the latent scores.

    The diagonal Fisher of the negative log-likelihood is estimated as the
    mean over ``n_batches`` batches of the squared STE gradient of each
    latent score::

        F_i = E[ ( d log p(y | x; theta) / d theta_i )^2 ]

    Since ``d log p / d theta_i = -d L / d theta_i``, we compute the
    gradient of the cross-entropy loss and square it.

    Args:
        model: Model containing ternary STE layers.
        dataloader: DataLoader over the task's training data. Batches are
            expected to be ``(x, y)`` tuples.
        n_batches: Maximum number of batches to sample (default 500).
        device: Device to run the Fisher estimation on. Defaults to the
            model's current device.

    Returns:
        List of tensors, one per ternary latent-score parameter (same order
        as :func:`get_ternary_latent_params`), each with the same shape as
        the parameter, containing the non-negative diagonal Fisher estimate.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)
    model.train()

    params = get_ternary_latent_params(model)
    fisher = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    batches_seen = 0

    for batch in dataloader:
        if batches_seen >= n_batches:
            break
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        for f, p in zip(fisher, params):
            if p.grad is not None:
                f += p.grad.detach().float() ** 2
        batches_seen += 1

    if batches_seen > 0:
        for f in fisher:
            f.div_(batches_seen)

    model.zero_grad(set_to_none=True)
    return fisher


def save_ewc_reference(model: nn.Module) -> list[torch.Tensor]:
    """Snapshot the current latent scores as detached tensors.

    Args:
        model: Model containing ternary STE layers.

    Returns:
        List of detached clones of ``latent_scores``, one per ternary layer
        (same order as :func:`get_ternary_latent_params`).
    """
    return [p.detach().clone() for p in get_ternary_latent_params(model)]


# ── EWC penalty ────────────────────────────────────────────────────


def ewc_penalty(
    model: nn.Module,
    ref_params: Sequence[torch.Tensor],
    fisher_diag: Sequence[torch.Tensor],
    ewc_lambda: float,
) -> torch.Tensor:
    """Compute the EWC regularization penalty.

    The penalty is::

        (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2

    summed over every latent-score parameter and every stored task
    reference.

    Args:
        model: Current model.
        ref_params: Reference (consolidated) latent scores, aligned with
            ``fisher_diag``.
        fisher_diag: Diagonal Fisher per latent score, aligned with
            ``ref_params``.
        ewc_lambda: EWC regularization strength.

    Returns:
        Scalar tensor (on the model's device) containing the penalty. The
        penalty is differentiable w.r.t. the model's latent scores.
    """
    params = get_ternary_latent_params(model)
    penalty = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for p, ref, f in zip(params, ref_params, fisher_diag):
        delta = p - ref.to(p.device)
        penalty = penalty + 0.5 * ewc_lambda * (f.to(p.device) * delta**2).sum()
    return penalty


# ── EWC managers ───────────────────────────────────────────────────


class OnlineEWC:
    """Online EWC: accumulate a single Fisher across tasks.

    After training each task, call :meth:`update` with that task's training
    loader. The Fisher is estimated on the just-trained model and the
    reference latent scores are snapshotted. The accumulated Fisher follows
    ``F_acc = gamma * F_prev + F_new`` and only the most recent reference is
    retained (Schwarz et al., 2018), so memory stays O(2 * |theta|).

    Args:
        model: Model containing ternary STE layers.
        gamma: Decay/weighting factor for the previous Fisher when
            accumulating (default 1.0 = equal weighting).
    """

    def __init__(self, model: nn.Module, gamma: float = 1.0):
        self._model = model
        self._gamma = float(gamma)
        self._fisher: list[torch.Tensor] | None = None
        self._ref_params: list[torch.Tensor] | None = None
        self._n_tasks = 0

    @property
    def n_tasks(self) -> int:
        """Number of tasks that have been consolidated so far."""
        return self._n_tasks

    @property
    def gamma(self) -> float:
        """The Fisher accumulation factor."""
        return self._gamma

    def has_penalty(self) -> bool:
        """Whether a penalty is available (i.e. at least one task seen)."""
        return self._fisher is not None and self._ref_params is not None

    def update(
        self,
        dataloader: DataLoader,
        n_batches: int = 500,
        device: torch.device | str | None = None,
    ) -> int:
        """Consolidate the current task into the EWC state.

        Args:
            dataloader: The current task's training loader (used for the
                Fisher estimate).
            n_batches: Number of batches to sample for the Fisher estimate.
            device: Device to run the Fisher estimation on.

        Returns:
            The new number of consolidated tasks.
        """
        fisher = compute_fisher_diag(self._model, dataloader, n_batches, device)
        ref = save_ewc_reference(self._model)

        if self._fisher is None:
            self._fisher = fisher
        else:
            self._fisher = [
                self._gamma * f_old + f_new for f_old, f_new in zip(self._fisher, fisher)
            ]
        self._ref_params = ref
        self._n_tasks += 1
        return self._n_tasks

    def penalty(self, model: nn.Module, ewc_lambda: float) -> torch.Tensor:
        """Compute the accumulated EWC penalty for ``model``.

        Args:
            model: Current model.
            ewc_lambda: EWC regularization strength.

        Returns:
            Scalar tensor; zero if no task has been consolidated yet.
        """
        if not self.has_penalty():
            return torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
        return ewc_penalty(model, self._ref_params, self._fisher, ewc_lambda)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the accumulated EWC state (Fisher + reference)."""
        return {
            "gamma": self._gamma,
            "n_tasks": self._n_tasks,
            "fisher": [f.clone() for f in self._fisher] if self._fisher else None,
            "ref_params": [r.clone() for r in self._ref_params] if self._ref_params else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the EWC state from :meth:`state_dict` output."""
        self._gamma = float(state["gamma"])
        self._n_tasks = int(state["n_tasks"])
        if state.get("fisher") is not None:
            self._fisher = [f.clone() for f in state["fisher"]]
        else:
            self._fisher = None
        if state.get("ref_params") is not None:
            self._ref_params = [r.clone() for r in state["ref_params"]]
        else:
            self._ref_params = None


class MultiTaskEWC:
    """Multi-task EWC: store a separate (Fisher, reference) per task.

    The penalty is the sum over every stored task, so memory cost is linear
    in the number of tasks: O(2 * T * |theta|). Provided mainly as an
    ablation against :class:`OnlineEWC`.

    Args:
        model: Model containing ternary STE layers.
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._fishers: list[list[torch.Tensor]] = []
        self._refs: list[list[torch.Tensor]] = []

    @property
    def n_tasks(self) -> int:
        """Number of tasks that have been consolidated so far."""
        return len(self._fishers)

    def has_penalty(self) -> bool:
        """Whether a penalty is available (i.e. at least one task seen)."""
        return len(self._fishers) > 0

    def update(
        self,
        dataloader: DataLoader,
        n_batches: int = 500,
        device: torch.device | str | None = None,
    ) -> int:
        """Consolidate the current task by appending its (Fisher, reference).

        Args:
            dataloader: The current task's training loader.
            n_batches: Number of batches to sample for the Fisher estimate.
            device: Device to run the Fisher estimation on.

        Returns:
            The new number of consolidated tasks.
        """
        fisher = compute_fisher_diag(self._model, dataloader, n_batches, device)
        ref = save_ewc_reference(self._model)
        self._fishers.append(fisher)
        self._refs.append(ref)
        return len(self._fishers)

    def penalty(self, model: nn.Module, ewc_lambda: float) -> torch.Tensor:
        """Compute the sum of EWC penalties over all stored tasks.

        Args:
            model: Current model.
            ewc_lambda: EWC regularization strength.

        Returns:
            Scalar tensor; zero if no task has been consolidated yet.
        """
        total = torch.zeros(
            (), device=next(model.parameters()).device, dtype=torch.float32
        )
        for ref, fisher in zip(self._refs, self._fishers):
            total = total + ewc_penalty(model, ref, fisher, ewc_lambda)
        return total
