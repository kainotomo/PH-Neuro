"""Supervised Hebbian classifier — winner-take-all Hebbian learning.

A lightweight wrapper around a single :class:`~ph_neuro.layers.linear.TernaryHebbianLinear`
layer trained via a winner-take-all (WTA) Hebbian rule:

- **Correct prediction**: strengthen the winning (correct) class connection
- **Wrong prediction**: strengthen the correct class, weaken the predicted class

This is a biologically-plausible approximation of the Perceptron algorithm
using local Hebbian plasticity. No ``.backward()``, no optimizers, no loss
functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.linear import TernaryHebbianLinear


class SupervisedHebbianClassifier:
    """Single-layer ternary Hebbian classifier with WTA learning.

    Uses a winner-take-all Hebbian rule:
    - Forward pass to get the predicted class
    - If correct: strengthen the correct class via Hebbian update
    - If wrong: strengthen the correct class, weaken the predicted class
      via anti-Hebbian update

    Args:
        in_features: Number of input features (e.g. 784 for MNIST).
        out_features: Number of output classes (e.g. 10 for MNIST).
        theta_upper: Hysteresis upper threshold. Default 1.0.
        theta_lower: Hysteresis lower threshold. Default 0.3.
        device: Device to place the model on.

    Attributes:
        model: The underlying ``TernaryHebbianLinear`` layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_upper: float = 1.0,
        theta_lower: float = 0.3,
        device: torch.device | str | None = None,
    ):
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = TernaryHebbianLinear(
            in_features=in_features,
            out_features=out_features,
            theta_upper=theta_upper,
            theta_lower=theta_lower,
        ).to(self._device)

    # ── Training ────────────────────────────────────────────────────

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        decay: float = 0.0,
        epsilon: float = 0.1,
    ) -> dict[str, float]:
        """Run one WTA Hebbian training step on a batch.

        For each sample:
        - **Correct prediction**: strengthen the correct class connection
        - **Wrong prediction**: strengthen the correct class AND
          weaken the predicted (wrong) class connection

        Args:
            x: Input tensor, shape ``(batch, *)``.
            y: Labels, shape ``(batch,)``.
            lr: Hebbian learning rate.
            decay: Homeostatic decay rate (0 = no decay).
            epsilon: Dead-zone width for ``ternary_sign``. Small values
                suppress noisy near-zero activations.

        Returns:
            Dict with ``flip_rate`` (fraction of weights that changed)
            and ``n_flips`` (absolute count).
        """
        # Guard: no autograd during Hebbian learning
        assert not torch.is_grad_enabled(), "Autograd must be disabled for Hebbian training"

        x = x.to(self._device)
        y = y.to(self._device)
        x_flat = x.view(x.size(0), -1)

        # Quantize input to ternary {-1, 0, +1} with noise filtering
        x_ternary = ternary_sign(x_flat, epsilon=epsilon)
        x_f = x_ternary.float()

        # Forward pass to get predictions
        out = self.model(x_f)
        pred = out.argmax(dim=1)

        # Snapshot current weights for flip tracking
        old_weights = self.model.weight.unpack().clone()

        # WTA Hebbian update: strengthen correct, weaken wrong prediction
        scores = self.model._latent_scores.scores
        wrong_mask = pred != y

        # For wrong predictions: Δ = lr × (correct_pre - pred_pre)
        if wrong_mask.any():
            wrong_idx = wrong_mask
            correct_hot = F.one_hot(y[wrong_idx], self.model._out_features).float()
            pred_hot = F.one_hot(pred[wrong_idx], self.model._out_features).float()
            delta = lr * (correct_hot.T @ x_f[wrong_idx] - pred_hot.T @ x_f[wrong_idx])
            scores += delta.to(scores.dtype)

        # Homeostatic decay
        if decay > 0:
            self.model.apply_decay(decay)

        # Refresh ternary weights via hysteresis
        self.model.refresh_weights()

        # Compute flip statistics
        new_weights = self.model.weight.unpack()
        flips = (old_weights != new_weights).sum().item()
        total = new_weights.numel()

        return {
            "flip_rate": flips / max(total, 1),
            "n_flips": flips,
        }

    # ── Inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """Predict class labels.

        Args:
            x: Input tensor, shape ``(batch, *)``.
            epsilon: Dead-zone width for ``ternary_sign`` (must match
                value used during training for consistency).

        Returns:
            Predicted class indices, shape ``(batch,)``.
        """
        x = x.to(self._device)
        x_flat = x.view(x.size(0), -1)
        x_ternary = ternary_sign(x_flat, epsilon=epsilon)
        out = self.model(x_ternary.float())
        return out.argmax(dim=1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epsilon: float = 0.1,
    ) -> float:
        """Evaluate accuracy on a data loader.

        Args:
            loader: DataLoader yielding ``(inputs, targets)``.
            epsilon: Dead-zone width for ``ternary_sign``.

        Returns:
            Accuracy as a fraction in ``[0.0, 1.0]``.
        """
        correct = 0
        total = 0
        for x, y in loader:
            pred = self.predict(x, epsilon=epsilon)
            y = y.to(self._device)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return correct / max(total, 1)

    # ── Weight statistics ────────────────────────────────────────────

    @torch.no_grad()
    def get_weight_stats(self) -> dict[str, float]:
        """Compute statistics of the current ternary weights.

        Returns:
            Dict with ``pos_pct``, ``neg_pct``, ``zero_pct`` (percentages
            of weights that are +1, -1, and 0 respectively).
        """
        w = self.model.weight.unpack()
        total = w.numel()
        return {
            "pos_pct": 100.0 * (w == 1).sum().item() / max(total, 1),
            "neg_pct": 100.0 * (w == -1).sum().item() / max(total, 1),
            "zero_pct": 100.0 * (w == 0).sum().item() / max(total, 1),
        }

    @staticmethod
    def get_flip_rate(
        old_weights: torch.Tensor,
        new_weights: torch.Tensor,
    ) -> float:
        """Compute fraction of weights that changed.

        Args:
            old_weights: Weight tensor before refresh.
            new_weights: Weight tensor after refresh.

        Returns:
            Fraction (0.0 to 1.0) of weights that differ.
        """
        return (old_weights != new_weights).sum().item() / max(new_weights.numel(), 1)

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._device

    def __repr__(self) -> str:
        return (
            f"SupervisedHebbianClassifier({self.model._in_features}"
            f"\u2192{self.model._out_features}, "
            f"\u03b8_u={self.model._theta_upper}, \u03b8_l={self.model._theta_lower})"
        )
