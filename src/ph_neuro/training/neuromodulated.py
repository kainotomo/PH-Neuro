"""Neuromodulated Ternary Hebbian (NTH) classifier.

Implements three-factor Hebbian learning (Frémaux & Gerstner, 2016):

    ΔW = η · M · pre · post

where M ∈ {-1, 0, +1} is a neuromodulator that controls whether the
pre×post correlation is strengthened, ignored, or weakened.

For the single-layer label-modulator case (NTH-1):
- For **wrong predictions**: M_c = +1 (correct class, strengthen), M_w = -1 (wrongly-predicted class, weaken)
- For **correct predictions**: M = 0 everywhere (no update — same as WTA)
- M = 0 for all other neurons (no update)

This is **mathematically identical** to the WTA Hebbian rule used in
:class:`~ph_neuro.training.supervised.SupervisedHebbianClassifier`,
but expressed as a *unified* update: a single matrix multiply instead
of separate Hebbian + anti-Hebbian operations. The unified form
generalizes naturally to arbitrary modulator sources (error signals,
prediction errors, novelty, reward).

No ``.backward()``, no optimizers, no loss functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.linear import TernaryHebbianLinear


def build_label_modulator(
    y: torch.Tensor,
    pred: torch.Tensor,
    out_features: int,
    *,
    positive_only: bool = False,
    negative_only: bool = False,
    full_target: bool = False,
) -> torch.Tensor:
    """Build a label-derived neuromodulator tensor.

    The standard label modulator assigns:
    - M_c = +1 for the correct class neuron (strengthen)
    - M_w = -1 for the wrongly-predicted neuron (weaken)
    - M = 0 for all other neurons (no update)

    Args:
        y: Ground-truth labels, shape ``(batch,)``.
        pred: Predicted labels, shape ``(batch,)``.
        out_features: Number of output classes.
        positive_only: If ``True``, only set M=+1 for correct class
            (no negative modulation). Default ``False``.
        negative_only: If ``True``, only set M=-1 for wrong predictions
            (no positive modulation). Default ``False``.
        full_target: If ``True``, set M=-1 for ALL wrong classes, not
            just the wrongly-predicted one. Default ``False``.

    Returns:
        Modulator tensor, shape ``(batch, out_features)``, values in
        {-1, 0, +1}.
    """
    batch_size = y.size(0)
    device = y.device

    modulator = torch.zeros(batch_size, out_features, device=device, dtype=torch.float32)

    # Identify wrong predictions
    wrong_mask = pred != y

    if not wrong_mask.any():
        # All predictions correct — no update needed (matches WTA behavior)
        return modulator

    wrong_idx = wrong_mask

    # For wrong predictions only:
    # - M_c = +1 for the correct class (strengthen)
    if not negative_only:
        modulator[wrong_idx, y[wrong_idx]] = 1.0

    # - M_w = -1 for the wrongly-predicted class (weaken)
    if not positive_only:
        if full_target:
            # Set M = -1 for ALL wrong classes (not just predicted)
            for i in wrong_idx.nonzero(as_tuple=True)[0]:
                modulator[i, :] = -1.0
                modulator[i, y[i]] = 1.0  # correct class stays +1
        else:
            # Standard: M = -1 only for the wrongly-predicted class
            modulator[wrong_idx, pred[wrong_idx]] = -1.0

    return modulator


class NeuromodulatedHebbianClassifier:
    """Single-layer ternary Hebbian classifier with neuromodulation.

    Uses the three-factor Hebbian rule:

        ΔW = η · M · pre · post

    where M is a per-neuron neuromodulator. For the default label
    modulator (NTH-1), this is equivalent to WTA but expressed as
    a single unified update.

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
        in_features: int = 784,
        out_features: int = 10,
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
        positive_only: bool = False,
        negative_only: bool = False,
        full_target: bool = False,
    ) -> dict[str, float]:
        """Run one NTH training step on a batch.

        1. **Forward pass** on real data to get predictions.
        2. **Build modulator** from labels: M_c=+1 (correct), M_w=-1 (wrong prediction).
        3. **Apply unified update**: ``Δscores = lr × Mᵀ @ pre / batch_size``
           — a single matrix multiply replacing the separate Hebbian/anti-Hebbian
           operations of WTA.

        Args:
            x: Input tensor, shape ``(batch, *)``.
            y: Labels, shape ``(batch,)``.
            lr: Hebbian learning rate.
            decay: Homeostatic decay rate (0 = no decay).
            epsilon: Dead-zone width for ``ternary_sign``.
            positive_only: If ``True``, only strengthen correct class
                (no anti-Hebbian weakening).
            negative_only: If ``True``, only weaken wrong predictions
                (no Hebbian strengthening).
            full_target: If ``True``, weaken ALL wrong classes, not
                just the wrongly-predicted one.

        Returns:
            Dict with ``flip_rate`` (fraction of weights that changed)
            and ``n_flips`` (absolute count).
        """
        # Guard: no autograd during Hebbian training
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

        # ── Build label modulator ───────────────────────────────────
        modulator = build_label_modulator(
            y, pred, self.model._out_features,
            positive_only=positive_only,
            negative_only=negative_only,
            full_target=full_target,
        )

        # ── Unified NTH update: Δscores = lr × Mᵀ @ pre ────────────
        # One single matrix multiply — theoretically equivalent to WTA's
        # two-step (correct_hot.T @ pre - pred_hot.T @ pre).
        scores = self.model._latent_scores.scores
        delta = lr * (modulator.T.to(scores.dtype) @ x_f.to(scores.dtype))
        scores += delta

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

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._device

    def __repr__(self) -> str:
        return (
            f"NeuromodulatedHebbianClassifier({self.model._in_features}"
            f"\u2192{self.model._out_features}, "
            f"\u03b8_u={self.model._theta_upper}, \u03b8_l={self.model._theta_lower})"
        )
