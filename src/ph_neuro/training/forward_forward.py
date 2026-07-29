"""Forward-Forward training for ternary Hebbian networks.

Implements the Forward-Forward algorithm (Hinton, 2022) for ternary
weight layers. Each layer learns through two forward passes:

- **Positive pass**: Real data → Hebbian update to increase goodness
- **Negative pass**: Corrupted/junk data → anti-Hebbian update to decrease goodness

The goodness function for ternary activations is **popcount**:
number of active neurons in the output.

No ``.backward()`` is called anywhere — learning is purely local
through Hebbian plasticity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.linear import TernaryHebbianLinear


def generate_negative_data(
    x: torch.Tensor,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """Generate negative (corrupted) data for Forward-Forward training.

    Takes the ternary-quantized input and corrupts it:
    1. Mask (set to 0) a random ``mask_ratio`` fraction of entries
    2. Overwrite the remaining entries with random {-1, 0, +1}

    This produces "junk" versions the network should learn to suppress.

    Args:
        x: Ternary input tensor, shape ``(batch, *)``, values in {-1, 0, +1}.
            Should already be passed through ``ternary_sign``.
        mask_ratio: Fraction of pixels to mask. Default 0.5.

    Returns:
        Corrupted tensor, same shape as ``x``, values in {-1, 0, +1}.
    """
    x_flat = x.view(x.size(0), -1)
    batch, n_pixels = x_flat.shape
    device = x.device

    # Start from clone of ternary-quantized input
    neg = x_flat.clone()

    # Random mask: mask_ratio fraction of pixels → set to 0
    mask = torch.rand(batch, n_pixels, device=device) < mask_ratio
    neg[mask] = 0

    # Random ternary noise for unmasked pixels: random {-1, 0, +1}
    noise = (torch.randint(0, 3, (batch, n_pixels), device=device) - 1).to(torch.int8)
    neg[~mask] = noise[~mask]

    return neg.view(x.shape)


class ForwardForwardClassifier:
    """Single-layer ternary classifier trained with Forward-Forward.

    Uses Forward-Forward positive/negative passes to learn:
    - **Positive pass**: For real data, only the correct-class output
      neuron fires (+1). Hebbian update strengthens that neuron's
      input connections.
    - **Negative pass**: For corrupted data (50% mask + random noise),
      any neuron that fires gets weakened via anti-Hebbian update.
      This teaches neurons to suppress responses to non-realistic inputs.

    Inference is a simple argmax over the 10 output neurons.

    No ``.backward()``, no optimizers, no loss functions.

    Args:
        in_features: Number of input features (e.g. 784 for MNIST).
        out_features: Number of output classes (e.g. 10 for MNIST).
        theta_upper: Hysteresis upper threshold. Default 1.0.
        theta_lower: Hysteresis lower threshold. Default 0.3.
        device: Device to place the model on.
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
        lr_pos: float = 0.01,
        lr_neg: float = 0.005,
        decay: float = 0.0,
        epsilon: float = 0.1,
        mask_ratio: float = 0.5,
    ) -> dict[str, float]:
        """Run one Forward-Forward training step on a batch.

        Combines WTA error correction with FF junk suppression:

        1. **Forward pass on real data**: Get class predictions.
        2. **WTA correction**: For wrong predictions, strengthen correct
           class AND anti-Hebbian weaken the wrongly predicted class.
        3. **Junk suppression (FF)**: Generate corrupted data, forward pass,
           anti-Hebbian weaken the most-active neuron per sample.

        This is the "FF-inspired WTA" — class-specific error correction
        (from WTA) + general junk regularization (from FF).

        Args:
            x: Input tensor, shape ``(batch, *)``.
            y: Labels, shape ``(batch,)``.
            lr_pos: Hebbian learning rate.
            lr_neg: Anti-Hebbian learning rate for negative pass.
            decay: Homeostatic decay rate (0 = no decay).
            epsilon: Dead-zone width for ``ternary_sign``.
            mask_ratio: Fraction of pixels to mask in negative data.

        Returns:
            Dict with ``flip_rate`` (fraction of weights that changed).
        """
        # Guard: no autograd during Hebbian training
        assert not torch.is_grad_enabled(), "Autograd must be disabled for Hebbian training"

        x = x.to(self._device)
        y = y.to(self._device)
        x_flat = x.view(x.size(0), -1)

        # ── Quantize input ──────────────────────────────────────────
        x_ternary = ternary_sign(x_flat, epsilon=epsilon)

        # ── Forward pass on real data ───────────────────────────────
        out = self.model(x_ternary.float())
        pred = out.argmax(dim=1)

        # ── WTA correction: strengthen correct, weaken wrong prediction ──
        wrong_mask = pred != y

        # Use the combined WTA update via latent scores directly:
        # Δ = lr_pos × (correct_onehot^T @ pre - pred_onehot^T @ pre) for wrong samples
        scores = self.model._latent_scores.scores

        if wrong_mask.any():
            wrong_idx = wrong_mask
            # One-hot: correct class
            correct_hot = F.one_hot(y[wrong_idx], self.model._out_features).float()
            # One-hot: predicted (wrong) class
            pred_hot = F.one_hot(pred[wrong_idx], self.model._out_features).float()
            # Combined delta: strengthen correct, weaken predicted
            delta = lr_pos * (
                correct_hot.T @ x_ternary[wrong_idx].float()
                - pred_hot.T @ x_ternary[wrong_idx].float()
            )
            scores += delta.to(scores.dtype)

        # ── Junk suppression (FF negative pass) ─────────────────────
        # Generate corrupted data from ternary input
        x_neg_ternary = generate_negative_data(x_ternary, mask_ratio=mask_ratio)

        # Forward pass on junk
        out_neg = self.model(x_neg_ternary.float())
        post_neg = ternary_sign(out_neg, epsilon=epsilon)

        # Only apply if at least one neuron fires on junk
        if post_neg.abs().sum() > 0:
            # For each sample, anti-Hebbian weaken the most-active neuron
            neg_winner = out_neg.abs().argmax(dim=1)
            post_neg_hot = F.one_hot(neg_winner, self.model._out_features).float()
            # Anti-Hebbian with separate learning rate
            scores -= lr_neg * (
                post_neg_hot.T @ x_neg_ternary.float()
            ).to(scores.dtype)

        # ── Post-update maintenance ────────────────────────────────
        # Snapshot old weights for flip tracking
        old_weights = self.model.weight.unpack().clone()

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

        Forward pass through the layer → argmax over output neurons.

        Args:
            x: Input tensor, shape ``(batch, *)``.
            epsilon: Dead-zone width for ``ternary_sign``.

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
            f"ForwardForwardClassifier({self.model._in_features}"
            f"\u2192{self.model._out_features}, "
            f"\u03b8_u={self.model._theta_upper}, \u03b8_l={self.model._theta_lower})"
        )
