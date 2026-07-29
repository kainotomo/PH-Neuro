"""Multi-layer Neuromodulated Ternary Hebbian (NTH) training.

Implements NTH-4: a 2-layer (784\u2192512\u219210) ternary Hebbian network trained
with neuromodulated learning on MNIST. The key innovation is propagating
the label-derived modulator signal to the hidden layer.

Three approaches for the hidden-layer modulator (\u0394W = \u03b7 \u00b7 M_hidden \u00b7 pre):

    **A: Label broadcast (simplest)**
        M_hidden[i, j] = -1 if prediction is wrong AND hidden neuron j is active
        M_hidden[i, j] = 0 otherwise
        The correctness signal is broadcast from the output to ALL active hidden
        neurons: "this representation was wrong, weaken it."

    **B: Weight-feedback modulator**
        M_hidden = M_output @ W_out  (where W_out is the ternary output weight)
        M_output encodes +1 for correct class, -1 for wrong prediction.
        This tells each hidden neuron: "contribute more to the correct class,
        contribute less to the wrongly-predicted class."
        NOT backprop \u2014 just a linear transformation through ternary weights.

    **C: Random feedback alignment**
        M_hidden = M_output @ B  (where B is a FIXED random ternary matrix)
        Inspired by Feedback Alignment (Lillicrap et al., 2016).
        The network must learn to align its hidden representations with B.

    **D: Latent score feedback (NTH-4b)**
        M_hidden = M_output @ S_out  (where S_out is the output layer's LATENT SCORES)
        Instead of using the sparse ternary weight matrix W_out (92% zeros), use
        the dense continuous latent scores S_out (fp16). Every synapse carries
        a latent score even when the ternary weight is 0, so the feedback signal
        propagates through a dense pathway instead of a near-zero sparse one.
        The latent scores encode confidence: a synapse with score +4.5 contributes
        strongly to the feedback; one with +0.1 contributes weakly.

Training is **joint** (both layers updated per batch, not greedy), because
the hidden modulator requires the output layer's predictions and weights.

No ``.backward()``, no optimizers, no loss functions.
"""

from __future__ import annotations

import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.core.activation import ternary_sign
from ph_neuro.core.hebbian_rules import neuromodulated_update
from ph_neuro.models.mlp import HebbianMLP
from ph_neuro.training.greedy import _init_hidden_layer_connectivity
from ph_neuro.training.neuromodulated import build_label_modulator

# Type alias for modulator modes
ModulatorMode = Literal["label_broadcast", "weight_feedback", "random_feedback", "latent_feedback"]


# ── Hidden modulator builders ──────────────────────────────────────


def _build_hidden_modulator_label_broadcast(
    y: torch.Tensor,
    pred: torch.Tensor,
    post_hidden: torch.Tensor,
) -> torch.Tensor:
    """Approach A: broadcast correctness signal to active hidden neurons.

    For wrong predictions, active hidden neurons (post != 0) get M = -1
    (anti-Hebbian: weaken their input connections). For correct predictions,
    M = 0 everywhere.

    This says: "the current hidden representation produced a wrong answer,
    so weaken whatever caused it."

    Args:
        y: Ground-truth labels, shape ``(batch,)``.
        pred: Predicted labels, shape ``(batch,)``.
        post_hidden: Hidden layer ternary activations, shape
            ``(batch, hidden_size)``, values in {-1, 0, +1}.

    Returns:
        Modulator tensor, shape ``(batch, hidden_size)``, values in {-1, 0, +1}.
    """
    batch_size = y.size(0)
    hidden_size = post_hidden.size(1)
    device = y.device

    modulator = torch.zeros(batch_size, hidden_size, device=device, dtype=torch.float32)

    # Identify wrong predictions
    wrong_mask = pred != y  # (batch,)

    if not wrong_mask.any():
        return modulator

    # For wrong predictions: anti-Hebbian on active hidden neurons
    # post_hidden[j] != 0 means the hidden neuron fired
    # We set M = -1 for active neurons, meaning ΔW = -lr × 1ᵀ @ pre
    # This weakens the input connections that caused this neuron to fire
    active_mask = post_hidden != 0  # (batch, hidden_size)
    modulator[wrong_mask] = -(active_mask[wrong_mask].float()) * post_hidden[wrong_mask].float().sign()

    return modulator


def _build_hidden_modulator_weight_feedback(
    M_output: torch.Tensor,
    W_out: torch.Tensor,
) -> torch.Tensor:
    """Approach B: propagate modulator through output weight matrix.

    M_hidden = M_output @ W_out

    where:
        M_output: (batch, 10)  \u2014 +1 for correct class, -1 for wrong prediction
        W_out:    (10, hidden_size) \u2014 ternary output weight matrix

    For wrong predictions, M_output has +1 at correct class index and -1 at
    the predicted (wrong) class index. Multiplying by W_out gives each hidden
    neuron a modulator proportional to W_out[correct] - W_out[predicted].

    This is NOT backprop \u2014 it's a simple linear transformation that has no
    gradients and computes exactly one direction. The weights are ternary so
    the computation is also cheap (popcount-compatible).

    Args:
        M_output: Output-layer modulator, shape ``(batch, n_classes)``,
            values in {-1, 0, +1}.
        W_out: Output layer ternary weight matrix (as dense float),
            shape ``(n_classes, hidden_size)``.

    Returns:
        Modulator tensor, shape ``(batch, hidden_size)`` (float, may be
        non-ternary since it's a linear combination).
    """
    # M_hidden = M_output @ W_out: (batch, 10) @ (10, hidden_size) = (batch, hidden_size)
    M_hidden = M_output.float() @ W_out.float()
    return M_hidden


def _build_hidden_modulator_random_feedback(
    M_output: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Approach C: random feedback alignment.

    M_hidden = M_output @ B

    where B is a FIXED random ternary matrix of shape (10, hidden_size),
    initialized once and never updated. Inspired by Feedback Alignment
    (Lillicrap et al., 2016): even random feedback weights work for
    backprop because the network learns to align its forward weights
    with the random feedback matrix.

    Args:
        M_output: Output-layer modulator, shape ``(batch, n_classes)``,
            values in {-1, 0, +1}.
        B: Fixed random ternary feedback matrix, shape
            ``(n_classes, hidden_size)``, values in {-1, 0, +1}.

    Returns:
        Modulator tensor, shape ``(batch, hidden_size)`` (float, may be
        non-ternary since it's a linear combination).
    """
    M_hidden = M_output.float() @ B.float()
    return M_hidden


def _build_hidden_modulator_latent_feedback(
    M_output: torch.Tensor,
    S_out: torch.Tensor,
) -> torch.Tensor:
    """Approach D: propagate modulator through output layer LATENT SCORES.

    M_hidden = M_output @ S_out

    where:
        M_output: (batch, 10)  — +1 for correct class, -1 for wrong prediction
        S_out:    (10, hidden_size) — output layer LATENT SCORES (fp16, DENSE)

    Unlike Approach B (which uses ternary weights W_out that are 92% zero),
    this uses the continuous fp16 latent scores. Every synapse has a latent
    score, even when its ternary weight is 0. The scores encode confidence:
    a score of +4.5 means "very confident this should be +1"; +0.2 means
    "barely positive." This provides a dense feedback pathway.

    The result is a continuous (not ternary) modulator — the
    ``neuromodulated_update`` function handles this correctly when
    ``post=None``.

    Args:
        M_output: Output-layer modulator, shape ``(batch, n_classes)``,
            values in {-1, 0, +1}.
        S_out: Output layer latent score matrix, shape
            ``(n_classes, hidden_size)`` (fp16, dense).

    Returns:
        Modulator tensor, shape ``(batch, hidden_size)`` (CONTINUOUS float,
        NOT restricted to {-1, 0, +1}).
    """
    # M_hidden = M_output @ S_out: (batch, 10) @ (10, hidden_size) = (batch, hidden_size)
    M_hidden = M_output.float() @ S_out.float()
    return M_hidden


def _init_random_feedback_matrix(
    n_classes: int,
    hidden_size: int,
    density: float = 0.5,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a fixed random ternary feedback matrix for Approach C.

    Args:
        n_classes: Number of output classes (10 for MNIST).
        hidden_size: Hidden layer size.
        density: Fraction of non-zero entries. Default 0.5 (50% dense).
        seed: Random seed for reproducibility.
        device: Torch device.

    Returns:
        Random ternary matrix, shape ``(n_classes, hidden_size)``,
        values in {-1, 0, +1}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    n_total = n_classes * hidden_size
    n_active = int(n_total * density)

    B = torch.zeros(n_classes, hidden_size, device=device, dtype=torch.int8)
    idx = torch.randperm(n_total, generator=rng, device=device)[:n_active]
    vals = (torch.randint(0, 2, (n_active,), generator=rng, device=device) * 2 - 1).to(torch.int8)
    B.flatten()[idx] = vals

    return B


# ── NTH multi-layer classifier ────────────────────────────────────


class NTHMultiLayerClassifier:
    """2-layer ternary Hebbian classifier with neuromodulated hidden training.

    Wraps a :class:`~ph_neuro.models.mlp.HebbianMLP` (784\u2192512\u219210) and
    trains both layers jointly with the NTH rule.

    The **output layer** always uses the standard label modulator:
        M_c = +1 (correct class), M_w = -1 (wrong prediction), 0 elsewhere
        Only updated on wrong predictions (matching NTH-1 / WTA behavior).

    The **hidden layer** uses one of three modulator propagation approaches:
        - ``label_broadcast``: global correctness signal broadcast to active
          hidden neurons (Approach A)
        - ``weight_feedback``: M_output @ W_out (Approach B)
        - ``random_feedback``: M_output @ B where B is a fixed random matrix
          (Approach C)
        - ``latent_feedback``: M_output @ S_out where S_out is the output
          layer's dense continuous latent scores (Approach D / NTH-4b)

    No ``.backward()``, no optimizers, no loss functions.

    Args:
        in_features: Input dimensionality (784 for MNIST).
        hidden_size: Hidden layer size (512).
        out_features: Number of output classes (10 for MNIST).
        modulator_mode: Hidden modulator approach.
        theta_upper: Hysteresis upper threshold.
        theta_lower: Hysteresis lower threshold.
        device: Device to place the model on.

    Attributes:
        model: The underlying ``HebbianMLP`` with 2 layers.
        hidden_layer: Shortcut to ``model.layers[0]``.
        output_layer: Shortcut to ``model.layers[1]``.
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_size: int = 512,
        out_features: int = 10,
        modulator_mode: ModulatorMode = "label_broadcast",
        theta_upper: float = 1.0,
        theta_lower: float = 0.3,
        device: torch.device | str | None = None,
    ):
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._modulator_mode = modulator_mode
        self._in_features = in_features
        self._hidden_size = hidden_size
        self._out_features = out_features

        # Create 2-layer MLP
        self.model = HebbianMLP(
            layer_sizes=[in_features, hidden_size, out_features],
            theta_upper=theta_upper,
            theta_lower=theta_lower,
        ).to(self._device)

        self.hidden_layer = self.model.layers[0]
        self.output_layer = self.model.layers[1]

        # For Approach C: fixed random feedback matrix
        if modulator_mode == "random_feedback":
            self._feedback_matrix = _init_random_feedback_matrix(
                n_classes=out_features,
                hidden_size=hidden_size,
                device=self._device,
            )
        else:
            self._feedback_matrix = None

    # ── Device ──────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._device

    # ── Modulator computation ──────────────────────────────────────

    def _compute_modulators(
        self,
        x_ternary: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both layers and compute all modulators.

        Args:
            x_ternary: Input tensor (ternary), shape ``(batch, in_features)``.
            y: Labels, shape ``(batch,)``.
            epsilon: Dead-zone for ``ternary_sign``.

        Returns:
            Tuple of:
            - ``M_output``: output-layer modulator, shape ``(batch, out_features)``
            - ``M_hidden``: hidden-layer modulator, shape ``(batch, hidden_size)``
            - ``h_hidden_ternary``: hidden layer ternary activations, shape ``(batch, hidden_size)``
            - ``pred``: predictions, shape ``(batch,)``
        """
        # Forward pass: input -> hidden -> output
        with torch.no_grad():
            # Hidden layer forward pass
            h_raw = self.hidden_layer(x_ternary.float())
            h_ternary = ternary_sign(h_raw, epsilon=epsilon)  # (batch, hidden_size)

            # Output layer forward pass
            out = self.output_layer(h_ternary.float())
            pred = out.argmax(dim=1)  # (batch,)

        # ── Output modulator ────────────────────────────────────
        # Standard label modulator (NTH-1 / WTA equivalent)
        M_output = build_label_modulator(y, pred, self._out_features)

        # ── Hidden modulator ────────────────────────────────────
        if self._modulator_mode == "label_broadcast":
            M_hidden = _build_hidden_modulator_label_broadcast(y, pred, h_ternary)
        elif self._modulator_mode == "weight_feedback":
            W_out = self.output_layer.weight.unpack()  # (out_features, hidden_size)
            M_hidden = _build_hidden_modulator_weight_feedback(M_output, W_out)
        elif self._modulator_mode == "random_feedback":
            M_hidden = _build_hidden_modulator_random_feedback(
                M_output, self._feedback_matrix
            )
        elif self._modulator_mode == "latent_feedback":
            S_out = self.output_layer._latent_scores.scores  # (out_features, hidden_size)
            M_hidden = _build_hidden_modulator_latent_feedback(M_output, S_out)
        else:
            raise ValueError(f"Unknown modulator mode: {self._modulator_mode}")

        return M_output, M_hidden, h_ternary, pred

    # ── Training ────────────────────────────────────────────────────

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lr_hidden: float = 0.01,
        lr_output: float = 0.01,
        decay: float = 0.0,
        epsilon: float = 0.1,
    ) -> dict[str, float]:
        """Run one NTH-4 training step on a batch.

        1. **Forward pass** through hidden layer \u2192 ternary activation \u2192 output layer
        2. **Build modulators**: M_output from labels, M_hidden from selected approach
        3. **Update output layer**: ``\u0394scores = lr_output \u00d7 M_output\u1d40 @ h_hidden``
        4. **Update hidden layer**: ``\u0394scores = lr_hidden \u00d7 M_hidden\u1d40 @ x_ternary``
        5. **Refresh weights** via hysteresis thresholding

        Args:
            x: Input tensor, shape ``(batch, *)``.
            y: Labels, shape ``(batch,)``.
            lr_hidden: Learning rate for hidden layer.
            lr_output: Learning rate for output layer.
            decay: Homeostatic decay rate (0 = no decay).
            epsilon: Dead-zone width for ``ternary_sign``.

        Returns:
            Dict with ``flip_rate_hidden``, ``flip_rate_output``,
            ``n_flips_hidden``, ``n_flips_output``, ``accuracy``.
        """
        # Guard: no autograd during Hebbian training
        assert not torch.is_grad_enabled(), "Autograd must be disabled for Hebbian training"

        x = x.to(self._device)
        y = y.to(self._device)
        x_flat = x.view(x.size(0), -1)

        # Quantize input to ternary {-1, 0, +1}
        x_ternary = ternary_sign(x_flat, epsilon=epsilon)

        # Snapshot weights for flip tracking
        old_w_hidden = self.hidden_layer.weight.unpack().clone()
        old_w_output = self.output_layer.weight.unpack().clone()

        # ── Compute modulators ─────────────────────────────────
        M_output, M_hidden, h_hidden_ternary, pred = self._compute_modulators(
            x_ternary, y, epsilon=epsilon
        )

        # ── Output layer update (NTH) ──────────────────────────
        # Δ = lr_output × M_outputᵀ @ h_hidden_ternary / batch_size
        output_scores = self.output_layer._latent_scores.scores
        output_scores = neuromodulated_update(
            output_scores,
            h_hidden_ternary,
            M_output,
            lr=lr_output,
            post=None,
        )
        self.output_layer._latent_scores.scores = output_scores

        # ── Hidden layer update (NTH) ──────────────────────────
        # Δ = lr_hidden × M_hiddenᵀ @ x_ternary / batch_size
        hidden_scores = self.hidden_layer._latent_scores.scores
        hidden_scores = neuromodulated_update(
            hidden_scores,
            x_ternary,
            M_hidden,
            lr=lr_hidden,
            post=None,
        )
        self.hidden_layer._latent_scores.scores = hidden_scores

        # Homeostatic decay
        if decay > 0:
            self.hidden_layer.apply_decay(decay)
            self.output_layer.apply_decay(decay)

        # Refresh ternary weights via hysteresis
        self.hidden_layer.refresh_weights()
        self.output_layer.refresh_weights()

        # Compute flip statistics
        new_w_hidden = self.hidden_layer.weight.unpack()
        new_w_output = self.output_layer.weight.unpack()
        n_flips_hidden = (old_w_hidden != new_w_hidden).sum().item()
        n_flips_output = (old_w_output != new_w_output).sum().item()
        total_hidden = new_w_hidden.numel()
        total_output = new_w_output.numel()

        return {
            "flip_rate_hidden": n_flips_hidden / max(total_hidden, 1),
            "flip_rate_output": n_flips_output / max(total_output, 1),
            "n_flips_hidden": n_flips_hidden,
            "n_flips_output": n_flips_output,
            "accuracy": (pred == y).float().mean().item(),
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
            epsilon: Dead-zone width for ``ternary_sign``.

        Returns:
            Predicted class indices, shape ``(batch,)``.
        """
        x = x.to(self._device)
        x_flat = x.view(x.size(0), -1)
        x_ternary = ternary_sign(x_flat, epsilon=epsilon)

        # Forward through hidden
        h = self.hidden_layer(x_ternary.float())
        h_ternary = ternary_sign(h, epsilon=epsilon)

        # Forward through output
        out = self.output_layer(h_ternary.float())
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

    # ── Training loop ──────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader | None = None,
        lr_hidden: float = 0.005,
        lr_output: float = 0.01,
        epochs: int = 10,
        decay: float = 0.0,
        epsilon: float = 0.1,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train both layers jointly for the specified number of epochs.

        Args:
            train_loader: DataLoader yielding ``(inputs, targets)``.
            test_loader: Optional DataLoader for per-epoch evaluation.
                If ``None``, evaluates on train_loader.
            lr_hidden: Learning rate for hidden layer.
            lr_output: Learning rate for output layer.
            epochs: Number of training epochs.
            decay: Homeostatic decay rate.
            epsilon: Dead-zone width for ``ternary_sign``.
            verbose: If ``True``, print per-epoch progress.

        Returns:
            Dict with per-epoch metrics:
            ``accuracy``, ``flip_rate_hidden``, ``flip_rate_output``.
        """
        history: dict[str, list[float]] = {
            "accuracy": [],
            "flip_rate_hidden": [],
            "flip_rate_output": [],
        }

        eval_loader = test_loader or train_loader

        # Bootstrap hidden layer with sparse random connectivity.
        # Without this, all-zero weights produce zero hidden output,
        # making Hebbian learning impossible for both layers.
        hidden_w = self.hidden_layer.weight.unpack()
        if torch.all(hidden_w == 0):
            _init_hidden_layer_connectivity(self.hidden_layer, density=0.1)

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            self.model.train()

            step_metrics: list[dict[str, float]] = []

            for x, y in train_loader:
                with torch.no_grad():
                    metrics = self.train_step(
                        x, y,
                        lr_hidden=lr_hidden,
                        lr_output=lr_output,
                        decay=decay,
                        epsilon=epsilon,
                    )
                    step_metrics.append(metrics)

            # Per-epoch evaluation
            acc = self.evaluate(eval_loader, epsilon=epsilon)
            avg_flip_h = sum(m["flip_rate_hidden"] for m in step_metrics) / max(len(step_metrics), 1)
            avg_flip_o = sum(m["flip_rate_output"] for m in step_metrics) / max(len(step_metrics), 1)

            history["accuracy"].append(acc)
            history["flip_rate_hidden"].append(avg_flip_h)
            history["flip_rate_output"].append(avg_flip_o)

            if verbose:
                epoch_time = time.time() - epoch_start
                print(
                    f"Epoch {epoch:2d}/{epochs}  "
                    f"Test Acc: {100 * acc:5.2f}%  "
                    f"Hidden Flips: {100 * avg_flip_h:6.3f}%/step  "
                    f"Output Flips: {100 * avg_flip_o:6.3f}%/step  "
                    f"Time: {epoch_time:.1f}s"
                )

        return history

    # ── Weight statistics ────────────────────────────────────────────

    @torch.no_grad()
    def get_weight_stats(self) -> dict[str, dict[str, float]]:
        """Compute weight statistics for both layers.

        Returns:
            Nested dict:
            ``{"hidden": {"pos_pct", "neg_pct", "zero_pct"},
              "output": {"pos_pct", "neg_pct", "zero_pct"}}``
        """
        stats = {}
        for name, layer in [("hidden", self.hidden_layer), ("output", self.output_layer)]:
            w = layer.weight.unpack()
            total = w.numel()
            stats[name] = {
                "pos_pct": 100.0 * (w == 1).sum().item() / max(total, 1),
                "neg_pct": 100.0 * (w == -1).sum().item() / max(total, 1),
                "zero_pct": 100.0 * (w == 0).sum().item() / max(total, 1),
            }
        return stats

    def __repr__(self) -> str:
        return (
            f"NTHMultiLayerClassifier("
            f"{self._in_features}\u2192{self._hidden_size}\u2192{self._out_features}, "
            f"modulator={self._modulator_mode})"
        )
