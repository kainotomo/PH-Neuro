"""Greedy layer-wise Hebbian training for multi-layer networks.

Trains a :class:`~ph_neuro.models.mlp.HebbianMLP` one layer at a time:

- **Hidden layers**: unsupervised self-organizing Hebbian — each layer
  learns the statistical structure of its input without labels.
- **Output layer**: supervised winner-take-all (WTA) Hebbian — strengthens
  the correct class and weakens the predicted (wrong) class.

After each layer is trained, it is frozen via ``requires_hebbian_(False)``
so subsequent layers learn from a stable representation.

No ``.backward()``, optimizers, or loss functions are used anywhere.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.core.activation import ternary_sign
from ph_neuro.layers.conv import TernaryHebbianConv2d
from ph_neuro.layers.linear import TernaryHebbianLinear
from ph_neuro.models.mlp import HebbianMLP


# ── Layer configuration ──────────────────────────────────────────


class LayerConfig:
    """Hyperparameters for training one layer in the greedy pipeline.

    Args:
        lr: Hebbian learning rate.
        epochs: Number of training epochs for this layer.
        hebbian_rule: Hebbian rule for hidden layers (``'basic'``,
            ``'oja'``, ``'bcm'``, ``'class_guided'``,
            ``'online_competitive'``). Ignored for output layer.
            ``'class_guided'`` uses label information (each hidden
            neuron learns the pattern of its assigned class).
            ``'online_competitive'`` uses winner-take-all with
            conscience — each sample, the best-matching neuron
            learns (online k-means with ternary prototypes).
        decay: Homeostatic decay rate.
        theta_upper: Hysteresis upper threshold. If ``None``, uses
            the layer's current value.
        theta_lower: Hysteresis lower threshold. If ``None``, uses
            the layer's current value.
        anti_hebbian: If ``True``, apply anti-Hebbian weakening to
            all non-target output classes (output layer only).
    """

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 5,
        hebbian_rule: str = "basic",
        decay: float = 0.0,
        theta_upper: float | None = None,
        theta_lower: float | None = None,
        anti_hebbian: bool = False,
    ):
        self.lr = lr
        self.epochs = epochs
        self.hebbian_rule = hebbian_rule
        self.decay = decay
        self.theta_upper = theta_upper
        self.theta_lower = theta_lower
        self.anti_hebbian = anti_hebbian

    @classmethod
    def default_hidden(cls) -> LayerConfig:
        """Default config for a hidden layer (unsupervised)."""
        return cls(lr=0.01, epochs=5, hebbian_rule="basic", decay=0.0)

    @classmethod
    def default_output(cls) -> LayerConfig:
        """Default config for the output layer (supervised WTA)."""
        return cls(lr=0.01, epochs=5, hebbian_rule="basic", decay=0.0, anti_hebbian=False)


def _init_hidden_layer_connectivity(
    layer: TernaryHebbianLinear,
    density: float = 0.1,
    seed: int | None = None,
) -> None:
    """Bootstrap an unsupervised hidden layer with random ternary weights.

    Before any Hebbian update, an all-zero weight layer produces zero
    output, making Hebbian learning impossible (no post-activation).

    This function sets a fraction of latent scores above ``theta_upper``
    (and the remainder below ``theta_lower``) so the first ``refresh_weights``
    call creates a sparse random connectivity pattern, bootstrapping
    non-zero output for self-organizing Hebbian.

    Args:
        layer: The ``TernaryHebbianLinear`` layer to initialize.
        density: Fraction of weights to activate (random +/-1).
            Default 0.1 (10% connectivity).
        seed: Optional random seed for reproducibility.
    """
    if seed is not None:
        torch.manual_seed(seed)

    scores = layer._latent_scores.scores
    n_total = scores.numel()
    n_active = max(1, int(n_total * density))

    # Random indices to activate
    idx = torch.randperm(n_total, device=scores.device)[:n_active]
    flat_scores = scores.flatten()
    # Set active scores above theta_upper with random sign
    active_values = (
        (torch.randint(0, 2, (n_active,), device=scores.device) * 2 - 1).float()
        * (layer.theta_upper + 1.0)
    ).to(scores.dtype)
    flat_scores[idx] = active_values

    # Force refresh to materialize ternary weights
    layer.refresh_weights()


def train_competitive_epoch(
    layer: TernaryHebbianLinear,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.0,
) -> dict[str, float]:
    """Train one hidden layer via competitive (winner-take-all) Hebbian.

    For each input, the hidden neuron with the strongest activation
    "wins" — only the winner's weights are updated (strengthening
    connections to active input pixels). Other neurons are inhibited
    via anti-Hebbian decay. This forces different hidden neurons to
    specialize on different input patterns (like online k-means).

    Args:
        layer: The ``TernaryHebbianLinear`` layer to train.
        loader: DataLoader yielding ``(inputs, targets)`` (targets ignored).
        frozen_encoder: Optional module running before ``layer``.
        device: Torch device.
        lr: Hebbian learning rate.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign`` (default 0 for hidden).

    Returns:
        Dict with ``flip_rate`` and ``n_flips`` averaged over batches.
    """
    total_flips = 0
    total_weights = 0
    n_batches = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)

        # Quantize input to ternary
        h = ternary_sign(x_flat, epsilon=epsilon)

        # Forward pass through frozen encoder (if any)
        if frozen_encoder is not None:
            with torch.no_grad():
                out = frozen_encoder(h.float())
                h = ternary_sign(out, epsilon=epsilon)

        # Get raw outputs (before ternary_sign) to find winners
        out = layer(h.float())

        # Snapshot weights before refresh
        old_w = layer.weight.unpack().clone()
        if old_w.device != device:
            old_w = old_w.to(device)

        # ── Competitive Hebbian ────────────────────────────────
        # Winner for each sample: hidden neuron with max activation
        winners = out.argmax(dim=1)  # (batch,)

        # Create one-hot winner mask: (batch, out_features)
        winner_mask = F.one_hot(winners, layer._out_features).float()

        # Winner update: strengthen connections to active (±1) inputs
        # Only the winner neuron's weights are updated
        # post = ternary_sign(out) -> use winner mask as "post" for winners
        post = winner_mask.to(device)

        # Basic Hebbian for winners only
        delta = lr * (post.T @ h.float())
        layer._latent_scores.scores += delta.to(layer._latent_scores.scores.dtype)

        # Anti-Hebbian for non-winners (weaken)
        non_winner_mask = 1.0 - winner_mask
        anti_delta = -lr * 0.1 * (non_winner_mask.T @ h.float())
        layer._latent_scores.scores += anti_delta.to(layer._latent_scores.scores.dtype)

        if decay > 0:
            layer.apply_decay(decay)

        layer.refresh_weights()

        # Track flips
        new_w = layer.weight.unpack()
        if new_w.device != device:
            new_w = new_w.to(device)
        total_flips += (old_w != new_w).sum().item()
        total_weights += new_w.numel()
        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
    }


# ── Class-guided Hebbian (supervised hidden layer) ─────────────


def _assign_neuron_classes(n_neurons: int, n_classes: int = 10) -> torch.Tensor:
    """Assign each hidden neuron to a fixed output class.

    Neurons are evenly distributed across classes. Returns an int64
    tensor of shape ``(n_neurons,)`` with values in ``[0, n_classes)``.

    Args:
        n_neurons: Number of hidden neurons.
        n_classes: Number of output classes (default 10 for MNIST).

    Returns:
        Tensor of neuron class assignments.
    """
    classes = torch.arange(n_classes).repeat((n_neurons + n_classes - 1) // n_classes)
    return classes[:n_neurons]


def train_class_guided_epoch(
    layer: TernaryHebbianLinear,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.1,
    neuron_classes: torch.Tensor | None = None,
    anti_lr: float = 0.01,
) -> dict[str, float]:
    """Train one hidden layer via class-guided Hebbian learning.

    Each hidden neuron is assigned a fixed output class. During training,
    for each input sample:

    - Neurons assigned to the **correct class**: Hebbian update — strengthen
      connections to active (+1) input pixels, weaken connections to
      inactive (-1/0) pixels.
    - Neurons assigned to **other classes**: anti-Hebbian update — weaken
      connections to active pixels, strengthen connections to inactive pixels.

    This creates class-specific feature detectors: each hidden neuron learns
    the average activation pattern of its assigned class.

    The Hebbian update uses the layer's raw float output as post (truncated
    to int8 by ``hebbian_update``), giving a richer learning signal than
    ternary {-1,0,+1} alone. Inter-layer communication remains ternary.

    Args:
        layer: The ``TernaryHebbianLinear`` layer to train.
        loader: DataLoader yielding ``(inputs, targets)``.
        frozen_encoder: Optional module running before ``layer``.
        device: Torch device.
        lr: Hebbian learning rate for class-matching neurons.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign``.
        neuron_classes: Tensor mapping each hidden neuron to a class
            index. If ``None``, uses evenly-distributed random assignment.
        anti_lr: Anti-Hebbian learning rate for non-matching neurons.

    Returns:
        Dict with ``flip_rate`` and ``n_flips`` averaged over batches.
    """
    n_out = layer._out_features
    if neuron_classes is None:
        neuron_classes = _assign_neuron_classes(n_out).to(device)
    else:
        neuron_classes = neuron_classes.to(device)

    # One-hot class assignments: (out_features, n_classes)
    class_onehot = F.one_hot(neuron_classes, num_classes=10).float()

    total_flips = 0
    total_weights = 0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_flat = x.view(x.size(0), -1)

        # Input is ternary {-1, 0, +1}
        h = ternary_sign(x_flat, epsilon=epsilon)

        # Forward pass through frozen encoder (ternary between layers)
        if frozen_encoder is not None:
            with torch.no_grad():
                for enc_layer in frozen_encoder:
                    h_float = enc_layer(h.float())
                    h = ternary_sign(h_float, epsilon=epsilon)

        # Forward pass through current layer (for output to next layer)
        out = layer(h.float())

        # Snapshot weights before refresh
        old_w = layer.weight.unpack().clone()
        if old_w.device != device:
            old_w = old_w.to(device)

        # ── Class-guided Hebbian update ────────────────────────
        # Use the CLASS LABEL as post, not the raw output.
        # Each hidden neuron is assigned a fixed class.
        # When input matches the neuron's class: post = +1 (Hebbian)
        # When input doesn't match: post = -1 (anti-Hebbian)
        # This gives a clean learning signal independent of the
        # current (possibly noisy) forward pass.
        scores = layer._latent_scores.scores

        # neuron_classes[k] = class assigned to hidden neuron k
        # class_onehot: (out_features, 10)
        class_onehot = F.one_hot(neuron_classes, num_classes=10).float()
        y_onehot = F.one_hot(y, num_classes=10).float()  # (batch, 10)

        # post: (batch, out_features) — +1 for matching, -1 for non-matching
        post = 2.0 * (y_onehot @ class_onehot.T) - 1.0  # {+1, -1}

        # Hebbian with ternary post: ΔW = lr × postᵀ @ pre
        delta = lr * (post.T @ h.float())
        scores += delta.to(scores.dtype)

        if decay > 0:
            layer.apply_decay(decay)

        layer.refresh_weights()

        # Track flips
        new_w = layer.weight.unpack()
        if new_w.device != device:
            new_w = new_w.to(device)
        total_flips += (old_w != new_w).sum().item()
        total_weights += new_w.numel()
        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
    }


# ── Online competitive Hebbian (brain-like prototype learning) ──


def train_online_competitive_epoch(
    layer: TernaryHebbianLinear,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.1,
) -> dict[str, float]:
    """Train one hidden layer via online competitive Hebbian with conscience.

    Processes samples **one at a time** (online — brain-like). Each hidden
    neuron acts as a prototype (a representative pattern). On each sample:

    1. Compute raw output for all neurons — the match score
    2. Apply **conscience bias**: penalize neurons that win too often,
       giving less-frequent winners a chance (ensures full codebook usage)
    3. **Winner**: the neuron with the highest adjusted score
    4. **Winner update**: move winner's weight toward the input pattern.
       ``Δscore[winner, i] = lr × input[i]``
    5. **Refresh weights**: hysteresis thresholding of ternary weights

    This is online k-means / vector quantization with ternary prototypes.
    No labels are used — purely unsupervised, purely local.

    Args:
        layer: The ``TernaryHebbianLinear`` layer to train.
        loader: DataLoader yielding ``(inputs, targets)`` (targets ignored).
        frozen_encoder: Optional module running before ``layer``.
        device: Torch device.
        lr: Learning rate for winner weight update.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign``.

    Returns:
        Dict with ``flip_rate`` and ``n_flips`` averaged over steps.
    """
    n_out = layer._out_features
    win_counts = torch.zeros(n_out, device=device)
    total_steps = 0
    total_flips = 0
    total_weights = 0
    n_batches = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)

        # Input is ternary {-1, 0, +1}
        h = ternary_sign(x_flat, epsilon=epsilon).float()

        # Forward pass through frozen encoder (ternary between layers)
        if frozen_encoder is not None:
            with torch.no_grad():
                for enc_layer in frozen_encoder:
                    h_float = enc_layer(h)
                    h = ternary_sign(h_float, epsilon=epsilon).float()

        # Process each sample individually (online learning)
        for s in range(h.shape[0]):
            single_input = h[s:s+1]  # (1, in_features)

            # Forward pass
            out = layer(single_input)  # (1, out_features)

            # Snapshot weights before update
            old_w = layer.weight.unpack().clone()
            if old_w.device != device:
                old_w = old_w.to(device)

            # ── Competitive selection with conscience ──────────
            # Conscience bias: penalize neurons that win too often
            # bias = 0.01  small penalty to ensure fairness
            if total_steps > 0:
                fair_share = 1.0 / n_out
                freq = win_counts / total_steps
                conscience_bias = 0.1 * (freq - fair_share)
            else:
                conscience_bias = torch.zeros(n_out, device=device)

            adjusted = out.flatten() - conscience_bias
            winner = adjusted.argmax().item()

            # ── Winner update ──────────────────────────────────
            # Move winner's weights toward the input pattern:
            # Δscore[winner, i] = lr × input[i]
            # For +1 input: strengthen (+1) weight
            # For -1 input: weaken toward -1 weight
            # For  0 input: no change
            scores = layer._latent_scores.scores
            delta = lr * single_input.flatten()  # (in_features,)
            scores[winner] += delta.to(scores.dtype)

            win_counts[winner] += 1.0
            total_steps += 1

            if decay > 0:
                layer.apply_decay(decay)

            layer.refresh_weights()

            # Track flips
            new_w = layer.weight.unpack()
            if new_w.device != device:
                new_w = new_w.to(device)
            total_flips += (old_w != new_w).sum().item()
            total_weights += new_w.numel()

        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
    }


# ── Training helpers (self-organizing Hebbian) ──────────────────


def train_unsupervised_epoch(
    layer: TernaryHebbianLinear,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.1,
) -> dict[str, float]:
    """Train one hidden layer for one epoch via self-organizing Hebbian.

    The pre-activation for the Hebbian update is always ternary {-1,0,+1}
    (bounded, memory-efficient). The post-activation is the layer's raw
    float output, which gets truncated to int8 by ``hebbian_update``,
    giving a richer range of values than ternary alone.

    Ternary ``ternary_sign`` is applied between hidden layers so that
    all inter-layer communication stays ternary (memory efficient).

    Args:
        layer: The ``TernaryHebbianLinear`` layer to train.
        loader: DataLoader yielding ``(inputs, targets)`` (targets ignored).
        frozen_encoder: Optional module running before ``layer`` (all
            earlier, frozen layers). ``None`` if this is the first layer.
        device: Torch device.
        lr: Hebbian learning rate.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign``.

    Returns:
        Dict with ``flip_rate`` and ``n_flips`` averaged over batches.
    """
    total_flips = 0
    total_weights = 0
    n_batches = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        x_flat = x.view(x.size(0), -1)

        # Input is always ternary {-1, 0, +1}
        h = ternary_sign(x_flat, epsilon=epsilon)

        # Forward pass through frozen encoder (ternary between layers)
        if frozen_encoder is not None:
            with torch.no_grad():
                for enc_layer in frozen_encoder:
                    h_float = enc_layer(h.float())
                    h = ternary_sign(h_float, epsilon=epsilon)

        # Forward pass through current layer — get raw float output
        out = layer(h.float())

        # Snapshot weights before refresh
        old_w = layer.weight.unpack().clone()
        if old_w.device != device:
            old_w = old_w.to(device)

        # Hebbian update: pre is ternary, post is raw float (truncated to int8)
        layer.hebbian_update(h, out, lr)

        if decay > 0:
            layer.apply_decay(decay)

        layer.refresh_weights()

        # Track flips
        new_w = layer.weight.unpack()
        if new_w.device != device:
            new_w = new_w.to(device)
        total_flips += (old_w != new_w).sum().item()
        total_weights += new_w.numel()
        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
    }


def train_supervised_wta_epoch(
    layer: TernaryHebbianLinear,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.1,
    anti_hebbian: bool = False,
) -> dict[str, float]:
    """Train the output layer for one epoch via WTA Hebbian.

    Uses the winner-take-all strategy proven in Phase 0:
    - Correct prediction: strengthen the winning class connection
    - Wrong prediction: strengthen the correct class AND weaken the
      predicted (wrong) class connection
    - Optionally (anti_hebbian): weaken all non-target classes

    Args:
        layer: The output ``TernaryHebbianLinear`` layer.
        loader: DataLoader yielding ``(inputs, targets)``.
        frozen_encoder: Module running before ``layer`` (all frozen
            hidden layers). ``None`` if no hidden layers.
        device: Torch device.
        lr: Hebbian learning rate.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign``.
        anti_hebbian: If ``True``, apply anti-Hebbian to all non-target
            output classes (not just the wrong prediction).

    Returns:
        Dict with ``flip_rate``, ``n_flips``, and ``accuracy``.
    """
    total_flips = 0
    total_weights = 0
    correct = 0
    total = 0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x_flat = x.view(x.size(0), -1)

        # Input is always ternary {-1, 0, +1}
        # Forward pass through frozen encoder (ternary between layers)
        h = ternary_sign(x_flat, epsilon=epsilon)
        if frozen_encoder is not None:
            with torch.no_grad():
                for enc_layer in frozen_encoder:
                    h_float = enc_layer(h.float())
                    h = ternary_sign(h_float, epsilon=epsilon)

        # Forward pass through output layer
        out = layer(h.float())
        pred = out.argmax(dim=1)

        # Track accuracy
        correct += (pred == y).sum().item()
        total += y.size(0)

        # Snapshot weights before refresh — ensure on correct device
        old_w = layer.weight.unpack().clone()
        if old_w.device != device:
            old_w = old_w.to(device)

        # ── WTA Hebbian update ─────────────────────────────────
        scores = layer._latent_scores.scores
        wrong_mask = pred != y

        if anti_hebbian:
            # Anti-Hebbian: weaken all non-target output classes
            target_hot = F.one_hot(y, layer._out_features).float()
            # Strengthen target, weaken everything else
            anti_targets = 1.0 - target_hot
            delta = lr * (
                target_hot.T @ h.float()
                - anti_targets.T @ h.float()
            )
            scores += delta.to(scores.dtype)
        elif wrong_mask.any():
            # Standard WTA: strengthen correct class, weaken wrong prediction
            correct_hot = F.one_hot(y[wrong_mask], layer._out_features).float()
            pred_hot = F.one_hot(pred[wrong_mask], layer._out_features).float()
            delta = lr * (correct_hot.T @ h.float()[wrong_mask]
                          - pred_hot.T @ h.float()[wrong_mask])
            scores += delta.to(scores.dtype)
        else:
            # All correct: strengthen each predicted class
            correct_hot = F.one_hot(y, layer._out_features).float()
            delta = lr * (correct_hot.T @ h.float())
            scores += delta.to(scores.dtype)

        if decay > 0:
            layer.apply_decay(decay)

        layer.refresh_weights()

        # Track flips — ensure on correct device
        new_w = layer.weight.unpack()
        if new_w.device != device:
            new_w = new_w.to(device)
        total_flips += (old_w != new_w).sum().item()
        total_weights += new_w.numel()
        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
        "accuracy": correct / max(total, 1),
    }


# ── Multi-layer Hebbian classifier ───────────────────────────────


class MultiLayerHebbianClassifier:
    """Multi-layer ternary Hebbian classifier with greedy training.

    Wraps a :class:`~ph_neuro.models.mlp.HebbianMLP` and trains it
    layer-by-layer using greedy layer-wise Hebbian learning.

    - **Hidden layers**: trained via unsupervised self-organizing Hebbian
      (learns statistical structure from input).
    - **Output layer**: trained via supervised WTA Hebbian (strengthen
      correct class, weaken wrong prediction).

    No ``.backward()``, optimizers, or loss functions are used.

    Args:
        layer_sizes: Sequence of layer sizes, e.g. ``[784, 256, 128, 10]``.
        theta_upper: Default hysteresis upper threshold for all layers.
        theta_lower: Default hysteresis lower threshold for all layers.
        device: Device to place the model on.

    Attributes:
        model: The underlying ``HebbianMLP``.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        theta_upper: float = 5.0,
        theta_lower: float = 1.0,
        device: torch.device | str | None = None,
    ):
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = HebbianMLP(
            layer_sizes=list(layer_sizes),
            theta_upper=theta_upper,
            theta_lower=theta_lower,
        ).to(self._device)

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._device

    @property
    def n_layers(self) -> int:
        """Number of Hebbian layers in the model."""
        return len(self.model.layers)

    # ── Greedy layer-wise training ────────────────────────────────

    def fit_greedy(
        self,
        train_loader: DataLoader,
        layer_configs: list[LayerConfig] | None = None,
        epsilon: float = 0.1,
        verbose: bool = True,
    ) -> dict[int, dict[str, list[float]]]:
        """Train all layers greedily, one at a time.

        Each layer's training uses its own ``LayerConfig``. Hidden layers
        (all but last) use unsupervised self-organizing Hebbian. The last
        layer uses supervised WTA Hebbian.

        Args:
            train_loader: DataLoader yielding ``(inputs, targets)``.
            layer_configs: Per-layer configs. If ``None``, uses defaults:
                hidden layers get :meth:`LayerConfig.default_hidden`, output
                layer gets :meth:`LayerConfig.default_output`. Length must
                equal ``n_layers``.
            epsilon: Dead-zone for ``ternary_sign``.
            verbose: If ``True``, print progress.

        Returns:
            Nested dict: ``{layer_idx: {"accuracy": [...], "flip_rate": [...]}}``
            where each value is a list of per-epoch metrics.
        """
        n = self.n_layers
        if layer_configs is None:
            layer_configs = [LayerConfig.default_hidden() for _ in range(n - 1)]
            layer_configs.append(LayerConfig.default_output())
        assert len(layer_configs) == n, (
            f"Expected {n} layer configs, got {len(layer_configs)}"
        )

        history: dict[int, dict[str, list[float]]] = {
            i: {"accuracy": [], "flip_rate": []} for i in range(n)
        }

        # Freeze all layers initially
        for i in range(n):
            self.model.get_layer(i).requires_hebbian_(False)

        for idx, cfg in enumerate(layer_configs):
            layer = self.model.get_layer(idx)
            is_output = idx == n - 1

            # Configure this layer
            layer.requires_hebbian_(True)
            if cfg.hebbian_rule is not None:
                layer._hebbian_rule = cfg.hebbian_rule
            if cfg.theta_upper is not None:
                layer.theta_upper = cfg.theta_upper
            if cfg.theta_lower is not None:
                layer.theta_lower = cfg.theta_lower

            # Bootstrap hidden layers with sparse random connectivity.
            # Only bootstrap if weights are all zero (not yet trained).
            if not is_output:
                w = layer.weight.unpack()
                if torch.all(w == 0):
                    _init_hidden_layer_connectivity(layer, density=0.1)

            # Build frozen encoder from all layers before this one
            if idx > 0:
                frozen_encoder = nn.Sequential(*list(self.model.layers[:idx]))
                frozen_encoder.requires_grad_(False)
            else:
                frozen_encoder = None

            if verbose:
                layer_name = f"Layer {idx + 1}/{n}"
                rule_str = f"WTA" if is_output else cfg.hebbian_rule
                print(
                    f"\n  Training {layer_name} [{layer._in_features}"
                    f"\u2192{layer._out_features}] "
                    f"rule={rule_str}, lr={cfg.lr}, epochs={cfg.epochs}"
                )
                if not is_output:
                    label = {
                        "class_guided": "class-guided",
                        "online_competitive": "online competitive (WTA + conscience)",
                    }.get(cfg.hebbian_rule, "self-organizing")
                    print(f"    Mode: {label}")
                else:
                    print(f"    Mode: supervised WTA (anti-Hebbian={cfg.anti_hebbian})")

            for epoch in range(1, cfg.epochs + 1):
                if not is_output:
                    if cfg.hebbian_rule == "class_guided":
                        metrics = train_class_guided_epoch(
                            layer=layer,
                            loader=train_loader,
                            frozen_encoder=frozen_encoder,
                            device=self._device,
                            lr=cfg.lr,
                            decay=cfg.decay,
                            epsilon=epsilon,
                            anti_lr=cfg.lr * 0.1,
                        )
                    elif cfg.hebbian_rule == "online_competitive":
                        metrics = train_online_competitive_epoch(
                            layer=layer,
                            loader=train_loader,
                            frozen_encoder=frozen_encoder,
                            device=self._device,
                            lr=cfg.lr,
                            decay=cfg.decay,
                            epsilon=epsilon,
                        )
                    else:
                        metrics = train_unsupervised_epoch(
                            layer=layer,
                            loader=train_loader,
                            frozen_encoder=frozen_encoder,
                            device=self._device,
                            lr=cfg.lr,
                            decay=cfg.decay,
                            epsilon=epsilon,
                        )
                else:
                    metrics = train_supervised_wta_epoch(
                        layer=layer,
                        loader=train_loader,
                        frozen_encoder=frozen_encoder,
                        device=self._device,
                        lr=cfg.lr,
                        decay=cfg.decay,
                        epsilon=epsilon,
                        anti_hebbian=cfg.anti_hebbian,
                    )

                # Store metrics
                history[idx]["flip_rate"].append(metrics["flip_rate"])
                if is_output:
                    history[idx]["accuracy"].append(metrics["accuracy"])
                elif frozen_encoder is not None:
                    acc = self.evaluate(train_loader, epsilon=epsilon)
                    history[idx]["accuracy"].append(acc)

                if verbose:
                    acc_str = (
                        f"Acc: {100 * history[idx]['accuracy'][-1]:5.2f}%"
                        if history[idx]["accuracy"]
                        else ""
                    )
                    print(
                        f"    Epoch {epoch:2d}/{cfg.epochs}  "
                        f"Flips: {100 * metrics['flip_rate']:6.3f}%/step  "
                        f"{acc_str}"
                    )

            # Freeze this layer before moving to the next
            layer.requires_hebbian_(False)

        return history

    def fit_supervised(
        self,
        train_loader: DataLoader,
        lr: float = 0.01,
        epochs: int = 10,
        decay: float = 0.0,
        epsilon: float = 0.1,
        anti_hebbian: bool = False,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train ALL layers simultaneously with supervised WTA Hebbian.

        All layers are updated simultaneously using the output layer's
        WTA error signal. Each hidden layer is updated based on its own
        pre/post activations during the forward pass, with the post
        modulated by whether the final prediction was correct.

        This avoids the "dead hidden layer" problem of unsupervised
        training because every layer gets a supervised signal through
        the forward pass activity pattern.

        Args:
            train_loader: DataLoader yielding ``(inputs, targets)``.
            lr: Hebbian learning rate (applied to all layers).
            epochs: Number of training epochs.
            decay: Homeostatic decay rate.
            epsilon: Dead-zone for ``ternary_sign``.
            anti_hebbian: Apply anti-Hebbian to all non-target output classes.
            verbose: If ``True``, print progress.

        Returns:
            Dict with ``accuracy`` and ``flip_rate`` per epoch.
        """
        n = self.n_layers

        # Enable all layers
        for i in range(n):
            self.model.get_layer(i).requires_hebbian_(True)
            # Bootstrap hidden layers with sparse random connectivity
            if i < n - 1:
                _init_hidden_layer_connectivity(self.model.get_layer(i), density=0.1)

        history: dict[str, list[float]] = {"accuracy": [], "flip_rate": []}

        for epoch in range(1, epochs + 1):
            total_flips = 0
            total_weights = 0
            correct = 0
            total = 0
            n_batches = 0

            for x, y in train_loader:
                x = x.to(self._device)
                y = y.to(self._device)
                x_flat = x.view(x.size(0), -1)

                # ── Forward pass through all layers ────────────
                # Store pre and post for each layer
                h = ternary_sign(x_flat, epsilon=epsilon)
                layer_pres: list[torch.Tensor] = []
                layer_posts: list[torch.Tensor] = []

                for i, layer in enumerate(self.model.layers):
                    layer_pres.append(h.clone())
                    out = layer(h.float())
                    if i < n - 1:
                        post = ternary_sign(out, epsilon=epsilon)
                    else:
                        post = ternary_sign(out, epsilon=0.0)  # output: no dead zone
                    layer_posts.append(post)
                    h = post

                # Final output and prediction
                final_out = layer_posts[-1]
                pred = final_out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                # Snapshot all weights
                old_weights = [
                    layer.weight.unpack().clone().to(self._device)
                    for layer in self.model.layers
                ]

                # ── Update each layer using WTA logic ──────────
                wrong_mask = pred != y

                for i, layer in enumerate(self.model.layers):
                    pre = layer_pres[i]
                    post = layer_posts[i]
                    scores = layer._latent_scores.scores

                    if i == n - 1:
                        # Output layer: standard WTA
                        if wrong_mask.any():
                            correct_hot = F.one_hot(y[wrong_mask], layer._out_features).float()
                            pred_hot = F.one_hot(pred[wrong_mask], layer._out_features).float()
                            delta = lr * (
                                correct_hot.T @ pre[wrong_mask].float()
                                - pred_hot.T @ pre[wrong_mask].float()
                            )
                            scores += delta.to(scores.dtype)
                        else:
                            correct_hot = F.one_hot(y, layer._out_features).float()
                            delta = lr * (correct_hot.T @ pre.float())
                            scores += delta.to(scores.dtype)
                    else:
                        # Hidden layer: sparse top-k competitive Hebbian.
                        # Only the top-k% most active hidden neurons get a
                        # Hebbian update per sample. This forces different
                        # hidden neurons to specialize on different inputs.
                        k = max(1, layer._out_features // 10)  # top 10%
                        post_float = post.float()  # (batch, hidden_size)

                        # For each sample, find top-k hidden neurons
                        _, topk_idx = post_float.topk(k, dim=1)
                        topk_mask = torch.zeros_like(post_float)
                        topk_mask.scatter_(1, topk_idx, 1.0)

                        # Hebbian for winners: strengthen active patterns
                        delta_win = lr * (topk_mask.T @ pre.float())
                        scores += delta_win.to(scores.dtype)

                        # Anti-Hebbian for non-winners
                        non_winner_mask = 1.0 - topk_mask
                        delta_lose = -lr * 0.01 * (non_winner_mask.T @ pre.float())
                        scores += delta_lose.to(scores.dtype)

                    if decay > 0:
                        layer.apply_decay(decay)
                    layer.refresh_weights()

                # Track flips
                for i, layer in enumerate(self.model.layers):
                    new_w = layer.weight.unpack().to(self._device)
                    total_flips += (old_weights[i] != new_w).sum().item()
                    total_weights += new_w.numel()
                n_batches += 1

            epoch_acc = correct / max(total, 1)
            epoch_flip = total_flips / max(total_weights, 1)
            history["accuracy"].append(epoch_acc)
            history["flip_rate"].append(epoch_flip)

            if verbose:
                print(
                    f"Epoch {epoch:2d}/{epochs}  "
                    f"Acc: {100 * epoch_acc:5.2f}%  "
                    f"Flips: {100 * epoch_flip:6.3f}%/step"
                )

        return history

    # ── Inference ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """Predict class labels through the full multi-layer stack.

        Applies ``ternary_sign(x, epsilon)`` at the input and between
        hidden layers for consistency with training.

        Args:
            x: Input tensor, shape ``(batch, *)``.
            epsilon: Dead-zone for ``ternary_sign``. Must match the
                value used during training.

        Returns:
            Predicted class indices, shape ``(batch,)``.
        """
        x = x.to(self._device)
        x_flat = x.view(x.size(0), -1)

        # Forward pass: ternary between hidden layers (memory efficient)
        # Output layer keeps raw float values for argmax
        h = ternary_sign(x_flat, epsilon=epsilon)
        for i, layer in enumerate(self.model.layers):
            h = layer(h.float())
            if i < len(self.model.layers) - 1:
                h = ternary_sign(h, epsilon=epsilon)
        return h.argmax(dim=1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epsilon: float = 0.1,
    ) -> float:
        """Evaluate accuracy on a data loader.

        Args:
            loader: DataLoader yielding ``(inputs, targets)``.
            epsilon: Dead-zone for ``ternary_sign``.

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

    # ── Weight statistics ──────────────────────────────────────────

    @torch.no_grad()
    def get_layer_weight_stats(self, layer_idx: int) -> dict[str, float]:
        """Compute weight statistics for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Dict with ``pos_pct``, ``neg_pct``, ``zero_pct``.
        """
        w = self.model.get_layer(layer_idx).weight.unpack()
        total = w.numel()
        return {
            "pos_pct": 100.0 * (w == 1).sum().item() / max(total, 1),
            "neg_pct": 100.0 * (w == -1).sum().item() / max(total, 1),
            "zero_pct": 100.0 * (w == 0).sum().item() / max(total, 1),
        }

    @torch.no_grad()
    def get_all_weight_stats(self) -> list[dict[str, float]]:
        """Compute weight statistics for all layers.

        Returns:
            List of dicts, one per layer.
        """
        return [self.get_layer_weight_stats(i) for i in range(self.n_layers)]

    # ── Convenience ────────────────────────────────────────────────

    def __repr__(self) -> str:
        sizes = [layer._in_features for layer in self.model.layers] + [
            self.model.layers[-1]._out_features
        ]
        return (
            f"MultiLayerHebbianClassifier({sizes[0]}"
            + "".join(f"\u2192{s}" for s in sizes[1:])
            + f", {self.n_layers} layers)"
        )


# ═══════════════════════════════════════════════════════════════════
# CNN Training Functions (Phase 1.2)
# ═══════════════════════════════════════════════════════════════════


def _init_conv_connectivity(
    conv_layer: TernaryHebbianConv2d,
    density: float = 0.1,
    seed: int | None = None,
) -> None:
    """Bootstrap a conv layer with sparse random ternary weights.

    Before any Hebbian update, all-zero weights produce zero output,
    making Hebbian learning impossible. This sets a fraction of latent
    scores above ``theta_upper`` to create random filter patterns.

    Args:
        conv_layer: ``TernaryHebbianConv2d`` to initialize.
        density: Fraction of weights to activate (default 0.1).
        seed: Optional random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)

    scores = conv_layer._latent_scores.scores
    n_total = scores.numel()
    n_active = max(1, int(n_total * density))

    idx = torch.randperm(n_total, device=scores.device)[:n_active]
    flat_scores = scores.flatten()
    active_values = (
        (torch.randint(0, 2, (n_active,), device=scores.device) * 2 - 1).float()
        * (conv_layer.theta_upper + 1.0)
    ).to(scores.dtype)
    flat_scores[idx] = active_values
    conv_layer.refresh_weights()


def train_conv_competitive_epoch(
    conv_layer: TernaryHebbianConv2d,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.1,
) -> dict[str, float]:
    """Train one conv hidden layer via per-position competitive Hebbian.

    At each spatial position ``(i,j)`` of the output feature map, the
    filter with the strongest activation "wins." The winner's weights
    are updated toward the input patch at that position; losers get a
    mild anti-Hebbian weakening.

    This is the convolutional analog of online competitive Hebbian
    from Phase 1.1 — different filters naturally specialize on different
    visual features (edges, colors, textures) because they win at
    different spatial positions.

    Args:
        conv_layer: The ``TernaryHebbianConv2d`` to train.
        loader: DataLoader yielding ``(inputs, targets)`` (targets ignored).
        frozen_encoder: Optional callable running before ``conv_layer``.
            If the encoder produces ``(N, C, H, W)`` output, it is used
            directly. If ``None``, input is treated as raw image.
        device: Torch device.
        lr: Hebbian learning rate.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign``.

    Returns:
        Dict with ``flip_rate`` and ``n_flips``.
    """
    total_flips = 0
    total_weights = 0
    n_batches = 0

    for batch in loader:
        # Support both labeled and unlabeled batches
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)

        # Quantize to ternary {-1, 0, +1}
        h = ternary_sign(x, epsilon=epsilon).float()

        # Forward through frozen encoder (if any)
        if frozen_encoder is not None:
            h = ternary_sign(frozen_encoder(h), epsilon=epsilon).float()

        # Forward through current conv layer — raw float output
        raw = conv_layer(h)  # (N, C_out, H_out, W_out)
        N, C_out, H_out, W_out = raw.shape
        L = H_out * W_out

        # Snapshot weights before refresh
        old_w = conv_layer.weight.unpack().clone()
        if old_w.device != device:
            old_w = old_w.to(device)

        # ── Per-position competitive Hebbian ────────────────────
        # Flatten spatial: raw → (N, C_out, L)
        raw_flat = raw.reshape(N, C_out, L)

        # Winner at each spatial position: filter with max activation
        winners = raw_flat.argmax(dim=1)  # (N, L) — winner filter index per position

        # One-hot winner mask: (N, L, C_out) → (N, C_out, L)
        winner_mask = F.one_hot(winners, C_out).float().permute(0, 2, 1)

        # ── Direct competitive Hebbian update ───────────────────
        # We update scores directly (not via hebbian_update) because
        # hebbian_update divides by (N * L), which kills the signal
        # when only 1/C_out of positions contribute to each filter.
        #
        # Δscore[f, c, kh, kw] = lr × Σ_{b,(i,j) where f wins} input_patch[b,i,j,c,kh,kw]
        #
        # Vectorized: patches is (N, C_in*kH*kW, L), winner_mask is (N, C_out, L)
        # delta_2d = lr × bmm(winner_mask, patches.T).sum(0)  → (C_out, C_in*kHW)

        # Unfold input to patches
        patches = torch.nn.functional.unfold(
            h,
            kernel_size=conv_layer.kernel_size,
            dilation=conv_layer._dilation,
            padding=conv_layer._padding,
            stride=conv_layer._stride,
        )  # (N, C_in*kH*kW, L)

        # Batch MatMul: (N, C_out, L) × (N, C_in*kHW, L)^T → (N, C_out, C_in*kHW)
        delta_2d = torch.bmm(winner_mask, patches.float().transpose(1, 2))  # (N, C_out, C_in*kHW)
        delta_2d = delta_2d.sum(dim=0)  # (C_out, C_in*kHW) — sum over batch
        delta_2d = lr * delta_2d  # no division by N*L!

        # Reshape to weight shape: (C_out, C_in, kH, kW)
        delta = delta_2d.reshape(
            conv_layer._out_channels, conv_layer._in_channels,
            *conv_layer.kernel_size,
        )

        scores = conv_layer._latent_scores.scores
        if scores.device != delta.device:
            scores = scores.to(delta.device)
        scores += delta.to(scores.dtype)

        if decay > 0:
            conv_layer.apply_decay(decay)

        conv_layer.refresh_weights()

        # Track flips
        new_w = conv_layer.weight.unpack()
        if new_w.device != device:
            new_w = new_w.to(device)
        total_flips += (old_w != new_w).sum().item()
        total_weights += new_w.numel()
        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
    }


def train_conv_class_guided_epoch(
    conv_layer: TernaryHebbianConv2d,
    loader: DataLoader,
    frozen_encoder: nn.Module | None,
    device: torch.device,
    lr: float,
    decay: float,
    epsilon: float = 0.1,
    n_classes: int = 10,
) -> dict[str, float]:
    """Train one conv layer via class-guided Hebbian learning.

    Each conv filter is assigned to a fixed output class. When a sample
    of the filter's class is presented, the filter strengthens toward
    the input pattern (Hebbian). When a different class is shown, the
    filter weakens (anti-Hebbian).

    This creates class-specific feature detectors in the conv layer,
    avoiding the "useful but not discriminative" problem of unsupervised
    competitive Hebbian.

    Args:
        conv_layer: The ``TernaryHebbianConv2d`` to train.
        loader: DataLoader yielding ``(inputs, targets)``.
        frozen_encoder: Optional callable running before ``conv_layer``.
        device: Torch device.
        lr: Hebbian learning rate for class-matching filters.
        decay: Homeostatic decay rate.
        epsilon: Dead-zone for ``ternary_sign``.
        n_classes: Number of output classes (default 10).

    Returns:
        Dict with ``flip_rate`` and ``n_flips``.
    """
    # Assign filters to classes
    n_filters = conv_layer._out_channels
    filter_classes = _assign_neuron_classes(n_filters, n_classes).to(device)
    class_onehot = F.one_hot(filter_classes, num_classes=n_classes).float()

    total_flips = 0
    total_weights = 0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Quantize to ternary {-1, 0, +1}
        h = ternary_sign(x, epsilon=epsilon).float()

        # Forward through frozen encoder (if any)
        if frozen_encoder is not None:
            h = ternary_sign(frozen_encoder(h), epsilon=epsilon).float()

        # Snapshot weights before refresh
        old_w = conv_layer.weight.unpack().clone()
        if old_w.device != device:
            old_w = old_w.to(device)

        # ── Class-guided Hebbian update (direct scores) ────────
        # For each filter: +1 for matching class, -0.1 for non-matching
        y_onehot = F.one_hot(y, num_classes=n_classes).float()  # (N, n_classes)
        match = y_onehot @ class_onehot.T  # (N, n_filters) — 1 if class matches
        post_2d = match - 0.1 * (1.0 - match)  # {+1.0 for match, -0.1 for non-match}

        # We update scores directly (not via hebbian_update which divides by N*L).
        # The update should be normalized by L (spatial positions) so each position
        # contributes proportionally:
        #   ΔW[f] = lr × Σ_b post[b,f] × mean_{i,j} input_patch[b,i,j]
        #
        # Unfold input to patches
        patches = torch.nn.functional.unfold(
            h,
            kernel_size=conv_layer.kernel_size,
            dilation=conv_layer._dilation,
            padding=conv_layer._padding,
            stride=conv_layer._stride,
        )  # (N, C_in*kH*kW, L)

        patches_mean = patches.mean(dim=2)  # (N, C_in*kH*kW) — avg over spatial positions

        # delta_2d: (n_filters, C_in*kH*kW)
        delta_2d = lr * (post_2d.T @ patches_mean)  # no N division (batch handles implicitly)

        # Reshape to weight shape
        delta = delta_2d.reshape(
            conv_layer._out_channels, conv_layer._in_channels,
            *conv_layer.kernel_size,
        )

        # Reshape to weight shape
        delta = delta_2d.reshape(
            conv_layer._out_channels, conv_layer._in_channels,
            *conv_layer.kernel_size,
        )

        scores = conv_layer._latent_scores.scores
        if scores.device != delta.device:
            scores = scores.to(delta.device)
        scores += delta.to(scores.dtype)

        if decay > 0:
            conv_layer.apply_decay(decay)

        conv_layer.refresh_weights()

        # Track flips
        new_w = conv_layer.weight.unpack()
        if new_w.device != device:
            new_w = new_w.to(device)
        total_flips += (old_w != new_w).sum().item()
        total_weights += new_w.numel()
        n_batches += 1

    return {
        "flip_rate": total_flips / max(total_weights, 1),
        "n_flips": total_flips,
    }


@torch.no_grad()
def evaluate_cnn(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.1,
) -> float:
    """Evaluate a HebbianCNN on a test set.

    Args:
        model: A ``HebbianCNN`` (or any ``nn.Module`` with a forward
            that returns ``(N, n_classes)`` logits).
        test_loader: DataLoader yielding ``(inputs, targets)``.
        device: Torch device.
        epsilon: Dead-zone for ``ternary_sign``.

    Returns:
        Accuracy as a float in ``[0.0, 1.0]``.
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x, epsilon=epsilon)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)
