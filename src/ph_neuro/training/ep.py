"""Equilibrium Propagation (EP) training for ternary Hebbian networks.

Implements TEP-1: a simplified EP algorithm for 2-layer ternary Hebbian
networks on MNIST (784→512→10).

EP contrasts two states of the same network on the same input:

1. **Free phase**: Standard forward pass → ``h_free``, ``y_free``
2. **Nudged phase**: Compute ``h_target = ternary_sign(S_out^T @ y_onehot)`` —
   what the hidden layer SHOULD produce for the correct class, derived
   from output layer latent scores (dense, not sparse ternary weights).
3. **EP update**: Difference of Hebbian correlations:

    ΔS_hidden = η_h × (h_target^T @ x - h_free^T @ x) / batch
    ΔS_output = η_o × (y_target^T @ h_free - y_free_ternary^T @ h_free) / batch

The key innovation over NTH-4b: ``h_target`` is a full TERNARY activation
vector ({-1,0,+1} per neuron), not a continuous scalar modulator. This
gives a structured, per-synapse signal.

No ``.backward()``, no optimizers, no loss functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.core.activation import ternary_sign
from ph_neuro.models.mlp import HebbianMLP
from ph_neuro.training.greedy import _init_hidden_layer_connectivity


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class EPConfig:
    """Hyperparameters for Equilibrium Propagation training.

    Args:
        lr_hidden: Hebbian learning rate for hidden layer EP update.
        lr_output: Hebbian learning rate for output layer WTA update.
        theta_upper: Hysteresis upper threshold.
        theta_lower: Hysteresis lower threshold.
        decay: Homeostatic decay rate (0 = no decay).
        epsilon: Dead-zone width for ``ternary_sign``.
        epochs: Number of training epochs.
        warmup_epochs: Number of initial epochs with only output WTA
            (no hidden layer EP update). This lets S_out grow meaningful
            structure before h_target becomes informative.
        hidden_update_on_correct: If True, apply hidden EP update even
            when the output prediction is correct. Default False (only
            wrong predictions trigger hidden update), matching WTA principle.
        hidden_density: Initial random connectivity for hidden layer.
    """

    lr_hidden: float = 0.005
    lr_output: float = 0.01
    theta_upper: float = 0.5
    theta_lower: float = 0.15
    decay: float = 0.0
    epsilon: float = 0.1
    epochs: int = 20
    warmup_epochs: int = 3
    hidden_update_on_correct: bool = False
    hidden_density: float = 0.1


# ── Helper: compute class-specific hidden target ─────────────────


def compute_hidden_target(
    y: torch.Tensor,
    S_out: torch.Tensor,
) -> torch.Tensor:
    """Compute class-specific ternary hidden targets from output latent scores.

    For each sample, ``h_target = ternary_sign(S_out^T @ y_onehot)``.

    This tells us: "given the output layer's current latent scores, what
    hidden activation pattern BEST produces the correct output class?"

    ``S_out`` has shape ``(n_classes, hidden_size)``. For class ``c`` row
    ``S_out[c, :]`` encodes how strongly the output neuron for class ``c``
    wants hidden neuron ``i`` to fire. Taking the sign of ``S_out[c, :]``
    gives the ideal ternary hidden state for class ``c``.

    The result is a ternary {-1, 0, +1} vector per sample — a full
    activation pattern, not a scalar modulator.

    Args:
        y: Ground-truth labels, shape ``(batch,)``.
        S_out: Output layer latent scores, shape
            ``(n_classes, hidden_size)`` (fp16, dense).

    Returns:
        Ternary hidden target tensor, shape ``(batch, hidden_size)``,
        values in {-1, 0, +1} as int8.
    """
    # S_out: (n_classes, hidden_size) -> transpose -> (hidden_size, n_classes)
    # y_onehot: (batch, n_classes)
    # h_target_raw: (batch, hidden_size)
    y_onehot = F.one_hot(y, num_classes=S_out.size(0)).float().to(S_out.device)
    h_target_raw = y_onehot @ S_out.float()  # (batch, hidden_size)
    h_target = ternary_sign(h_target_raw)
    return h_target


# ── EP training epoch ─────────────────────────────────────────────


def train_ep_epoch(
    hidden_layer: nn.Module,
    output_layer: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: EPConfig,
    epoch: int,
    verbose: bool = True,
) -> dict[str, float]:
    """Run one epoch of Equilibrium Propagation training.

    Both layers are updated jointly (not greedily). The hidden layer uses
    the EP difference-of-correlations update; the output layer uses
    standard WTA (same as Phase 0).

    Args:
        hidden_layer: The ``TernaryHebbianLinear`` hidden layer (784→512).
        output_layer: The ``TernaryHebbianLinear`` output layer (512→10).
        loader: DataLoader yielding ``(inputs, targets)``.
        device: Torch device.
        cfg: EP configuration.
        epoch: Current epoch number (1-indexed, for warmup logic).
        verbose: If True, print per-batch progress.

    Returns:
        Dict with ``accuracy``, ``flip_rate_hidden``, ``flip_rate_output``,
        ``h_target_correlation``, ``h_sparsity``, ``out_sparsity``.
    """
    # Guard: no autograd during Hebbian training
    assert not torch.is_grad_enabled(), "Autograd must be disabled for Hebbian training"

    total_correct = 0
    total_samples = 0
    total_flips_hidden = 0
    total_flips_output = 0
    total_weights_hidden = 0
    total_weights_output = 0
    total_target_corr = 0.0
    n_batches = 0

    is_warmup = epoch <= cfg.warmup_epochs
    hidden_dim = hidden_layer._out_features
    n_classes = output_layer._out_features

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x_flat = x.view(x.size(0), -1)

        # Quantize input to ternary {-1, 0, +1}
        x_ternary = ternary_sign(x_flat, epsilon=cfg.epsilon)

        # Snapshot weights for flip tracking
        old_w_hidden = hidden_layer.weight.unpack().clone()
        if old_w_hidden.device != device:
            old_w_hidden = old_w_hidden.to(device)
        old_w_output = output_layer.weight.unpack().clone()
        if old_w_output.device != device:
            old_w_output = old_w_output.to(device)

        # ── Free phase: forward pass ────────────────────────────
        h_raw = hidden_layer(x_ternary.float())  # (batch, hidden_size)
        h_free = ternary_sign(h_raw, epsilon=0.0)  # (batch, hidden_size), ternary

        out_raw = output_layer(h_free.float())  # (batch, n_classes)
        y_free = ternary_sign(out_raw, epsilon=0.0)  # (batch, n_classes), ternary
        pred = out_raw.argmax(dim=1)

        # Track accuracy
        batch_correct = (pred == y).sum().item()
        total_correct += batch_correct
        total_samples += y.size(0)

        # ── Output layer update (WTA / EP output difference) ────
        # ΔS_out = η_out × (y_target^T @ h_free - y_free^T @ h_free) / batch
        # For wrong predictions only: strengthen target, weaken predicted
        wrong_mask = pred != y
        output_scores = output_layer._latent_scores.scores

        if wrong_mask.any():
            wrong_idx = wrong_mask
            y_onehot = F.one_hot(y[wrong_idx], n_classes).float()
            y_pred_onehot = F.one_hot(pred[wrong_idx], n_classes).float()

            # EP difference: target correlation - free correlation
            delta_out = cfg.lr_output * (
                y_onehot.T @ h_free[wrong_idx].float()
                - y_pred_onehot.T @ h_free[wrong_idx].float()
            )
            output_scores += delta_out.to(output_scores.dtype)
        elif not is_warmup:
            # All correct: still strengthen correct class connections
            y_onehot = F.one_hot(y, n_classes).float()
            delta_out = cfg.lr_output * (y_onehot.T @ h_free.float())
            output_scores += delta_out.to(output_scores.dtype)

        # ── Hidden layer EP update ──────────────────────────────
        if not is_warmup:
            # Compute ternary hidden targets from output latent scores
            S_out = output_layer._latent_scores.scores  # (n_classes, hidden_size)
            h_target = compute_hidden_target(y, S_out)  # (batch, hidden_size), ternary

            # Track correlation between h_target and h_free
            # Higher correlation = hidden layer is aligning with targets
            batch_corr = (h_target == h_free).float().mean().item()
            total_target_corr += batch_corr

            # EP difference: target correlation - free correlation
            # ΔS_hidden = η_h × (h_target^T @ x_ternary - h_free^T @ x_ternary) / batch
            hidden_scores = hidden_layer._latent_scores.scores

            if cfg.hidden_update_on_correct:
                # Apply EP update for ALL samples
                delta_hidden = cfg.lr_hidden * (
                    h_target.float().T @ x_ternary.float()
                    - h_free.float().T @ x_ternary.float()
                )
                hidden_scores += delta_hidden.to(hidden_scores.dtype)
            else:
                # Apply EP update only for wrong predictions (WTA principle)
                if wrong_mask.any():
                    delta_hidden = cfg.lr_hidden * (
                        h_target[wrong_mask].float().T @ x_ternary[wrong_mask].float()
                        - h_free[wrong_mask].float().T @ x_ternary[wrong_mask].float()
                    )
                    hidden_scores += delta_hidden.to(hidden_scores.dtype)
        else:
            # Warmup: track correlation but don't update hidden
            S_out = output_layer._latent_scores.scores
            h_target = compute_hidden_target(y, S_out)
            batch_corr = (h_target == h_free).float().mean().item()
            total_target_corr += batch_corr

        # ── Post-update maintenance ─────────────────────────────
        if cfg.decay > 0:
            hidden_layer.apply_decay(cfg.decay)
            output_layer.apply_decay(cfg.decay)

        hidden_layer.refresh_weights()
        output_layer.refresh_weights()

        # Track flips
        new_w_hidden = hidden_layer.weight.unpack()
        if new_w_hidden.device != device:
            new_w_hidden = new_w_hidden.to(device)
        new_w_output = output_layer.weight.unpack()
        if new_w_output.device != device:
            new_w_output = new_w_output.to(device)

        total_flips_hidden += (old_w_hidden != new_w_hidden).sum().item()
        total_weights_hidden += new_w_hidden.numel()
        total_flips_output += (old_w_output != new_w_output).sum().item()
        total_weights_output += new_w_output.numel()
        n_batches += 1

    # Compute layer sparsity
    w_hidden = hidden_layer.weight.unpack()
    w_output = output_layer.weight.unpack()
    h_sparsity = (w_hidden == 0).float().mean().item()
    out_sparsity = (w_output == 0).float().mean().item()

    return {
        "accuracy": total_correct / max(total_samples, 1),
        "flip_rate_hidden": total_flips_hidden / max(total_weights_hidden, 1),
        "flip_rate_output": total_flips_output / max(total_weights_output, 1),
        "h_target_corr": total_target_corr / max(n_batches, 1),
        "h_sparsity": h_sparsity,
        "out_sparsity": out_sparsity,
    }


# ── Evaluation ────────────────────────────────────────────────────


def evaluate(
    hidden_layer: nn.Module,
    output_layer: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.1,
) -> float:
    """Evaluate the 2-layer EP network on a data loader.

    Standard forward pass: input → ternary → hidden → ternary → output → argmax.

    Args:
        hidden_layer: ``TernaryHebbianLinear`` hidden layer.
        output_layer: ``TernaryHebbianLinear`` output layer.
        loader: DataLoader yielding ``(inputs, targets)``.
        device: Torch device.
        epsilon: Dead-zone for ``ternary_sign``.

    Returns:
        Accuracy as a fraction (0.0 to 1.0).
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_flat = x.view(x.size(0), -1)

            x_ternary = ternary_sign(x_flat, epsilon=epsilon)
            h = hidden_layer(x_ternary.float())
            h_ternary = ternary_sign(h, epsilon=0.0)
            out = output_layer(h_ternary.float())
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / max(total, 1)


def get_weight_stats(module: nn.Module) -> dict[str, float]:
    """Get aggregated weight distribution stats for a module with ``weight``.

    Args:
        module: A module with a ``weight`` attribute (``TernaryTensor``).

    Returns:
        Dict with ``pos_pct``, ``neg_pct``, ``zero_pct``.
    """
    w = module.weight.unpack()
    n = w.numel()
    if n == 0:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "zero_pct": 100.0}
    return {
        "pos_pct": (w == 1).sum().item() / n * 100,
        "neg_pct": (w == -1).sum().item() / n * 100,
        "zero_pct": (w == 0).sum().item() / n * 100,
    }


# ── Equilibrium Propagation Classifier ────────────────────────────


class EquilibriumPropagationClassifier:
    """2-layer ternary Hebbian classifier trained with EP.

    Wraps an ``HebbianMLP`` (784→512→10) and trains both layers jointly
    using Equilibrium Propagation:

    - **Hidden layer**: EP difference-of-correlations update:
        ΔS_hidden = η_h × (h_target^T @ x - h_free^T @ x)
      where ``h_target = ternary_sign(S_out^T @ y_onehot)``.
    - **Output layer**: Standard WTA (Phase 0):
        ΔS_output = η_o × (y_target^T @ h_free - y_pred^T @ h_free)

    No ``.backward()``, no optimizers, no loss functions.

    Args:
        in_features: Input dimensionality (784 for MNIST).
        hidden_size: Hidden layer size (512).
        out_features: Number of output classes (10 for MNIST).
        cfg: ``EPConfig`` with hyperparameters.
        device: Device to place the model on.
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_size: int = 512,
        out_features: int = 10,
        cfg: EPConfig | None = None,
        device: torch.device | str | None = None,
    ):
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._cfg = cfg or EPConfig()
        self._in_features = in_features
        self._hidden_size = hidden_size
        self._out_features = out_features

        # Create 2-layer MLP
        self.model = HebbianMLP(
            layer_sizes=[in_features, hidden_size, out_features],
            theta_upper=self._cfg.theta_upper,
            theta_lower=self._cfg.theta_lower,
        ).to(self._device)

        self.hidden_layer = self.model.layers[0]
        self.output_layer = self.model.layers[1]

        # Bootstrap hidden layer with sparse random connectivity
        w_h = self.hidden_layer.weight.unpack()
        if torch.all(w_h == 0):
            _init_hidden_layer_connectivity(
                self.hidden_layer,
                density=self._cfg.hidden_density,
            )

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._device

    # ── Training ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Run EP training for the configured number of epochs.

        Args:
            train_loader: DataLoader yielding ``(inputs, targets)``.
            test_loader: Optional DataLoader for per-epoch evaluation.
            verbose: If True, print per-epoch progress.

        Returns:
            Dict of per-epoch metric lists: ``accuracy``, ``flip_rate_hidden``,
            ``flip_rate_output``, ``h_target_corr``, ``h_sparsity``, ``out_sparsity``,
            and ``test_accuracy`` (if test_loader provided).
        """
        history: dict[str, list[float]] = {
            "accuracy": [],
            "flip_rate_hidden": [],
            "flip_rate_output": [],
            "h_target_corr": [],
            "h_sparsity": [],
            "out_sparsity": [],
        }
        if test_loader is not None:
            history["test_accuracy"] = []

        total_start = time.time()

        for epoch in range(1, self._cfg.epochs + 1):
            is_warmup = epoch <= self._cfg.warmup_epochs

            epoch_start = time.time()

            with torch.no_grad():
                metrics = train_ep_epoch(
                    hidden_layer=self.hidden_layer,
                    output_layer=self.output_layer,
                    loader=train_loader,
                    device=self._device,
                    cfg=self._cfg,
                    epoch=epoch,
                    verbose=verbose,
                )

            epoch_time = time.time() - epoch_start

            # Evaluate on test set if provided
            test_acc = 0.0
            if test_loader is not None:
                test_acc = evaluate(
                    hidden_layer=self.hidden_layer,
                    output_layer=self.output_layer,
                    loader=test_loader,
                    device=self._device,
                    epsilon=self._cfg.epsilon,
                )
                history["test_accuracy"].append(test_acc)

            # Store history
            for key in ["accuracy", "flip_rate_hidden", "flip_rate_output",
                         "h_target_corr", "h_sparsity", "out_sparsity"]:
                history[key].append(metrics[key])

            # Print progress
            if verbose:
                warmup_tag = " [WARMUP]" if is_warmup else ""
                ep_tag = f" | EP-hidden lr={self._cfg.lr_hidden}" if not is_warmup else ""
                corr_str = f" | h-corr={metrics['h_target_corr']:.3f}"
                test_str = f" | test={test_acc*100:.2f}%" if test_loader else ""
                print(
                    f"  Epoch {epoch:2d}/{self._cfg.epochs}"
                    f"{warmup_tag}"
                    f" | train-acc={metrics['accuracy']*100:.2f}%"
                    f"{test_str}"
                    f" | h-flip={metrics['flip_rate_hidden']:.4f}"
                    f" | o-flip={metrics['flip_rate_output']:.4f}"
                    f"{corr_str}"
                    f" | h-sparse={metrics['h_sparsity']*100:.1f}%"
                    f" | o-sparse={metrics['out_sparsity']*100:.1f}%"
                    f" | {epoch_time:.1f}s"
                    f"{ep_tag}"
                )

        total_time = time.time() - total_start

        if verbose:
            final_acc = history.get("test_accuracy", history["accuracy"])[-1]
            print(f"\n  ── Done! Final accuracy: {final_acc*100:.2f}% [{total_time:.1f}s]")

        return history

    def evaluate(
        self,
        loader: DataLoader,
        epsilon: float | None = None,
    ) -> float:
        """Evaluate accuracy.

        Args:
            loader: DataLoader yielding ``(inputs, targets)``.
            epsilon: Dead-zone for ``ternary_sign``. Uses config value if None.

        Returns:
            Accuracy as a fraction.
        """
        eps = epsilon if epsilon is not None else self._cfg.epsilon
        return evaluate(
            hidden_layer=self.hidden_layer,
            output_layer=self.output_layer,
            loader=loader,
            device=self._device,
            epsilon=eps,
        )

    def get_weight_distribution(self) -> dict[str, dict[str, float]]:
        """Get weight distribution for both layers.

        Returns:
            Dict with ``hidden`` and ``output`` keys, each containing
            ``pos_pct``, ``neg_pct``, ``zero_pct``.
        """
        return {
            "hidden": get_weight_stats(self.hidden_layer),
            "output": get_weight_stats(self.output_layer),
        }
