"""Hebbian trainer — trains Hebbian networks without backpropagation.

The trainer orchestrates the training loop, calling ``hebbian_update``
and ``refresh_weights`` on each layer, without ever calling ``.backward()``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from ph_neuro.layers.linear import TernaryHebbianLinear, ternary_sign


class HebbianTrainer:
    """Trainer for Hebbian networks.

    Args:
        model: A PyTorch module containing ``TernaryHebbianLinear`` layers.
        lr: Learning rate for Hebbian updates.
        decay: Homeostatic decay rate for latent scores.
        device: Device to train on (``'cpu'`` or ``'cuda'``).

    Example::

        model = HebbianMLP([784, 256, 10])
        trainer = HebbianTrainer(model, lr=0.01, decay=1e-5)
        trainer.fit(train_loader, epochs=10)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.01,
        decay: float = 1e-5,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.lr = lr
        self.decay = decay
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # Collect all Hebbian layers
        self._hebbian_layers: list[TernaryHebbianLinear] = []
        for module in self.model.modules():
            if isinstance(module, TernaryHebbianLinear):
                self._hebbian_layers.append(module)

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, list[float]]:
        """Train the model using Hebbian updates.

        No ``.backward()`` is called anywhere. Each layer's latent
        scores are updated using the Hebbian rule, and ternary weights
        are refreshed periodically via hysteresis.

        Args:
            train_loader: DataLoader yielding ``(inputs, targets)``.
            epochs: Number of training epochs.
            callbacks: Optional list of callables ``(epoch, metrics) -> None``.

        Returns:
            Dictionary of training metrics per epoch.
        """
        history: dict[str, list[float]] = {"loss": []}

        for epoch in range(epochs):
            epoch_metrics = self._run_epoch(train_loader)
            history["loss"].append(epoch_metrics.get("loss", 0.0))

            if callbacks:
                for cb in callbacks:
                    cb(epoch, epoch_metrics)

        return history

    def _run_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training epoch."""
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            elif isinstance(batch, torch.Tensor):
                x = batch
            else:
                x = batch[0]

            x = x.to(self.device)
            x_flat = x.view(x.size(0), -1)

            # Quantize input to ternary
            x_ternary = ternary_sign(x_flat)

            # Forward pass through each Hebbian layer
            h = x_ternary
            for layer in self._hebbian_layers:
                out = layer(h.float())
                post = ternary_sign(out)
                # Hebbian update
                layer.hebbian_update(h, post, self.lr)
                layer.apply_decay(self.decay)
                layer.refresh_weights()
                h = post

            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    def evaluate(
        self,
        loader: DataLoader,
    ) -> float:
        """Evaluate accuracy on a data loader.

        Args:
            loader: DataLoader yielding ``(inputs, targets)``.

        Returns:
            Accuracy as a fraction (0.0 to 1.0).
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                x_flat = x.view(x.size(0), -1)

                out = self.model(x_flat)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / max(total, 1)
