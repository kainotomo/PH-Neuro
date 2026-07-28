"""MNIST MLP example — single-layer Hebbian classifier.

Usage:
    python -m ph_neuro.examples.mnist_mlp

This script trains a single :class:`~ph_neuro.layers.linear.TernaryHebbianLinear`
layer (784 → 10) on MNIST using Hebbian updates. No ``.backward()`` is called.
"""

from __future__ import annotations

import torch
from torch.nn.functional import one_hot

from ph_neuro.layers.linear import TernaryHebbianLinear, ternary_sign
from ph_neuro.training.data import get_mnist_loaders


def main():
    """Run the MNIST MLP Hebbian experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    print("MNIST data loaded.")

    # Model: single layer 784 → 10
    model = TernaryHebbianLinear(
        in_features=784,
        out_features=10,
        theta_upper=5.0,
        theta_lower=1.0,
    ).to(device)

    # Training
    epochs = 10
    lr = 0.01

    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device).view(-1, 784)
            y = y.to(device)

            # Forward pass
            x_ternary = ternary_sign(x)
            out = model(x_ternary)

            # Supervised Hebbian update
            target = one_hot(y, num_classes=10).to(torch.int8)  # {0, 1}
            target = target * 2 - 1  # {0, 1} → {-1, +1}

            # Positive Hebbian for correct class
            correct_mask = target == 1
            model.hebbian_update(x_ternary, target * correct_mask, lr)

            model.refresh_weights()

        # Evaluate
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).view(-1, 784)
                y = y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch + 1:2d}/{epochs}  Accuracy: {acc:.2f}%")

    print("Done!")


if __name__ == "__main__":
    main()
