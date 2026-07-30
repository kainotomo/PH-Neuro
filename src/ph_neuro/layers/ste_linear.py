"""STE-based ternary linear layer.

Replaces the Hebbian update with standard backpropagation through
a Straight-Through Estimator (STE). Ternary weights {-1, 0, +1} are
derived from float latent scores; gradients flow through the ``sign()``
nonlinearity via the STE trick.

Usage::

    layer = TernarySTELinear(784, 256)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)

    for x, y in dataloader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

.. seealso::
    :class:`TernarySTEConv2d` in :mod:`ph_neuro.layers.ste_conv` for
    the convolutional variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


# ── Straight-Through Estimator (STE) ────────────────────────────────


class _STESign(torch.autograd.Function):
    """Straight-Through Estimator for the sign function.

    Forward:
        ``y = sign(x)`` → values in {-1, 0, +1}

    Backward:
        ``∂L/∂x = ∂L/∂y`` — identity pass-through (STE trick).

    This allows gradients to flow through the ``sign()`` step despite
    ``sign()`` having zero derivative almost everywhere.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Apply Straight-Through Estimator sign.

    Args:
        x: Input tensor.

    Returns:
        Tensor with same shape, values in {-1, 0, +1} in forward pass,
        identity gradient in backward pass.
    """
    return _STESign.apply(x)


# ── STE Linear Layer ────────────────────────────────────────────────


class TernarySTELinear(nn.Module):
    """Linear layer with ternary weights trained via STE backpropagation.

    Stores float latent scores as a learnable ``nn.Parameter``. During
    the forward pass, ternary weights are derived via ``sign()``; the
    backward pass uses the STE trick so gradients update the latent scores
    directly.

    After training, the ternary weights can be extracted via
    ``layer.ternary_weight()`` for memory-efficient inference.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias.
        device: Torch device.
        dtype: Torch dtype for the latent scores (default ``torch.float32``).

    Attributes:
        latent_scores: The underlying float ``nn.Parameter``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features

        # Latent scores: the actual learnable parameters
        self.latent_scores = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.normal_(self.latent_scores, mean=0.0, std=0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def ternary_weight(self) -> torch.Tensor:
        """The ternary weight matrix {-1, 0, +1} derived from latent scores.

        Returns:
            int8 tensor of shape ``(out_features, in_features)``.
        """
        return self.latent_scores.sign().to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary weights and STE backward.

        Args:
            x: Input tensor, shape ``(batch, *, in_features)``.

        Returns:
            Output tensor, shape ``(batch, *, out_features)``.
        """
        w_tern = ste_sign(self.latent_scores)
        out = F.linear(x, w_tern)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"bias={self.bias is not None}"
        )
