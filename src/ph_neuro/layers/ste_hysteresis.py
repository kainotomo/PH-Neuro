"""Hysteresis-STE ternary layers.

Implements the Hysteresis-STE algorithm: a Straight-Through Estimator
that applies dual-threshold hysteresis when deriving ternary weights
from latent scores.

Standard STE:
  forward:  W_tern = sign(W_latent)
  backward: dL/dW_latent = dL/dW_tern  (STE)

Hysteresis-STE:
  forward:  W_tern = tern_hyst(W_latent, theta_upper, theta_lower, prev_ternary)
            -> |W_latent| < theta_lower: W_tern = 0
            -> |W_latent| > theta_upper: W_tern = sign(W_latent)
            -> otherwise: unchanged from prev_ternary
  backward: dL/dW_latent = dL/dW_tern  (STE)

Usage::

    layer = HysteresisSTELinear(784, 256, theta_upper=1.0, theta_lower=0.3)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)

    for x, y in dataloader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

.. seealso::
    :class:`HysteresisSTEConv2d` for the convolutional variant.
    :class:`TernarySTELinear` in :mod:`ph_neuro.layers.ste_linear` for
    standard STE (without hysteresis).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_linear import ste_sign


# ── Straight-Through Estimator with Hysteresis ─────────────────────


class _STESignHysteresis(torch.autograd.Function):
    """Straight-Through Estimator for sign with hysteresis.

    Forward:
        Applies dual-threshold hysteresis to derive ternary weights:
        ``{-1, 0, +1}``, where values between the lower and upper
        thresholds retain their previous ternary state.

    Backward:
        Identity pass-through (STE trick): ``dL/dx = dL/dy``.

    This allows gradients to flow through the hysteresis step despite
    ``sign()`` having zero derivative almost everywhere.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        prev_ternary: torch.Tensor,
        theta_upper: float,
        theta_lower: float,
    ) -> torch.Tensor:
        """Forward pass: hysteresis thresholding.

        Args:
            x: Latent scores (float tensor).
            prev_ternary: Previous ternary weights (int8 tensor, {-1,0,+1}).
            theta_upper: Upper threshold for activation (0 -> +/-1).
            theta_lower: Lower threshold for deactivation (+/-1 -> 0).

        Returns:
            Float tensor with values in {-1, 0, +1} (same dtype as ``x``).
            The autograd graph is preserved for gradient flow.
        """
        # Start with previous ternary state (cast to float for autograd)
        prev = prev_ternary.to(x.dtype)

        # ── Hysteresis logic using torch.where for autograd compatibility ──
        # Condition 1: prev == 0 and |x| > theta_upper -> sign(x)
        activate_mask = (prev_ternary == 0) & (x.abs() > theta_upper)
        activated = torch.where(activate_mask, x.sign(), prev)

        # Condition 2: prev != 0 and |x| < theta_lower -> 0
        deactivate_mask = (prev_ternary != 0) & (x.abs() < theta_lower)
        result = torch.where(deactivate_mask, torch.zeros_like(activated), activated)

        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Backward: STE identity pass-through.

        Returns:
            Gradients for (x, prev_ternary, theta_upper, theta_lower).
            Only ``x`` receives a gradient (STE identity).
        """
        # STE: pass gradient through for x, None for non-differentiable inputs
        return grad_output, None, None, None


def ste_sign_hysteresis(
    x: torch.Tensor,
    prev_ternary: torch.Tensor,
    theta_upper: float = 1.0,
    theta_lower: float = 0.3,
) -> torch.Tensor:
    """Apply Straight-Through Estimator with dual-threshold hysteresis.

    Args:
        x: Latent scores (float tensor).
        prev_ternary: Previous ternary weights (int8 tensor, {-1,0,+1}).
        theta_upper: Upper threshold — latent score must exceed this
            to activate a synapse (0 -> +/-1).
        theta_lower: Lower threshold — latent score must fall below
            this to deactivate a synapse (+/-1 -> 0).

    Returns:
        Float tensor with same shape and dtype as ``x``, values in
        ``{-1, 0, +1}`` in forward, identity gradient in backward pass.
    """
    return _STESignHysteresis.apply(x, prev_ternary, theta_upper, theta_lower)


# ── Hysteresis-STE Linear Layer ────────────────────────────────────


class HysteresisSTELinear(nn.Module):
    """Linear layer with hysteresis-regularised STE backpropagation.

    Stores float latent scores as a learnable ``nn.Parameter``. During
    the forward pass, ternary weights are derived via dual-threshold
    hysteresis instead of plain ``sign()``. The backward pass uses the
    STE trick so gradients update the latent scores directly.

    The hysteresis mechanism is inherited from PH-Neuro's ternary layers
    and promotes weight sparsity and reduces oscillatory flipping.

    After training, the ternary weights can be extracted via
    ``layer.ternary_weight()`` for memory-efficient inference.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        theta_upper: Hysteresis upper threshold (default 1.0).
        theta_lower: Hysteresis lower threshold (default 0.3).
        bias: If ``True``, adds a learnable bias.
        device: Torch device.
        dtype: Torch dtype for the latent scores (default ``torch.float32``).

    Attributes:
        latent_scores: The underlying float ``nn.Parameter``.
        prev_ternary: The previous ternary state ``nn.Buffer`` (int8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_upper: float = 1.0,
        theta_lower: float = 0.3,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._theta_upper = theta_upper
        self._theta_lower = theta_lower

        # Latent scores: the actual learnable parameters
        self.latent_scores = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.normal_(self.latent_scores, mean=0.0, std=0.1)

        # Previous ternary state: persistent buffer (not a Parameter)
        # Initialized to all zeros (all weights start deactivated)
        self.register_buffer(
            "prev_ternary",
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device),
        )

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

    @property
    def theta_upper(self) -> float:
        """Hysteresis upper threshold."""
        return self._theta_upper

    @theta_upper.setter
    def theta_upper(self, value: float) -> None:
        self._theta_upper = value

    @property
    def theta_lower(self) -> float:
        """Hysteresis lower threshold."""
        return self._theta_lower

    @theta_lower.setter
    def theta_lower(self, value: float) -> None:
        self._theta_lower = value

    def ternary_weight(self) -> torch.Tensor:
        """The ternary weight matrix {-1, 0, +1} from the current state.

        Returns:
            int8 tensor of shape ``(out_features, in_features)``.
        """
        return self.prev_ternary.clone()

    def reset_hysteresis_state(self) -> None:
        """Reset the hysteresis state to all zeros.

        Call this when loading pretrained weights or manually changing
        latent scores so the previous ternary state stays consistent.
        """
        self.prev_ternary.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hysteresis-STE weights and STE backward.

        Args:
            x: Input tensor, shape ``(batch, *, in_features)``.

        Returns:
            Output tensor, shape ``(batch, *, out_features)``.
        """
        # Apply hysteresis-STE to derive ternary weights (float tensor)
        w_tern = ste_sign_hysteresis(
            self.latent_scores,
            self.prev_ternary,
            self._theta_upper,
            self._theta_lower,
        )

        # Update the persistent buffer (int8 copy, detached — not part of graph)
        self.prev_ternary.copy_(w_tern.to(torch.int8))

        out = F.linear(x, w_tern)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"theta_upper={self._theta_upper}, "
            f"theta_lower={self._theta_lower}, "
            f"bias={self.bias is not None}"
        )


# ── Hysteresis-STE Convolutional Layer ─────────────────────────────


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an int to a 2-tuple if needed."""
    if isinstance(x, tuple):
        return x
    return (x, x)


class HysteresisSTEConv2d(nn.Module):
    """2D convolution with hysteresis-regularised STE backpropagation.

    Convolutional variant of ``HysteresisSTELinear``. Ternary weights
    {-1, 0, +1} are derived from float latent scores via dual-threshold
    hysteresis, and gradients flow through via STE.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
        theta_upper: Hysteresis upper threshold (default 1.0).
        theta_lower: Hysteresis lower threshold (default 0.3).
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        bias: If ``True``, adds a learnable bias.
        device: Torch device.
        dtype: Torch dtype for the latent scores (default ``torch.float32``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        theta_upper: float = 1.0,
        theta_lower: float = 0.3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = _pair(kernel_size)
        self._stride = _pair(stride)
        self._padding = _pair(padding)
        self._dilation = _pair(dilation)
        self._theta_upper = theta_upper
        self._theta_lower = theta_lower

        # Latent scores: learnable parameter (shape: out, in, kH, kW)
        self.latent_scores = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                self._kernel_size[0],
                self._kernel_size[1],
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.latent_scores, mean=0.0, std=0.1)

        # Previous ternary state: persistent buffer
        self.register_buffer(
            "prev_ternary",
            torch.zeros(
                out_channels,
                in_channels,
                self._kernel_size[0],
                self._kernel_size[1],
                dtype=torch.int8,
                device=device,
            ),
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def theta_upper(self) -> float:
        return self._theta_upper

    @theta_upper.setter
    def theta_upper(self, value: float) -> None:
        self._theta_upper = value

    @property
    def theta_lower(self) -> float:
        return self._theta_lower

    @theta_lower.setter
    def theta_lower(self, value: float) -> None:
        self._theta_lower = value

    def ternary_weight(self) -> torch.Tensor:
        """The ternary weight kernel {-1, 0, +1} from the current state.

        Returns:
            int8 tensor of shape ``(out_channels, in_channels, kH, kW)``.
        """
        return self.prev_ternary.clone()

    def reset_hysteresis_state(self) -> None:
        """Reset the hysteresis state to all zeros."""
        self.prev_ternary.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hysteresis-STE weights.

        Args:
            x: Input tensor, shape ``(batch, in_channels, H, W)``.

        Returns:
            Output tensor, shape ``(batch, out_channels, H_out, W_out)``.
        """
        # Apply hysteresis-STE to derive ternary weights (float tensor)
        w_tern = ste_sign_hysteresis(
            self.latent_scores,
            self.prev_ternary,
            self._theta_upper,
            self._theta_lower,
        )

        # Update the persistent buffer (int8 copy)
        self.prev_ternary.copy_(w_tern.to(torch.int8))

        out = F.conv2d(
            x,
            w_tern,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
        )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self._in_channels}, "
            f"out_channels={self._out_channels}, "
            f"kernel_size={self._kernel_size}, "
            f"theta_upper={self._theta_upper}, "
            f"theta_lower={self._theta_lower}, "
            f"stride={self._stride}, "
            f"bias={self.bias is not None}"
        )
