"""STE-based ternary 2D convolutional layer.

Straight-Through Estimator variant for convolutions. Ternary weights
{-1, 0, +1} are derived from float latent scores, and gradients flow
through the ``sign()`` via STE.

Usage::

    layer = TernarySTEConv2d(3, 64, kernel_size=3)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)

    for x, y in dataloader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

.. seealso::
    :class:`TernarySTELinear` in :mod:`ph_neuro.layers.ste_linear` for
    the linear (MLP) variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_linear import ste_sign


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an int to a 2-tuple if needed."""
    if isinstance(x, tuple):
        return x
    return (x, x)


class TernarySTEConv2d(nn.Module):
    """2D convolution with ternary weights trained via STE backpropagation.

    Stores float latent scores as a learnable ``nn.Parameter`` of shape
    ``(out_channels, in_channels, kH, kW)``. During forward, ternary
    weights are derived via ``sign()``; backward uses STE.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
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

        # Latent scores: the actual learnable parameters
        self.latent_scores = nn.Parameter(
            torch.empty(
                out_channels, in_channels, self._kernel_size[0], self._kernel_size[1],
                device=device, dtype=dtype,
            )
        )
        nn.init.normal_(self.latent_scores, mean=0.0, std=0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def ternary_weight(self) -> torch.Tensor:
        """The ternary weight tensor {-1, 0, +1} derived from latent scores.

        Returns:
            int8 tensor of shape ``(out_channels, in_channels, kH, kW)``.
        """
        return self.latent_scores.sign().to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary weights and STE backward.

        Args:
            x: Input tensor, shape ``(batch, in_channels, H, W)``.

        Returns:
            Output tensor, shape ``(batch, out_channels, H_out, W_out)``.
        """
        w_tern = ste_sign(self.latent_scores)
        out = F.conv2d(x, w_tern, stride=self._stride, padding=self._padding,
                       dilation=self._dilation)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self._in_channels} -> {self._out_channels}, "
            f"kernel={self._kernel_size}, stride={self._stride}, "
            f"padding={self._padding}, dilation={self._dilation}, "
            f"bias={self.bias is not None}"
        )
