"""Direct Quantized Training (DQT) 2D convolutional layer.

Implements :class:`TernaryDQTConv2d` — the convolutional counterpart of
:class:`TernaryDQTLinear` (see :mod:`ph_neuro.layers.ste_dqt`). Ternary
weights {-1, 0, +1} are stored directly as int8 and updated via stochastic
rounding of accumulated float gradients, WITHOUT persistent latent float
scores during training.

The forward pass is ``F.conv2d`` using the int8 ternary weights (cast to
float). The backward pass is a custom autograd Function
(``_DQTConvGradFn``) that routes gradients to the float accumulation
buffer ``weight_float``:

- **grad_input** via ``torch.nn.grad.conv2d_input`` (the exact adjoint used
  by PyTorch autograd, correct for any stride/padding/dilation).
- **grad_weight_float** via the im2col correlation identity
  (``F.unfold`` + einsum): for each output channel ``co`` the weight
  gradient is the correlation of the input patches with ``grad_output``.

Usage::

    layer = TernaryDQTConv2d(3, 64, kernel_size=3, padding=1, bias=False)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)

    for x, y in dataloader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        layer.apply_stochastic_rounding()  # float buffer -> int8 ternary
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_dqt import stochastic_round


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an int to a 2-tuple if needed."""
    if isinstance(x, tuple):
        return x
    return (x, x)


def _conv2d_weight_grad(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> torch.Tensor:
    """Compute the weight gradient of ``conv2d`` via the im2col identity.

    For ``conv2d(input, weight)`` the gradient w.r.t. the weight is::

        grad_weight[co, ci, kh, kw] =
            sum_{n, i, j} grad_output[n, co, i, j]
                          * input[n, ci, i*sh + kh*dh - ph, j*sw + kw*dw - pw]

    This is exactly ``sum_n conv2d_cross(input[n], grad_output[n])`` where
    ``grad_output`` plays the role of the filter. We compute it with
    ``F.unfold`` (im2col) followed by an einsum contraction, which is
    correct for any stride/padding/dilation (groups=1).

    Args:
        input: Input tensor, shape ``(N, C_in, H_in, W_in)``.
        grad_output: Output gradient, shape ``(N, C_out, H_out, W_out)``.
        kernel_size: ``(kH, kW)`` of the convolution kernel.
        stride, padding, dilation: Conv2d parameters.

    Returns:
        Weight gradient tensor, shape ``(C_out, C_in, kH, kW)``.
    """
    n_batch, c_in = input.shape[0], input.shape[1]
    c_out = grad_output.shape[1]
    k_h, k_w = kernel_size

    # (N, C_in*kH*kW, L) where L = H_out * W_out
    patches = F.unfold(
        input,
        kernel_size=(k_h, k_w),
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    n_pos = patches.shape[2]

    # (N, C_out, L)
    g = grad_output.reshape(n_batch, c_out, n_pos)

    # grad_weight_flat[co, ck] = sum_{n, l} g[n, co, l] * patches[n, ck, l]
    grad_weight_flat = torch.einsum("nol,nkl->ok", g, patches)

    return grad_weight_flat.view(c_out, c_in, k_h, k_w)


# ── DQT Conv Autograd Function ─────────────────────────────────────


class _DQTConvGradFn(torch.autograd.Function):
    """Custom autograd Function for Direct Quantized Training on Conv2d.

    Forward uses the int8 ternary weights (``weight_ternary``) for
    ``F.conv2d``. Backward computes the input gradient via
    ``torch.nn.grad.conv2d_input`` and the weight gradient via the im2col
    correlation identity, routing gradients to the float accumulation
    buffer ``weight_float`` (STE principle — gradients pass through the
    quantization step as if it were identity).

    The int8 weights themselves do NOT receive gradients; they are updated
    via stochastic rounding of the float buffer after each optimizer step.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight_float: torch.Tensor,
        weight_ternary: torch.Tensor,
        bias: torch.Tensor | None,
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
    ) -> torch.Tensor:
        # Save for backward: need input and ternary weights for grad computation
        ctx.save_for_backward(input, weight_ternary)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        # Forward: use int8 ternary weights (cast to float for conv2d)
        w = weight_ternary.float()
        output = F.conv2d(input, w, bias, stride=stride, padding=padding, dilation=dilation)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight_ternary = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        w = weight_ternary.float()

        # Gradient w.r.t. input (adjoint of conv2d — exact, any stride/padding)
        grad_input = torch.nn.grad.conv2d_input(
            input.shape,
            w,
            grad_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # STE gradient w.r.t. weight_float (im2col correlation identity)
        grad_weight_float = _conv2d_weight_grad(
            input, grad_output, (w.shape[2], w.shape[3]), stride, padding, dilation
        )

        # weight_ternary is a buffer (no gradient)
        grad_weight_ternary = None

        # Bias gradient (only if a bias was used)
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if ctx.needs_input_grad[3] else None

        # stride / padding / dilation are non-tensor args — no gradient
        return grad_input, grad_weight_float, grad_weight_ternary, grad_bias, None, None, None


# ── DQT Conv Layer ─────────────────────────────────────────────────


class TernaryDQTConv2d(nn.Module):
    """2D convolution with ternary weights trained via DQT + stochastic rounding.

    Stores ternary weights directly as int8 (``weight_ternary`` buffer).
    A float ``weight_float`` parameter serves as the gradient accumulation
    buffer — it is the only learnable parameter. After each optimizer step,
    ``apply_stochastic_rounding()`` discretizes the float buffer into
    ternary int8 weights.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        bias: If ``True``, adds a learnable bias (default ``False`` — in the
            M1.1 CNN BatchNorm handles the per-channel shift).
        device: Torch device.
        dtype: Torch dtype for the float accumulation buffer (default float32).

    Attributes:
        weight_float: The float ``nn.Parameter`` for gradient accumulation.
        weight_ternary: The int8 buffer storing ternary weights {-1, 0, +1}.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
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

        # Float accumulation buffer — the ONLY learnable parameter
        # Initialized with small random values for symmetry breaking
        self.weight_float = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                self._kernel_size[0],
                self._kernel_size[1],
                device=device,
                dtype=dtype,
            )
        )
        nn.init.normal_(self.weight_float, mean=0.0, std=0.1)

        # Ternary weights stored as int8 buffer (-1, 0, +1)
        # Initialized via stochastic rounding of the initial float values
        init_ternary = stochastic_round(self.weight_float.data)
        self.register_buffer("weight_ternary", init_ternary)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Track flip statistics
        self.register_buffer("_prev_ternary", self.weight_ternary.clone())

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self._kernel_size

    def ternary_weight(self) -> torch.Tensor:
        """The ternary weight tensor {-1, 0, +1}.

        Returns:
            int8 tensor of shape ``(out_channels, in_channels, kH, kW)``.
        """
        return self.weight_ternary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using int8 ternary weights with DQT gradient routing.

        The forward pass uses ``weight_ternary`` (int8 buffer) for
        ``F.conv2d``. Gradients are routed to ``weight_float`` via the
        custom autograd Function ``_DQTConvGradFn`` — the DQT version of
        the STE trick.

        Args:
            x: Input tensor, shape ``(batch, in_channels, H, W)``.

        Returns:
            Output tensor, shape ``(batch, out_channels, H_out, W_out)``.
        """
        return _DQTConvGradFn.apply(
            x,
            self.weight_float,
            self.weight_ternary,
            self.bias,
            self._stride,
            self._padding,
            self._dilation,
        )

    @torch.no_grad()
    def apply_stochastic_rounding(self) -> dict[str, float]:
        """Apply stochastic rounding to discretize float buffer into ternary weights.

        Must be called AFTER ``optimizer.step()``. Uses stochastic rounding
        (rather than deterministic sign) to allow the optimizer to explore
        the weight space more effectively.

        Returns:
            Dict with flip statistics:
            - ``flip_rate``: fraction of ternary weights that changed
            - ``n_flips``: absolute number of flips
        """
        # Save previous state for flip tracking
        self._prev_ternary = self.weight_ternary.clone()

        # Stochastic rounding: float -> {-1, 0, +1}
        w_float = self.weight_float.data
        w_new = stochastic_round(w_float)

        # Compute flip statistics
        n_flips = (self.weight_ternary != w_new).sum().item()
        total = w_new.numel()

        # Update ternary weights
        self.weight_ternary.copy_(w_new)

        return {
            "flip_rate": n_flips / max(total, 1),
            "n_flips": n_flips,
        }

    @torch.no_grad()
    def apply_deterministic_rounding(self) -> dict[str, float]:
        """Apply DETERMINISTIC rounding: sign() instead of stochastic_round().

        Used during the annealing (fine-tuning) phase: once training switches
        to deterministic rounding, the float accumulation buffer is snapped
        to ``sign(float)`` so the ternary weights stop jittering and the
        network can settle into a clean fine-tuning regime (no more
        stochastic flip noise near the end of training).

        Returns:
            Dict with flip statistics (identical interface to
            ``apply_stochastic_rounding``):
            - ``flip_rate``: fraction of ternary weights that changed
            - ``n_flips``: absolute number of flips
        """
        self._prev_ternary = self.weight_ternary.clone()
        w_new = self.weight_float.data.sign().clamp(-1, 1).to(torch.int8)
        n_flips = (self.weight_ternary != w_new).sum().item()
        total = w_new.numel()
        self.weight_ternary.copy_(w_new)
        return {"flip_rate": n_flips / max(total, 1), "n_flips": n_flips}

    @torch.no_grad()
    def get_flip_rate(self) -> float:
        """Get the flip rate since the last stochastic rounding step.

        Returns:
            Fraction of ternary weights that changed.
        """
        n_flips = (self._prev_ternary != self.weight_ternary).sum().item()
        total = self.weight_ternary.numel()
        return n_flips / max(total, 1)

    @torch.no_grad()
    def get_weight_stats(self) -> dict[str, float]:
        """Get statistics of the current ternary weights.

        Returns:
            Dict with ``pos_pct``, ``neg_pct``, ``zero_pct``.
        """
        w = self.weight_ternary
        total = w.numel()
        return {
            "pos_pct": 100.0 * (w == 1).sum().item() / max(total, 1),
            "neg_pct": 100.0 * (w == -1).sum().item() / max(total, 1),
            "zero_pct": 100.0 * (w == 0).sum().item() / max(total, 1),
        }

    def extra_repr(self) -> str:
        return (
            f"{self._in_channels} -> {self._out_channels}, "
            f"kernel={self._kernel_size}, stride={self._stride}, "
            f"padding={self._padding}, dilation={self._dilation}, "
            f"bias={self.bias is not None}"
        )
