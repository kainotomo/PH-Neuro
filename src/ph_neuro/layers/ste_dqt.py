"""Direct Quantized Training (DQT) with stochastic rounding for ternary weights.

Implements :class:`TernaryDQTLinear` — a linear layer that stores ternary
weights directly as int8 and updates them via stochastic rounding of
accumulated gradients, WITHOUT maintaining persistent float latent scores.

The key insight: only a float *buffer* (not a learnable parameter) is kept
for gradient accumulation during the backward pass. Ternary weights are
stored as int8 {-1, 0, +1} and updated *directly* via stochastic rounding.

Usage::

    layer = TernaryDQTLinear(784, 256)
    # weight_float is an nn.Parameter — the optimizer tracks it
    # weight_ternary is an int8 buffer — the actual ternary weights
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)

    for x, y in dataloader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        layer.apply_stochastic_rounding()  # discretize float → ternary

Comparison with STE (TernarySTELinear):
    - STE: persistent float latent scores → deterministic sign() → ternary
    - DQT: float accumulation buffer → stochastic_round() → ternary (int8)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


# ── Stochastic Rounding ─────────────────────────────────────────────


def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding toward {-1, 0, +1}.

    For each element ``v`` (clamped to [-1, 1]):
        - floor(v) ∈ {-1, 0}  and  ceil(v) ∈ {0, 1}
        - frac = v - floor(v) ∈ [0, 1)
        - Round to ceil(v) with probability ``frac``, else floor(v)

    This is the key mechanism that allows DQT to explore the weight space
    without getting stuck at zero (unlike deterministic sign).

    Args:
        x: Float tensor of any shape.

    Returns:
        int8 tensor with values in {-1, 0, +1}, same shape as ``x``.
    """
    x_clamped = x.clamp(-1.0, 1.0)
    floor_val = torch.floor(x_clamped)  # values in {-1, 0}
    ceil_val = torch.ceil(x_clamped)    # values in {0, 1}
    frac = x_clamped - floor_val  # fractional part in [0, 1)

    # Random mask: round up (to ceil) where random < frac
    random_mask = torch.rand_like(x_clamped) < frac
    result = floor_val.clone()
    result[random_mask] = ceil_val[random_mask]

    return result.to(torch.int8)


# ── STE Sign (same as ste_linear.py) ────────────────────────────────


class _STESign(torch.autograd.Function):
    """Straight-Through Estimator for the sign function."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Apply Straight-Through Estimator sign."""
    return _STESign.apply(x)


# ── DQT Autograd Function ───────────────────────────────────────────


class _DQTGradFn(torch.autograd.Function):
    """Custom autograd Function for Direct Quantized Training.

    Forward uses the int8 ternary weights (``weight_ternary``) for the
    matmul. Backward computes gradients and routes them to the float
    accumulation buffer (``weight_float``) — this is the STE principle:
    gradients pass through the quantization step as if it were identity.

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
    ) -> torch.Tensor:
        # Save for backward: need input and ternary weights for grad computation
        ctx.save_for_backward(input, weight_ternary)
        # Forward: use int8 ternary weights (cast to float for matmul)
        w = weight_ternary.float()
        output = input.mm(w.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight_ternary = ctx.saved_tensors

        # Gradient w.r.t. input: grad_output @ W
        grad_input = grad_output.mm(weight_ternary.float())

        # STE gradient w.r.t. weight_float: grad_output^T @ input
        # (same shape as weight_float — optimizer will use this)
        grad_weight_float = grad_output.t().mm(input)

        # weight_ternary is a buffer (no gradient)
        grad_weight_ternary = None

        # Bias gradient
        grad_bias = grad_output.sum(0) if ctx.needs_input_grad[3] else None

        return grad_input, grad_weight_float, grad_weight_ternary, grad_bias


# ── DQT Linear Layer ────────────────────────────────────────────────


class TernaryDQTLinear(nn.Module):
    """Linear layer with ternary weights trained via DQT + stochastic rounding.

    Stores ternary weights directly as int8 (``weight_ternary`` buffer).
    A float ``weight_float`` parameter serves as the gradient accumulation
    buffer — it is the only learnable parameter, using 4 bytes per weight
    (float32) instead of 8 bytes (float64). After each optimizer step,
    ``apply_stochastic_rounding()`` discretizes the float buffer into
    ternary int8 weights.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias.
        device: Torch device.
        dtype: Torch dtype for the float accumulation buffer (default float32).

    Attributes:
        weight_float: The float ``nn.Parameter`` for gradient accumulation.
        weight_ternary: The int8 buffer storing ternary weights {-1, 0, +1}.
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

        # Float accumulation buffer — the ONLY learnable parameter
        # Initialized with small random values for symmetry breaking
        self.weight_float = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight_float, mean=0.0, std=0.1)

        # Ternary weights stored as int8 buffer (-1, 0, +1)
        # Initialized via stochastic rounding of the initial float values
        init_ternary = stochastic_round(self.weight_float.data)
        self.register_buffer("weight_ternary", init_ternary)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # Track flip statistics
        self.register_buffer("_prev_ternary", self.weight_ternary.clone())

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using int8 ternary weights with DQT gradient routing.

        The forward pass uses ``weight_ternary`` (int8 buffer) for the matmul.
        Gradients are routed to ``weight_float`` via the custom autograd
        Function ``_DQTGradFn`` — this is the DQT version of the STE trick.

        Args:
            x: Input tensor, shape ``(batch, *, in_features)``.

        Returns:
            Output tensor, shape ``(batch, *, out_features)``.
        """
        # Flatten input if needed (handles (batch, *, in_features) shapes)
        if x.dim() > 2:
            x = x.flatten(1)

        # Use custom autograd Function: forward uses weight_ternary,
        # backward routes gradient to weight_float (STE-style)
        return _DQTGradFn.apply(x, self.weight_float, self.weight_ternary, self.bias)

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

        # Stochastic rounding: float → {-1, 0, +1}
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
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"bias={self.bias is not None}"
        )
