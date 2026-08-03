"""DQT + Hysteresis-STE: Direct Quantized Training with hysteresis thresholds.

Implements :class:`TernaryDQTHysteresisLinear` — a linear layer that combines
two techniques that were previously validated independently:

1. **DQT (E017)** — direct quantized training: ternary weights stored as int8,
   float buffer used only for gradient accumulation, updated via **stochastic
   rounding** (no persistent latent float scores).
2. **Hysteresis-STE (E016/L2)** — dual-threshold hysteresis as a sparsity
   regularizer: ``|w| > theta_upper -> sign(w)``, ``|w| < theta_lower -> 0``,
   otherwise the previous ternary state is kept.

The combination keeps the memory advantage of DQT (only int8 ternary weights
are the trained state — no latent scores) while inheriting the high sparsity
of hysteresis.

**Ternary update rule (after each ``optimizer.step()``):**

.. code-block:: text

    |w_float| < theta_lower  ->  0                          (deactivate)
    |w_float| > theta_upper  ->  stochastic_round(w_float)  (activate)
    else (hysteresis gap)    ->  keep current ternary       (memory)

The stochastic rounding in the upper zone is the key DQT mechanism: weights
just above ``theta_upper`` activate probabilistically, which mitigates the
L2 deadzone problem where deterministic hysteresis never activates synapses
that sit below the activation threshold.

Usage::

    layer = TernaryDQTHysteresisLinear(784, 256, theta_upper=0.3, theta_lower=0.15)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)

    for x, y in dataloader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        layer.apply_stochastic_rounding()  # hysteresis + stochastic update

Comparison with the two parents:

- ``TernaryDQTLinear`` (E017): same autograd routing + stochastic rounding,
  but no hysteresis -> lower sparsity (56%).
- ``HysteresisSTELinear`` (E016/L2): same dual thresholds, but keeps latent
  float scores and uses deterministic STE sign -> higher sparsity (95%) at
  a small accuracy cost and 4.5x more training memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import stochastic_round

# ── Combined Autograd Function ─────────────────────────────────────


class _DQTHysteresisGradFn(torch.autograd.Function):
    """Autograd function routing gradients to the float accumulation buffer.

    Forward uses the int8 ternary weights (``weight_ternary``) for the matmul.
    Backward computes gradients and routes them to the float accumulation
    buffer (``weight_float``) — the same STE principle as DQT: gradients pass
    through the quantisation step as if it were identity.

    The int8 weights themselves do NOT receive gradients; they are updated
    via the hysteresis-gated stochastic rounding after each optimizer step.
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
        # (same shape as weight_float — the optimizer will use this)
        grad_weight_float = grad_output.t().mm(input)

        # weight_ternary is a buffer (no gradient)
        grad_weight_ternary = None

        # Bias gradient
        grad_bias = grad_output.sum(0) if ctx.needs_input_grad[3] else None

        return grad_input, grad_weight_float, grad_weight_ternary, grad_bias


# ── Hysteresis-gated stochastic rounding ───────────────────────────


@torch.no_grad()
def hysteresis_stochastic_round(
    w_float: torch.Tensor,
    prev_ternary: torch.Tensor,
    theta_upper: float,
    theta_lower: float,
    explore_gap: bool = False,
) -> torch.Tensor:
    """Discretise a float weight buffer into ternary weights via hysteresis.

    Combines the L2 dual-threshold hysteresis with DQT stochastic rounding:

    - ``|w| < theta_lower``  -> ``0`` (deterministic deactivation -> sparsity)
    - ``|w| > theta_upper``  -> ``stochastic_round(w)`` (DQT exploration;
      near the boundary activation is probabilistic, high ``|w|`` is ~``+/-1``)
    - ``theta_lower <= |w| <= theta_upper`` -> keep ``prev_ternary`` (memory).
      If ``explore_gap=True``, the gap is also stochastically rounded instead
      (a deadzone-mitigation ablation).

    Args:
        w_float: Float accumulation buffer, any shape.
        prev_ternary: Current ternary weights (int8, ``{-1, 0, +1}``).
        theta_upper: Hysteresis upper threshold (activation).
        theta_lower: Hysteresis lower threshold (deactivation).
        explore_gap: If ``True``, stochastic-round the hysteresis gap too.

    Returns:
        int8 tensor with values in ``{-1, 0, +1}``, same shape.
    """
    w = w_float.clamp(-1.0, 1.0)
    abs_w = w.abs()

    lower_mask = abs_w < theta_lower          # deactivate -> 0
    upper_mask = abs_w > theta_upper          # activate -> stochastic round
    gap_mask = ~(lower_mask | upper_mask)     # hysteresis memory zone

    # Start from previous ternary state (hysteresis memory in the gap)
    result = prev_ternary.clone()

    # Deactivate
    result[lower_mask] = 0

    # Activate via DQT stochastic rounding (exploration near the boundary)
    stochastic = stochastic_round(w)
    result[upper_mask] = stochastic[upper_mask]

    # Optional: stochastic rounding in the gap to break the deadzone
    if explore_gap:
        result[gap_mask] = stochastic[gap_mask]

    return result.to(torch.int8)


# ── Combined Linear Layer ──────────────────────────────────────────


class TernaryDQTHysteresisLinear(nn.Module):
    """Linear layer with ternary weights trained via DQT + hysteresis.

    Stores ternary weights directly as int8 (``weight_ternary`` buffer) —
    this is the trained state, there are **no latent float scores**. A float
    ``weight_float`` parameter serves only as the gradient accumulation buffer
    during training (it can be discarded after training for inference).

    After each ``optimizer.step()``, ``apply_stochastic_rounding()``
    discretises the float buffer into ternary weights using the L2
    dual-threshold hysteresis rule combined with DQT stochastic rounding.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        theta_upper: Hysteresis upper threshold (default 0.3 — L2 best).
        theta_lower: Hysteresis lower threshold (default 0.15 — L2 best).
        explore_gap: If ``True``, apply stochastic rounding inside the
            hysteresis gap too (deadzone-mitigation ablation).
        bias: If ``True``, adds a learnable bias.
        init_std: Std of the normal init for the float accumulation buffer.
        device: Torch device.
        dtype: Torch dtype for the float accumulation buffer.

    Attributes:
        weight_float: The float ``nn.Parameter`` for gradient accumulation.
        weight_ternary: The int8 buffer storing ternary weights {-1, 0, +1}.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_upper: float = 0.3,
        theta_lower: float = 0.15,
        explore_gap: bool = False,
        bias: bool = True,
        init_std: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if theta_lower >= theta_upper:
            raise ValueError(
                f"theta_lower ({theta_lower}) must be < theta_upper ({theta_upper})"
            )
        self._in_features = in_features
        self._out_features = out_features
        self._theta_upper = theta_upper
        self._theta_lower = theta_lower
        self._explore_gap = explore_gap

        # Float accumulation buffer — the ONLY learnable parameter
        self.weight_float = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight_float, mean=0.0, std=init_std)

        # Ternary weights stored as int8 buffer (-1, 0, +1)
        # Initialised via stochastic rounding of the initial float values
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using int8 ternary weights with DQT gradient routing.

        The forward pass uses ``weight_ternary`` (int8 buffer) for the matmul.
        Gradients are routed to ``weight_float`` via the custom autograd
        Function ``_DQTHysteresisGradFn`` — the DQT-style STE trick.

        Args:
            x: Input tensor, shape ``(batch, *, in_features)``.

        Returns:
            Output tensor, shape ``(batch, *, out_features)``.
        """
        # Flatten input if needed (handles (batch, *, in_features) shapes)
        if x.dim() > 2:
            x = x.flatten(1)

        return _DQTHysteresisGradFn.apply(
            x, self.weight_float, self.weight_ternary, self.bias
        )

    @torch.no_grad()
    def apply_stochastic_rounding(self) -> dict[str, float]:
        """Discretise the float buffer into ternary weights (hysteresis + stoch).

        Must be called AFTER ``optimizer.step()``. Combines the L2 dual
        threshold hysteresis with DQT stochastic rounding:

        - ``|w| < theta_lower`` -> ``0``
        - ``|w| > theta_upper`` -> ``stochastic_round(w)``
        - hysteresis gap -> keep current ternary (unless ``explore_gap``)

        Returns:
            Dict with flip statistics:
            - ``flip_rate``: fraction of ternary weights that changed
            - ``n_flips``: absolute number of flips
        """
        # Save previous state for flip tracking
        self._prev_ternary = self.weight_ternary.clone()

        w_new = hysteresis_stochastic_round(
            self.weight_float.data,
            self.weight_ternary,
            self._theta_upper,
            self._theta_lower,
            explore_gap=self._explore_gap,
        )

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
            f"theta_upper={self._theta_upper}, "
            f"theta_lower={self._theta_lower}, "
            f"explore_gap={self._explore_gap}, "
            f"bias={self.bias is not None}"
        )
