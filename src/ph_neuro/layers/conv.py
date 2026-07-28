"""Ternary Hebbian 2D convolutional layer.

Convolutional variant of :class:`~ph_neuro.layers.linear.TernaryHebbianLinear`.
The Hebbian rule for convolutions is naturally local — each filter weight
connects a local patch of the input to one output neuron.

Forward: uses ``torch.nn.functional.unfold`` for efficient patch extraction,
then ``F.linear`` with reshaped ternary weights.

Hebbian update: ``ΔW = lr × Σ_{batch,spatial} output[b,f,i,j] × input_patch[b,i,j]``
— summed over all batch and spatial positions where the filter fires.

See Also:
    :class:`~ph_neuro.layers.linear.TernaryHebbianLinear` — the linear (MLP) variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.core.latent_scores import LatentScoreTensor
from ph_neuro.core.ternary_tensor import TernaryTensor


class TernaryHebbianConv2d(nn.Module):
    """Ternary Hebbian 2D convolution layer.

    Stores ternary weights of shape ``(out_channels, in_channels, kH, kW)``
    with associated latent float scores. The forward pass extracts local
    patches via ``F.unfold`` and performs a ternary MatMul. Learning updates
    the latent scores via a spatial Hebbian rule; ternary weights are
    periodically refreshed via hysteresis thresholding.

    No ``.backward()`` is called for learning — this layer learns purely
    through local Hebbian plasticity.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        theta_upper: Hysteresis upper threshold — latent score must exceed
            this to activate a synapse (0 → ±1).
        theta_lower: Hysteresis lower threshold — latent score must fall
            below this to deactivate a synapse (±1 → 0).

    Attributes:
        weight: Ternary weight tensor (read-only via ``TernaryTensor``).
        latent_scores: fp16 latent scores accessible for inspection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        theta_upper: float = 7.0,
        theta_lower: float = 1.5,
        device: torch.device | str | None = None,
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
        self._hebbian_enabled = True

        # Weight shape: (out_channels, in_channels, kH, kW)
        weight_shape = (out_channels, in_channels, self._kernel_size[0], self._kernel_size[1])

        # Ternary weight storage (starts all zeros)
        self._ternary_weight = TernaryTensor(weight_shape, packed=False, device=device)

        # Latent scores (small random init)
        self._latent_scores = LatentScoreTensor(weight_shape, init_std=0.1, device=device)

    # ── Properties ───────────────────────────────────────────────

    @property
    def weight(self) -> TernaryTensor:
        """Ternary weight tensor (read-only)."""
        return self._ternary_weight

    @property
    def latent_scores(self) -> torch.Tensor:
        """fp16 latent scores (read-only)."""
        return self._latent_scores.scores

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

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self._kernel_size

    # ── Device movement ──────────────────────────────────────────

    def _apply(self, fn: callable) -> TernaryHebbianConv2d:
        """Override ``_apply`` to move custom tensors alongside registered ones."""
        self._ternary_weight._data = fn(self._ternary_weight._data)
        self._latent_scores.scores = fn(self._latent_scores.scores)
        return super()._apply(fn)

    # ── Forward pass ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: unfold → ternary MatMul → reshape.

        Extracts local patches via ``F.unfold`` and performs a
        linear transformation using the ternary weights.

        Args:
            x: Input tensor, shape ``(N, in_channels, H, W)``.

        Returns:
            Output tensor, shape ``(N, out_channels, H_out, W_out)``.
        """
        N = x.shape[0]

        # Unfold input: (N, C_in*kH*kW, L) where L = H_out * W_out
        patches = F.unfold(
            x,
            kernel_size=self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )  # (N, C_in*kH*kW, L)

        # Weight: (C_out, C_in, kH, kW) → 2D: (C_out, C_in*kH*kW)
        w = self._ternary_weight.to_dense()
        if w.device != x.device:
            w = w.to(x.device)
        w_2d = w.reshape(self._out_channels, -1)

        # MatMul: (N, C_in*kH*kW, L) → transpose to (N, L, C_in*kH*kW)
        # Then F.linear(patches, w_2d) = patches @ w_2d.T → (N, L, C_out)
        patches_t = patches.transpose(1, 2)  # (N, L, C_in*kH*kW)
        out = F.linear(patches_t, w_2d)  # (N, L, C_out)

        # Reshape to spatial: (N, C_out, H_out, W_out)
        H_out = _conv_output_size(
            x.shape[2], self._kernel_size[0], self._stride[0], self._padding[0], self._dilation[0]
        )
        W_out = _conv_output_size(
            x.shape[3], self._kernel_size[1], self._stride[1], self._padding[1], self._dilation[1]
        )
        out = out.transpose(1, 2).reshape(N, self._out_channels, H_out, W_out)
        return out

    # ── Hebbian update ───────────────────────────────────────────

    def hebbian_update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        lr: float,
    ) -> None:
        """Apply Hebbian update to latent scores.

        The update aggregates over ALL batch and spatial positions:

            ΔW = lr / (N · L) × Σ_{b,i,j} output[b,:,i,j] × input_patch[b,i,j]

        where ``input_patch`` is the ``(C_in · kH · kW)``-dimensional patch
        at spatial position ``(i,j)`` and ``output[b,:,i,j]`` is the
        ``C_out``-dimensional filter response at the same position.

        Args:
            pre_activation: Pre-synaptic input, shape ``(N, C_in, H, W)``.
                This is the **raw input** (not yet unfolded).
            post_activation: Post-synaptic activations, shape
                ``(N, C_out, H_out, W_out)``. Should be ternary {-1, 0, +1}
                (int8).
            lr: Learning rate for the Hebbian update.
        """
        if not self._hebbian_enabled:
            return

        N = pre_activation.shape[0]

        # Unfold input to patches: (N, C_in*kH*kW, L)
        patches = F.unfold(
            pre_activation,
            kernel_size=self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )  # (N, C_in*kH*kW, L)

        L = patches.shape[2]  # number of spatial positions = H_out * W_out

        # Flatten post: (N, C_out, H_out, W_out) → (N, C_out, L)
        post_flat = post_activation.float().reshape(N, self._out_channels, L)

        # ΔW_2d = lr / (N * L) × Σ_b post_flat[b] @ patches[b].T
        # post_flat: (N, C_out, L), patches: (N, C_in*kH*kW, L)
        # bmm: (N, C_out, C_in*kH*kW) → sum over batch → (C_out, C_in*kH*kW)
        delta_2d = torch.bmm(post_flat, patches.float().transpose(1, 2))  # (N, C_out, C_in*kHW)
        delta_2d = delta_2d.sum(dim=0)  # (C_out, C_in*kHW)
        delta_2d = lr * delta_2d / (N * L)

        # Reshape to weight shape: (C_out, C_in, kH, kW)
        delta = delta_2d.reshape(self._out_channels, self._in_channels, *self._kernel_size)

        if self._latent_scores.scores.device != delta.device:
            self._latent_scores.scores = self._latent_scores.scores.to(delta.device)
        self._latent_scores.scores += delta.to(self._latent_scores.scores.dtype)

    # ── Weight refresh (hysteresis) ──────────────────────────────

    def refresh_weights(self) -> None:
        """Refresh ternary weights from latent scores using hysteresis.

        For each synapse:
        - If weight is 0 and ``|score| > theta_upper`` → flip to ``sign(score)``
        - If weight is ±1 and ``|score| < theta_lower`` → flip to 0

        The hysteresis gap ``(theta_upper - theta_lower)`` prevents oscillation.
        """
        scores = self._latent_scores.scores
        current_weights = self._ternary_weight.unpack()

        if scores.device != current_weights.device:
            current_weights = current_weights.to(scores.device)

        new_weights = current_weights.clone()

        # Activate: weight == 0 and |score| > theta_upper
        activate_mask = (current_weights == 0) & (scores.abs() > self._theta_upper)
        new_weights[activate_mask] = scores[activate_mask].sign().to(torch.int8)

        # Deactivate: weight != 0 and |score| < theta_lower
        deactivate_mask = (current_weights != 0) & (scores.abs() < self._theta_lower)
        new_weights[deactivate_mask] = 0

        # Write back new weights
        if self._ternary_weight.packed:
            self._ternary_weight = TernaryTensor.pack(new_weights)
        else:
            self._ternary_weight._data = new_weights

    # ── Freeze control ───────────────────────────────────────────

    def requires_hebbian_(self, enabled: bool) -> TernaryHebbianConv2d:
        """Enable or disable Hebbian learning for this layer.

        When disabled, ``hebbian_update`` and ``apply_decay`` are no-ops.

        Args:
            enabled: If ``True`` (default), Hebbian learning is active.

        Returns:
            ``self`` for chaining.
        """
        self._hebbian_enabled = enabled
        return self

    def apply_decay(self, decay_rate: float) -> None:
        """Apply homeostatic decay to latent scores.

        Args:
            decay_rate: Decay factor.
        """
        if not self._hebbian_enabled:
            return
        self._latent_scores.apply_decay(decay_rate)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self._in_channels}, "
            f"out_channels={self._out_channels}, "
            f"kernel_size={self._kernel_size}, "
            f"stride={self._stride}, "
            f"padding={self._padding}, "
            f"theta_upper={self._theta_upper}, "
            f"theta_lower={self._theta_lower}"
        )


# ── Helpers ──────────────────────────────────────────────────────


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an int to a pair, or pass through a pair unchanged."""
    if isinstance(x, tuple):
        return x
    return (x, x)


def _conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Compute the output spatial size for one dimension of a convolution.

    Formula: ``⌊(input_size + 2·padding - dilation·(kernel_size - 1) - 1) / stride + 1⌋``
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
