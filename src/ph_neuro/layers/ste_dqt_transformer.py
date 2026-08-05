"""DQT Transformer layers (Milestone M2.1).

Implements GPT-2-style transformer building blocks whose weight-bearing
linear projections are trained with Direct Quantized Training (DQT) — the
same int8 ternary weights + stochastic rounding + annealing machinery as
:mod:`ph_neuro.layers.ste_dqt`. These are the NEW DQT transformer layers
(the Hebbian-era placeholders in :mod:`ph_neuro.layers.attention` and
:mod:`ph_neuro.models.transformer` are a different era and are NOT used).

Design rules (M2.1 brief):
    - Pre-norm: ``x = x + attn(RMSNorm(x))``, ``x = x + ffn(RMSNorm(x))``.
      Pre-norm is more stable for DQT (M1.1 showed DQT wants stability).
    - RMSNorm is a FLOAT module (scale parameter, no bias) — never ternary.
    - Q, K, V, O projections are ``TernaryDQTLinear`` (DQT weights).
    - Attention scores are float softmax — never quantized.
    - GELU activation is float — never quantized.
    - RoPE (rotary position embeddings) — parameter-free position encoding.

Module summary::

    TernaryDQTRMSNorm              — float RMSNorm (x / rms(x) * scale)
    TernaryDQTLinear3D             — shape-preserving DQT linear (B, T, C)->(B, T, O)
    TernaryDQTMultiheadAttention   — GPT-2 style causal self-attention, RoPE, DQT proj
    TernaryDQTFeedForward          — DQT FFN: DQTLinear -> GELU -> DQTLinear
    TernaryDQTTransformerBlock     — pre-norm block: attn + ffn with residual add
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_dqt import TernaryDQTLinear

__all__ = [
    "TernaryDQTRMSNorm",
    "TernaryDQTLinear3D",
    "precompute_rotary_embeddings",
    "apply_rotary_embeddings",
    "TernaryDQTMultiheadAttention",
    "TernaryDQTFeedForward",
    "TernaryDQTTransformerBlock",
]


# ── RMSNorm (float — NOT ternary) ──────────────────────────────────


class TernaryDQTRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Used instead of LayerNorm: simpler and faster (no mean subtraction,
    no bias). ``x / sqrt(mean(x^2) + eps) * weight``. The ``weight`` scale
    is a standard float ``nn.Parameter`` — this module is intentionally
    NOT quantized (normalization layers stay float in DQT, like M1.1's
    BatchNorm).

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant added to the variance for numerical stability.
        device: Torch device for the scale parameter.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension.

        Args:
            x: Input tensor, shape ``(..., dim)``.

        Returns:
            Normalized tensor, same shape.
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ── Shape-preserving DQT linear ────────────────────────────────────


class TernaryDQTLinear3D(nn.Module):
    """Shape-preserving wrapper around :class:`TernaryDQTLinear`.

    ``TernaryDQTLinear`` flattens inputs with ``dim > 2`` to 2D and does
    NOT restore the leading dimensions (its M1.1 contract). Transformers
    need ``(batch, seq, features) -> (batch, seq, out)``, so this wrapper
    reshapes around the inner DQT linear while reusing ALL of its DQT
    machinery (``weight_float`` param, int8 ``weight_ternary`` buffer,
    ``apply_stochastic_rounding``, ``apply_deterministic_rounding``).

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias (default ``False`` —
            RMSNorm handles the per-token shift, GPT-2 style).
        scale: Output scaling factor. If ``None`` (default), uses the
            variance-preserving ``1 / sqrt(in_features)``. This is
            CRITICAL for DQT transformers: ternary weights are ±1 with
            only ~10% nonzero, so a raw matmul over ``in_features``
            amplifies activations by ~``sqrt(in_features)``, which makes
            the residual stream explode (measured block output std ~20-30
            without scaling). ``1/sqrt(in_features)`` keeps block outputs
            ~O(0.1) and is the same stabilization BitNet b1.58 gets from
            its per-tensor activation scaling.
        device: Torch device.
        dtype: Float dtype for the accumulation buffer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.linear = TernaryDQTLinear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        if scale is None:
            scale = 1.0 / math.sqrt(in_features)
        self._output_scale = float(scale)

    @property
    def output_scale(self) -> float:
        """The output scaling factor applied to the matmul result."""
        return self._output_scale

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def weight_ternary(self) -> torch.Tensor:
        """The int8 ternary weight buffer (delegated to the inner layer)."""
        return self.linear.weight_ternary

    @property
    def weight_float(self) -> torch.Tensor:
        """The float accumulation parameter (delegated to the inner layer)."""
        return self.linear.weight_float

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DQT linear that preserves leading (batch, seq, ...) dims.

        Args:
            x: Input tensor, shape ``(*leading, in_features)``.

        Returns:
            Output tensor, shape ``(*leading, out_features)``, scaled by
            ``output_scale`` (variance-preserving 1/sqrt(in_features)).
        """
        *leading, in_f = x.shape
        flat = x.reshape(-1, in_f)
        out = self.linear(flat)  # (prod(leading), out_features)
        out = out.reshape(*leading, self.out_features)
        return out * self._output_scale

    def apply_stochastic_rounding(self) -> dict[str, float]:
        """Delegate to the inner DQT linear (see ``TernaryDQTLinear``)."""
        return self.linear.apply_stochastic_rounding()

    def apply_deterministic_rounding(self) -> dict[str, float]:
        """Delegate to the inner DQT linear (annealing / fine-tuning phase)."""
        return self.linear.apply_deterministic_rounding()

    def get_flip_rate(self) -> float:
        """Delegate to the inner DQT linear."""
        return self.linear.get_flip_rate()

    def get_weight_stats(self) -> dict[str, float]:
        """Delegate to the inner DQT linear."""
        return self.linear.get_weight_stats()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ── Rotary position embeddings (RoPE — parameter-free) ─────────────


def precompute_rotary_embeddings(
    dim: int,
    max_seq_len: int,
    theta_base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cosine/sine tables (Su et al., 2021).

    Returns a pair of buffers of shape ``(max_seq_len, dim)``. Each row is
    the (cos, sin) rotation angles for a position; the last half of the
    table mirrors the first half (``cat([freq, freq])``) so it broadcasts
    against the head dimension in :func:`apply_rotary_embeddings`.

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta_base: Rotary base (default 10000, same as RoFormer).
        device: Torch device for the buffers.
        dtype: Float dtype for the buffers.

    Returns:
        Tuple ``(cos, sin)`` of shape ``(max_seq_len, dim)``.
    """
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )  # (half,)
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, half)
    # GPT-J style: repeat the halves so the table has shape (seq, dim).
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension (RoPE helper)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embeddings(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query/key tensors.

    Args:
        q: Query, shape ``(batch, heads, seq, d_head)``.
        k: Key, shape ``(batch, heads, seq, d_head)``.
        cos: Cosine table, shape ``(max_seq, d_head)``.
        sin: Sine table, shape ``(max_seq, d_head)``.

    Returns:
        Tuple of rotated ``(q, k)`` with unchanged shapes.
    """
    # Broadcast (seq, d_head) -> (1, 1, seq, d_head)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ── Multi-head self-attention ──────────────────────────────────────


class TernaryDQTMultiheadAttention(nn.Module):
    """GPT-2 style causal self-attention with DQT projections.

    Q, K, V and the output projection are :class:`TernaryDQTLinear3D`
    (ternary int8 DQT weights, trained with stochastic rounding). The
    attention scores themselves stay float (softmax is never quantized).
    RoPE is applied to Q/K for position encoding (parameter-free).

    Args:
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads (``d_model % n_heads == 0``).
        max_seq_len: Maximum sequence length (for the RoPE tables).
        dropout: Dropout applied to attention weights (default 0.0).
        theta_base: RoPE base (default 10000).
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        theta_base: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, "d_head must be even for RoPE"

        # DQT projections (bias=False — RMSNorm handles the shift)
        self.q_proj = TernaryDQTLinear3D(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = TernaryDQTLinear3D(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = TernaryDQTLinear3D(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = TernaryDQTLinear3D(d_model, d_model, device=device, dtype=dtype)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

        # RoPE tables (buffers — not parameters)
        cos, sin = precompute_rotary_embeddings(
            self.d_head, max_seq_len, theta_base=theta_base, device=device, dtype=dtype
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(
        self, x: torch.Tensor, past_length: int = 0
    ) -> torch.Tensor:
        """Causal self-attention forward pass.

        Args:
            x: Input, shape ``(batch, seq, d_model)``.
            past_length: Number of already-processed tokens (0 for
                training on full sequences).

        Returns:
            Attention output, shape ``(batch, seq, d_model)``.
        """
        batch, seq, _ = x.shape

        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE on Q/K
        cos = self.rope_cos[past_length : past_length + seq]
        sin = self.rope_sin[past_length : past_length + seq]
        q, k = apply_rotary_embeddings(q, k, cos, sin)

        # Scaled dot-product attention with causal mask
        attn = q @ k.transpose(-2, -1) * self.scale  # (B, H, T, T)
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        return self.o_proj(out)


# ── Feed-forward network ───────────────────────────────────────────


class TernaryDQTFeedForward(nn.Module):
    """Pointwise feed-forward network with DQT projections.

    ``TernaryDQTLinear3D(d_model, d_ff) -> GELU -> TernaryDQTLinear3D(d_ff, d_model)``
    with ``d_ff ~ 4 * d_model``. GELU is float (never quantized).

    Args:
        d_model: Model (embedding) dimension.
        d_ff: Hidden feed-forward dimension (typically ``4 * d_model``).
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.fc_in = TernaryDQTLinear3D(d_model, d_ff, device=device, dtype=dtype)
        self.fc_out = TernaryDQTLinear3D(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward pass (unchanged shape).

        Args:
            x: Input, shape ``(batch, seq, d_model)``.

        Returns:
            Output, shape ``(batch, seq, d_model)``.
        """
        x = self.fc_in(x)
        x = F.gelu(x)
        return self.fc_out(x)


# ── Transformer block ──────────────────────────────────────────────


class TernaryDQTTransformerBlock(nn.Module):
    """Pre-norm transformer block (GPT-2 style).

    ``x = x + attention(RMSNorm(x))`` then ``x = x + feedforward(RMSNorm(x))``.
    Pre-norm (rather than post-norm) is used because M1.1 showed DQT wants
    training stability.

    Args:
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum sequence length (RoPE tables).
        dropout: Dropout probability (default 0.0).
        theta_base: RoPE base (default 10000).
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        theta_base: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.attn_norm = TernaryDQTRMSNorm(d_model, device=device)
        self.ffn_norm = TernaryDQTRMSNorm(d_model, device=device)
        self.attention = TernaryDQTMultiheadAttention(
            d_model,
            n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            theta_base=theta_base,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = TernaryDQTFeedForward(
            d_model, d_ff, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm residual block.

        Args:
            x: Input, shape ``(batch, seq, d_model)``.

        Returns:
            Output, shape ``(batch, seq, d_model)``.
        """
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
