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
    "TernaryDQTMoEFeedForward",
    "TernaryDQTMoETransformerBlock",
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


# ── Mixture-of-Experts feed-forward (M2.3) ─────────────────────────


class TernaryDQTMoEFeedForward(nn.Module):
    """Sparse Mixture-of-Experts feed-forward with DQT ternary experts.

    A switch of ``n_experts`` identical feed-forward networks, each an
    ``TernaryDQTLinear3D(d_model, d_ff) -> GELU -> TernaryDQTLinear3D(d_ff,
    d_model)`` (the same FFN shape as :class:`TernaryDQTFeedForward`,
    replicated per expert, each with the 1/sqrt(in) output scaling that is
    REQUIRED for DQT transformers — the M2.1 finding). A tiny FLOAT router
    (``nn.Linear(d_model, n_experts)``) selects the ``top_k`` experts per
    token and the experts' outputs are combined as a weighted
    (re-normalized softmax) sum. Only the selected experts run in the
    forward pass (grouped per-expert execution), so the active parameter
    count is ``top_k / n_experts`` of the full expert stack.

    Key M2.3 design rules:
        - The router is FLOAT (``nn.Linear``) — never ternary, never DQT.
          It is a tiny fraction of the params (``d_model * n_experts``)
          and needs full precision for stable top-K selection. The training
          runner gives it its own (0.1×) learning rate.
        - Load balancing: Switch-Transformer auxiliary loss
          ``n_experts * sum_i(f_i * P_i)`` (scaled by ``lb_coef=0.1`` in
          the runner), which discourages expert collapse.
        - Per-expert execution: tokens are grouped by their top-K expert
          choice and each expert runs only on its own tokens, so expert
          FLOPs scale with ``top_k / n_experts``.

    Args:
        d_model: Model (embedding) dimension.
        d_ff: Hidden feed-forward dimension (per expert).
        n_experts: Number of experts.
        top_k: Active experts per token (``1 <= top_k <= n_experts``).
        router_init_std: Init std of the (float) router weights.
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.

    Attributes:
        router: Float ``nn.Linear(d_model, n_experts)`` — the router.
        experts: ``ModuleList`` of ``n_experts`` FFN ``nn.Sequential``s.
        selection_counts / n_selections / coverage_counts / n_samples:
            Load-balancing tracking buffers (usage metrics).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 6,
        top_k: int = 2,
        router_init_std: float = 0.02,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(
                f"top_k ({top_k}) must be in [1, n_experts={n_experts}]"
            )
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k

        # Tiny FLOAT router (not quantized — needs full precision for
        # stable top-K selection, same as E019).
        self.router = nn.Linear(d_model, n_experts, bias=False, device=device)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)

        # DQT ternary experts — each an FFN identical in shape to the
        # dense TernaryDQTFeedForward (with the 1/sqrt(in) scaling).
        self.experts = nn.ModuleList(
            nn.Sequential(
                TernaryDQTLinear3D(d_model, d_ff, device=device, dtype=dtype),
                nn.GELU(),
                TernaryDQTLinear3D(d_ff, d_model, device=device, dtype=dtype),
            )
            for _ in range(n_experts)
        )

        # Load balancing tracking buffers (usage metrics) — registered on
        # the same device as the model so the CUDA/CPU indices match.
        self.register_buffer("selection_counts", torch.zeros(n_experts, device=device))
        self.register_buffer("n_selections", torch.zeros(1, device=device))
        self.register_buffer("coverage_counts", torch.zeros(n_experts, device=device))
        self.register_buffer("n_samples", torch.zeros(1, device=device))

    # ── Forward ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse MoE forward with top-K routing and grouped execution.

        Only the selected experts run: tokens are grouped per expert and
        each expert is invoked on exactly the tokens that selected it, so
        the FLOPs and memory of the expert stack scale with
        ``top_k / n_experts``.

        Args:
            x: Input, shape ``(batch, seq, d_model)``.

        Returns:
            Tuple ``(output, aux_loss)`` where ``output`` is the weighted
            (re-normalized softmax) sum of the top-K expert outputs with
            shape ``(batch, seq, d_model)`` and ``aux_loss`` is the
            Switch-Transformer load balancing loss (part of the graph).
        """
        batch, seq, _ = x.shape
        flat = x.reshape(-1, self.d_model)  # (B*T, d_model)
        n_tokens = flat.shape[0]

        # Router: top-K selection per token (softmax, top-K, re-normalize)
        logits = self.router(flat)  # (N, n_experts)
        probs = torch.softmax(logits, dim=-1)  # (N, n_experts)
        topk_probs, indices = torch.topk(probs, self.top_k, dim=-1)  # (N, K)
        weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balancing bookkeeping — only during real training forwards
        # (NOT eval — is_grad_enabled is False under @torch.no_grad — and
        # NOT gradient-checkpoint recomputation, which re-runs this forward
        # inside backward). Keeps the metrics honest.
        if torch.is_grad_enabled():
            self._update_usage_stats(indices)

        # Grouped expert execution — only run experts that were selected.
        out = torch.zeros(n_tokens, self.d_model, device=x.device, dtype=x.dtype)
        for e in range(self.n_experts):
            rows, k_pos = (indices == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            expert_out = self.experts[e](flat[rows])  # (n, d_model)
            w = weights[rows, k_pos].unsqueeze(-1)  # (n, 1)
            out = out.index_add(0, rows, expert_out * w)

        # Switch-Transformer auxiliary load balancing loss (part of graph).
        aux_loss = self._aux_load_balance_loss(logits, indices)

        return out.reshape(batch, seq, self.d_model), aux_loss

    # ── Load balancing helpers ──────────────────────────────────────

    @torch.no_grad()
    def _update_usage_stats(self, indices: torch.Tensor) -> None:
        """Accumulate selection/coverage counters for one forward pass."""
        self.selection_counts += torch.bincount(
            indices.flatten(), minlength=self.n_experts
        ).float()
        self.n_selections += indices.numel()
        sel_mask = (
            indices.unsqueeze(-1)
            == torch.arange(self.n_experts, device=indices.device)
        ).any(dim=1)  # (N, E) — which experts each token picked
        self.coverage_counts += sel_mask.float().sum(dim=0)
        self.n_samples += indices.shape[0]

    @torch.no_grad()
    def selection_fractions(self) -> torch.Tensor:
        """Share of all selections that went to each expert (sums to 1)."""
        if self.n_selections.item() <= 0:
            return torch.full((self.n_experts,), 1.0 / self.n_experts)
        return self.selection_counts / self.n_selections

    @torch.no_grad()
    def coverage_fractions(self) -> torch.Tensor:
        """Fraction of tokens where each expert was among the top-K."""
        if self.n_samples.item() <= 0:
            return torch.full((self.n_experts,), float(self.top_k) / self.n_experts)
        return self.coverage_counts / self.n_samples

    def _aux_load_balance_loss(
        self, logits: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Switch-Transformer auxiliary load balancing loss.

        ``L = n_experts * sum_i(f_i * P_i)`` where ``f_i`` is the fraction
        of selections dispatched to expert ``i`` and ``P_i`` is the mean
        router probability of expert ``i``. Minimized (value 1.0) when
        routing is perfectly uniform. Part of the computation graph so the
        optimizer can reduce it (scaled by ``lb_coef`` in the runner).
        """
        probs = torch.softmax(logits, dim=-1)  # (N, n_experts)
        f = (
            torch.bincount(indices.flatten(), minlength=self.n_experts).float()
            / indices.numel()
        )
        p = probs.mean(dim=0)
        return self.n_experts * (f * p).sum()

    # ── Utilities ───────────────────────────────────────────────────

    @torch.no_grad()
    def reset_usage_stats(self) -> None:
        """Reset the load balancing counters (e.g. between training/eval)."""
        self.selection_counts.zero_()
        self.n_selections.zero_()
        self.coverage_counts.zero_()
        self.n_samples.zero_()

    @torch.no_grad()
    def get_weight_stats(self) -> dict[str, float]:
        """Aggregate ternary weight stats across all experts."""
        total = zeros = pos = neg = 0
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, TernaryDQTLinear):
                    w = m.weight_ternary
                    n = w.numel()
                    total += n
                    zeros += int((w == 0).sum())
                    pos += int((w == 1).sum())
                    neg += int((w == -1).sum())
        if total == 0:
            return {"pos_pct": 0.0, "neg_pct": 0.0, "zero_pct": 0.0}
        return {
            "pos_pct": 100.0 * pos / total,
            "neg_pct": 100.0 * neg / total,
            "zero_pct": 100.0 * zeros / total,
        }

    @torch.no_grad()
    def balance_report(self) -> dict[str, float]:
        """Per-expert utilization for monitoring (dead-expert detector).

        Returns a dict with ``balance_ratio`` (max/min selection share —
        1.0 is perfectly balanced, ``inf`` means a dead expert) and
        ``min_share`` (the smallest expert's selection share — ``~0``
        flags expert collapse).
        """
        fracs = self.selection_fractions()
        mn, mx = float(fracs.min()), float(fracs.max())
        return {
            "balance_ratio": float(mx / mn) if mn > 0 else float("inf"),
            "min_share": mn,
            "max_share": mx,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count router (float) vs expert (ternary buffers) parameters."""
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = sum(
            p.numel() for p in self.experts.parameters()
        )  # includes the int8 ternary buffers
        return {
            "router": router_params,
            "experts": expert_params,
            "total": router_params + expert_params,
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"n_experts={self.n_experts}, top_k={self.top_k}"
        )


class TernaryDQTMoETransformerBlock(nn.Module):
    """Pre-norm transformer block with a sparse MoE feed-forward (M2.3).

    ``x = x + attention(RMSNorm(x))`` then ``x = x + moe_ffn(RMSNorm(x))`` —
    identical to :class:`TernaryDQTTransformerBlock` except the dense FFN is
    replaced by :class:`TernaryDQTMoEFeedForward`. Returns the block output
    AND the MoE auxiliary load-balancing loss so the training loop can add
    ``lb_coef * aux_loss`` to the LM loss.

    Args:
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (per expert).
        n_experts: Number of MoE experts.
        top_k: Active experts per token.
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
        n_experts: int = 6,
        top_k: int = 2,
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
        self.moe_ffn = TernaryDQTMoEFeedForward(
            d_model, d_ff, n_experts=n_experts, top_k=top_k, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-norm residual block with a sparse MoE FFN.

        Args:
            x: Input, shape ``(batch, seq, d_model)``.

        Returns:
            Tuple ``(output, aux_loss)`` where ``output`` has shape
            ``(batch, seq, d_model)`` and ``aux_loss`` is the block's MoE
            load balancing loss (0 for the dense variant — here it is the
            MoE FFN's aux loss).
        """
        x = x + self.attention(self.attn_norm(x))
        h, aux_loss = self.moe_ffn(self.ffn_norm(x))
        x = x + h
        return x, aux_loss
