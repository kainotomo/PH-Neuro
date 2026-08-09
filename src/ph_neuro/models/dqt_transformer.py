"""DQT Transformer model factory (Milestone M2.1).

``dqt_gpt2()`` builds a GPT-2-style decoder-only transformer whose
weight-bearing linear projections are trained with Direct Quantized
Training (DQT) — ternary int8 weights + stochastic rounding + annealing.

Architecture (GPT-2 / BitNet-style):
    ``TokenEmbedding (float) -> Dropout -> [TransformerBlock x n_layers]
     -> RMSNorm -> LM Head (DQT ternary)``

Key M2.1 design decisions:
    - Token embedding is a float ``nn.Embedding`` (a lookup table, NOT a
      matmul) — never ternary.
    - LM Head is a DQT ternary ``TernaryDQTLinear3D``.
    - No weight tying: the embedding is float and the LM head is ternary,
      so they cannot share weights (kept separate).
    - Position encoding is RoPE (parameter-free) inside the attention.

The recommended FULL config (d_model=768, n_heads=12, n_layers=9,
d_ff=3072, vocab=50257) gives ~102M ternary weights (target ~100M) +
~39M float embedding ≈ 141M total parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_transformer import (
    TernaryDQTLinear3D,
    TernaryDQTMoETransformerBlock,
    TernaryDQTRMSNorm,
    TernaryDQTTransformerBlock,
)

__all__ = [
    "DQTTransformer",
    "dqt_gpt2",
    "DQTMoETransformer",
    "dqt_gpt2_moe",
    "SMOKE_CONFIG",
    "FULL_CONFIG",
    "M2_2_CONFIG",
    "SMOKE_MOE_CONFIG",
    "M2_3_CONFIG",
    "count_ternary_weights",
    "count_parameters",
    "count_float_parameters",
    "build_config",
]

# ── Recommended configurations ─────────────────────────────────────

# Phase 1 (smoke test): ~16M ternary weights. Validates convergence fast.
SMOKE_CONFIG: dict = {
    "vocab_size": 50257,  # GPT-2 BPE
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 4,
    "d_ff": 1024,
    "max_seq_len": 256,
}

# Phase 2 (FULL run): ~102M ternary weights (target ~100M) + ~39M float
# embedding ≈ 141M total. n_heads=12 keeps d_head = 768 // 12 = 64.
FULL_CONFIG: dict = {
    "vocab_size": 50257,  # GPT-2 BPE
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 9,
    "d_ff": 3072,
    "max_seq_len": 512,
}

# Phase M2.2 (scaling test 102M → 250M): ~252.8M ternary weights + ~51.5M
# float embedding ≈ 304M total. Per block: 4·1024² + 2·1024·4096 =
# 12,582,912 ternary; 16 blocks = 201,326,592; LM Head 1024·50257 =
# 51,463,168 → total ternary 252,789,760. n_heads=16 keeps d_head = 64.
# Trained on WikiText-2 (GPT-2 BPE). Gradient checkpointing is REQUIRED at
# batch 8 / seq 256 to fit 8 GB (see the E026 memory budget).
M2_2_CONFIG: dict = {
    "vocab_size": 50257,  # GPT-2 BPE
    "d_model": 1024,
    "n_heads": 16,
    "n_layers": 16,
    "d_ff": 4096,
    "max_seq_len": 256,
}

# Smoke config for the MoE model (M2.3): small so unit tests / smoke runs
# stay fast. 2 dense + 2 MoE blocks (4 experts, top-2), d=256.
SMOKE_MOE_CONFIG: dict = {
    "vocab_size": 50257,  # GPT-2 BPE
    "d_model": 256,
    "n_heads": 4,
    "n_layers": 4,  # dense_layers + moe_layers
    "dense_layers": 2,
    "moe_layers": 2,
    "d_ff": 1024,
    "n_experts": 4,
    "top_k": 2,
    "max_seq_len": 256,
    "lb_coef": 0.1,
    "router_lr_ratio": 0.1,
}

# Milestone M2.3 FULL config: MoE DQT Transformer — first MoE DQT model.
#   d_model=768, n_layers=12, n_heads=12, d_ff=3072.
#   Layers 0-5: dense FFN (6 layers — cheap, good early features).
#   Layers 6-11: MoE FFN (6 layers x 6 experts, top-2 routing).
#
# NOTE (2026-08-06, revised from 4 dense + 8 MoE): the original 8-MoE-layer
# config (312.3M ternary) measured 7.40 GB torch peak / ~7.9-8.0 GB
# nvidia-smi at batch 4 — AT the 8.2 GB card's physical limit, so it crashed
# intermittently with `CUDA driver error: device not ready` (NOT gaming —
# reproduced with the Windows GPU idle). Per the brief's memory-budget rule
# ("ΑΝ ο υπολογισμός σου βγάζει >7.5 GB → ΜΕΙΩΣΕ n_experts ή n_layers"), the
# MoE stack was cut 8→6 layers. The revised config still meets the milestone
# envelope (300-400M total / 150-200M active):
#   total ternary ~265M + ~39M float embed ≈ 304M total (300-400M ✓)
#   active ~152M (top-2 of 6 experts, 57%) (150-200M ✓)
#   torch peak ~6.7 GB / nvidia-smi ~7.2 GB → ~1 GB headroom ✓
#
# Parameter budget (verified numerically):
#   per expert FFN  = 2 * 768 * 3072 = 4,718,592 (4.72M)
#   dense FFN       = 6 * 4.72M = 28.31M
#   MoE FFN (total) = 6 * 6 * 4.72M = 169.87M
#   MoE FFN (active)= 6 * 2 * 4.72M = 56.62M  (top-2 of 6)
#   attention       = 12 * 4 * 768^2 = 28.31M
#   LM head         = 768 * 50257 = 38.60M
#   TOTAL ternary   = 265.09M,  active 151.84M (57%)
#   float embedding = 50257 * 768 = 38.60M (no AdamW — SGD, ~0.15 GB)
M2_3_CONFIG: dict = {
    "vocab_size": 50257,  # GPT-2 BPE
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,  # dense_layers + moe_layers
    "dense_layers": 6,
    "moe_layers": 6,
    "d_ff": 3072,
    "n_experts": 6,
    "top_k": 2,
    "max_seq_len": 256,
    "lb_coef": 0.1,          # Switch-Transformer aux loss weight
    "router_lr_ratio": 0.1,   # router lr = 0.1 x expert lr
}


def build_moe_config(
    vocab_size: int = 50257,
    d_model: int = 768,
    n_heads: int = 12,
    d_ff: int = 3072,
    dense_layers: int = 4,
    moe_layers: int = 8,
    n_experts: int = 6,
    top_k: int = 2,
    max_seq_len: int = 256,
    lb_coef: float = 0.1,
    router_lr_ratio: float = 0.1,
) -> dict:
    """Assemble an M2.3 MoE model config dict (useful for CLI overrides)."""
    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": dense_layers + moe_layers,
        "dense_layers": dense_layers,
        "moe_layers": moe_layers,
        "d_ff": d_ff,
        "n_experts": n_experts,
        "top_k": top_k,
        "max_seq_len": max_seq_len,
        "lb_coef": lb_coef,
        "router_lr_ratio": router_lr_ratio,
    }


def build_config(
    vocab_size: int = 50257,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 9,
    d_ff: int = 3072,
    max_seq_len: int = 512,
) -> dict:
    """Assemble a model config dict (useful for CLI overrides)."""
    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "max_seq_len": max_seq_len,
    }


# ── Model ──────────────────────────────────────────────────────────


class DQTTransformer(nn.Module):
    """Decoder-only GPT-2-style transformer with DQT ternary projections.

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum sequence length (RoPE tables).
        dropout: Dropout probability (default 0.0).
        theta_base: RoPE base (default 10000).
        use_grad_checkpointing: If True, wrap each block in
            ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
            so activations are recomputed during backward instead of stored.
            This trades ~30-40% compute for a large activation-memory cut —
            REQUIRED for the 250M M2.2 config on 8 GB. Blocks are pure
            (no in-place ops, no side effects) so checkpointing is safe.
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        theta_base: float = 10000.0,
        use_grad_checkpointing: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.use_grad_checkpointing = use_grad_checkpointing

        # Float token embedding (lookup table — NOT ternary)
        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TernaryDQTTransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    theta_base=theta_base,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = TernaryDQTRMSNorm(d_model, device=device)
        # LM head is DQT ternary (no weight tying with the float embedding)
        self.lm_head = TernaryDQTLinear3D(
            d_model, vocab_size, bias=False, device=device, dtype=dtype
        )

    def _checkpoint_block(
        self, x: torch.Tensor, block: nn.Module
    ) -> torch.Tensor:
        """Run a single block through gradient checkpointing."""
        return block(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Language-model forward pass.

        Args:
            tokens: Token ids, shape ``(batch, seq_len)``.

        Returns:
            Logits, shape ``(batch, seq_len, vocab_size)``.
        """
        x = self.token_embedding(tokens)  # (B, T, d_model)
        x = self.pos_dropout(x)
        if self.use_grad_checkpointing:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self._checkpoint_block, x, block, use_reentrant=False
                )
        else:
            for block in self.blocks:
                x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)  # (B, T, vocab_size)


def dqt_gpt2(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_seq_len: int,
    dropout: float = 0.0,
    theta_base: float = 10000.0,
    use_grad_checkpointing: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> DQTTransformer:
    """Build a DQT transformer (factory for :class:`DQTTransformer`).

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: Feed-forward hidden dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability (default 0.0).
        theta_base: RoPE base (default 10000).
        use_grad_checkpointing: If True, wrap each block in gradient
            checkpointing (recompute activations in backward) — see
            :class:`DQTTransformer` for details.
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.

    Returns:
        A ``DQTTransformer`` instance.
    """
    return DQTTransformer(
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        max_seq_len,
        dropout=dropout,
        theta_base=theta_base,
        use_grad_checkpointing=use_grad_checkpointing,
        device=device,
        dtype=dtype,
    )


# ── MoE DQT Transformer (M2.3) ────────────────────────────────────


class DQTMoETransformer(nn.Module):
    """Decoder-only GPT-2-style DQT transformer with a hybrid dense+MoE stack.

    Same as :class:`DQTTransformer` (float token embedding, DQT ternary
    projections, RMSNorm, RoPE, pre-norm, DQT LM head) EXCEPT the block
    stack is hybrid:

        - The first ``dense_layers`` blocks use the plain dense
          :class:`TernaryDQTTransformerBlock` (cheap, good early features).
        - The remaining ``moe_layers`` blocks use the sparse
          :class:`TernaryDQTMoETransformerBlock` (top-K of ``n_experts``
          DQT experts per token, float router, Switch-Transformer aux loss).

    The forward returns ``(logits, aux_loss)`` — ``aux_loss`` is the SUM of
    all MoE blocks' load-balancing losses (the dense blocks contribute 0),
    so the training loop adds ``lb_coef * aux_loss`` to the LM loss.

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (per expert).
        dense_layers: Number of leading dense blocks.
        moe_layers: Number of trailing MoE blocks (total = dense + moe).
        n_experts: Number of MoE experts per MoE block.
        top_k: Active experts per token.
        max_seq_len: Maximum sequence length (RoPE tables).
        dropout: Dropout probability (default 0.0).
        theta_base: RoPE base (default 10000).
        use_grad_checkpointing: If True, wrap each block in
            ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``.
            NOTE: M2.3 fits 8 GB WITHOUT it (~6.7 GB est.), so it defaults
            to False. If enabled, the MoE usage-stat buffers are still
            correct (both numerator and denominator scale together).
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dense_layers: int,
        moe_layers: int,
        n_experts: int,
        top_k: int,
        max_seq_len: int,
        dropout: float = 0.0,
        theta_base: float = 10000.0,
        use_grad_checkpointing: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dense_layers = dense_layers
        self.moe_layers = moe_layers
        self.n_layers = dense_layers + moe_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.use_grad_checkpointing = use_grad_checkpointing

        # Float token embedding (lookup table — NOT ternary)
        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.pos_dropout = nn.Dropout(dropout)

        # Hybrid block stack: leading dense blocks, trailing MoE blocks.
        self.blocks = nn.ModuleList()
        for _ in range(dense_layers):
            self.blocks.append(
                TernaryDQTTransformerBlock(
                    d_model, n_heads, d_ff, max_seq_len=max_seq_len,
                    dropout=dropout, theta_base=theta_base, device=device,
                    dtype=dtype,
                )
            )
        for _ in range(moe_layers):
            self.blocks.append(
                TernaryDQTMoETransformerBlock(
                    d_model, n_heads, d_ff, n_experts=n_experts, top_k=top_k,
                    max_seq_len=max_seq_len, dropout=dropout,
                    theta_base=theta_base, device=device, dtype=dtype,
                )
            )

        self.final_norm = TernaryDQTRMSNorm(d_model, device=device)
        # LM head is DQT ternary (no weight tying with the float embedding)
        self.lm_head = TernaryDQTLinear3D(
            d_model, vocab_size, bias=False, device=device, dtype=dtype
        )

    @staticmethod
    def _ckpt_dense(x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        return block(x)

    @staticmethod
    def _ckpt_moe(
        x: torch.Tensor, block: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return block(x)

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Language-model forward pass with hybrid dense+MoE blocks.

        Args:
            tokens: Token ids, shape ``(batch, seq_len)``.

        Returns:
            Tuple ``(logits, aux_loss)`` where ``logits`` has shape
            ``(batch, seq_len, vocab_size)`` and ``aux_loss`` is the summed
            MoE load-balancing loss (a scalar requiring grad when any MoE
            block exists, else a constant 0.0).
        """
        x = self.token_embedding(tokens)  # (B, T, d_model)
        x = self.pos_dropout(x)

        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            if isinstance(block, TernaryDQTMoETransformerBlock):
                if self.use_grad_checkpointing:
                    x, b_aux = torch.utils.checkpoint.checkpoint(
                        self._ckpt_moe, x, block, use_reentrant=False
                    )
                else:
                    x, b_aux = block(x)
                aux_losses.append(b_aux)
            else:
                if self.use_grad_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        self._ckpt_dense, x, block, use_reentrant=False
                    )
                else:
                    x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if aux_losses:
            aux_loss = torch.stack(aux_losses).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device, requires_grad=False)
        return logits, aux_loss


def dqt_gpt2_moe(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    dense_layers: int,
    moe_layers: int,
    n_experts: int,
    top_k: int,
    max_seq_len: int,
    dropout: float = 0.0,
    theta_base: float = 10000.0,
    use_grad_checkpointing: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> DQTMoETransformer:
    """Build a hybrid dense+MoE DQT transformer (factory for M2.3).

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Model (embedding) dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (per expert).
        dense_layers: Number of leading dense blocks.
        moe_layers: Number of trailing MoE blocks.
        n_experts: Number of MoE experts per MoE block.
        top_k: Active experts per token.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability (default 0.0).
        theta_base: RoPE base (default 10000).
        use_grad_checkpointing: If True, wrap each block in gradient
            checkpointing (default False — M2.3 fits 8 GB without it).
        device: Torch device.
        dtype: Float dtype for the DQT accumulation buffers.

    Returns:
        A ``DQTMoETransformer`` instance.
    """
    return DQTMoETransformer(
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        dense_layers,
        moe_layers,
        n_experts,
        top_k,
        max_seq_len,
        dropout=dropout,
        theta_base=theta_base,
        use_grad_checkpointing=use_grad_checkpointing,
        device=device,
        dtype=dtype,
    )


# ── Parameter counting ─────────────────────────────────────────────


def count_ternary_weights(model: nn.Module) -> int:
    """Total number of ternary (int8) weights across all DQT linear layers."""
    return sum(
        m.weight_ternary.numel()
        for m in model.modules()
        if isinstance(m, TernaryDQTLinear)
    )


def count_parameters(model: nn.Module) -> int:
    """Total number of float parameters (optimizer-tracked)."""
    return sum(p.numel() for p in model.parameters())


def count_float_parameters(model: nn.Module) -> int:
    """Float parameters not counted as ternary weights (embedding + norms)."""
    return count_parameters(model) - count_ternary_weights(model)


def model_summary(model: nn.Module) -> dict[str, int]:
    """Summary of a DQT transformer's parameter distribution."""
    total = count_parameters(model)
    ternary = count_ternary_weights(model)
    return {
        "total_params": total,
        "ternary_weights": ternary,
        "float_params": total - ternary,
    }
