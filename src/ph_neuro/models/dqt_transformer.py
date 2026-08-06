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
    TernaryDQTRMSNorm,
    TernaryDQTTransformerBlock,
)

__all__ = [
    "DQTTransformer",
    "dqt_gpt2",
    "SMOKE_CONFIG",
    "FULL_CONFIG",
    "M2_2_CONFIG",
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
