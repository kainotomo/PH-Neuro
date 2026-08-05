"""Unit tests for the DQT Transformer layers (Milestone M2.1).

Covers :mod:`ph_neuro.layers.ste_dqt_transformer`: RMSNorm, the
shape-preserving DQT linear, multi-head causal attention (with RoPE),
the DQT feed-forward, the pre-norm transformer block, a full (mini)
transformer assembled from the layers, the ternary-weight invariant,
and DQT stochastic rounding.

All tests run on CPU with small dims so they stay fast.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_transformer import (
    TernaryDQTFeedForward,
    TernaryDQTLinear3D,
    TernaryDQTMultiheadAttention,
    TernaryDQTRMSNorm,
    TernaryDQTTransformerBlock,
)

D_MODEL = 64
N_HEADS = 4
D_FF = 256


def _iter_dqt_linear(model: nn.Module):
    """Yield every inner TernaryDQTLinear module (recursively)."""
    for m in model.modules():
        if isinstance(m, TernaryDQTLinear):
            yield m


def _ternary_invariant_holds(model: nn.Module) -> bool:
    """All DQT weights are int8 in {-1, 0, +1}."""
    for m in _iter_dqt_linear(model):
        w = m.weight_ternary
        if w.dtype != torch.int8:
            return False
        if not bool(torch.all((w >= -1) & (w <= 1)).item()):
            return False
    return True


def _build_mini_transformer(seq_len: int = 32) -> nn.Module:
    """A tiny end-to-end DQT transformer assembled from the layers.

    Mirrors the real model factory: float token embedding -> blocks ->
    final RMSNorm -> DQT LM head.
    """
    vocab = 128
    d_model = D_MODEL
    embedding = nn.Embedding(vocab, d_model)
    block = TernaryDQTTransformerBlock(
        d_model, N_HEADS, D_FF, max_seq_len=seq_len
    )
    final_norm = TernaryDQTRMSNorm(d_model)
    lm_head = TernaryDQTLinear3D(d_model, vocab, bias=False)

    class _Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            self.block = block
            self.final_norm = final_norm
            self.lm_head = lm_head

        def forward(self, tokens):
            x = self.embedding(tokens)
            x = self.block(x)
            x = self.final_norm(x)
            return self.lm_head(x)

    return _Mini()


class TestTernaryDQTRMSNorm:
    """RMSNorm module tests."""

    def test_rms_norm_forward(self):
        """RMSNorm should preserve shape and have unit RMS output."""
        rms = TernaryDQTRMSNorm(D_MODEL)
        x = torch.randn(4, 8, D_MODEL)
        out = rms(x)
        assert out.shape == x.shape, f"Got shape {out.shape}"
        # RMS of the output (before the scale, weight=1) should be ~1
        out_rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(out_rms, torch.ones_like(out_rms), atol=1e-2), (
            f"RMSNorm output RMS not ~1: {out_rms.mean().item():.4f}"
        )

    def test_rms_norm_vs_pytorch(self):
        """RMSNorm ≈ PyTorch LayerNorm on zero-mean input (same order)."""
        dim = 32
        eps = 1e-6
        rms = TernaryDQTRMSNorm(dim, eps=eps)
        ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        x = torch.randn(4, 8, dim)
        x_zero = x - x.mean(-1, keepdim=True)  # zero mean -> var == mean(x^2)
        assert torch.allclose(rms(x_zero), ln(x_zero), atol=1e-3), (
            "RMSNorm should match LayerNorm on zero-mean inputs"
        )


class TestTernaryDQTLinear3D:
    """Shape-preserving DQT linear wrapper."""

    def test_linear3d_shape(self):
        """(B, T, C) input should produce (B, T, O) output."""
        layer = TernaryDQTLinear3D(D_MODEL, D_MODEL, bias=False)
        x = torch.randn(2, 10, D_MODEL)
        out = layer(x)
        assert out.shape == (2, 10, D_MODEL), f"Got shape {out.shape}"
        assert layer.linear.weight_ternary.dtype == torch.int8


class TestTernaryDQTMultiheadAttention:
    """Multi-head causal self-attention tests."""

    def test_attention_forward(self):
        """Attention forward with causal mask should preserve shape."""
        attn = TernaryDQTMultiheadAttention(D_MODEL, N_HEADS, max_seq_len=32)
        x = torch.randn(2, 8, D_MODEL)
        out = attn(x)
        assert out.shape == x.shape, f"Got shape {out.shape}"
        assert torch.isfinite(out).all(), "Attention output contains NaN/inf"

    def test_attention_causal(self):
        """Tokens must not see the future."""
        attn = TernaryDQTMultiheadAttention(D_MODEL, N_HEADS, max_seq_len=32)
        torch.manual_seed(0)
        x = torch.randn(2, 8, D_MODEL)
        x2 = x.clone()
        x2[:, 4:, :] = 0.0  # change only future tokens (positions 4..7)
        out1 = attn(x)
        out2 = attn(x2)
        # Positions 0..3 attend only to 0..3 (unchanged) -> identical outputs
        assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5), (
            "Early tokens changed when future tokens changed — causal mask broken"
        )

    def test_attention_multihead(self):
        """Multi-head attention works and keeps correct shapes."""
        n_heads = 8
        d_model = 64
        attn = TernaryDQTMultiheadAttention(d_model, n_heads, max_seq_len=16)
        x = torch.randn(3, 12, d_model)
        out = attn(x)
        assert out.shape == (3, 12, d_model)
        # Sanity: heads were actually split (d_head correct)
        assert attn.d_head == d_model // n_heads


class TestTernaryDQTFeedForward:
    """DQT feed-forward network tests."""

    def test_feedforward_forward(self):
        """FFN should preserve shape and contain DQT linear layers."""
        ffn = TernaryDQTFeedForward(D_MODEL, D_FF)
        x = torch.randn(2, 10, D_MODEL)
        out = ffn(x)
        assert out.shape == x.shape, f"Got shape {out.shape}"
        assert torch.isfinite(out).all()


class TestTernaryDQTTransformerBlock:
    """Pre-norm transformer block tests."""

    def test_transformer_block_forward(self):
        """Block should preserve shape with residual connections."""
        block = TernaryDQTTransformerBlock(
            D_MODEL, N_HEADS, D_FF, max_seq_len=32
        )
        x = torch.randn(2, 10, D_MODEL)
        out = block(x)
        assert out.shape == x.shape, f"Got shape {out.shape}"
        assert torch.isfinite(out).all()
        assert _ternary_invariant_holds(block)


class TestTernaryDQTTransformerModel:
    """Full (mini) transformer assembled from the layers."""

    def test_transformer_model_forward(self):
        """Whole model (embedding + block + norm + LM head) works."""
        seq_len = 32
        model = _build_mini_transformer(seq_len=seq_len)
        model.eval()
        tokens = torch.randint(0, 128, (2, seq_len))
        logits = model(tokens)
        assert logits.shape == (2, seq_len, 128), f"Got shape {logits.shape}"
        assert torch.isfinite(logits).all()

    def test_transformer_dqt_weights(self):
        """All DQT weights in the model are ternary int8 in {-1, 0, +1}."""
        model = _build_mini_transformer()
        assert _ternary_invariant_holds(model), (
            "Some DQT weights are not int8 ternary in {-1, 0, +1}"
        )
        # The model must actually contain DQT linear layers
        n_dqt = sum(1 for _ in _iter_dqt_linear(model))
        assert n_dqt >= 6, f"Expected >= 6 DQT linear layers, got {n_dqt}"

    def test_transformer_stochastic_rounding(self):
        """DQT stochastic rounding keeps weights ternary and tracks flips."""
        model = _build_mini_transformer()
        for _ in range(3):
            for m in _iter_dqt_linear(model):
                stats = m.apply_stochastic_rounding()
                assert 0.0 <= stats["flip_rate"] <= 1.0
            assert _ternary_invariant_holds(model)
