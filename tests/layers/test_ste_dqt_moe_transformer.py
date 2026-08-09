"""Unit tests for the DQT transformer MoE layers (Milestone M2.3).

Covers :mod:`ph_neuro.layers.ste_dqt_transformer`: the sparse
:class:`TernaryDQTMoEFeedForward` (float router + top-K routing + grouped
per-expert DQT execution + Switch-Transformer load-balancing loss) and the
:class:`TernaryDQTMoETransformerBlock` (pre-norm block with a MoE FFN), plus
a hybrid dense+MoE transformer assembled from the layers.

All tests run on CPU with small dims so they stay fast.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_transformer import (
    TernaryDQTMoEFeedForward,
    TernaryDQTMoETransformerBlock,
    TernaryDQTTransformerBlock,
)

D_MODEL = 64
N_HEADS = 4
D_FF = 256
N_EXPERTS = 4
TOP_K = 2


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


def _count_ternary(model: nn.Module) -> int:
    """Total number of int8 ternary weights (for active-param accounting)."""
    return sum(m.weight_ternary.numel() for m in _iter_dqt_linear(model))


# ── MoE FFN: forward ───────────────────────────────────────────────


class TestMoeFfnForward:
    """TernaryDQTMoEFeedForward forward pass."""

    def test_moe_ffn_forward(self):
        """(B, T, D) input produces (B, T, D) output + a scalar aux loss."""
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        x = torch.randn(2, 8, D_MODEL)
        out, aux_loss = moe(x)
        assert out.shape == x.shape, f"Got shape {out.shape}"
        assert torch.isfinite(out).all(), "MoE output contains NaN/inf"
        assert torch.isfinite(aux_loss).all(), "aux loss contains NaN/inf"
        assert aux_loss.ndim == 0, "aux loss must be a scalar"
        # Switch aux loss is minimized at 1.0; a random router should be
        # somewhere >= 1.0 and finite.
        assert aux_loss.item() >= 1.0 - 1e-3

    def test_moe_ffn_router_is_float(self):
        """The router is a plain float nn.Linear (never DQT/ternary)."""
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        assert isinstance(moe.router, nn.Linear)
        assert moe.router.weight.dtype == torch.float32
        assert moe.router.bias is None
        assert moe.router.weight.requires_grad
        # Router params are NOT DQT layers (not counted as ternary weights)
        assert not any(
            isinstance(m, TernaryDQTLinear) for m in moe.router.modules()
        )

    def test_moe_ffn_experts_are_dqt(self):
        """Each expert is an FFN of TernaryDQTLinear3D (DQT) layers."""
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        assert len(moe.experts) == N_EXPERTS
        # 2 ternary linears per expert (fc_in + fc_out)
        assert sum(
            1 for m in moe.experts[0].modules() if isinstance(m, TernaryDQTLinear)
        ) == 2
        assert _ternary_invariant_holds(moe)
        # Per-expert ternary count == 2 * d_model * d_ff
        assert _count_ternary(moe) == N_EXPERTS * 2 * D_MODEL * D_FF

    def test_moe_ffn_invalid_top_k(self):
        """top_k must be in [1, n_experts]."""
        for bad in (0, N_EXPERTS + 1):
            try:
                TernaryDQTMoEFeedForward(
                    D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=bad
                )
            except ValueError:
                pass
            else:  # pragma: no cover - failure branch
                raise AssertionError(f"top_k={bad} should raise ValueError")


# ── MoE FFN: top-K routing ─────────────────────────────────────────


class TestMoeFfnTopK:
    """Exactly top_k experts are active per token."""

    def test_moe_ffn_topk(self):
        """Output equals the weighted sum of exactly top_k experts."""
        torch.manual_seed(0)
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        x = torch.randn(4, 6, D_MODEL)  # 24 tokens
        out, _ = moe(x)

        flat = x.reshape(-1, D_MODEL)
        logits = moe.router(flat)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, indices = torch.topk(probs, TOP_K, dim=-1)
        weights = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-8)

        # Run ALL experts and manually compute the same weighted sum.
        # Gather per-expert outputs on the full token set, then mask.
        all_outs = []
        for e in range(N_EXPERTS):
            # full-batch expert output (B*T, D)
            full = moe.experts[e](flat)
            all_outs.append(full)
        all_outs = torch.stack(all_outs, dim=-1)  # (N, D, E)
        manual = torch.zeros_like(flat)
        for k in range(TOP_K):
            manual += all_outs[
                torch.arange(flat.size(0)), :, indices[:, k]
            ] * weights[:, k].unsqueeze(-1)

        assert torch.allclose(out.reshape(-1, D_MODEL), manual, atol=1e-5)

        # Every token selected exactly top_k DISTINCT experts
        assert indices.shape == (flat.size(0), TOP_K)
        assert (indices.unique(dim=-1).shape == indices.shape)

    def test_moe_ffn_gradients_flow(self):
        """Gradients flow to both the float router and all DQT experts."""
        torch.manual_seed(1)
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        x = torch.randn(4, 4, D_MODEL)
        out, aux_loss = moe(x)
        (out.sum() + aux_loss).backward()

        assert moe.router.weight.grad is not None
        assert torch.isfinite(moe.router.weight.grad).all()
        for expert in moe.experts:
            for m in expert.modules():
                if isinstance(m, TernaryDQTLinear):
                    assert m.weight_float.grad is not None
                    assert torch.isfinite(m.weight_float.grad).all()

    def test_moe_ffn_usage_stats(self):
        """Selection/coverage counters accumulate only in training forwards."""
        torch.manual_seed(2)
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        for _ in range(3):
            moe(torch.randn(16, 4, D_MODEL))
        fracs = moe.selection_fractions()
        assert fracs.shape == (N_EXPERTS,)
        assert torch.allclose(fracs.sum(), torch.tensor(1.0), atol=1e-5)
        cov = moe.coverage_fractions()
        assert torch.allclose(cov.sum(), torch.tensor(float(TOP_K)), atol=1e-4)

        # Eval forwards (no grad) do NOT accumulate
        n_before = moe.n_selections.item()
        with torch.no_grad():
            moe(torch.randn(16, 4, D_MODEL))
        assert moe.n_selections.item() == n_before


# ── MoE FFN: load balancing ────────────────────────────────────────


class TestMoeFfnLoadBalance:
    """Switch-Transformer aux loss is computed and is trainable."""

    def test_moe_ffn_load_balance(self):
        """aux loss is the Switch-Transformer objective and stays >= 1."""
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        x = torch.randn(8, 4, D_MODEL)
        _out, aux = moe(x)
        # L = n_experts * sum_i(f_i * P_i) >= 1 with equality iff uniform
        assert aux.item() >= 1.0 - 1e-3

    def test_moe_ffn_lb_loss_trainable(self):
        """Optimizing only the aux loss drives routing toward balance."""
        torch.manual_seed(3)
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        opt = torch.optim.AdamW(moe.router.parameters(), lr=0.05)
        x = torch.randn(32, 4, D_MODEL)

        first = None
        for _ in range(60):
            opt.zero_grad()
            _out, aux = moe(x)
            if first is None:
                first = aux.item()
            aux.backward()
            opt.step()
        last = aux.item()

        # The router should learn a (near-)uniform policy → aux -> 1.0
        assert last < first, f"aux did not decrease: {first:.4f} -> {last:.4f}"
        assert last < 1.5, f"aux not near uniform: {last:.4f}"

    def test_moe_ffn_balance_report(self):
        """balance_report detects a (near-)uniform router as balanced."""
        moe = TernaryDQTMoEFeedForward(
            D_MODEL, D_FF, n_experts=N_EXPERTS, top_k=TOP_K
        )
        torch.manual_seed(5)
        for _ in range(50):
            moe(torch.randn(32, 4, D_MODEL))
        rep = moe.balance_report()
        assert set(rep) == {"balance_ratio", "min_share", "max_share"}
        assert rep["min_share"] > 0.0
        assert rep["balance_ratio"] < 5.0  # random router is roughly balanced


# ── MoE transformer block ──────────────────────────────────────────


class TestMoeBlockForward:
    """TernaryDQTMoETransformerBlock works end-to-end."""

    def test_moe_block_forward(self):
        """Pre-norm MoE block: shape preserved, residual + aux loss returned."""
        block = TernaryDQTMoETransformerBlock(
            D_MODEL, N_HEADS, D_FF, n_experts=N_EXPERTS, top_k=TOP_K,
            max_seq_len=32,
        )
        x = torch.randn(2, 8, D_MODEL)
        out, aux = block(x)
        assert out.shape == x.shape, f"Got shape {out.shape}"
        assert torch.isfinite(out).all()
        assert torch.isfinite(aux).all()
        assert aux.ndim == 0
        assert _ternary_invariant_holds(block)

    def test_moe_block_gradient_flow(self):
        """Backward through the whole MoE block reaches router + experts."""
        torch.manual_seed(4)
        block = TernaryDQTMoETransformerBlock(
            D_MODEL, N_HEADS, D_FF, n_experts=N_EXPERTS, top_k=TOP_K,
            max_seq_len=16,
        )
        x = torch.randn(2, 8, D_MODEL, requires_grad=True)
        out, aux = block(x)
        (out.sum() + aux).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert block.moe_ffn.router.weight.grad is not None
        for m in _iter_dqt_linear(block):
            assert m.weight_float.grad is not None


# ── Hybrid dense + MoE transformer ─────────────────────────────────


class TestHybridModel:
    """A mini transformer with leading dense blocks and trailing MoE blocks."""

    def _build_hybrid(self, vocab=64, seq_len=32, dense=1, moe=2):
        d_model, heads, d_ff = D_MODEL, N_HEADS, D_FF
        embedding = nn.Embedding(vocab, d_model)
        blocks = nn.ModuleList()
        for _ in range(dense):
            blocks.append(
                TernaryDQTTransformerBlock(
                    d_model, heads, d_ff, max_seq_len=seq_len
                )
            )
        for _ in range(moe):
            blocks.append(
                TernaryDQTMoETransformerBlock(
                    d_model, heads, d_ff, n_experts=N_EXPERTS, top_k=TOP_K,
                    max_seq_len=seq_len,
                )
            )
        final_norm = nn.LayerNorm(d_model)
        lm_head = nn.Linear(d_model, vocab)

        class _Hybrid(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = embedding
                self.blocks = blocks
                self.final_norm = final_norm
                self.lm_head = lm_head

            def forward(self, tokens):
                x = self.embedding(tokens)
                aux = torch.tensor(0.0, requires_grad=False)
                for block in self.blocks:
                    if isinstance(block, TernaryDQTMoETransformerBlock):
                        x, b_aux = block(x)
                        aux = aux + b_aux
                    else:
                        x = block(x)
                x = self.final_norm(x)
                return self.lm_head(x), aux

        return _Hybrid()

    def test_hybrid_model(self):
        """Dense + MoE blocks together produce logits and a summed aux loss."""
        model = self._build_hybrid()
        tokens = torch.randint(0, 64, (2, 8))
        logits, aux = model(tokens)
        assert logits.shape == (2, 8, 64)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux).all()
        assert aux.ndim == 0
        assert _ternary_invariant_holds(model)

    def test_hybrid_model_backward(self):
        """Training loss (LM + lb_coef*aux) backprops through both block kinds."""
        torch.manual_seed(6)
        model = self._build_hybrid()
        tokens = torch.randint(0, 64, (2, 8))
        targets = torch.randint(0, 64, (2, 8))
        logits, aux = model(tokens)
        lm = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 64), targets.reshape(-1)
        )
        total = lm + 0.1 * aux
        total.backward()

        # Every DQT layer got a gradient (dense + MoE experts)
        for m in _iter_dqt_linear(model):
            assert m.weight_float.grad is not None
        # Every MoE router got a gradient
        for block in model.blocks:
            if isinstance(block, TernaryDQTMoETransformerBlock):
                assert block.moe_ffn.router.weight.grad is not None

    def test_hybrid_model_learns(self):
        """The hybrid model can overfit one learnable batch (sanity)."""
        vocab, seq_len = 64, 16
        model = self._build_hybrid(vocab=vocab, seq_len=seq_len, dense=1, moe=1)
        torch.manual_seed(7)
        tokens = torch.randint(0, vocab, (4, seq_len))
        targets = torch.randint(0, vocab, (4, seq_len))
        opt = torch.optim.AdamW(model.parameters(), lr=0.02)

        def step():
            opt.zero_grad()
            logits, aux = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab), targets.reshape(-1)
            ) + 0.1 * aux
            loss.backward()
            opt.step()
            # DQT: stochastic rounding after every optimizer step
            for m in _iter_dqt_linear(model):
                m.apply_stochastic_rounding()
            return float(loss.item())

        initial = step()
        final = initial
        for _ in range(100):
            final = step()

        assert _ternary_invariant_holds(model)
        assert final < initial * 0.9, (
            f"Hybrid model did not learn: {initial:.4f} -> {final:.4f}"
        )
