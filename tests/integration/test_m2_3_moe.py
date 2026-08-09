"""Integration tests for Milestone M2.3 — MoE DQT Transformer on TinyStories.

Verifies the end-to-end hybrid dense+MoE DQT transformer pipeline: the
model factory + M2_3_CONFIG parameter budget (~265M ternary, ~152M active,
303.7M total — revised 2026-08-06 from 4+8 to 6+6 MoE layers to fit the
8 GB card reliably), single-batch overfitting (sanity that DQT attention +
MoE experts + router + aux loss can learn), a short training loop where
loss decreases, the two-param-group optimizer (experts at lr, routers at
0.1x lr), the lb_coef-weighted Switch-Transformer aux loss, expert
utilization balancing, checkpoint pruning (keep latest N + best.pt), and
pause/resume.

All tests run on CPU with synthetic (learnable) data so they stay fast and
never require downloading TinyStories.
"""

from __future__ import annotations

import math
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.examples.run_m2_3_dqt_moe import (
    ANNEAL_FRACTION,
    apply_dqt_rounding,
    collect_expert_utilization,
    compute_perplexity,
    evaluate_perplexity,
    find_latest_checkpoint,
    load_checkpoint,
    should_use_stochastic,
    train_dqt_transformer,
)
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_transformer import TernaryDQTMoETransformerBlock
from ph_neuro.models.dqt_transformer import (
    M2_3_CONFIG,
    count_parameters,
    count_ternary_weights,
    dqt_gpt2_moe,
)
from ph_neuro.training.tinystories import make_synthetic_lm_loader

DEVICE = torch.device("cpu")


def _build_small_model(vocab: int = 64, seq_len: int = 32) -> nn.Module:
    """A tiny hybrid dense+MoE DQT transformer for fast integration tests."""
    return dqt_gpt2_moe(
        vocab_size=vocab,
        d_model=64,
        n_heads=4,
        d_ff=256,
        dense_layers=1,
        moe_layers=2,
        n_experts=4,
        top_k=2,
        max_seq_len=seq_len + 1,
        device=DEVICE,
    )


def _check_ternary_invariants(model: nn.Module) -> bool:
    """All DQT weights are int8 ternary in {-1, 0, +1}."""
    for m in model.modules():
        if isinstance(m, TernaryDQTLinear):
            w = m.weight_ternary
            if w.dtype != torch.int8:
                return False
            if not bool(torch.all((w >= -1) & (w <= 1)).item()):
                return False
    return True


def _count_moe_blocks(model: nn.Module) -> int:
    return sum(
        1
        for b in model.blocks
        if isinstance(b, TernaryDQTMoETransformerBlock)
    )


def _make_optimizers(model: nn.Module, lr: float = 0.03):
    """Build the M2.3 two-group AdamW + embed-SGD (mirrors the runner)."""
    router_params = [p for n, p in model.named_parameters() if "router" in n]
    main_params = [
        p
        for n, p in model.named_parameters()
        if "router" not in n and n != "token_embedding.weight"
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": lr},
            {"params": router_params, "lr": lr * 0.1},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    emb_optimizer = torch.optim.SGD([model.token_embedding.weight], lr=lr)
    return optimizer, emb_optimizer


# ── 1. Config + factory ────────────────────────────────────────────


class TestM2_3Config:
    """M2_3_CONFIG builds the right parameter budget."""

    def test_m2_3_config_budget(self):
        """~265M ternary total, ~152M active (57%), 300-400M total incl. embed."""
        cfg = M2_3_CONFIG
        assert cfg["dense_layers"] == 6
        assert cfg["moe_layers"] == 6
        assert cfg["n_experts"] == 6
        assert cfg["top_k"] == 2
        assert cfg["d_model"] == 768
        model = dqt_gpt2_moe(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
            dense_layers=cfg["dense_layers"], moe_layers=cfg["moe_layers"],
            n_experts=cfg["n_experts"], top_k=cfg["top_k"],
            max_seq_len=cfg["max_seq_len"], device=DEVICE,
        )
        n_ternary = count_ternary_weights(model)
        n_total = count_parameters(model)
        # Total (incl. float embedding) in the 300-400M milestone envelope
        assert 300e6 <= n_total <= 400e6, f"total {n_total:,} outside 300-400M"
        # Active params per token ~152M (150-200M)
        moe_expert_ternary = sum(
            block.moe_ffn.count_parameters()["experts"]
            for block in model.blocks
            if isinstance(block, TernaryDQTMoETransformerBlock)
        )
        active_ternary = (n_ternary - moe_expert_ternary) + moe_expert_ternary * (
            cfg["top_k"] / cfg["n_experts"]
        )
        n_active = int(round((n_total - n_ternary) + active_ternary))
        assert 150e6 <= n_active <= 200e6, f"active {n_active:,} outside 150-200M"
        assert 0.45 <= n_active / n_total <= 0.65, (
            f"active fraction {n_active/n_total:.2f} not ~52%"
        )
        assert _count_moe_blocks(model) == 6
        assert _check_ternary_invariants(model)

    def test_hybrid_has_dense_and_moe_blocks(self):
        """The factory builds dense blocks first, then MoE blocks."""
        model = _build_small_model()
        kinds = [
            "moe" if isinstance(b, TernaryDQTMoETransformerBlock) else "dense"
            for b in model.blocks
        ]
        assert kinds == ["dense", "moe", "moe"], f"unexpected block layout {kinds}"

    def test_forward_returns_logits_and_aux(self):
        """Forward returns (logits, aux_loss) — the M2.3 contract."""
        model = _build_small_model()
        tokens = torch.randint(0, 64, (2, 8))
        logits, aux = model(tokens)
        assert logits.shape == (2, 8, 64)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(aux).all()
        assert aux.ndim == 0


# ── 2. Optimizer (router 0.1x LR) ─────────────────────────────────


class TestMoeOptimizer:
    """The runner's two-group optimizer gives routers a 0.1x LR."""

    def test_router_param_group(self):
        """Routers are in their own AdamW group at 0.1x the expert LR."""
        model = _build_small_model()
        optimizer, _emb = _make_optimizers(model, lr=0.03)
        groups = optimizer.param_groups
        assert len(groups) == 2
        assert abs(groups[0]["lr"] - 0.03) < 1e-9
        assert abs(groups[1]["lr"] - 0.003) < 1e-9  # 0.1x
        # Group 1 holds exactly the router weights
        router_names = {n for n, _p in model.named_parameters() if "router" in n}
        assert len(groups[1]["params"]) == len(router_names)

    def test_router_grad_flows(self):
        """Backprop reaches both the router and the DQT experts."""
        torch.manual_seed(0)
        model = _build_small_model()
        tokens = torch.randint(0, 64, (2, 8))
        targets = torch.randint(0, 64, (2, 8))
        logits, aux = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1)) + 0.1 * aux
        loss.backward()
        for n, p in model.named_parameters():
            if "router" in n:
                assert p.grad is not None, f"no grad for {n}"
                assert torch.isfinite(p.grad).all()


# ── 3. Overfit a single batch ──────────────────────────────────────


class TestMoeOverfitBatch:
    """The hybrid MoE DQT model can overfit one learnable batch (sanity)."""

    def test_hybrid_overfit_batch(self):
        """Loss on one learnable batch drops well below its start."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        optimizer, emb_optimizer = _make_optimizers(model, lr=0.03)

        torch.manual_seed(7)
        input_ids = torch.randint(0, vocab, (8, seq_len))
        targets = torch.randint(0, vocab, (8, seq_len))

        def step_loss() -> float:
            optimizer.zero_grad()
            emb_optimizer.zero_grad()
            logits, aux = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab), targets.reshape(-1)
            ) + 0.1 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            emb_optimizer.step()
            apply_dqt_rounding(model, use_stochastic=True)
            return float(loss.item())

        initial = step_loss()
        final = initial
        for _ in range(200):
            final = step_loss()

        assert _check_ternary_invariants(model)
        assert final < initial * 0.5, (
            f"Hybrid loss did not drop: {initial:.3f} -> {final:.3f}"
        )
        assert final < math.log(vocab), (
            f"Not below uniform baseline ln({vocab})={math.log(vocab):.3f}: {final:.3f}"
        )


# ── 4. Short training loop ─────────────────────────────────────────


class TestMoeTrainingLoop:
    """Short training loop: loss decreases, utilization is balanced."""

    def test_hybrid_training_loop(self):
        """60 steps: windowed loss decreases and experts stay utilized."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        optimizer, emb_optimizer = _make_optimizers(model, lr=0.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=10, seed=3
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=4
        )
        results = train_dqt_transformer(
            model, train_loader, val_loader, optimizer, scheduler, DEVICE,
            epochs=6, max_steps=60, anneal_fraction=1.0,
            lb_coef=0.1, emb_optimizer=emb_optimizer,
            record_steps=True, verbose=False,
        )
        step_losses = results["step_loss_history"]
        assert len(step_losses) == 60
        window = 10
        first = sum(step_losses[:window]) / window
        last = sum(step_losses[-window:]) / window
        assert last < first, (
            f"Windowed step loss did not decrease: first10={first:.4f} last10={last:.4f}"
        )
        assert results["steps_trained"] == 60
        assert math.isfinite(results["final_val_ppl"])
        assert _check_ternary_invariants(model)

        # Expert utilization: no dead experts after training
        util = collect_expert_utilization(model)
        assert len(util) == 2  # 2 MoE blocks
        for layer in util:
            assert layer["min_share"] > 0.05, (
                f"layer {layer['layer']} has a (near-)dead expert: "
                f"{layer['selection_fractions']}"
            )
            assert layer["balance_ratio"] < 20.0

    def test_lb_coef_zero_disables_aux(self):
        """lb_coef=0.0 trains without the aux loss (no crash)."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        optimizer, emb_optimizer = _make_optimizers(model, lr=0.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=6, seed=9
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=2, seed=10
        )
        results = train_dqt_transformer(
            model, train_loader, val_loader, optimizer, scheduler, DEVICE,
            epochs=2, max_steps=10, anneal_fraction=1.0,
            lb_coef=0.0, emb_optimizer=emb_optimizer,
            record_steps=True, verbose=False,
        )
        assert len(results["step_loss_history"]) == 10
        assert math.isfinite(results["final_val_ppl"])


# ── 5. Evaluation + perplexity ─────────────────────────────────────


class TestMoePerplexity:
    """Perplexity with the (logits, aux) forward contract."""

    def test_evaluate_perplexity(self):
        """evaluate_perplexity() handles the MoE tuple forward."""
        vocab, seq_len = 64, 32
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=5
        )
        ppl = evaluate_perplexity(model, loader, DEVICE)
        assert math.isfinite(ppl)
        assert ppl > 1.0
        assert ppl < math.exp(vocab)

    def test_annealing_schedule(self):
        """should_use_stochastic() switches at the anneal fraction (unchanged)."""
        total = 100
        anneal_start = int(total * ANNEAL_FRACTION)
        assert should_use_stochastic(anneal_start - 1, total, ANNEAL_FRACTION) is True
        assert should_use_stochastic(anneal_start, total, ANNEAL_FRACTION) is False


# ── 6. Pause/resume ────────────────────────────────────────────────


class TestMoeResume:
    """Checkpointed MoE training can be paused and resumed."""

    def _make_runner(self, vocab=64, seq_len=32):
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        optimizer, emb_optimizer = _make_optimizers(model, lr=0.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=10, seed=3
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=4
        )
        return model, train_loader, val_loader, optimizer, scheduler, emb_optimizer

    def test_checkpoint_and_find_latest(self):
        """Training writes checkpoints; find_latest picks the newest."""
        model, tl, vl, opt, sch, emb_opt = self._make_runner()
        with tempfile.TemporaryDirectory() as ckpt_dir:
            train_dqt_transformer(
                model, tl, vl, opt, sch, DEVICE,
                epochs=2, max_steps=4, anneal_fraction=1.0,
                lb_coef=0.1, emb_optimizer=emb_opt,
                checkpoint_every=2, checkpoint_dir=ckpt_dir, verbose=False,
            )
            latest = find_latest_checkpoint(ckpt_dir)
            assert latest is not None
            assert "ckpt_step4.pt" in latest, f"Expected step-4 checkpoint, got {latest}"

    def test_resume_continues_training(self):
        """A fresh run resumed from a checkpoint continues the step count."""
        with tempfile.TemporaryDirectory() as ckpt_dir:
            m1, tl, vl, o1, s1, e1 = self._make_runner()
            r1 = train_dqt_transformer(
                m1, tl, vl, o1, s1, DEVICE,
                epochs=2, max_steps=4, anneal_fraction=1.0,
                lb_coef=0.1, emb_optimizer=e1,
                checkpoint_every=2, checkpoint_dir=ckpt_dir, verbose=False,
            )
            assert r1["steps_trained"] == 4

            m2, tl, vl, o2, s2, e2 = self._make_runner()
            loaded = load_checkpoint(
                "auto", ckpt_dir, m2, o2, s2, DEVICE, emb_optimizer=e2
            )
            assert loaded["step"] == 4
            r2 = train_dqt_transformer(
                m2, tl, vl, o2, s2, DEVICE,
                epochs=3, max_steps=8, anneal_fraction=1.0,
                lb_coef=0.1, emb_optimizer=e2,
                checkpoint_every=2, checkpoint_dir=ckpt_dir,
                start_step=loaded["step"],
                start_epoch=loaded["epoch"] + 1,
                best_val_ppl=loaded["best_val_ppl"],
                best_step=loaded["best_step"],
                verbose=False,
            )
            assert r2["steps_trained"] == 8
            assert _check_ternary_invariants(m2)


# ── 7. Checkpoint pruning (disk-bounding) ──────────────────────────


class TestCheckpointPruning:
    """Keep the latest N periodic checkpoints + best.pt (bounded disk)."""

    def _make_runner(self, vocab=64, seq_len=32):
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        optimizer, emb_optimizer = _make_optimizers(model, lr=0.03)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=10, seed=3
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=4, n_batches=4, seed=4
        )
        return model, train_loader, val_loader, optimizer, scheduler, emb_optimizer

    def test_prunes_old_checkpoints(self):
        """After saving, only the newest keep_last periodic ckpts remain."""
        model, tl, vl, opt, sch, emb_opt = self._make_runner()
        with tempfile.TemporaryDirectory() as ckpt_dir:
            train_dqt_transformer(
                model, tl, vl, opt, sch, DEVICE,
                epochs=2, max_steps=8, anneal_fraction=1.0,
                lb_coef=0.1, emb_optimizer=emb_opt,
                checkpoint_every=2, checkpoint_dir=ckpt_dir,
                keep_last_checkpoints=2, verbose=False,
            )
            # Steps 2,4,6,8 are written; pruning keeps only the newest 2.
            ckpts = sorted(
                int(p.replace("ckpt_step", "").replace(".pt", ""))
                for p in os.listdir(ckpt_dir)
                if p.startswith("ckpt_step")
            )
            assert ckpts == [6, 8], f"expected [6, 8] after pruning, got {ckpts}"

    def test_best_checkpoint_saved(self):
        """Val improvements write a persistent best.pt (never pruned)."""
        model, tl, vl, opt, sch, emb_opt = self._make_runner()
        with tempfile.TemporaryDirectory() as ckpt_dir:
            train_dqt_transformer(
                model, tl, vl, opt, sch, DEVICE,
                epochs=2, max_steps=8, anneal_fraction=1.0,
                lb_coef=0.1, emb_optimizer=emb_opt,
                val_every=2, checkpoint_every=2, checkpoint_dir=ckpt_dir,
                keep_last_checkpoints=2, verbose=False,
            )
            assert os.path.exists(os.path.join(ckpt_dir, "best.pt")), "best.pt missing"
            # Periodic pruning never touches best.pt
            assert os.path.exists(os.path.join(ckpt_dir, "best.pt"))

    def test_find_latest_skips_corrupted(self):
        """--resume auto falls back to the highest VALID checkpoint."""
        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Valid checkpoint at step 10
            torch.save({"step": 10}, os.path.join(ckpt_dir, "ckpt_step10.pt"))
            # Corrupt (truncated) checkpoint at step 20 — must be skipped
            with open(os.path.join(ckpt_dir, "ckpt_step20.pt"), "wb") as f:
                f.write(b"\x00\x01\x02\x03")  # garbage, not a valid zip
            latest = find_latest_checkpoint(ckpt_dir)
            assert latest is not None
            assert latest.endswith("ckpt_step10.pt"), (
                f"should fall back to step10, got {latest}"
            )
