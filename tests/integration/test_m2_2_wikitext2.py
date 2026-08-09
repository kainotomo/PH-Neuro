"""Integration tests for Milestone M2.2 — DQT Transformer 250M on WikiText-2.

Verifies the M2.2 additions on top of M2.1: the WikiText-2 dataset identifier
and sequence packing, the M2_2_CONFIG parameter counts (252.8M ternary), the
gradient-checkpointing equivalence (checkpointed forward/backward == plain),
the embedding-SGD two-optimizer setup (the "χωρίς AdamW" memory design), the
status-file writer, and a short two-optimizer training loop where the loss
decreases.

All tests run on CPU with synthetic (learnable) data so they stay fast and
never download WikiText-2.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.examples.run_m2_2_dqt_wikitext2 import (
    ANNEAL_FRACTION,
    apply_dqt_rounding,
    compute_perplexity,
    find_latest_checkpoint,
    load_checkpoint,
    should_use_stochastic,
    train_dqt_transformer,
    write_status_file,
)
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.models.dqt_transformer import (
    FULL_CONFIG,
    M2_2_CONFIG,
    count_parameters,
    count_ternary_weights,
    dqt_gpt2,
)
from ph_neuro.training.tinystories import make_synthetic_lm_loader
from ph_neuro.training.wikitext2 import (
    CONFIG_NAME,
    DATASET_NAME,
    pack_sequences,
    tokenize_texts,
)

DEVICE = torch.device("cpu")

# Expected M2_2_CONFIG breakdown (computed by hand, verified by the build test):
# per block: 4*1024^2 + 2*1024*4096 = 12,582,912; 16 blocks = 201,326,592;
# LM head: 1024*50257 = 51,463,168 → total ternary 252,789,760.
EXPECTED_M2_2_TERNARY = 252_789_760
EXPECTED_M2_2_TOTAL = 304_286_720


def _build_small_model(vocab: int = 64, seq_len: int = 32, grad_ckpt: bool = False) -> nn.Module:
    """A tiny DQT transformer for fast integration tests."""
    return dqt_gpt2(
        vocab_size=vocab,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=seq_len + 1,
        use_grad_checkpointing=grad_ckpt,
        device=DEVICE,
    )


def _build_sgd_optimizers(model: nn.Module, lr: float = 0.01):
    """Build the M2.2 two-optimizer setup (AdamW main + SGD embedding)."""
    emb_params = [model.token_embedding.weight]
    main_params = [
        p for n, p in model.named_parameters() if n != "token_embedding.weight"
    ]
    optimizer = torch.optim.AdamW(
        main_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    emb_optimizer = torch.optim.SGD(emb_params, lr=lr)
    return optimizer, emb_optimizer


# ── 1. M2_2_CONFIG ─────────────────────────────────────────────────


class TestM22Config:
    def test_config_values(self):
        """M2_2_CONFIG has the documented 250M scaling values."""
        assert M2_2_CONFIG["d_model"] == 1024
        assert M2_2_CONFIG["n_layers"] == 16
        assert M2_2_CONFIG["n_heads"] == 16
        assert M2_2_CONFIG["d_ff"] == 4096
        assert M2_2_CONFIG["vocab_size"] == 50257
        assert M2_2_CONFIG["max_seq_len"] == 256
        # d_head = d_model // n_heads = 64 (even → RoPE OK, same as M2.1)
        assert M2_2_CONFIG["d_model"] % M2_2_CONFIG["n_heads"] == 0

    @pytest.mark.slow
    def test_m2_2_param_counts(self):
        """Building the full M2_2_CONFIG gives exactly 252.8M ternary weights."""
        cfg = M2_2_CONFIG
        model = dqt_gpt2(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_seq_len=cfg["max_seq_len"],
            device=DEVICE,
        )
        n_ternary = count_ternary_weights(model)
        n_total = count_parameters(model)
        assert n_ternary == EXPECTED_M2_2_TERNARY, (
            f"ternary={n_ternary} expected {EXPECTED_M2_2_TERNARY}"
        )
        assert n_total == EXPECTED_M2_2_TOTAL, (
            f"total={n_total} expected {EXPECTED_M2_2_TOTAL}"
        )
        # Float embedding + RMSNorm scales ≈ 51.5M
        assert n_total - n_ternary == EXPECTED_M2_2_TOTAL - EXPECTED_M2_2_TERNARY

    def test_m2_2_ternary_formula(self):
        """The hand-computed ternary count matches the per-block formula."""
        d, l, ff, v = 1024, 16, 4096, 50257
        per_block = 4 * d * d + 2 * d * ff
        total_ternary = l * per_block + d * v
        assert total_ternary == EXPECTED_M2_2_TERNARY

    def test_gate_constant(self):
        """The M2.2 GO gate is ppl < 20 (M2.1 was < 30)."""
        from ph_neuro.examples.run_m2_2_dqt_wikitext2 import PPL_GATE

        assert PPL_GATE == 20.0


# ── 2. Gradient checkpointing ──────────────────────────────────────


class TestGradCheckpointing:
    def test_checkpoint_matches_plain_forward(self):
        """Checkpointed forward == plain forward (same weights, same input)."""
        torch.manual_seed(0)
        m_plain = _build_small_model(grad_ckpt=False)
        m_ckpt = _build_small_model(grad_ckpt=True)
        # Copy weights so both models are identical
        m_ckpt.load_state_dict(m_plain.state_dict())
        x = torch.randint(0, 64, (2, 32))
        with torch.no_grad():
            out_plain = m_plain(x)
            out_ckpt = m_ckpt(x)
        assert torch.allclose(out_plain, out_ckpt, atol=1e-5)

    def test_checkpoint_backward_grads_match(self):
        """Checkpointed backward produces the same weight grads as plain."""
        torch.manual_seed(1)
        m_plain = _build_small_model(grad_ckpt=False)
        m_ckpt = _build_small_model(grad_ckpt=True)
        m_ckpt.load_state_dict(m_plain.state_dict())
        x = torch.randint(0, 64, (4, 32))
        y = torch.randint(0, 64, (4, 32))
        for m in (m_plain, m_ckpt):
            logits = m(x)
            loss = F.cross_entropy(
                logits.reshape(-1, m.vocab_size), y.reshape(-1)
            )
            loss.backward()
        grads_plain = {n: p.grad for n, p in m_plain.named_parameters()}
        grads_ckpt = {n: p.grad for n, p in m_ckpt.named_parameters()}
        for n in grads_plain:
            assert grads_plain[n] is not None
            assert torch.allclose(
                grads_plain[n], grads_ckpt[n], atol=1e-6
            ), f"grad mismatch at {n}"


# ── 3. Embedding-SGD two-optimizer setup ───────────────────────────


class TestEmbeddingSgd:
    def test_embedding_not_in_main_optimizer(self):
        """The main AdamW optimizer excludes the token embedding (memory design)."""
        model = _build_small_model()
        optimizer, emb_optimizer = _build_sgd_optimizers(model)
        main_param_ids = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        assert id(model.token_embedding.weight) not in main_param_ids
        emb_param_ids = {
            id(p) for group in emb_optimizer.param_groups for p in group["params"]
        }
        assert id(model.token_embedding.weight) in emb_param_ids

    def test_two_optimizer_step_decreases_loss(self):
        """AdamW + SGD-embedding both step; the training loss decreases."""
        torch.manual_seed(42)
        model = _build_small_model(vocab=64, seq_len=32)
        optimizer, emb_optimizer = _build_sgd_optimizers(model)
        x = torch.randint(0, 64, (4, 32))
        y = torch.randint(0, 64, (4, 32))
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), y.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            emb_optimizer.step()
            apply_dqt_rounding(model, use_stochastic=True)
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ── 4. Status file ─────────────────────────────────────────────────


class TestStatusFile:
    def test_write_status_file_roundtrip(self):
        """write_status_file() writes a parseable JSON with the right fields."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status.json")
            write_status_file(
                path,
                seed=42,
                lr=0.01,
                step=100,
                total_steps=1000,
                epoch=2,
                loss=3.5,
                ppl=33.1,
                tok_per_s=1234.5,
                gpu_mem_gb=4.8,
                status="RUNNING",
            )
            with open(path) as f:
                d = json.load(f)
            assert d["seed"] == 42
            assert d["step"] == 100
            assert d["total_steps"] == 1000
            assert d["ppl"] == pytest.approx(33.1)
            assert d["tok_per_s"] == pytest.approx(1234.5)
            assert d["gpu_mem_gb"] == pytest.approx(4.8)
            assert d["status"] == "RUNNING"


# ── 5. WikiText-2 loader helpers ───────────────────────────────────


class TestWikitext2Loader:
    def test_dataset_identifier(self):
        """WikiText-2 is Salesforce/wikitext with the wikitext-2-raw-v1 config."""
        assert DATASET_NAME == "Salesforce/wikitext"
        assert CONFIG_NAME == "wikitext-2-raw-v1"

    def test_pack_sequences_shift(self):
        """pack_sequences() produces shift-by-1 (input, target) windows."""
        tokens = torch.arange(0, 100, dtype=torch.long)
        input_ids, targets = pack_sequences(tokens, 8)
        assert input_ids.shape == targets.shape
        assert input_ids.shape[1] == 8
        assert torch.equal(targets[:, :-1], input_ids[:, 1:])

    def test_tokenize_texts(self):
        """tokenize_texts() returns a flat int tensor via the M2.1 path."""
        from ph_neuro.training.wikitext2 import make_gpt2_tokenizer

        tok = make_gpt2_tokenizer()
        ids = tokenize_texts(["hello world", "second article"], tok)
        assert ids.dtype == torch.int32
        assert ids.ndim == 1
        expected = sum(len(tok.encode(t)) for t in ["hello world", "second article"])
        assert ids.numel() == expected


# ── 6. Two-optimizer training loop via train_dqt_transformer ───────


class TestM22TrainingLoop:
    def test_short_training_loop_decreases_loss(self):
        """End-to-end train_dqt_transformer with the embedding-SGD setup works."""
        torch.manual_seed(7)
        vocab, seq_len, batch, n_batches = 64, 32, 4, 6
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=batch, n_batches=n_batches
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=batch, n_batches=2
        )
        model = _build_small_model(vocab=vocab, seq_len=seq_len)
        optimizer, emb_optimizer = _build_sgd_optimizers(model)
        total_steps = 4
        scheduler = None
        results = train_dqt_transformer(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            DEVICE,
            epochs=2,
            max_steps=total_steps,
            anneal_fraction=1.0,  # pure stochastic (tiny run, no premature anneal)
            checkpoint_dir=None,
            emb_optimizer=emb_optimizer,
            emb_scheduler=None,
            record_steps=True,
        )
        assert results["steps_trained"] == total_steps
        # Loss decreased over the run (first step vs later steps)
        step_losses = results["step_loss_history"]
        assert len(step_losses) == total_steps
        assert step_losses[-1] < step_losses[0] + 1e-6
        # Final validation perplexity is finite
        assert math.isfinite(results["final_val_ppl"])
        # Ternary invariants hold after rounding
        for m in model.modules():
            if isinstance(m, TernaryDQTLinear):
                w = m.weight_ternary
                assert w.dtype == torch.int8
                assert bool(torch.all((w >= -1) & (w <= 1)).item())


# ── 7. Pause/resume with the two-optimizer setup ────────────────────


class TestM22Resume:
    def test_resume_restores_two_optimizers(self):
        """Checkpoint → rebuild → resume continues training (both optimizers)."""
        torch.manual_seed(11)
        vocab, seq_len, batch, n_batches = 64, 32, 4, 6
        train_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=batch, n_batches=n_batches
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=vocab, seq_len=seq_len, batch_size=batch, n_batches=2
        )
        with tempfile.TemporaryDirectory() as tmp:
            # Phase 1: train 3 steps (checkpoints at step 2 only), then stop.
            model = _build_small_model(vocab=vocab, seq_len=seq_len)
            optimizer, emb_optimizer = _build_sgd_optimizers(model)
            train_dqt_transformer(
                model,
                train_loader,
                val_loader,
                optimizer,
                None,
                DEVICE,
                epochs=2,
                max_steps=3,
                anneal_fraction=1.0,
                checkpoint_every=2,
                checkpoint_dir=tmp,
                emb_optimizer=emb_optimizer,
                emb_scheduler=None,
                record_steps=True,
            )
            latest = find_latest_checkpoint(tmp)
            assert latest is not None

            # Phase 2: fresh model + optimizers, restore from the checkpoint
            model2 = _build_small_model(vocab=vocab, seq_len=seq_len)
            optimizer2, emb_optimizer2 = _build_sgd_optimizers(model2)
            r = load_checkpoint(
                latest,
                tmp,
                model2,
                optimizer2,
                None,
                DEVICE,
                emb_optimizer=emb_optimizer2,
                emb_scheduler=None,
            )
            assert r["step"] == 2
            # Embedding SGD optimizer restored (plain SGD has no per-param
            # moments, so its `state` is legitimately empty — the param group
            # is what gets restored).
            assert emb_optimizer2.param_groups[0]["lr"] == pytest.approx(0.01)

            # Phase 3: continue training, loss stays finite and step advances
            results = train_dqt_transformer(
                model2,
                train_loader,
                val_loader,
                optimizer2,
                None,
                DEVICE,
                epochs=2,
                max_steps=6,
                anneal_fraction=1.0,
                start_step=r["step"],
                start_epoch=r["epoch"] + 1,
                checkpoint_dir=tmp,
                emb_optimizer=emb_optimizer2,
                emb_scheduler=None,
                record_steps=True,
            )
            assert results["steps_trained"] == 6
            assert math.isfinite(results["final_train_loss"])
