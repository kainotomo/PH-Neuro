#!/usr/bin/env python3
"""Milestone M2.1 — DQT Transformer on TinyStories (GO/NO-GO perplexity <30).

First demonstration of Direct Quantized Training (DQT) on a Transformer
language model. The model is a GPT-2-style decoder-only transformer whose
Q/K/V/O and FFN projections (plus the LM head) use ternary int8 DQT weights
(:class:`TernaryDQTLinear` + stochastic rounding). This is the language
counterpart of the M1.1/M1.2 vision validation, and the CRITICAL GO/NO-GO
for the whole project: if a DQT transformer trains stably on TinyStories
and reaches mean validation perplexity < 30 (3 seeds), we GO to MoE scaling
(M2.3); otherwise the plan pivots.

The critical DQT mechanic (validated in M1.1): after EVERY
``optimizer.step()`` we discretize the float accumulation buffer into int8
ternary weights. For the first ``ANNEAL_FRACTION`` (80%) of steps this uses
stochastic rounding (exploration); the final 20% anneals to deterministic
``sign()`` so the ternary weights stop jittering and the network settles
into a clean fine-tuning regime (this removed the M1.1 flip noise).

Usage::

    python -m ph_neuro.examples.run_m2_1_dqt_transformer \\
        --d-model 768 --n-layers 9 --n-heads 12 --d-ff 3072 \\
        --lr 0.01 --epochs 3 --batch-size 8 --seq-len 256 --seed 42

    # Smoke test (no TinyStories download — synthetic learnable corpus):
    python -m ph_neuro.examples.run_m2_1_dqt_transformer --smoke --synthetic

Output:
    JSON file: ``{output_dir}/results_m2_1_dqt_transformer_lr{lr}_seed{seed}.json``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.models.dqt_transformer import (
    FULL_CONFIG,
    SMOKE_CONFIG,
    build_config,
    count_parameters,
    count_ternary_weights,
    dqt_gpt2,
)
from ph_neuro.training.tinystories import (
    get_tinystories_data,
    make_synthetic_lm_loader,
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")

DQT_LAYERS = (TernaryDQTLinear,)

# Fraction of training steps with stochastic rounding before annealing to
# deterministic sign (validated in M1.1-RETRY). The final
# (1 - ANNEAL_FRACTION) of steps run in a clean deterministic tail.
ANNEAL_FRACTION = 0.80

# GO/NO-GO gate: mean validation perplexity across 3 seeds must be < 30.
PPL_GATE = 30.0


# ── DQT helpers ────────────────────────────────────────────────────


def is_dqt_module(module: nn.Module) -> bool:
    """Whether a module is a DQT weight-bearing layer."""
    return isinstance(module, DQT_LAYERS)


@torch.no_grad()
def apply_dqt_rounding(model: nn.Module, use_stochastic: bool = True) -> float:
    """Apply DQT rounding to every DQT linear layer.

    Args:
        model: The DQT transformer.
        use_stochastic: If True, use ``stochastic_round()``. If False, use
            deterministic ``sign()`` (annealing / fine-tuning phase).

    Returns:
        Mean flip rate across all DQT layers.
    """
    flips: list[float] = []
    for module in model.modules():
        if is_dqt_module(module):
            if use_stochastic:
                flips.append(module.apply_stochastic_rounding()["flip_rate"])
            else:
                flips.append(module.apply_deterministic_rounding()["flip_rate"])
    return sum(flips) / max(len(flips), 1) if flips else 0.0


def should_use_stochastic(
    step: int, total_steps: int, anneal_fraction: float = ANNEAL_FRACTION
) -> bool:
    """Whether step ``step`` uses stochastic (vs deterministic) rounding."""
    return step < int(total_steps * anneal_fraction)


# ── Evaluation ─────────────────────────────────────────────────────


def compute_perplexity(loss: float) -> float:
    """Perplexity from a mean cross-entropy loss: ``exp(loss)``."""
    return float(math.exp(loss))


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, val_loader: DataLoader, device: torch.device
) -> float:
    """Mean validation cross-entropy loss, converted to perplexity.

    Args:
        model: The DQT transformer.
        val_loader: Validation loader yielding ``(input_ids, targets)``.
        device: Torch device.

    Returns:
        Validation perplexity (``exp(mean loss)``).
    """
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    for input_ids, targets in val_loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), targets.reshape(-1)
        )
        total_loss += loss.item() * targets.numel()
        n_tokens += targets.numel()
    if n_tokens == 0:
        return float("inf")
    return compute_perplexity(total_loss / max(n_tokens, 1))


# ── Training loop ──────────────────────────────────────────────────


def train_dqt_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epochs: int,
    max_steps: int | None = None,
    anneal_fraction: float = ANNEAL_FRACTION,
    grad_clip: float = 1.0,
    val_every: int | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | None = None,
    record_steps: bool = False,
    verbose: bool = True,
) -> dict:
    """Train a DQT transformer for language modeling.

    Key DQT difference: after each ``optimizer.step()`` we call
    ``apply_dqt_rounding()`` on every DQT layer. For the first
    ``int(total_steps * anneal_fraction)`` steps this is stochastic
    rounding (exploration); afterwards it switches to deterministic
    ``sign()`` (clean fine-tuning tail — the M1.1-RETRY fix).

    Args:
        model: The DQT transformer.
        train_loader: Training loader yielding ``(input_ids, targets)``.
        val_loader: Validation loader.
        optimizer: AdamW optimizer (tracks the DQT float buffers).
        scheduler: LR scheduler (stepped once per step, cosine+warmup).
        device: Torch device.
        epochs: Number of epochs.
        max_steps: If set, cap total training steps (tests/smoke).
        anneal_fraction: Fraction of steps with stochastic rounding.
        grad_clip: Max gradient norm (critical for transformers).
        val_every: Validate every N steps (default: every epoch).
        checkpoint_every: Save a checkpoint every N steps (None = off).
        checkpoint_dir: Directory for checkpoints.
        record_steps: If True, record every step's loss in
            ``step_loss_history`` (useful for tests/short runs).
        verbose: Print per-epoch progress.

    Returns:
        Dict with per-step/epoch histories and final metrics:
        - ``best_val_ppl``, ``best_step``, ``final_val_ppl``
        - ``final_train_loss``, ``anneal_start_step``, ``steps_trained``
        - ``training_time_seconds``
        - ``train_loss_history``, ``val_ppl_history``, ``lr_history``,
          ``flip_history``
    """
    total_start = time.time()
    total_steps = (
        max_steps
        if max_steps is not None
        else epochs * max(len(train_loader), 1)
    )
    anneal_start_step = int(total_steps * anneal_fraction)
    deterministic_active = False

    step = 0
    best_val_ppl = float("inf")
    best_step = 0
    ema_loss: float | None = None

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "lr": [],
        "flip": [],
        "step": [],
        "step_loss": [],
    }
    os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_tokens = 0
        running_flip = 0.0
        n_steps_epoch = 0

        for input_ids, targets in train_loader:
            if max_steps is not None and step >= max_steps:
                break
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)  # (B, T, V)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), targets.reshape(-1)
            )
            loss.backward()
            # Gradient clipping is CRITICAL for DQT transformers (the
            # stochastic rounding can inject spikes into the optimization).
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # ── DQT: rounding after EVERY optimizer step ──
            use_stochastic = should_use_stochastic(step, total_steps, anneal_fraction)
            if (not use_stochastic) and (not deterministic_active):
                deterministic_active = True
                if verbose:
                    print(
                        f"  🔒 Step {step}: switching to DETERMINISTIC sign "
                        "(no more stochastic rounding)"
                    )
            mean_flip = apply_dqt_rounding(model, use_stochastic=use_stochastic)

            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
            running_loss += loss.item() * targets.numel()
            n_tokens += targets.numel()
            running_flip += mean_flip
            n_steps_epoch += 1
            step += 1

            if record_steps:
                history["step_loss"].append(float(loss.item()))

            if scheduler is not None:
                scheduler.step()

            if checkpoint_every and checkpoint_dir and step % checkpoint_every == 0:
                _save_checkpoint(
                    model, optimizer, step, checkpoint_dir, epoch
                )

            if val_every and step % val_every == 0:
                val_ppl = evaluate_perplexity(model, val_loader, device)
                model.train()
                step_val_loss = math.log(val_ppl)  # exp(mean loss) -> mean loss
                history["val_loss"].append(step_val_loss)
                history["val_ppl"].append(val_ppl)
                history["step"].append(step)
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    best_step = step

        if n_steps_epoch == 0:
            break  # max_steps already reached

        epoch_loss = running_loss / max(n_tokens, 1)
        history["train_loss"].append(epoch_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["flip"].append(running_flip / max(n_steps_epoch, 1))

        # Per-epoch validation (unless already done step-wise this epoch)
        if val_every is None:
            val_loss = _mean_val_loss(model, val_loader, device)
            val_ppl = compute_perplexity(val_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)
            history["step"].append(step)
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_step = step
        else:
            val_ppl = history["val_ppl"][-1] if history["val_ppl"] else float("inf")

        if verbose:
            print(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"Train Loss: {epoch_loss:.4f}  "
                f"Val PPL: {val_ppl:.2f}  "
                f"Flip: {history['flip'][-1]:.4f}  "
                f"LR: {history['lr'][-1]:.2e}"
            )

    steps_trained = step
    total_time = time.time() - total_start

    # Final evaluation on the val set
    final_val_ppl = evaluate_perplexity(model, val_loader, device)
    if final_val_ppl < best_val_ppl:
        best_val_ppl = final_val_ppl
        best_step = steps_trained

    return {
        "best_val_ppl": float(best_val_ppl),
        "best_step": best_step,
        "final_val_ppl": float(final_val_ppl),
        "final_train_loss": float(
            history["train_loss"][-1]
            if history["train_loss"]
            else ema_loss or float("inf")
        ),
        "anneal_start_step": anneal_start_step,
        "anneal_fraction": float(anneal_fraction),
        "steps_trained": steps_trained,
        "epochs": epochs,
        "training_time_seconds": float(total_time),
        "final_flip_rate": float(history["flip"][-1] if history["flip"] else 0.0),
        "train_loss_history": [float(x) for x in history["train_loss"]],
        "step_loss_history": [float(x) for x in history["step_loss"]],
        "val_loss_history": [float(x) for x in history["val_loss"]],
        "val_ppl_history": [float(x) for x in history["val_ppl"]],
        "lr_history": [float(x) for x in history["lr"]],
        "flip_history": [float(x) for x in history["flip"]],
        "step_history": [int(x) for x in history["step"]],
    }


@torch.no_grad()
def _mean_val_loss(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Mean validation cross-entropy loss (no perplexity conversion)."""
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    for input_ids, targets in val_loader:
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), targets.reshape(-1)
        )
        total_loss += loss.item() * targets.numel()
        n_tokens += targets.numel()
    model.train()
    return total_loss / max(n_tokens, 1)


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    epoch: int,
) -> None:
    """Save a training checkpoint to ``{dir}/ckpt_step{step}.pt``."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step{step}.pt")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# ── LR schedule: warmup + cosine ───────────────────────────────────


def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine LR schedule with linear warmup (GPT-style).

    Args:
        optimizer: The optimizer.
        warmup_steps: Linear warmup over the first N steps.
        total_steps: Total training steps (cosine anneals over the rest).
        min_lr_ratio: Fraction of the base LR at the end of cosine.

    Returns:
        A ``LambdaLR`` scheduler (step it once per training step).
    """

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ── Main ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone M2.1: DQT Transformer on TinyStories (GO/NO-GO ppl<30)"
    )
    # Architecture (defaults = FULL_CONFIG → ~102M ternary weights)
    parser.add_argument("--d-model", type=int, default=FULL_CONFIG["d_model"])
    parser.add_argument("--n-heads", type=int, default=FULL_CONFIG["n_heads"])
    parser.add_argument("--n-layers", type=int, default=FULL_CONFIG["n_layers"])
    parser.add_argument("--d-ff", type=int, default=FULL_CONFIG["d_ff"])
    parser.add_argument("--vocab-size", type=int, default=FULL_CONFIG["vocab_size"])
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="RoPE max seq len (default: seq-len)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use the SMOKE_CONFIG architecture (~16M ternary)")
    # Training
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (DQT best: 0.01)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anneal-fraction", type=float, default=ANNEAL_FRACTION)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap total training steps (tests/smoke)")
    parser.add_argument("--val-every", type=int, default=None,
                        help="Validate every N steps (default: per epoch)")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save a checkpoint every N steps")
    # Data
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic learnable data (no TinyStories download)")
    parser.add_argument("--synthetic-vocab", type=int, default=64)
    parser.add_argument("--synthetic-batches", type=int, default=8)
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Cap TinyStories stories downloaded (None=all)")
    parser.add_argument("--num-workers", type=int, default=0)
    # Output
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--output-dir", default="m2_1_results")
    parser.add_argument("--checkpoint-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ── Architecture ───────────────────────────────────────────────
    if args.smoke:
        cfg = build_config(
            vocab_size=args.vocab_size,
            d_model=SMOKE_CONFIG["d_model"],
            n_heads=SMOKE_CONFIG["n_heads"],
            n_layers=SMOKE_CONFIG["n_layers"],
            d_ff=SMOKE_CONFIG["d_ff"],
            max_seq_len=args.max_seq_len or args.seq_len,
        )
    else:
        cfg = build_config(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len or args.seq_len,
        )

    print_header(
        f"M2.1 DQT Transformer TinyStories (GO/NO-GO ppl<30): "
        f"d={cfg['d_model']} L={cfg['n_layers']} H={cfg['n_heads']} "
        f"ff={cfg['d_ff']} lr={args.lr}, {args.epochs}ep, seed={args.seed}, "
        f"anneal@{int(100 * args.anneal_fraction)}%"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Data ───────────────────────────────────────────────────────
    if args.synthetic:
        seq_len = min(args.seq_len, 64)  # keep smoke runs small
        train_loader = make_synthetic_lm_loader(
            vocab_size=args.synthetic_vocab,
            seq_len=seq_len,
            batch_size=args.batch_size,
            n_batches=args.synthetic_batches,
            seed=args.seed,
        )
        val_loader = make_synthetic_lm_loader(
            vocab_size=args.synthetic_vocab,
            seq_len=seq_len,
            batch_size=args.batch_size,
            n_batches=max(2, args.synthetic_batches // 2),
            seed=args.seed + 1,
        )
        cfg["vocab_size"] = args.synthetic_vocab
        data_meta = {
            "mode": "synthetic",
            "vocab_size": args.synthetic_vocab,
            "seq_len": seq_len,
            "n_train_batches": args.synthetic_batches,
        }
    else:
        train_loader, val_loader, data_meta = get_tinystories_data(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            # 0 or None => download the whole dataset (no cap)
            max_samples=args.max_samples or None,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        data_meta["mode"] = "tinystories"
        cfg["vocab_size"] = data_meta["vocab_size"]
        seq_len = args.seq_len
    print(f"Data: {data_meta['mode']} | vocab={cfg['vocab_size']} | seq_len={seq_len}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print()

    # ── Model ──────────────────────────────────────────────────────
    model = dqt_gpt2(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        dropout=args.dropout,
        device=device,
    )
    n_total = count_parameters(model)
    n_ternary = count_ternary_weights(model)
    print(f"Model: d_model={cfg['d_model']}, n_layers={cfg['n_layers']}, "
          f"n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}")
    print(f"  Float parameters:  {n_total - n_ternary:,}")
    print(f"  Ternary weights:   {n_ternary:,}  (int8)")
    print(f"  Total:             {n_total:,}")
    print()

    # ── Optimizer + scheduler ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    total_steps = args.max_steps or args.epochs * len(train_loader)
    scheduler = make_cosine_warmup_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
    )

    # ── Train ──────────────────────────────────────────────────────
    print("Training...")
    print()
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    results = train_dqt_transformer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        epochs=args.epochs,
        max_steps=args.max_steps,
        anneal_fraction=args.anneal_fraction,
        grad_clip=args.grad_clip,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=checkpoint_dir,
    )

    # ── Peak GPU memory ────────────────────────────────────────────
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    # ── Result dict ────────────────────────────────────────────────
    result = {
        "experiment": "m2_1_dqt_transformer",
        "dataset": data_meta["mode"],
        "tokenizer": data_meta.get("tokenizer", "synthetic"),
        "seed": args.seed,
        "device": str(device),
        "config": cfg,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "anneal_fraction": args.anneal_fraction,
        "architecture": (
            f"DQT Transformer: emb({cfg['vocab_size']}->{cfg['d_model']}) "
            f"+ {cfg['n_layers']}x[Attn({cfg['n_heads']}h, RoPE) + FFN({cfg['d_ff']})] "
            f"+ RMSNorm + DQT LM Head"
        ),
        "method": (
            "Direct Quantized Training (DQT): int8 ternary weights + stochastic "
            f"rounding annealed to deterministic sign for final "
            f"{int(100 * (1 - args.anneal_fraction))}% of steps"
        ),
        "n_float_params": n_total - n_ternary,
        "n_ternary_weights": n_ternary,
        "n_total_params": n_total,
        "perplexity_gate": PPL_GATE,
        "peak_gpu_memory_mb": peak_mem_mb,
        **results,
    }

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"results_m2_1_dqt_transformer_lr{args.lr}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────
    print()
    print_header("Results Summary")
    print(f"  Best Val Perplexity:   {results['best_val_ppl']:.2f}  (step {results['best_step']})")
    print(f"  Final Val Perplexity:  {results['final_val_ppl']:.2f}")
    print(f"  Final Train Loss:      {results['final_train_loss']:.4f}")
    print(f"  Anneal Start Step:     {results['anneal_start_step']} (deterministic sign after)")
    print(f"  Steps Trained:         {results['steps_trained']}")
    print(f"  Training Time:         {results['training_time_seconds']:.1f}s")
    print(f"  Final Flip Rate:       {results['final_flip_rate']:.4f}")
    if peak_mem_mb > 0:
        print(f"  Peak GPU Memory:       {peak_mem_mb:.1f} MB")
    print()
    verdict = (
        f"GO ✅ (ppl < {PPL_GATE})"
        if results["best_val_ppl"] < PPL_GATE
        else f"NO-GO 🔴 (ppl >= {PPL_GATE})"
    )
    print(f"  M2.1 Verdict:          ppl={results['best_val_ppl']:.2f}  →  {verdict}")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
