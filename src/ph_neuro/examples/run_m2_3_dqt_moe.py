#!/usr/bin/env python3
"""Milestone M2.3 — MoE DQT Transformer on TinyStories (GO/NO-GO ppl<20).

First MoE (Mixture of Experts) DQT Transformer: ~312M ternary params total,
~161M active per token (52%). The M2.1 GPT-2-style DQT transformer (int8
ternary weights + stochastic rounding + annealing) is extended with a
hybrid dense+MoE block stack:

    - layers 0-3: dense FFN (cheap, good early features)
    - layers 4-11: MoE FFN (8 layers x 6 experts, top-2 routing) — a float
      router selects the top-2 of 6 DQT ternary experts per token, and only
      the selected experts run (grouped execution), so the active parameter
      count is ~top_k/n_experts of the expert stack.

Trained on TinyStories (GPT-2 BPE). GO if the mean validation perplexity
across 3 seeds is < 20 (M2.1 dense 102M got 11.35; the 3x-param MoE must
approach or beat it).

The M2.3 config (~312.3M ternary) is expected to fit 8 GB without gradient
checkpointing (~6.7 GB est. — see the M2.3 memory budget). Memory levers
retained from M2.2: the float token embedding trains with plain SGD (no
AdamW moments — "embedding χωρίς AdamW") and the shell script sets
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Two optimizers in lockstep: AdamW with TWO param groups — experts/attention/
norms/LM head at ``lr`` and the MoE routers at ``0.1 x lr`` (E019 finding:
slow router is required to avoid expert collapse) — plus a plain SGD
optimizer for the float token embedding. Every step:

    loss = CE(logits, targets) + lb_coef * aux_load_balance_loss
    optimizer.step(); emb_optimizer.step(); apply_dqt_rounding()

Pause/resume (kept from M2.1/M2.2):
    - ``--resume auto`` resumes from the latest checkpoint
    - SIGINT / SIGTERM / **SIGUSR1** → graceful pause: finish the step,
      save a checkpoint, print the resume command, exit 130
    - ``--pause-file PATH`` → external pause control: while the file exists,
      the loop pauses at the next step boundary
    - ``--checkpoint-every N`` → periodic checkpoints
    - writes ``status.json`` every ``--progress-every`` steps for the
      ``status`` command in ``scripts/run_m2_3_dqt_moe.sh``

Usage::

    python -m ph_neuro.examples.run_m2_3_dqt_moe \\
        --lr 0.01 --epochs 3 --batch-size 8 --seq-len 256 --seed 42

    # Smoke test (10 steps, no TinyStories download — synthetic corpus):
    python -m ph_neuro.examples.run_m2_3_dqt_moe \\
        --max-steps 10 --seed 42 --synthetic

Output:
    JSON file: ``{output_dir}/results_m2_3_dqt_moe_lr{lr}_seed{seed}.json``
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import signal
import sys
import time
import warnings
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_transformer import TernaryDQTMoETransformerBlock
from ph_neuro.utils.optimizers import make_adamw
from ph_neuro.models.dqt_transformer import (
    M2_2_CONFIG,
    M2_3_CONFIG,
    SMOKE_MOE_CONFIG,
    build_moe_config,
    count_parameters,
    count_ternary_weights,
    dqt_gpt2_moe,
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

# GO/NO-GO gate: mean validation perplexity across 3 seeds must be < 20.
PPL_GATE = 20.0

# Low-disk warning threshold (GB free on the WSL filesystem). Checkpoint
# pruning bounds the disk, but this catches unexpected growth and reminds to
# compact the WSL VHDX (the M2.3 disk-fill / WSL-reboot root cause).
MIN_FREE_GB = 25.0

# ── Graceful pause (SIGINT / SIGTERM / SIGUSR1 / pause file) ───────
# A SIGINT (Ctrl+C), SIGTERM or SIGUSR1 sets this flag instead of killing
# the process mid-step. The training loop checks it between steps, saves a
# checkpoint and exits cleanly, so the run can be resumed with
# ``--resume auto`` with only a few seconds of progress lost. The M2.2
# external-pause flow uses SIGUSR1 (the gaming co-use pause signal) — see
# the pause/resume section of the M2.2 brief.
_PAUSE_REQUESTED = False


def _pause_signal_handler(signum, frame):
    """Request a graceful pause at the next step boundary."""
    global _PAUSE_REQUESTED
    _PAUSE_REQUESTED = True
    print(
        f"\n  ⏸️  Received signal {signum} — finishing current step, "
        "then pausing and saving a checkpoint...",
        flush=True,
    )


def install_pause_handlers() -> None:
    """Install graceful-pause handlers for SIGINT, SIGTERM and SIGUSR1."""
    signal.signal(signal.SIGINT, _pause_signal_handler)
    signal.signal(signal.SIGTERM, _pause_signal_handler)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _pause_signal_handler)


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
        model: The DQT (MoE) transformer — forward returns ``(logits, aux)``.
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
        logits, _ = model(input_ids)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), targets.reshape(-1)
        )
        total_loss += loss.item() * targets.numel()
        n_tokens += targets.numel()
    if n_tokens == 0:
        return float("inf")
    return compute_perplexity(total_loss / max(n_tokens, 1))


# ── MoE expert utilization (for monitoring / result JSON) ──────────


@torch.no_grad()
def collect_expert_utilization(model: nn.Module) -> list[dict]:
    """Per-MoE-layer routing utilization for the result JSON.

    Args:
        model: The hybrid DQT MoE transformer.

    Returns:
        List of dicts (one per MoE block) with ``layer`` index and
        per-expert ``selection_fractions`` / ``coverage_fractions`` plus a
        ``balance_ratio`` (max/min selection share; ``inf`` = dead expert).
    """
    layers: list[dict] = []
    for i, block in enumerate(model.blocks):
        if isinstance(block, TernaryDQTMoETransformerBlock):
            moe = block.moe_ffn
            report = moe.balance_report()
            layers.append(
                {
                    "layer": i,
                    "selection_fractions": moe.selection_fractions().tolist(),
                    "coverage_fractions": moe.coverage_fractions().tolist(),
                    "balance_ratio": report["balance_ratio"],
                    "min_share": report["min_share"],
                    "max_share": report["max_share"],
                }
            )
    return layers


def _log_expert_utilization(model: nn.Module) -> None:
    """Print a compact per-MoE-layer balance line (dead-expert detector)."""
    for i, block in enumerate(model.blocks):
        if isinstance(block, TernaryDQTMoETransformerBlock):
            moe = block.moe_ffn
            fracs = moe.selection_fractions()
            fracs_s = " ".join(f"{f:.2f}" for f in fracs.tolist())
            rep = moe.balance_report()
            print(
                f"      MoE L{i}: sel=[{fracs_s}] "
                f"balance={rep['balance_ratio']:.2f} min={rep['min_share']:.3f}",
                flush=True,
            )


# ── Status file (for the ``status`` shell command) ──────────────────


def _gpu_mem_gb(device: torch.device) -> float:
    """Current GPU memory used in GB (0.0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024**3)


def write_status_file(
    status_file: str | None,
    *,
    seed: int,
    lr: float,
    step: int,
    total_steps: int,
    epoch: int,
    loss: float,
    ppl: float,
    tok_per_s: float,
    gpu_mem_gb: float,
    status: str,
) -> None:
    """Write the M2.3 progress status JSON for ``run_m2_3_dqt_moe.sh status``."""
    if status_file is None:
        return
    payload = {
        "experiment": "m2_3_dqt_moe",
        "seed": seed,
        "lr": lr,
        "step": int(step),
        "total_steps": int(total_steps),
        "epoch": int(epoch),
        "loss": float(loss),
        "ppl": float(ppl),
        "tok_per_s": float(tok_per_s),
        "gpu_mem_gb": float(gpu_mem_gb),
        "gpu_total_gb": (
            float(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            if torch.cuda.is_available()
            else 0.0
        ),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    tmp = status_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, status_file)


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
    lb_coef: float = 0.1,
    val_every: int | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | None = None,
    keep_last_checkpoints: int = 2,
    status_file: str | None = None,
    progress_every: int = 50,
    pause_file: str | None = None,
    seed: int = 0,
    emb_optimizer: torch.optim.Optimizer | None = None,
    emb_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    record_steps: bool = False,
    start_step: int = 0,
    start_epoch: int = 1,
    best_val_ppl: float = float("inf"),
    best_step: int = 0,
    verbose: bool = True,
    use_autocast: bool = False,
) -> dict:
    """Train a DQT transformer for language modeling.

    Key DQT difference: after each ``optimizer.step()`` we call
    ``apply_dqt_rounding()`` on every DQT layer. For the first
    ``int(total_steps * anneal_fraction)`` steps this is stochastic
    rounding (exploration); afterwards it switches to deterministic
    ``sign()`` (clean fine-tuning tail — the M1.1-RETRY fix).

    M2.3 MoE differences: the model forward returns ``(logits, aux_loss)``
    and the total loss is ``CE(logits, targets) + lb_coef * aux_loss``
    (Switch-Transformer load balancing, ``lb_coef=0.1``). Every
    ``progress_every`` steps a per-MoE-layer expert utilization line is
    printed (dead-expert detector).

    Args:
        model: The DQT (MoE) transformer — forward returns (logits, aux).
        train_loader: Training loader yielding ``(input_ids, targets)``.
        val_loader: Validation loader.
        optimizer: AdamW optimizer with TWO param groups (experts at ``lr``
            and MoE routers at ``0.1 x lr`` — the E019 slow-router rule).
        scheduler: LR scheduler (stepped once per step, cosine+warmup).
        device: Torch device.
        epochs: Number of epochs.
        max_steps: If set, cap total training steps (tests/smoke).
        anneal_fraction: Fraction of steps with stochastic rounding.
        grad_clip: Max gradient norm (critical for transformers).
        lb_coef: Weight of the MoE aux load-balancing loss (default 0.1).
        val_every: Validate every N steps (default: every epoch).
        checkpoint_every: Save a checkpoint every N steps (None = off).
        checkpoint_dir: Directory for checkpoints.
        keep_last_checkpoints: Number of newest periodic checkpoints to keep
            (older ones are pruned after each save — bounds disk; the latest
            + ``best.pt`` are all that are needed to resume).
        status_file: Path to write progress ``status.json`` every
            ``progress_every`` steps (for the ``status`` shell command).
        progress_every: Steps between status writes / progress prints.
        pause_file: If set, the loop pauses gracefully (saves a checkpoint)
            as soon as this file exists — external pause control.
        seed: Experiment seed (recorded in the status file).
        emb_optimizer: Optional SGD optimizer for the float token embedding
            (trained WITHOUT AdamW moments — the M2.2 memory budget saves
            ~0.4 GB this way). Stepped once per step like the main one.
        emb_scheduler: Optional LR scheduler for ``emb_optimizer`` (same
            cosine+warmup schedule as the main optimizer).
        record_steps: If True, record every step's loss in
            ``step_loss_history`` (useful for tests/short runs).
        start_step: Step counter to continue from (resume).
        start_epoch: First epoch to run (resume).
        best_val_ppl: Best validation perplexity seen so far (resume).
        best_step: Step at which ``best_val_ppl`` was reached (resume).
        verbose: Print per-epoch progress.

    Returns:
        Dict with per-step/epoch histories and final metrics:
        - ``best_val_ppl``, ``best_step``, ``final_val_ppl``
        - ``final_train_loss``, ``anneal_start_step``, ``steps_trained``
        - ``training_time_seconds``
        - ``train_loss_history``, ``val_ppl_history``, ``lr_history``,
          ``flip_history``
    """
    global _PAUSE_REQUESTED
    _PAUSE_REQUESTED = False

    total_start = time.time()
    total_steps = (
        max_steps
        if max_steps is not None
        else epochs * max(len(train_loader), 1)
    )
    anneal_start_step = int(total_steps * anneal_fraction)
    # If resuming from the deterministic phase, don't re-print the switch.
    deterministic_active = start_step >= anneal_start_step

    step = start_step
    ema_loss: float | None = None
    step_t0 = time.time()
    tokens_this_step = 0

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

    if verbose and start_step > 0:
        print(
            f"  ↩️ Resuming from step {start_step} (epoch {start_epoch}), "
            f"best val ppl so far: {best_val_ppl:.2f}"
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_tokens = 0
        running_flip = 0.0
        n_steps_epoch = 0

        for input_ids, targets in train_loader:
            if max_steps is not None and step >= max_steps:
                break
            if _PAUSE_REQUESTED:
                break
            if pause_file and os.path.exists(pause_file):
                print(
                    f"\n  ⏸️  Pause file detected ({pause_file}) — pausing at "
                    "next step boundary...",
                    flush=True,
                )
                _PAUSE_REQUESTED = True
                break
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            tokens_this_step = targets.numel()

            optimizer.zero_grad()
            # OPT-3: bf16 autocast (enabled only when weight_float is bf16 —
            # halves activation memory + ~1.2x faster via tensor cores).
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast
            ):
                logits, aux_loss = model(input_ids)  # (B, T, V), scalar
                loss = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size), targets.reshape(-1)
                )
                # MoE: add the Switch-Transformer load balancing loss so the
                # router learns a uniform dispatch (E019: without it, expert
                # collapse in the first epoch). lb_coef=0.1 by default.
                if lb_coef > 0.0 and aux_loss is not None:
                    loss = loss + lb_coef * aux_loss
            loss.backward()
            # Gradient clipping is CRITICAL for DQT transformers (the
            # stochastic rounding can inject spikes into the optimization).
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if emb_optimizer is not None:
                emb_optimizer.step()

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
            running_loss += loss.item() * tokens_this_step
            n_tokens += tokens_this_step
            running_flip += mean_flip
            n_steps_epoch += 1
            step += 1

            if record_steps:
                history["step_loss"].append(float(loss.item()))

            if scheduler is not None:
                scheduler.step()
            if emb_scheduler is not None:
                emb_scheduler.step()

            # ── Periodic progress + status file (for `status` cmd) ──
            if progress_every and step % progress_every == 0:
                elapsed = time.time() - total_start
                tok_per_s = step * tokens_this_step / max(elapsed, 1e-6)
                eta_s = (total_steps - step) * (elapsed / max(step, 1))
                cur_ppl = compute_perplexity(ema_loss or 0.0)
                gpu_gb = _gpu_mem_gb(device)
                write_status_file(
                    status_file,
                    seed=seed,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    step=step,
                    total_steps=total_steps,
                    epoch=epoch,
                    loss=ema_loss or 0.0,
                    ppl=cur_ppl,
                    tok_per_s=tok_per_s,
                    gpu_mem_gb=gpu_gb,
                    status="RUNNING",
                )
                if verbose:
                    print(
                        f"  Step {step:6d}/{total_steps}  loss {ema_loss:.4f}  "
                        f"ppl {cur_ppl:7.1f}  {tok_per_s:6.0f} tok/s  "
                        f"flip {mean_flip:.4f}  ETA {eta_s/60:5.1f} min  "
                        f"GPU {gpu_gb:.1f} GB",
                        flush=True,
                    )
                    # Expert utilization (dead-expert detector)
                    _log_expert_utilization(model)

            if checkpoint_every and checkpoint_dir and step % checkpoint_every == 0:
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    checkpoint_dir,
                    epoch,
                    best_val_ppl=best_val_ppl,
                    best_step=best_step,
                    emb_optimizer=emb_optimizer,
                    emb_scheduler=emb_scheduler,
                    keep_last=keep_last_checkpoints,
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
                    if checkpoint_dir:
                        _save_best_checkpoint(
                            model, optimizer, scheduler, step, checkpoint_dir,
                            epoch, best_val_ppl=best_val_ppl, best_step=best_step,
                            emb_optimizer=emb_optimizer,
                            emb_scheduler=emb_scheduler,
                        )

        if _PAUSE_REQUESTED:
            break  # skip epoch-end work and pause

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
                if checkpoint_dir:
                    _save_best_checkpoint(
                        model, optimizer, scheduler, step, checkpoint_dir,
                        epoch, best_val_ppl=best_val_ppl, best_step=best_step,
                        emb_optimizer=emb_optimizer,
                        emb_scheduler=emb_scheduler,
                    )
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

    # Graceful pause: save a checkpoint and exit so ``--resume`` can continue.
    if _PAUSE_REQUESTED:
        if checkpoint_dir:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                checkpoint_dir,
                epoch,
                best_val_ppl=best_val_ppl,
                best_step=best_step,
                emb_optimizer=emb_optimizer,
                emb_scheduler=emb_scheduler,
                keep_last=keep_last_checkpoints,
            )
            write_status_file(
                status_file,
                seed=seed,
                lr=float(optimizer.param_groups[0]["lr"]),
                step=step,
                total_steps=total_steps,
                epoch=epoch,
                loss=ema_loss or 0.0,
                ppl=compute_perplexity(ema_loss or 0.0),
                tok_per_s=0.0,
                gpu_mem_gb=_gpu_mem_gb(device),
                status="PAUSED",
            )
            print(
                f"\n  ⏸️  Paused at step {step} (epoch {epoch}). "
                "Checkpoint saved. Resume with --resume auto."
            )
        else:
            print(
                f"\n  ⏸️  Paused at step {step} (epoch {epoch}). "
                "No checkpoint dir set — progress since the last periodic "
                "checkpoint is lost."
            )
        raise KeyboardInterrupt

    # Final evaluation on the val set
    final_val_ppl = evaluate_perplexity(model, val_loader, device)
    if final_val_ppl < best_val_ppl:
        best_val_ppl = final_val_ppl
        best_step = steps_trained
        if checkpoint_dir:
            _save_best_checkpoint(
                model, optimizer, scheduler, steps_trained, checkpoint_dir,
                epoch, best_val_ppl=best_val_ppl, best_step=best_step,
                emb_optimizer=emb_optimizer, emb_scheduler=emb_scheduler,
            )

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
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), targets.reshape(-1)
        )
        total_loss += loss.item() * targets.numel()
        n_tokens += targets.numel()
    model.train()
    return total_loss / max(n_tokens, 1)


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    epoch: int,
    best_val_ppl: float,
    best_step: int,
    emb_optimizer: torch.optim.Optimizer | None = None,
    emb_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    """Assemble the checkpoint dict (model + optimizers + schedulers)."""
    return {
        "step": step,
        "epoch": epoch,
        "best_val_ppl": float(best_val_ppl),
        "best_step": best_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "emb_optimizer_state_dict": (
            emb_optimizer.state_dict() if emb_optimizer is not None else None
        ),
        "emb_scheduler_state_dict": (
            emb_scheduler.state_dict() if emb_scheduler is not None else None
        ),
    }


def _prune_checkpoints(checkpoint_dir: str, keep_last: int) -> int:
    """Delete all but the newest ``keep_last`` ``ckpt_step*.pt`` files.

    The DISK fix for the M2.3 WSL reboots: with ``CHECKPOINT_EVERY=500`` and
    ~3.9 GB/checkpoint, an unpruned run grows ~754 GB/seed and fills the
    1 TB disk (the observed root cause). Keeping only the newest
    ``keep_last`` periodic checkpoints + ``best.pt`` bounds per-seed disk to
    ``(keep_last + 1) * ~3.9 GB``. ``best.pt`` is never pruned. The newest
    checkpoint is all that is needed to resume, so this loses nothing but
    stale files.

    Args:
        checkpoint_dir: Directory containing ``ckpt_step*.pt`` files.
        keep_last: Number of newest periodic checkpoints to keep.

    Returns:
        Number of files deleted.
    """
    matches = glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt"))
    if len(matches) <= keep_last:
        return 0

    def _step_of(path: str) -> int:
        base = os.path.basename(path)
        try:
            return int(base.replace("ckpt_step", "").replace(".pt", ""))
        except ValueError:
            return -1

    matches.sort(key=_step_of)
    deleted = 0
    for path in matches[:-keep_last]:
        try:
            os.remove(path)
            deleted += 1
        except OSError:  # pragma: no cover - best-effort cleanup
            pass
    return deleted


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    checkpoint_dir: str,
    epoch: int,
    best_val_ppl: float = float("inf"),
    best_step: int = 0,
    emb_optimizer: torch.optim.Optimizer | None = None,
    emb_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    keep_last: int = 2,
) -> None:
    """Save a training checkpoint to ``{dir}/ckpt_step{step}.pt``.

    Includes the model, optimizer and scheduler state (main + optional
    embedding optimizer) so training can be paused and later resumed with
    ``--resume``. After saving, old periodic checkpoints are pruned so only
    the newest ``keep_last`` remain (disk-bounding — see
    :func:`_prune_checkpoints`).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step{step}.pt")
    torch.save(
        _checkpoint_payload(
            model, optimizer, scheduler, step, epoch, best_val_ppl, best_step,
            emb_optimizer=emb_optimizer, emb_scheduler=emb_scheduler,
        ),
        path,
    )
    n_deleted = _prune_checkpoints(checkpoint_dir, keep_last)
    if n_deleted:
        print(
            f"  🧹 Pruned {n_deleted} old checkpoint(s) "
            f"(keeping latest {keep_last})",
            flush=True,
        )

    # Disk safety net: warn when the WSL filesystem free space is low (the
    # M2.3 disk-fill → WSL-reboot root cause). With pruning this should never
    # trigger, but it gives an early signal to compact the VHDX.
    try:
        free_gb = shutil.disk_usage(checkpoint_dir).free / (1024**3)
        if free_gb < MIN_FREE_GB:
            print(
                f"  ⚠️  LOW DISK: only {free_gb:.0f} GB free on the WSL filesystem. "
                "Checkpoints are pruned, but consider compacting the WSL VHDX: "
                "run `wsl --shutdown` then `wsl --manage Ubuntu-24.04 --shrink` "
                "from Windows PowerShell.",
                flush=True,
            )
    except OSError:  # pragma: no cover - best-effort
        pass


def _save_best_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    checkpoint_dir: str,
    epoch: int,
    best_val_ppl: float,
    best_step: int,
    emb_optimizer: torch.optim.Optimizer | None = None,
    emb_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Overwrite ``{dir}/best.pt`` with the best-val checkpoint.

    Called whenever ``best_val_ppl`` improves, so ``best.pt`` always holds
    the best model seen (useful for the E027 evaluation). It is a single
    file (overwritten) and is never pruned.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model, optimizer, scheduler, step, epoch, best_val_ppl, best_step,
            emb_optimizer=emb_optimizer, emb_scheduler=emb_scheduler,
        ),
        os.path.join(checkpoint_dir, "best.pt"),
    )


def find_latest_checkpoint(checkpoint_dir: str | None) -> str | None:
    """Return the highest-step VALID checkpoint in ``checkpoint_dir``.

    Corrupted (e.g. truncated) checkpoint files are skipped, so ``--resume
    auto`` falls back to the last checkpoint that was written completely.
    This is the M2.3 lesson: a checkpoint being written when the disk filled
    / WSL crashed was saved as a truncated (unloadable) file, which broke
    ``--resume auto``.

    Args:
        checkpoint_dir: Directory containing ``ckpt_step*.pt`` files.

    Returns:
        Path of the highest-step readable checkpoint, or ``None``.
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None
    matches = glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt"))
    if not matches:
        return None

    def _step_of(path: str) -> int:
        base = os.path.basename(path)
        try:
            return int(base.replace("ckpt_step", "").replace(".pt", ""))
        except ValueError:
            return -1

    # Try highest-step first; cheap zip-structure check catches truncated
    # files (torch checkpoints are zip archives; a truncated save has no
    # central directory).
    for path in sorted(matches, key=_step_of, reverse=True):
        if zipfile.is_zipfile(path):
            return path
        print(
            f"  ⚠️  Skipping corrupted checkpoint {os.path.basename(path)} "
            "(truncated/invalid) — trying the previous one...",
            flush=True,
        )
    return None


def load_checkpoint(
    resume: str,
    checkpoint_dir: str | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    emb_optimizer: torch.optim.Optimizer | None = None,
    emb_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    """Load optimizer/model/scheduler state from a checkpoint for resume.

    Args:
        resume: Path to a checkpoint file, or ``"auto"`` to resume the
            latest checkpoint in ``checkpoint_dir``.
        checkpoint_dir: Default checkpoint directory (for ``"auto"``).
        model: The (freshly built) model to restore into.
        optimizer: The (freshly built) optimizer to restore into.
        scheduler: The (freshly built) scheduler to restore into.
        device: Torch device.
        emb_optimizer: The (freshly built) embedding optimizer to restore.
        emb_scheduler: The (freshly built) embedding scheduler to restore.

    Returns:
        Dict with ``step``, ``epoch``, ``best_val_ppl``, ``best_step``.
    """
    path = resume
    if path == "auto":
        path = find_latest_checkpoint(checkpoint_dir)
        if path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir!r} to resume from"
            )
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if emb_optimizer is not None and ckpt.get("emb_optimizer_state_dict") is not None:
        emb_optimizer.load_state_dict(ckpt["emb_optimizer_state_dict"])
    if emb_scheduler is not None and ckpt.get("emb_scheduler_state_dict") is not None:
        emb_scheduler.load_state_dict(ckpt["emb_scheduler_state_dict"])

    print(f"  ↩️ Loaded checkpoint {os.path.basename(path)}")
    return {
        "step": int(ckpt["step"]),
        "epoch": int(ckpt["epoch"]),
        "best_val_ppl": float(ckpt.get("best_val_ppl", float("inf"))),
        "best_step": int(ckpt.get("best_step", 0)),
    }


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
        description="Milestone M2.3: MoE DQT Transformer on TinyStories (GO/NO-GO ppl<20)"
    )
    # Architecture (defaults = M2_3_CONFIG → ~312M ternary, ~161M active)
    parser.add_argument("--d-model", type=int, default=M2_3_CONFIG["d_model"])
    parser.add_argument("--n-heads", type=int, default=M2_3_CONFIG["n_heads"])
    parser.add_argument("--d-ff", type=int, default=M2_3_CONFIG["d_ff"])
    parser.add_argument("--dense-layers", type=int, default=M2_3_CONFIG["dense_layers"],
                        help="Leading dense FFN blocks (cheap early features)")
    parser.add_argument("--moe-layers", type=int, default=M2_3_CONFIG["moe_layers"],
                        help="Trailing MoE FFN blocks (top-K of n-experts)")
    parser.add_argument("--n-experts", type=int, default=M2_3_CONFIG["n_experts"])
    parser.add_argument("--top-k", type=int, default=M2_3_CONFIG["top_k"])
    parser.add_argument("--lb-coef", type=float, default=M2_3_CONFIG["lb_coef"],
                        help="Weight of the Switch-Transformer aux load-balancing loss")
    parser.add_argument("--router-lr-ratio", type=float, default=M2_3_CONFIG["router_lr_ratio"],
                        help="MoE router lr = ratio x expert lr (E019: 0.1 avoids collapse)")
    parser.add_argument("--vocab-size", type=int, default=M2_3_CONFIG["vocab_size"])
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="RoPE max seq len (default: seq-len)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use the SMOKE_MOE_CONFIG architecture (small hybrid)")
    parser.add_argument("--m2-2-config", action="store_true",
                        help="Use the M2.2 dense-only config via the MoE factory "
                             "(dense_layers=n_layers, moe_layers=0 — baseline check)")
    parser.add_argument("--grad-checkpoint", action="store_true", default=False,
                        help="Gradient checkpointing (recompute block activations in "
                             "backward) — M2.3 fits 8 GB without it (~6.7 GB est.), "
                             "so this defaults OFF")
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                        help="Disable gradient checkpointing (default)")
    # Training
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (DQT best: 0.01; router gets 0.1x)")
    parser.add_argument("--epochs", type=int, default=3)
    # batch 4 (not 8): VERIFIED memory-safe for the ~312M config. torch peak
    # ~7.4 GB (weights + AdamW moments + grads + logits at vocab 50257 — fixed
    # costs dominate; activations are ~1.4 GB). batch 8 pushes past the 8 GB
    # card's limit under shared-GPU conditions (crashes with "device not
    # ready"), so 4 is the verified default — same call M2.2 made.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anneal-fraction", type=float, default=ANNEAL_FRACTION)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16"], default="fp32",
        help="DQT weight-buffer dtype: fp32 (default) or bf16 (OPT-3: halves "
             "weight_float memory 4→2 B/param, pairs with autocast)",
    )
    parser.add_argument("--embed-adamw", action="store_true",
                        help="Opt-in: train the float token embedding with AdamW too "
                             "(M2.1 behavior, ~0.4 GB more VRAM). Default: the embedding "
                             "uses plain SGD (no AdamW moments) to fit 312M in 8 GB.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap total training steps (tests/smoke)")
    parser.add_argument("--val-every", type=int, default=None,
                        help="Validate every N steps (default: per epoch)")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save a checkpoint every N steps")
    parser.add_argument("--keep-last-checkpoints", type=int, default=2,
                        help="Number of newest periodic checkpoints to keep per seed "
                             "(older ones are pruned after each save to bound disk; "
                             "the latest + best.pt are enough to resume)")
    parser.add_argument("--progress-every", type=int, default=50,
                        help="Write status.json + print progress every N steps")
    parser.add_argument("--pause-file", default=None,
                        help="External pause control: while this file exists, "
                             "the loop pauses gracefully at the next step boundary")
    parser.add_argument("--resume", default=None,
                        help="Resume from a checkpoint path, or 'auto' for the "
                             "latest checkpoint in --checkpoint-dir")
    # Data
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic learnable data (no TinyStories download)")
    parser.add_argument("--synthetic-vocab", type=int, default=64)
    parser.add_argument("--synthetic-batches", type=int, default=8)
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap TinyStories stories to load (None = runner default 50000; "
                             "full run scripts pass 150000)")
    parser.add_argument("--num-workers", type=int, default=0)
    # Output
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--output-dir", default="m2_3_results")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--pid-file", default=None,
                        help="Write this process's PID to a file at startup "
                             "(for external pause/resume control)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Record our PID so an external supervisor can send SIGUSR1/SIGINT to pause.
    if args.pid_file:
        with open(args.pid_file, "w") as f:
            f.write(str(os.getpid()))

    # ── Architecture ───────────────────────────────────────────────
    if args.smoke:
        cfg = build_moe_config(
            vocab_size=args.vocab_size,
            d_model=SMOKE_MOE_CONFIG["d_model"],
            n_heads=SMOKE_MOE_CONFIG["n_heads"],
            d_ff=SMOKE_MOE_CONFIG["d_ff"],
            dense_layers=SMOKE_MOE_CONFIG["dense_layers"],
            moe_layers=SMOKE_MOE_CONFIG["moe_layers"],
            n_experts=SMOKE_MOE_CONFIG["n_experts"],
            top_k=SMOKE_MOE_CONFIG["top_k"],
            max_seq_len=args.max_seq_len or args.seq_len,
            lb_coef=args.lb_coef,
            router_lr_ratio=args.router_lr_ratio,
        )
    elif args.m2_2_config:
        # Dense-only baseline via the MoE factory (moe_layers=0).
        cfg = build_moe_config(
            vocab_size=args.vocab_size,
            d_model=M2_2_CONFIG["d_model"],
            n_heads=M2_2_CONFIG["n_heads"],
            d_ff=M2_2_CONFIG["d_ff"],
            dense_layers=M2_2_CONFIG["n_layers"],
            moe_layers=0,
            n_experts=args.n_experts,
            top_k=args.top_k,
            max_seq_len=args.max_seq_len or args.seq_len,
            lb_coef=args.lb_coef,
            router_lr_ratio=args.router_lr_ratio,
        )
    else:
        cfg = build_moe_config(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dense_layers=args.dense_layers,
            moe_layers=args.moe_layers,
            n_experts=args.n_experts,
            top_k=args.top_k,
            max_seq_len=args.max_seq_len or args.seq_len,
            lb_coef=args.lb_coef,
            router_lr_ratio=args.router_lr_ratio,
        )

    print_header(
        f"M2.3 MoE DQT Transformer TinyStories (GO/NO-GO ppl<20): "
        f"d={cfg['d_model']} L={cfg['n_layers']} "
        f"(dense {cfg['dense_layers']} + MoE {cfg['moe_layers']}x{cfg['n_experts']}e/"
        f"top-{cfg['top_k']}) H={cfg['n_heads']} ff={cfg['d_ff']} "
        f"lr={args.lr} (router {args.router_lr_ratio}x, lb {args.lb_coef}), "
        f"{args.epochs}ep, seed={args.seed}, "
        f"anneal@{int(100 * args.anneal_fraction)}%, "
        f"grad_ckpt={args.grad_checkpoint}"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Data (TinyStories) ─────────────────────────────────────────
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
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        data_meta["mode"] = "tinystories"
        cfg["vocab_size"] = data_meta["vocab_size"]
        seq_len = args.seq_len
    print(f"Data: {data_meta['mode']} | vocab={cfg['vocab_size']} | seq_len={seq_len}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    if "n_train_tokens" in data_meta:
        print(f"  Tokens: train={data_meta['n_train_tokens']:,} "
              f"val={data_meta['n_val_tokens']:,}")
    print()

    # ── Model ──────────────────────────────────────────────────────
    # OPT-3: bf16 weight_float buffers when --dtype bf16 (4→2 B/param).
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = dqt_gpt2_moe(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        dense_layers=cfg["dense_layers"],
        moe_layers=cfg["moe_layers"],
        n_experts=cfg["n_experts"],
        top_k=cfg["top_k"],
        max_seq_len=cfg["max_seq_len"],
        dropout=args.dropout,
        use_grad_checkpointing=args.grad_checkpoint,
        device=device,
        dtype=dtype,
    )
    n_total = count_parameters(model)
    n_ternary = count_ternary_weights(model)
    # Active params per token: all float params are active (embedding + norms
    # + routers); dense/attention/LM-head ternary are active; only the MoE
    # expert ternary scales by top_k / n_experts (grouped execution).
    moe_expert_ternary = sum(
        block.moe_ffn.count_parameters()["experts"]
        for block in model.blocks
        if isinstance(block, TernaryDQTMoETransformerBlock)
    )
    active_ternary = (n_ternary - moe_expert_ternary) + moe_expert_ternary * (
        cfg["top_k"] / cfg["n_experts"]
    )
    n_active = int(round((n_total - n_ternary) + active_ternary))
    print(f"Model: d_model={cfg['d_model']}, L={cfg['n_layers']} "
          f"(dense {cfg['dense_layers']} + MoE {cfg['moe_layers']}), "
          f"n_heads={cfg['n_heads']}, d_ff={cfg['d_ff']}, "
          f"n_experts={cfg['n_experts']}, top_k={cfg['top_k']}")
    print(f"  Float parameters:  {n_total - n_ternary:,}")
    print(f"  Ternary weights:   {n_ternary:,}  (int8, total)")
    print(f"  Active / token:    ~{n_active:,} (top-{cfg['top_k']}/{cfg['n_experts']})")
    print(f"  Total:             {n_total:,}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print()

    # ── Optimizer + scheduler ──────────────────────────────────────
    # M2.3 design (memory + E019 MoE lesson):
    #   1. The float token embedding trains with plain SGD (no AdamW moments
    #      — "embedding χωρίς AdamW", saves ~0.4 GB at d=1024+).
    #   2. The MoE routers are FLOAT nn.Linear with their OWN (0.1x) LR — a
    #      separate AdamW param group (E019: slow router is REQUIRED to
    #      avoid expert collapse in the first epoch).
    #   3. Everything else (DQT float buffers, RMSNorm scales, LM head) is
    #      AdamW at the base lr.
    if args.embed_adamw:
        # OPT-2: 8-bit AdamW (states 8→2 B/param). Falls back to fp32.
        optimizer = make_adamw(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        emb_optimizer = None
    else:
        emb_params = [model.token_embedding.weight]
        router_params = [
            p for n, p in model.named_parameters() if "router" in n
        ]
        main_params = [
            p
            for n, p in model.named_parameters()
            if "router" not in n and n != "token_embedding.weight"
        ]
        # OPT-2: 8-bit AdamW for DQT params + float routers (two param
        # groups, slow router 0.1x); embedding stays plain SGD.
        optimizer = make_adamw(
            [
                {"params": main_params, "lr": args.lr},
                {
                    "params": router_params,
                    "lr": args.lr * args.router_lr_ratio,
                },
            ],
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        emb_optimizer = torch.optim.SGD(emb_params, lr=args.lr)
    emb_scheduler = None
    total_steps = args.max_steps or args.epochs * len(train_loader)
    scheduler = make_cosine_warmup_scheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
    )
    if emb_optimizer is not None:
        emb_scheduler = make_cosine_warmup_scheduler(
            emb_optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps
        )

    # ── Checkpoint dir (per-seed so seeds don't overwrite each other) ──
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        args.output_dir, "checkpoints", f"seed{args.seed}"
    )
    # Status file for the `status` shell command.
    status_file = os.path.join(checkpoint_dir, "status.json")

    # ── Resume (pause/resume support) ──────────────────────────────
    orig_epochs = args.epochs
    start_step, start_epoch = 0, 1
    best_val_ppl, best_step = float("inf"), 0
    if args.resume:
        try:
            r = load_checkpoint(
                args.resume,
                checkpoint_dir,
                model,
                optimizer,
                scheduler,
                device,
                emb_optimizer=emb_optimizer,
                emb_scheduler=emb_scheduler,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        start_step, start_epoch = r["step"], r["epoch"] + 1
        best_val_ppl, best_step = r["best_val_ppl"], r["best_step"]
        if start_step >= total_steps:
            print(
                f"Checkpoint is already at/past the step budget "
                f"({total_steps} steps). Nothing to do."
            )
            sys.exit(0)
        # Resume must run until the ORIGINAL step budget — the checkpoint may
        # be mid-epoch, so bounding training with max_steps is what keeps the
        # total training length correct. The epoch label becomes cosmetic; the
        # outer epoch loop gets enough passes to reach the budget.
        args.max_steps = total_steps
        args.epochs = max(orig_epochs, start_epoch + 2)
        print(
            f"  ↩️ Resume: continuing from step {start_step} towards the step "
            f"budget of {total_steps} ({total_steps - start_step} steps "
            f"remaining, anneal at step {int(total_steps * args.anneal_fraction)})."
        )

    # ── Train ──────────────────────────────────────────────────────
    print("Training...")
    print()
    install_pause_handlers()  # Ctrl+C / SIGTERM / SIGUSR1 -> graceful checkpointed pause
    try:
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
            lb_coef=args.lb_coef,
            val_every=args.val_every,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            keep_last_checkpoints=args.keep_last_checkpoints,
            status_file=status_file,
            progress_every=args.progress_every,
            pause_file=args.pause_file,
            seed=args.seed,
            emb_optimizer=emb_optimizer,
            emb_scheduler=emb_scheduler,
            start_step=start_step,
            start_epoch=start_epoch,
            best_val_ppl=best_val_ppl,
            best_step=best_step,
            use_autocast=args.dtype == "bf16" and device.type == "cuda",
        )
    except KeyboardInterrupt:
        # Pause requested (SIGINT/SIGTERM/SIGUSR1). The checkpoint was already
        # saved inside train_dqt_transformer; do NOT write a (partial) result
        # JSON, so the run can be resumed cleanly with --resume auto.
        print("\n⏹️  Training paused by request — no result JSON written.")
        print(
            f"  Resume: bash scripts/run_m2_3_dqt_moe.sh "
            f"resume {args.lr} {args.seed}"
        )
        print(f"  Checkpoints: {checkpoint_dir}")
        sys.exit(130)

    # ── Peak GPU memory ────────────────────────────────────────────
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()

    # ── Result dict ────────────────────────────────────────────────
    result = {
        "experiment": "m2_3_dqt_moe",
        "dataset": data_meta["mode"],
        "tokenizer": data_meta.get("tokenizer", "synthetic"),
        "seed": args.seed,
        "device": str(device),
        "config": cfg,
        "learning_rate": args.lr,
        "router_lr_ratio": args.router_lr_ratio,
        "lb_coef": args.lb_coef,
        "epochs": orig_epochs,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
        "anneal_fraction": args.anneal_fraction,
        "embed_adamw": bool(args.embed_adamw),
        "grad_checkpointing": bool(args.grad_checkpoint),
        "architecture": (
            f"MoE DQT Transformer: emb({cfg['vocab_size']}->{cfg['d_model']}) "
            f"+ {cfg['dense_layers']}x[dense Attn+FFN] + "
            f"{cfg['moe_layers']}x[Attn + MoE({cfg['n_experts']}e, top-{cfg['top_k']})] "
            f"+ RMSNorm + DQT LM Head"
        ),
        "method": (
            "Direct Quantized Training (DQT): int8 ternary weights + stochastic "
            f"rounding annealed to deterministic sign for final "
            f"{int(100 * (1 - args.anneal_fraction))}% of steps; "
            f"MoE routers float @ {args.router_lr_ratio}x lr; "
            f"Switch-Transformer aux loss @ lb_coef={args.lb_coef}"
        ),
        "n_float_params": n_total - n_ternary,
        "n_ternary_weights": n_ternary,
        "n_total_params": n_total,
        "n_active_params": n_active,
        "active_fraction": float(n_active) / max(n_total, 1),
        "perplexity_gate": PPL_GATE,
        "expert_utilization": collect_expert_utilization(model),
        "peak_gpu_memory_mb": peak_mem_mb,
        "resumed": args.resume is not None,
        "resume_step": start_step if args.resume else 0,
        **results,
    }

    # ── Save result JSON + mark COMPLETED in the status file ──────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"results_m2_3_dqt_moe_lr{args.lr}_seed{args.seed}.json",
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    write_status_file(
        status_file,
        seed=args.seed,
        lr=args.lr,
        step=int(results["steps_trained"]),
        total_steps=total_steps,
        epoch=orig_epochs,
        loss=float(results["final_train_loss"]),
        ppl=float(results["final_val_ppl"]),
        tok_per_s=0.0,
        gpu_mem_gb=_gpu_mem_gb(device),
        status="COMPLETED",
    )

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
    print("  Expert utilization (per MoE layer, selection fractions):")
    for layer in collect_expert_utilization(model):
        fracs = " ".join(f"{f:.3f}" for f in layer["selection_fractions"])
        print(
            f"    MoE L{layer['layer']}: [{fracs}]  "
            f"balance={layer['balance_ratio']:.2f} (min {layer['min_share']:.3f})"
        )
    print()
    verdict = (
        f"GO ✅ (ppl < {PPL_GATE})"
        if results["best_val_ppl"] < PPL_GATE
        else f"NO-GO 🔴 (ppl >= {PPL_GATE})"
    )
    print(f"  M2.3 Verdict:          ppl={results['best_val_ppl']:.2f}  →  {verdict}")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
