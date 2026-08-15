#!/usr/bin/env python3
"""E035 — Ternary LoRA experiment runner (Step 2.2).

The on-device product test: does 2-bit ternary adaptation preserve **≥90%**
of float gated-LoRA quality at **16× smaller** storage? E034 proved the
surprise gate (gated lr ``η·M_t``) makes float LoRA selective (single-domain
Δppl +0.902, two-domain BT −0.009). E035 keeps the **exact same** gated-LoRA
protocol (rank-1 344,064-param budget, o_proj+down_proj, AdamW wd=0, WikiText
warmup M=0 → PubMed 100K, 3 seeds, eval window 512/stride 256) and changes
only the **adapter representation** to ternary {-1, 0, +1}:

* ``--ternary float`` — E034 gated LoRA (base; also a bit-identical
  determinism check against the E034 result).
* ``--ternary ta`` — POST-TRAINING quantization (CAT-Q style): load the E034
  float gated checkpoint (``--float-ckpt``), quantize A and B to ternary with
  per-matrix scale factors, then either eval immediately (``--calib-steps 0``,
  ``ta_q``) or run a short calibration fine-tune (``--calib-steps N``, STE
  through the ternary weights, constant lr ``--calib-lr``, re-quantize —
  ``ta_qft``).
* ``--ternary tb`` — DQT-style training (``ste_dqt.py`` mechanics on the
  adapter weights): int8 ternary buffers + float accumulation buffers via a
  custom autograd Function (STE routing), ``apply_stochastic_rounding()``
  after each optimizer step, trainable per-matrix scales, gated lr.
* ``--ternary tc`` — STE with latent scores (``ste_linear.py`` mechanics):
  float latent scores + ``sign()`` forward with identity backward, trainable
  per-matrix scales, gated lr.

``--phases 2`` runs the sequential two-domain stream (WikiText warmup →
PubMed 100K → CNN/DailyMail 100K) for the selectivity test. T-A reuses the
E034 **two-domain** float checkpoint in that case.

Each cell reports: Δppl (target/source), forgetting %, M-trace, per-step
training overhead, the 2-bit packed storage size on disk (vs float32), and
the full protocol-schema metrics.

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e035_lora \\
        --ternary tb --phases 1 --tag tb --budget-tokens 100000 --seed 42

Output: ``results/brain/e035/{mshort}_pubmed_{budget}_{tag}_seed{seed}.json``
(phases=2 adds ``_cnn`` to the filename).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from collections import OrderedDict

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── quiet HF cache chatter ─────────────────────────────────────────
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from ph_neuro.brain.brain_wrapper import check_gpu_free, gpu_free_mb  # noqa: E402
from ph_neuro.brain.datasets import (  # noqa: E402
    cnn_dailymail_eval_ids,
    cnn_dailymail_train_ids,
    make_combined_batch_iter,
    make_three_domain_batch_iter,
    pubmed_eval_ids,
    pubmed_train_ids,
    wikitext_ids,
)
from ph_neuro.brain.lora import (  # noqa: E402
    LoRAAdapter,
    TernaryLoRAAdapter,
    all_lora_weights,
    build_lora_adapters,
    build_ternary_lora_adapters,
    n_lora_params,
    pack_ternary_adapters,
    ternary_storage_report,
)
from ph_neuro.brain.modulator import SurpriseModulator  # noqa: E402
from ph_neuro.brain.stats import block_paired_stats  # noqa: E402
from ph_neuro.examples.run_e034_lora import (  # noqa: E402
    compute_effective_lr,
    default_min_free_gb,
    disable_triton_bmm,
    frozen_eval,
    load_model,
    model_short,
    plastic_eval,
)

log = logging.getLogger("e035")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 1024

# E031 locked surprise defaults (identical to E034).
DEFAULT_S0 = 0.05
DEFAULT_K = 60.0
DEFAULT_M_MAX = 1.0
DEFAULT_ALPHA = 0.99

# T-A calibration defaults (short, constant-lr fine-tune of the quantized
# adapter to recover quantization noise — not an adaptation).
DEFAULT_CALIB_STEPS = 20
DEFAULT_CALIB_LR = 1e-4

CHECKPOINT_FORMAT = "ph_neuro_e035_lora_checkpoint"

TERNARY_MODES = ("float", "ta", "tb", "tc")


def setup_logging(tag: str, budget: int, seed: int, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"e035_{tag}_budget{budget}_seed{seed}.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return path


# ── E035 checkpointing (format-tagged; mirrors E034) ───────────────


def _save_checkpoint(path: str, step: int, adapters, optimizer, modulator,
                     config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "plastic": {i: ad.state_dict() for i, ad in enumerate(adapters)},
        "optimizer": optimizer.state_dict(),
        "modulator": modulator.state_dict() if modulator is not None else None,
        # RNG state → T-B's stochastic rounding resumes bit-exactly (a
        # checkpoint without the RNG state would re-draw different rounding
        # noise after a resume, breaking determinism).
        "rng_state": torch.get_rng_state().cpu(),
        "config": config,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _resume(checkpoint_dir: str, steps: int, adapters, optimizer,
            modulator) -> int:
    """Return the step to resume from (largest checkpoint < ``steps``, or 0).

    Restores plastic + optimizer + EMA (+ RNG for T-B's stochastic rounding).
    "Already complete" is decided by the **caller** via the result JSON, NOT
    by the presence of a final checkpoint: a crash between the final
    checkpoint and the JSON write must not make a cell with no result look
    complete (E035 smoke caught this — a stale final checkpoint from a
    cancelled run caused cells to be skipped without producing results).
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return 0
    import glob
    import re

    best_step, best_path = -1, None
    for p in glob.glob(os.path.join(checkpoint_dir, "brain_ckpt_step*.pt")):
        m = re.match(r".*brain_ckpt_step(\d+)\.pt$", p)
        if not m:
            continue
        n = int(m.group(1))
        if n < steps and n > best_step:
            best_step, best_path = n, p
    if best_path is None:
        return 0
    ckpt = torch.load(best_path, weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        log.warning("unrecognized checkpoint format in %s — ignoring", best_path)
        return 0
    plastic = ckpt["plastic"]
    for i, ad in enumerate(adapters):
        ad.load_state_dict(plastic[str(i) if str(i) in plastic else i])
    optimizer.load_state_dict(ckpt["optimizer"])
    if modulator is not None and ckpt.get("modulator") is not None:
        modulator.load_state_dict(ckpt["modulator"])
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"].to(torch.uint8).cpu())
    log.info("resumed from step %d (%s)", best_step, best_path)
    return best_step


def load_float_checkpoint_into(ckpt_path: str, adapters) -> None:
    """Load an E034 float LoRA checkpoint's A/B into float-phase adapters."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    plastic = ckpt["plastic"]
    for i, ad in enumerate(adapters):
        st = plastic[str(i) if str(i) in plastic else i]
        ad.A.data.copy_(st["A"].to(ad.device))
        ad.B.data.copy_(st["B"].to(ad.device))
    log.info("loaded float adapter state from %s (%d adapters)",
             ckpt_path, len(adapters))


# ── training loop (gated lr, checkpointing, T-B rounding) ─────────


def train_gated(
    args, model, adapters, optimizer, modulator, const_scale,
    data_iter, start_step, total_steps, warmup_steps, ckpt_dir,
) -> list[dict]:
    """Run the E034 gated-lr training loop; returns train_metrics.

    Mirrors E034 exactly (gate → effective lr → AdamW step), plus:
    * per-step wall time in ``train_metrics`` (for the overhead comparison);
    * ``apply_stochastic_rounding()`` after each step for ``tb`` adapters;
    * full checkpoint save/resume (format-tagged, SIGINT/SIGTERM safe).
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def checkpointed_forward(ids, mask):
        def _fwd(i, m):
            return model(input_ids=i, attention_mask=m, use_cache=False)

        if args.no_checkpoint:
            return _fwd(ids, mask)
        return torch.utils.checkpoint.checkpoint(_fwd, ids, mask, use_reentrant=False)

    prev_handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    current_step = [start_step]

    def _on_signal(signum, frame):
        log.warning("signal %s received — saving checkpoint at step %d",
                    signum, current_step[0])
        _save_checkpoint(
            os.path.join(ckpt_dir, f"brain_ckpt_step{current_step[0]}.pt"),
            current_step[0], adapters, optimizer, modulator, {"tag": args.tag},
        )
        os._exit(130)  # noqa: PLR1722

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    train_metrics: list[dict] = []
    optimizer.zero_grad()
    t0 = time.time()
    try:
        for step in range(start_step, total_steps):
            current_step[0] = step
            t_step0 = time.time()
            batch = next(data_iter)
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask")
            mask = torch.ones_like(ids) if mask is None else mask.to(device)

            logits = checkpointed_forward(ids, mask).logits
            V = logits.size(-1)
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].to(torch.float32).reshape(-1, V),
                ids[..., 1:].reshape(-1),
            )

            # ── surprise gate → effective lr ──────────────────────
            surprise_s, M = 0.0, 1.0
            if modulator is not None:
                surprise_s, M = modulator.update(loss)
            in_warmup = step < warmup_steps
            effective_lr, do_update = compute_effective_lr(
                "surprise", args.lr, M, const_scale, in_warmup,
            )
            flip_stats: dict = {}
            if do_update:
                optimizer.param_groups[0]["lr"] = effective_lr
                loss.backward()
                if (step - start_step + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    # T-B: stochastic-round the float buffers into int8.
                    for ad in adapters:
                        if isinstance(ad, TernaryLoRAAdapter):
                            fs = ad.apply_after_step()
                            if fs:
                                for k, v in fs.items():
                                    flip_stats[k] = float(v)

            mean_abs = sum(ad.mean_abs() for ad in adapters) / len(adapters)
            train_metrics.append({
                "step": step,
                "loss": float(loss.item()),
                "ema_loss": float(modulator.ema_loss) if modulator is not None and
                            modulator.ema_loss is not None else float(loss.item()),
                "surprise_s": float(surprise_s),
                "modulator_M": float(M),
                "effective_lr": float(effective_lr),
                "mean_abs_b": mean_abs,
                "step_time_s": float(time.time() - t_step0),
                "tokens_seen": (step + 1) * ids.numel(),
                **{f"flip_{k}": v for k, v in flip_stats.items()},
            })
            if (step + 1) % max(args.grad_accum * 100, 1) == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"brain_ckpt_step{step + 1}.pt"),
                    step + 1, adapters, optimizer, modulator, {"tag": args.tag},
                )
            if step % 10 == 0 or step == total_steps - 1:
                m = train_metrics[-1]
                log.info(
                    "step %d/%d loss=%.4f ema=%.4f s=%+.4f M=%.3f lr=%.3e |w|=%.3e "
                    "%.2fs/step",
                    step, total_steps, m["loss"], m["ema_loss"], m["surprise_s"],
                    m["modulator_M"], m["effective_lr"], m["mean_abs_b"],
                    m["step_time_s"],
                )
    finally:
        signal.signal(signal.SIGINT, prev_handlers[0])
        signal.signal(signal.SIGTERM, prev_handlers[1])

    _save_checkpoint(
        os.path.join(ckpt_dir, f"brain_ckpt_step{total_steps}.pt"),
        total_steps, adapters, optimizer, modulator, {"tag": args.tag},
    )
    log.info("learning done in %.1f s (%d steps)", time.time() - t0, len(train_metrics))
    return train_metrics


def run_calibration(args, model, tok, adapters, ckpt_dir, seed) -> list[dict]:
    """T-A short calibration fine-tune of the quantized adapter (ta_qft).

    Phase ``calib``: forward uses ``sign(float A/B)`` (STE), scales fixed,
    constant lr ``args.calib_lr`` for ``args.calib_steps`` PubMed steps.
    Re-quantizes the float latents to the int8 snapshot at the end. Returns
    the calibration train metrics (small).
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for ad in adapters:
        ad.set_phase("calib")
    params = [p for ad in adapters for p in ad.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.calib_lr, weight_decay=0.0)
    wiki_ids = wikitext_ids("train", tok)
    pub_ids = pubmed_train_ids(tok)
    data_iter = make_combined_batch_iter(
        wiki_ids, pub_ids, 0, args.batch_size, args.seq_len, seed
    )

    def checkpointed_forward(ids, mask):
        def _fwd(i, m):
            return model(input_ids=i, attention_mask=m, use_cache=False)

        return torch.utils.checkpoint.checkpoint(_fwd, ids, mask, use_reentrant=False)

    metrics: list[dict] = []
    optimizer.zero_grad()
    for step in range(args.calib_steps):
        batch = next(data_iter)
        ids = batch["input_ids"].to(device)
        mask = batch.get("attention_mask")
        mask = torch.ones_like(ids) if mask is None else mask.to(device)
        logits = checkpointed_forward(ids, mask).logits
        V = logits.size(-1)
        loss = torch.nn.functional.cross_entropy(
            logits[..., :-1, :].to(torch.float32).reshape(-1, V),
            ids[..., 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        metrics.append({"step": step, "loss": float(loss.item())})
        if step % 10 == 0 or step == args.calib_steps - 1:
            log.info("calib step %d/%d loss=%.4f", step, args.calib_steps, loss.item())
    # Re-quantize the fine-tuned float latents → final int8 snapshot.
    for ad in adapters:
        ad.quantize()
    log.info("calibration done (%d steps, lr=%.2e); re-quantized",
             args.calib_steps, args.calib_lr)
    return metrics


def _adapter_weights(adapters) -> torch.Tensor:
    """Concatenated effective A/B weights (float or ternary, per mode)."""
    parts: list[torch.Tensor] = []
    for ad in adapters:
        if isinstance(ad, TernaryLoRAAdapter):
            if ad.mode == "ta" and ad.phase in ("float", "calib"):
                parts.append(ad.A.detach().flatten())
                parts.append(ad.B.detach().flatten())
            else:
                A_tern, B_tern, _, _ = ad.ternary_snapshot()
                parts.append(A_tern.float().flatten())
                parts.append(B_tern.float().flatten())
        else:
            parts.append(ad.A.detach().flatten())
            parts.append(ad.B.detach().flatten())
    return torch.cat(parts)


# ── arg parser ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--ternary", choices=TERNARY_MODES, default="tb")
    p.add_argument("--phases", type=int, choices=(1, 2), default=1)
    p.add_argument("--tag", required=True)
    p.add_argument("--budget-tokens", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-3)  # η (E034 base)
    p.add_argument("--calib-steps", type=int, default=0,
                   help="T-A: calibration fine-tune steps (0 = ta_q, >0 = ta_qft)")
    p.add_argument("--calib-lr", type=float, default=DEFAULT_CALIB_LR)
    p.add_argument("--float-ckpt", default=None,
                   help="T-A: E034 float gated-LoRA checkpoint to reuse")
    p.add_argument("--s0", type=float, default=DEFAULT_S0)
    p.add_argument("--k", type=float, default=DEFAULT_K)
    p.add_argument("--m-max", type=float, default=DEFAULT_M_MAX)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-checkpoint", action="store_true", help="disable grad checkpointing")
    p.add_argument("--output-dir", default="results/brain/e035")
    p.add_argument("--log-dir", default="logs/brain/e035")
    p.add_argument("--frozen-cache-dir", default="results/brain/e035/cache")
    p.add_argument("--gpu-policy", choices=("exit", "wait", "warn"), default="exit")
    p.add_argument("--device", default=None)
    p.add_argument("--min-free-gb", type=float, default=None)
    p.add_argument("--eval-window", type=int, default=512)
    p.add_argument("--eval-stride", type=int, default=256)
    p.add_argument("--eval-pubmed-tokens", type=int, default=500_000)
    p.add_argument("--no-deregister", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    mshort = model_short(args.model)
    seed = args.seed
    output_dir = os.path.abspath(args.output_dir)
    log_path = setup_logging(args.tag, args.budget_tokens, seed, os.path.abspath(args.log_dir))
    log.info(
        "E035 start ternary=%s phases=%d tag=%s rank=%d budget=%d seed=%d "
        "model=%s calib_steps=%d (%s)",
        args.ternary, args.phases, args.tag, args.rank, args.budget_tokens, seed,
        args.model, args.calib_steps, log_path,
    )
    torch.manual_seed(seed)

    adapt_steps = math.ceil(args.budget_tokens / (args.batch_size * args.seq_len))
    total_steps = args.warmup_steps + args.phases * adapt_steps
    phase1_steps = adapt_steps
    log.info("adapt steps/phase=%d phase1=%d total=%d", adapt_steps, phase1_steps, total_steps)

    if not args.no_deregister:
        disable_triton_bmm()

    min_free_gb = args.min_free_gb or default_min_free_gb(args.model)
    log.info("GPU pre-check: need >= %.1f GiB free", min_free_gb)
    check_gpu_free(min_free_gb, args.gpu_policy, log)
    free_mb = gpu_free_mb()
    log.info("GPU free: %s MiB", free_mb if free_mb is not None else "n/a")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.model, device)
    model.requires_grad_(False)

    if args.ternary == "float":
        adapters = build_lora_adapters(model, args.rank, device)
    else:
        adapters = build_ternary_lora_adapters(model, args.rank, device, mode=args.ternary)
    n_params = n_lora_params(adapters)
    log.info(
        "%d ternary-mode(%s) adapters, %d trainable params (%.1f KB fp32) — "
        "budget match to E034 (rank %d)",
        len(adapters), args.ternary, n_params, n_params * 4 / 1024, args.rank,
    )

    # ── optimizer + modulator ─────────────────────────────────────
    params = [p for ad in adapters for p in ad.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    modulator = SurpriseModulator(
        mode="surprise_ema", alpha=args.alpha, s0=args.s0, k=args.k, M_max=args.m_max,
    )

    ckpt_dir = os.path.join(
        output_dir, "checkpoints",
        f"{args.ternary}_{args.phases}p_{args.tag}_budget{args.budget_tokens}_seed{seed}",
    )

    # Result JSON path — computed early because "already complete" is decided
    # by its existence (not by a stale final checkpoint, see _resume).
    budget_tag = f"{args.budget_tokens // 1000}k"
    fname = f"{mshort}_pubmed_{budget_tag}_{args.tag}_seed{seed}.json"
    if args.phases == 2:
        fname = f"{mshort}_pubmed_cnn_{budget_tag}_{args.tag}_seed{seed}.json"
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        log.info("already complete (result JSON exists): %s", out_path)
        return 0

    train_metrics: list[dict] = []
    calib_metrics: list[dict] = []
    load_note: str | None = None

    if args.ternary == "ta":
        # T-A: reuse a float gated-LoRA checkpoint (or train float from scratch).
        if args.float_ckpt and os.path.exists(args.float_ckpt):
            load_float_checkpoint_into(args.float_ckpt, adapters)
            load_note = f"reused float ckpt {args.float_ckpt}"
            for ad in adapters:
                ad.quantize()
        else:
            if args.float_ckpt:
                log.warning("float checkpoint %s not found — training float from scratch",
                            args.float_ckpt)
            start_step = _resume(ckpt_dir, total_steps, adapters, optimizer, modulator)
            wiki_ids = wikitext_ids("train", tok)
            pub_ids = pubmed_train_ids(tok)
            data_iter = make_combined_batch_iter(
                wiki_ids, pub_ids, args.warmup_steps, args.batch_size, args.seq_len, seed,
            )
            for _ in range(start_step):
                next(data_iter)
            train_metrics = train_gated(
                args, model, adapters, optimizer, modulator, None,
                data_iter, start_step, total_steps, args.warmup_steps, ckpt_dir,
            )
            load_note = "float trained from scratch (E034-identical protocol)"
            for ad in adapters:
                ad.quantize()
        if args.calib_steps > 0:
            calib_metrics = run_calibration(args, model, tok, adapters, ckpt_dir, seed)
    else:
        # float / tb / tc: E034 gated training loop.
        start_step = _resume(ckpt_dir, total_steps, adapters, optimizer, modulator)
        wiki_ids = wikitext_ids("train", tok)
        pub_ids = pubmed_train_ids(tok)
        if args.phases == 2:
            cnn_ids = cnn_dailymail_train_ids(tok)
            data_iter = make_three_domain_batch_iter(
                wiki_ids, pub_ids, cnn_ids, args.warmup_steps, phase1_steps,
                args.batch_size, args.seq_len, seed,
            )
        else:
            data_iter = make_combined_batch_iter(
                wiki_ids, pub_ids, args.warmup_steps, args.batch_size, args.seq_len, seed,
            )
        for _ in range(start_step):
            next(data_iter)
        train_metrics = train_gated(
            args, model, adapters, optimizer, modulator, None,
            data_iter, start_step, total_steps, args.warmup_steps, ckpt_dir,
        )
        if args.ternary == "float":
            load_note = "float gated LoRA (E034 base)"

    # ── evaluation ─────────────────────────────────────────────────
    log.info("tokenizing eval corpora…")
    wiki_test = wikitext_ids("test", tok)
    pub_test = pubmed_eval_ids(tok, max_tokens=args.eval_pubmed_tokens)
    if args.phases == 2:
        cnn_test = cnn_dailymail_eval_ids(tok, max_tokens=args.eval_pubmed_tokens)
    log.info("eval tokens: wiki_test=%d, pubmed_eval=%d%s",
             wiki_test.numel(), pub_test.numel(),
             f", cnn_eval={cnn_test.numel()}" if args.phases == 2 else "")

    def eval_domain(domain_ids: torch.Tensor, domain: str):
        frozen = frozen_eval(model, tok, domain_ids, domain, args.frozen_cache_dir,
                             args.eval_window, args.eval_stride, adapters)
        plastic = plastic_eval(model, domain_ids, args.eval_window, args.eval_stride)
        stats = block_paired_stats(
            frozen["per_block"]["nll"],
            plastic["per_block"]["nll"],
            frozen["per_block"]["tokens"],
        )
        return frozen, plastic, stats

    src_frozen, src_plastic, src_stats = eval_domain(wiki_test, "wikitext2")
    tgt_frozen, tgt_plastic, tgt_stats = eval_domain(pub_test, "pubmed")

    source_ppl_delta = src_plastic["ppl"] - src_frozen["ppl"]  # + = forgetting
    target_ppl_delta = tgt_frozen["ppl"] - tgt_plastic["ppl"]  # + = improved
    forgetting_pct = (src_plastic["ppl"] / src_frozen["ppl"] - 1.0) * 100.0

    # ── gate / timing / storage statistics ─────────────────────────
    gate_stats: dict = {}
    if train_metrics:
        ms = [m["modulator_M"] for m in train_metrics]
        eff = [m["effective_lr"] for m in train_metrics]
        times = [m["step_time_s"] for m in train_metrics]
        adapt_ms = ms[args.warmup_steps:] if len(ms) > args.warmup_steps else ms
        gate_stats = {
            "mean_surprise_M": float(sum(ms) / len(ms)),
            "mean_M_adapt": float(sum(adapt_ms) / max(len(adapt_ms), 1)),
            "pct_steps_M_gt_05": float(sum(1.0 for m in ms if m > 0.5) / len(ms)),
            "final_M": float(ms[-1]),
            "effective_mean_lr": float(sum(eff) / len(eff)),
            "n_steps": len(ms),
            "mean_step_time_s": float(sum(times) / len(times)),
        }
        if args.ternary == "tb":
            a_flips = [m.get("flip_A_flip_rate", 0.0) for m in train_metrics]
            b_flips = [m.get("flip_B_flip_rate", 0.0) for m in train_metrics]
            gate_stats["mean_A_flip_rate"] = float(sum(a_flips) / len(a_flips))
            gate_stats["mean_B_flip_rate"] = float(sum(b_flips) / len(b_flips))

    if calib_metrics:
        gate_stats["calib_mean_loss"] = float(
            sum(m["loss"] for m in calib_metrics) / len(calib_metrics))
        gate_stats["calib_loss_first"] = float(calib_metrics[0]["loss"])
        gate_stats["calib_loss_last"] = float(calib_metrics[-1]["loss"])
        gate_stats["calib_n_steps"] = len(calib_metrics)

    # ── storage (2-bit packed vs float32) ──────────────────────────
    if args.ternary == "float":
        storage = {
            "n_params": n_params,
            "float32_bytes": n_params * 4,
            "packed_bytes": n_params * 4,
            "reduction_factor": 1.0,
            "packed": False,
            "disk_bytes": n_params * 4,
        }
    else:
        packed = pack_ternary_adapters(adapters)
        report = ternary_storage_report(adapters, packed)
        os.makedirs(output_dir, exist_ok=True)
        packed_path = os.path.join(
            output_dir, f"{mshort}_pubmed_{args.tag}_seed{seed}.ternary2bit"
        )
        with open(packed_path, "wb") as fh:
            fh.write(packed.cpu().numpy().tobytes())
        disk_bytes = os.path.getsize(packed_path)
        storage = {
            **report,
            "packed": True,
            "disk_bytes": int(disk_bytes),
            "packed_path": packed_path,
        }
        log.info("ternary storage: fp32=%.2f KB, packed=%.2f KB (disk=%d B), "
                 "%dx smaller", report["float32_bytes"] / 1024,
                 report["packed_bytes"] / 1024, disk_bytes,
                 report["reduction_factor"])

    allw = _adapter_weights(adapters)
    plastic_weights = {
        "count": n_params,
        "bytes": n_params * 4,
        "mean_magnitude": float(allw.abs().mean()),
        "max_magnitude": float(allw.abs().max()),
        "sparsity": float((allw.abs() < 1e-8).float().mean()),
    }

    result = {
        "experiment": "e035_ternary_lora",
        "step": "2.2",
        "tag": args.tag,
        "adapter": args.ternary,
        "ternary_mode": None if args.ternary == "float" else args.ternary,
        "calib_steps": args.calib_steps,
        "float_ckpt_used": bool(args.float_ckpt and os.path.exists(args.float_ckpt)),
        "load_note": load_note,
        "gate": "surprise_ema",
        "phases": args.phases,
        "phase2_domain": "cnn_dailymail" if args.phases == 2 else None,
        "rank": args.rank,
        "model": args.model,
        "model_short": mshort,
        "modulator": {
            "mode": "surprise_ema", "alpha": args.alpha, "s0": args.s0,
            "k": args.k, "M_max": args.m_max,
        },
        "source_domain": "wikitext2",
        "target_domain": "pubmed",
        "adaptation_tokens": args.budget_tokens,
        "adaptation_steps": adapt_steps,
        "phase1_steps": phase1_steps,
        "phase2_tokens": args.budget_tokens if args.phases == 2 else None,
        "warmup_steps": args.warmup_steps,
        "seed": seed,
        "lr": args.lr,
        "decay_rate": 0.0,
        "eval": {"window": args.eval_window, "stride": args.eval_stride,
                 "aggregation": "unweighted"},
        "metrics": {
            "source_ppl_frozen": src_frozen["ppl"],
            "source_ppl_plastic": src_plastic["ppl"],
            "source_ppl_delta": source_ppl_delta,
            "target_ppl_frozen": tgt_frozen["ppl"],
            "target_ppl_plastic": tgt_plastic["ppl"],
            "target_ppl_delta": target_ppl_delta,
            "target_ppl_delta_ci95": tgt_stats["delta_ppl_ci95"],
            "target_block_paired_t": tgt_stats["paired_t"],
            "target_block_paired_p": tgt_stats["paired_p"],
            "target_block_cohens_d": tgt_stats["cohens_d"],
            "source_block_paired_t": src_stats["paired_t"],
            "source_block_paired_p": src_stats["paired_p"],
            "source_block_cohens_d": src_stats["cohens_d"],
            "forgetting_pct": forgetting_pct,
            **gate_stats,
        },
        "storage": storage,
        "plastic_weights": plastic_weights,
        "n_target_blocks": tgt_stats["n_blocks"],
        "n_source_blocks": src_stats["n_blocks"],
        "train_metrics": train_metrics,
        "calib_metrics": calib_metrics,
    }

    if args.phases == 2:
        cnn_frozen, cnn_plastic, cnn_stats = eval_domain(cnn_test, "cnn_dailymail")
        result["phase2_metrics"] = {
            "phase2_domain": "cnn_dailymail",
            "phase2_ppl_frozen": cnn_frozen["ppl"],
            "phase2_ppl_plastic": cnn_plastic["ppl"],
            "phase2_ppl_delta": cnn_frozen["ppl"] - cnn_plastic["ppl"],  # + = improved
            "phase2_ppl_delta_ci95": cnn_stats["delta_ppl_ci95"],
            "phase2_block_paired_t": cnn_stats["paired_t"],
            "phase2_block_paired_p": cnn_stats["paired_p"],
            "phase2_block_cohens_d": cnn_stats["cohens_d"],
            "n_blocks": cnn_stats["n_blocks"],
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    extra = ""
    if args.phases == 2:
        extra = (f" | phase2 Δppl={result['phase2_metrics']['phase2_ppl_delta']:+.3f}"
                 f" (ci {[f'{x:.3f}' for x in cnn_stats['delta_ppl_ci95']]})")
    log.info(
        "RESULT -> %s | ternary=%s target Δppl=%+.3f (ci %s) | source forgetting="
        "%+.3f%% | mean M=%.3f | %.2fs/step | storage=%dx%s",
        out_path, args.ternary, target_ppl_delta,
        [f"{x:.3f}" for x in tgt_stats["delta_ppl_ci95"]], forgetting_pct,
        gate_stats.get("mean_surprise_M", 0.0) or 0.0,
        gate_stats.get("mean_step_time_s", 0.0) or 0.0,
        storage.get("reduction_factor", 1.0), extra,
    )
    log.info("E035 complete ternary=%s phases=%d tag=%s seed=%d",
             args.ternary, args.phases, args.tag, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
