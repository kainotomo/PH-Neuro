#!/usr/bin/env python3
"""E034 — Surprise-Gated LoRA experiment runner (Step 2.1).

The first value-add of PH-Neuro's brain machinery on top of backprop LoRA:
reuse E032 Part D's minimal manual LoRA (o_proj + down_proj, rank 1 = 344,064
params, A ~ N(0,1/d_in), B = 0, AdamW wd=0.0) and replace the **constant lr**
with the **surprise-gated lr** — E031's one validated brain mechanism:

    L_t   = cross-entropy loss of the training step        # float32
    L̂_t  ← α·L̂_{t−1} + (1−α)·L_t                          # EMA, α = 0.99
    s_t   = (L_t − L̂_t) / L̂_t                              # relative deviation
    M_t   = M_max / (1 + exp(−k·(s_t − s₀)))               # sigmoid, float32
    lr_t  = η · M_t                                         # gated lr
    optimizer.param_groups[0]["lr"] = lr_t   (before each AdamW step)

Methods:
* ``plain`` — constant lr (E032 convention; learns during warmup). For the
  two-domain plain baseline (E032 single-domain numbers are reused).
* ``surprise`` — gated lr; **M = 0 during warmup** (no warmup learning, as in
  E031), then ``lr = η·M_t`` over the adapt phase(s). EMA runs continuously
  (warmup → phase 1 → phase 2), so each domain boundary opens a bounded
  plasticity window and in-domain steps anneal to near-zero.
* ``const_reduced`` — control: **no warmup learning** (matching ``surprise``),
  then constant ``lr = η·const_scale`` over the adapt phase. With
  ``const_scale = mean(M_t)`` of the gated run, this isolates "gating is just
  a lower average lr" — same total effective learning, only the temporal
  shape differs.

Phases:
* ``--phases 1`` — single-domain: WikiText warmup → PubMed (100K).
* ``--phases 2`` — sequential two-domain: WikiText warmup → PubMed (100K) →
  CNN/DailyMail (100K). The EMA runs continuously through both adapt phases
  (no reset at the second boundary).

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e034_lora \\
        --method surprise --phases 1 --tag gated --budget-tokens 100000 \\
        --seed 42

Output: ``results/brain/e034/{mshort}_pubmed_{budget}_{tag}_seed{seed}.json``
(phases=2 adds ``_{phase2_domain}`` to the filename).
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
    all_lora_weights,
    build_lora_adapters,
    n_lora_params,
)
from ph_neuro.brain.modulator import SurpriseModulator  # noqa: E402
from ph_neuro.brain.stats import block_paired_stats  # noqa: E402

log = logging.getLogger("e034")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 1024

# E031 locked surprise defaults.
DEFAULT_S0 = 0.05
DEFAULT_K = 60.0
DEFAULT_M_MAX = 1.0
DEFAULT_ALPHA = 0.99

CHECKPOINT_FORMAT = "ph_neuro_e034_lora_checkpoint"

METHODS = ("plain", "surprise", "const_reduced")


def model_short(model_id: str) -> str:
    if "SmolLM2-1.7B" in model_id:
        return "smolllm2_1p7b"
    if "gpt2" in model_id:
        return "gpt2_124m"
    return model_id.replace("/", "__")


def default_min_free_gb(model_id: str) -> float:
    if "SmolLM2" in model_id:
        return 6.0
    if "gpt2" in model_id:
        return 2.0
    return 4.0


def setup_logging(tag: str, budget: int, seed: int, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"e034_{tag}_budget{budget}_seed{seed}.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return path


def disable_triton_bmm() -> None:
    """Workaround for the no-C-compiler Triton fused-bmm failure (RoPE path)."""
    try:
        torch.backends.python_native.disable_operations("bmm")
        log.info("disabled torch native override for 'bmm' (Triton workaround)")
    except Exception as exc:  # noqa: BLE001 - API may differ across torch builds
        log.warning("could not disable 'bmm' native override: %s", exc)


def load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading tokenizer + model %s (bf16, eager attention)", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok


# ── eval with frozen cache ─────────────────────────────────────────


def _frozen_cache_path(cache_dir: str, mshort: str, domain: str) -> str:
    return os.path.join(cache_dir, f"frozen_{mshort}_{domain}.json")


def frozen_eval(model, tok, domain_ids: torch.Tensor, domain: str, cache_dir: str,
                window: int, stride: int, adapters: list[LoRAAdapter] | None = None) -> dict:
    """Frozen eval for a domain; cached to disk (seed-independent, reused).

    Reuses the E031/E032-produced cache when present (wiki + pubmed are
    bit-identical across experiments); CNN/DailyMail is computed once here.
    When computing fresh, the LoRA hooks are disabled so the result is the
    true frozen model (bit-identical to the raw unwrapped model).
    """
    cache_path = _frozen_cache_path(cache_dir, model_short(
        model.config._name_or_path or model.config.model_type
    ), domain)
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            cached = json.load(fh)
        log.info("frozen %s eval reused from cache (%d blocks)", domain, cached["n_blocks"])
        return cached
    log.info("computing frozen %s eval (seed-independent baseline)", domain)
    ids = domain_ids.to(next(model.parameters()).device)
    if adapters is not None:
        for ad in adapters:
            ad.set_enabled(False)
    try:
        n = ids.numel()
        blocks: list[tuple[float, int]] = []
        with torch.no_grad():
            for begin in range(0, n, stride):
                end = min(begin + window, n)
                chunk = ids[begin:end].unsqueeze(0)
                if chunk.size(-1) < 2:
                    continue
                logits = model(input_ids=chunk).logits
                shift_l = logits[..., :-1, :].to(torch.float32).reshape(-1, logits.size(-1))
                shift_t = chunk[..., 1:].reshape(-1)
                nll = torch.nn.functional.cross_entropy(shift_l, shift_t, reduction="sum").item()
                blocks.append((nll, int(shift_t.numel())))
    finally:
        if adapters is not None:
            for ad in adapters:
                ad.set_enabled(True)
    total_nll = sum(b[0] for b in blocks)
    total_tok = sum(b[1] for b in blocks)
    payload = {
        "domain": domain,
        "ppl": float(math.exp(total_nll / total_tok)),
        "mean_nll": float(total_nll / total_tok),
        "n_tokens": int(total_tok),
        "n_blocks": len(blocks),
        "per_block": {"nll": [b[0] for b in blocks], "tokens": [b[1] for b in blocks]},
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, cache_path)
    return payload


def plastic_eval(model, ids: torch.Tensor, window: int, stride: int) -> dict:
    """Sliding-window ppl with the LoRA hooks active (no_grad)."""
    ids = ids.to(next(model.parameters()).device)
    n = ids.numel()
    blocks: list[tuple[float, int]] = []
    with torch.no_grad():
        for begin in range(0, n, stride):
            end = min(begin + window, n)
            chunk = ids[begin:end].unsqueeze(0)
            if chunk.size(-1) < 2:
                continue
            logits = model(input_ids=chunk).logits
            shift_l = logits[..., :-1, :].to(torch.float32).reshape(-1, logits.size(-1))
            shift_t = chunk[..., 1:].reshape(-1)
            nll = torch.nn.functional.cross_entropy(shift_l, shift_t, reduction="sum").item()
            blocks.append((nll, int(shift_t.numel())))
    total_nll = sum(b[0] for b in blocks)
    total_tok = sum(b[1] for b in blocks)
    return {
        "ppl": float(math.exp(total_nll / total_tok)),
        "mean_nll": float(total_nll / total_tok),
        "n_tokens": int(total_tok),
        "per_block": {"nll": [b[0] for b in blocks], "tokens": [b[1] for b in blocks]},
    }


# ── checkpointing (persists modulator EMA for bit-identical resume) ──


def _save_checkpoint(path: str, step: int, adapters, optimizer, modulator,
                     config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "plastic": {i: ad.state_dict() for i, ad in enumerate(adapters)},
        "optimizer": optimizer.state_dict(),
        "modulator": modulator.state_dict() if modulator is not None else None,
        "config": config,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _resume(checkpoint_dir: str, steps: int, adapters, optimizer,
            modulator) -> int:
    """Return step to resume from; restore plastic + optimizer + EMA state."""
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
    for p in glob.glob(os.path.join(checkpoint_dir, "brain_ckpt_step*.pt")):
        m = re.match(r".*brain_ckpt_step(\d+)\.pt$", p)
        if m and int(m.group(1)) >= steps:
            return steps  # already complete
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
    log.info("resumed from step %d (%s)", best_step, best_path)
    return best_step


# ── gated-lr step logic (pure, testable) ──────────────────────────


def compute_effective_lr(
    method: str, base_lr: float, M: float, const_scale: float | None,
    in_warmup: bool,
) -> tuple[float, bool]:
    """Return ``(effective_lr, do_update)`` for one training step.

    * ``plain`` — constant ``base_lr``; **learns during warmup too** (E032
      convention — the maximal-upper-bound reading of "same warmup").
    * ``surprise`` — gated: ``lr = η·M``, and ``M = 0`` during warmup (E031:
      EMA settles on the source loss, no plastic update). ``do_update`` is
      False during warmup.
    * ``const_reduced`` — control: ``lr = η·const_scale`` over the adapt
      phase, **no warmup learning** (matching ``surprise`` so the comparison
      isolates the lr's temporal shape, not the total effective learning).
    """
    if method == "surprise":
        M = 0.0 if in_warmup else M
        return base_lr * M, not in_warmup
    if method == "const_reduced":
        lr = base_lr * (const_scale or 0.0) if not in_warmup else 0.0
        return lr, not in_warmup
    return base_lr, True


# ── main ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--method", choices=METHODS, default="surprise")
    p.add_argument("--phases", type=int, choices=(1, 2), default=1)
    p.add_argument("--tag", required=True)
    p.add_argument("--budget-tokens", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-3)  # η (E032 best plain lr)
    p.add_argument("--const-scale", type=float, default=None,
                   help="const_reduced: lr = η·const_scale over adapt steps")
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
    p.add_argument("--output-dir", default="results/brain/e034")
    p.add_argument("--log-dir", default="logs/brain/e034")
    p.add_argument("--frozen-cache-dir", default="results/brain/e034/cache")
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
        "E034 start method=%s phases=%d tag=%s rank=%d budget=%d seed=%d model=%s (%s)",
        args.method, args.phases, args.tag, args.rank, args.budget_tokens, seed,
        args.model, log_path,
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

    adapters = build_lora_adapters(model, args.rank, device)
    n_params = n_lora_params(adapters)
    log.info(
        "%d LoRA adapters, %d trainable params (%.1f KB fp32) — budget match "
        "to E032 Part D (rank %d)",
        len(adapters), n_params, n_params * 4 / 1024, args.rank,
    )

    params = [p for ad in adapters for p in ad.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # Surprise modulator (E031 locked defaults); None unless method=surprise.
    modulator = None
    if args.method == "surprise":
        modulator = SurpriseModulator(
            mode="surprise_ema",
            alpha=args.alpha,
            s0=args.s0,
            k=args.k,
            M_max=args.m_max,
        )
    const_scale = args.const_scale if args.method == "const_reduced" else None
    if args.method == "const_reduced" and const_scale is None:
        log.error("const_reduced requires --const-scale")
        return 2

    ckpt_dir = os.path.join(
        output_dir, "checkpoints",
        f"{args.method}_{args.phases}p_{args.tag}_budget{args.budget_tokens}_seed{seed}",
    )

    start_step = _resume(ckpt_dir, total_steps, adapters, optimizer, modulator)
    if start_step >= total_steps:
        log.info("already complete (step %d >= %d); skipping", start_step, total_steps)
        return 0

    # ── learn stream ───────────────────────────────────────────────
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
            wiki_ids, pub_ids, args.warmup_steps, args.batch_size, args.seq_len, seed
        )
    for _ in range(start_step):
        next(data_iter)

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
            in_warmup = step < args.warmup_steps
            effective_lr, do_update = compute_effective_lr(
                args.method, args.lr, M, const_scale, in_warmup,
            )
            if do_update:
                optimizer.param_groups[0]["lr"] = effective_lr
                loss.backward()
                if (step - start_step + 1) % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

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
                "tokens_seen": (step + 1) * ids.numel(),
            })
            if (step + 1) % max(args.grad_accum * 100, 1) == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"brain_ckpt_step{step + 1}.pt"),
                    step + 1, adapters, optimizer, modulator, {"tag": args.tag},
                )
            if step % 10 == 0 or step == total_steps - 1:
                m = train_metrics[-1]
                log.info(
                    "step %d/%d loss=%.4f ema=%.4f s=%+.4f M=%.3f lr=%.3e |w|=%.3e",
                    step, total_steps, m["loss"], m["ema_loss"], m["surprise_s"],
                    m["modulator_M"], m["effective_lr"], m["mean_abs_b"],
                )
    finally:
        signal.signal(signal.SIGINT, prev_handlers[0])
        signal.signal(signal.SIGTERM, prev_handlers[1])

    _save_checkpoint(
        os.path.join(ckpt_dir, f"brain_ckpt_step{total_steps}.pt"),
        total_steps, adapters, optimizer, modulator, {"tag": args.tag},
    )
    log.info("learning done in %.1f s (%d steps)", time.time() - t0, len(train_metrics))

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

    # Surprise/effective-lr statistics over the training trace.
    gate_stats: dict = {}
    if train_metrics:
        ms = [m["modulator_M"] for m in train_metrics]
        eff = [m["effective_lr"] for m in train_metrics]
        gate_stats = {
            "mean_surprise_M": float(sum(ms) / len(ms)),
            "mean_M_adapt": float(sum(ms[args.warmup_steps:]) /
                                  max(len(ms[args.warmup_steps:]), 1))
            if len(ms) > args.warmup_steps else None,
            "pct_steps_M_gt_05": float(
                sum(1.0 for m in ms if m > 0.5) / len(ms)),
            "final_M": float(ms[-1]),
            "effective_mean_lr": float(sum(eff) / len(eff)),
            "n_steps": len(ms),
        }

    allw = all_lora_weights(adapters)
    plastic_weights = {
        "count": n_params,
        "bytes": n_params * 4,
        "mean_magnitude": float(allw.abs().mean()),
        "max_magnitude": float(allw.abs().max()),
        "sparsity": float((allw.abs() < 1e-8).float().mean()),
    }

    result = {
        "experiment": "e034_surprise_gated_lora",
        "step": "2.1",
        "tag": args.tag,
        "method": {"plain": "lora", "surprise": "lora_gated",
                   "const_reduced": "lora_const_reduced"}[args.method],
        "gate": {"plain": "none", "surprise": "surprise_ema",
                 "const_reduced": "const_reduced"}[args.method],
        "phases": args.phases,
        "phase2_domain": "cnn_dailymail" if args.phases == 2 else None,
        "rank": args.rank,
        "model": args.model,
        "model_short": mshort,
        "modulator": {
            "mode": "surprise_ema" if args.method == "surprise" else
                    ("const_reduced" if args.method == "const_reduced" else "none"),
            "alpha": args.alpha, "s0": args.s0, "k": args.k, "M_max": args.m_max,
            "const_scale": const_scale,
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
        "plastic_weights": plastic_weights,
        "n_target_blocks": tgt_stats["n_blocks"],
        "n_source_blocks": src_stats["n_blocks"],
        "train_metrics": train_metrics,
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
    budget_tag = f"{args.budget_tokens // 1000}k"
    fname = f"{mshort}_pubmed_{budget_tag}_{args.tag}_seed{seed}.json"
    if args.phases == 2:
        fname = f"{mshort}_pubmed_cnn_{budget_tag}_{args.tag}_seed{seed}.json"
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    extra = ""
    if args.phases == 2:
        extra = (f" | phase2 Δppl={result['phase2_metrics']['phase2_ppl_delta']:+.3f}"
                 f" (ci {[f'{x:.3f}' for x in cnn_stats['delta_ppl_ci95']]})")
    log.info(
        "RESULT -> %s | target Δppl=%+.3f (ci %s) | source forgetting=%+.3f%%"
        " | mean M=%.3f%s",
        out_path, target_ppl_delta,
        [f"{x:.3f}" for x in tgt_stats["delta_ppl_ci95"]], forgetting_pct,
        gate_stats.get("mean_surprise_M", 0.0) or 0.0, extra,
    )
    log.info("E034 complete method=%s phases=%d tag=%s seed=%d",
             args.method, args.phases, args.tag, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
