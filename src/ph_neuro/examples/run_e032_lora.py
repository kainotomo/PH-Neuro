#!/usr/bin/env python3
"""E032 — LoRA backprop baseline (Part D, the comparison).

Real LoRA (backprop; ``peft`` is not installed, so this is a minimal manual
LoRA, ~50 lines) at the **same parameter budget** as the best low-rank local
config from Part A: same 48 injection sites (``o_proj`` + ``down_proj``),
same rank, same ``A: (r, d_in)`` / ``B: (d_out, r)`` structure → identical
param count. Only the update rule differs (AdamW backprop vs local Hebbian).

* Frozen backbone ``requires_grad_(False)``; only A/B train.
* Init: A ~ N(0, 1/d_in), B = 0 (identical to the local low-rank mode → the
  comparison isolates the learning rule, not the init).
* Optimizer: AdamW (wd = 0.0, matching protocol §4 B5), constant lr from the
  sweep ``{1e-4, 3e-4, 1e-3}``.
* Data: the **same** combined stream as the local runner (WikiText warmup
  steps → PubMed adapt steps) via ``make_combined_batch_iter``. Note: unlike
  the local method (whose warmup steps have M = 0 and therefore no update),
  backprop LoRA *does* update during the warmup steps — the honest, maximal
  upper-bound reading of "same warmup procedure".
* Gradient checkpointing wraps the model forward so activation memory stays
  near forward-only levels on the 8 GB card.
* Model runs in ``eval()`` (no dropout) — deterministic, matching the local
  method's conditions.

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e032_lora \\
        --rank 4 --lr 3e-4 --tag lora_lr3e4 --budget-tokens 100000 --seed 42

Output: ``results/brain/e032/smolllm2_1p7b_pubmed_{budget}_{tag}_seed{seed}.json``
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

from ph_neuro.brain.brain_wrapper import (  # noqa: E402
    check_gpu_free,
    gpu_free_mb,
)
from ph_neuro.brain.block_wrappers import _get_in_features, _get_out_features  # noqa: E402
from ph_neuro.brain.datasets import (  # noqa: E402
    make_combined_batch_iter,
    pubmed_eval_ids,
    pubmed_train_ids,
    wikitext_ids,
)
from ph_neuro.brain.stats import block_paired_stats  # noqa: E402

log = logging.getLogger("e032_lora")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN

CHECKPOINT_FORMAT = "ph_neuro_e032_lora_checkpoint"


def model_short(model_id: str) -> str:
    if "SmolLM2-1.7B" in model_id:
        return "smolllm2_1p7b"
    return model_id.replace("/", "__")


def default_min_free_gb(model_id: str) -> float:
    return 6.0 if "SmolLM2" in model_id else 2.0


def setup_logging(tag: str, budget: int, seed: int, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"e032_lora_{tag}_budget{budget}_seed{seed}.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return path


def disable_triton_bmm() -> None:
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


# ── LoRA adapter + hooks ───────────────────────────────────────────


class LoRAAdapter:
    """One trainable LoRA pair injected at one frozen projection module.

    ``output + (B @ (A @ x))`` via a forward hook; A/B are real
    ``nn.Parameters`` so gradients flow back through the hook.
    """

    def __init__(self, module, rank: int, device, dtype=torch.float32):
        self.name = module.__class__.__name__
        self.out_features = _get_out_features(module)
        self.in_features = _get_in_features(module)
        self.rank = int(rank)
        self.device = torch.device(device)
        # Scaled random projection init (matches the local low-rank mode:
        # same init, only the update rule differs).
        self.A = torch.randn(
            self.rank, self.in_features, dtype=dtype, device=self.device
        ) * (1.0 / math.sqrt(self.in_features))
        self.B = torch.zeros(self.out_features, self.rank, dtype=dtype, device=self.device)
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        x = args[0]
        t = torch.einsum("ri,bsi->bsr", self.A.to(output.dtype), x)
        return output + torch.einsum("or,bsr->bso", self.B.to(output.dtype), t)

    def parameters(self):
        yield self.A
        yield self.B

    def remove(self):
        self.handle.remove()

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            [("A", self.A.detach().clone().to(torch.float32)),
             ("B", self.B.detach().clone().to(torch.float32))]
        )

    def load_state_dict(self, state) -> None:
        self.A.data.copy_(state["A"].to(self.device))
        self.B.data.copy_(state["B"].to(self.device))

    def n_params(self) -> int:
        return self.A.numel() + self.B.numel()


def build_lora_adapters(model, rank: int, device) -> list[LoRAAdapter]:
    """Attach LoRA at every o_proj/down_proj (llama) / c_proj (gpt2) site."""
    from ph_neuro.brain.block_wrappers import get_block_container, get_block_wrapper

    container = get_block_container(model)
    wrapper = get_block_wrapper(model)
    adapters: list[LoRAAdapter] = []
    for i, block in enumerate(container):
        for path in wrapper.block_paths:
            mod = block
            for part in path.split("."):
                mod = getattr(mod, part)
            adapters.append(LoRAAdapter(mod, rank, device))
    return adapters


# ── eval (LoRA active = "plastic"; frozen from cache) ──────────────


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
    mean_nll = total_nll / total_tok
    return {
        "ppl": float(math.exp(mean_nll)),
        "mean_nll": float(mean_nll),
        "n_tokens": int(total_tok),
        "per_block": {"nll": [b[0] for b in blocks], "tokens": [b[1] for b in blocks]},
    }


# ── checkpointing ──────────────────────────────────────────────────


def _save_checkpoint(path: str, step: int, adapters, optimizer, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "plastic": {i: ad.state_dict() for i, ad in enumerate(adapters)},
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _resume(checkpoint_dir: str, steps: int, adapters, optimizer) -> int:
    """Return step to resume from; restore plastic + optimizer state."""
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
    for i, ad in enumerate(adapters):
        ad.load_state_dict(ckpt["plastic"][str(i) if str(i) in ckpt["plastic"] else i])
    optimizer.load_state_dict(ckpt["optimizer"])
    log.info("resumed from step %d (%s)", best_step, best_path)
    return best_step


# ── main ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--tag", required=True)
    p.add_argument("--budget-tokens", type=int, default=100_000)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-checkpoint", action="store_true", help="disable grad checkpointing")
    p.add_argument("--output-dir", default="results/brain/e032")
    p.add_argument("--log-dir", default="logs/brain/e032")
    p.add_argument("--frozen-cache-dir", default="results/brain/e032/cache")
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
        "E032 LoRA start tag=%s rank=%d lr=%g budget=%d seed=%d (%s)",
        args.tag, args.rank, args.lr, args.budget_tokens, seed, log_path,
    )
    torch.manual_seed(seed)

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
    n_params = sum(ad.n_params() for ad in adapters)
    log.info(
        "%d LoRA adapters, %d trainable params (%.1f KB fp32), budget match to "
        "local low-rank rank=%d",
        len(adapters), n_params, n_params * 4 / 1024, args.rank,
    )

    params = [p for ad in adapters for p in ad.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    adapt_steps = math.ceil(args.budget_tokens / (args.batch_size * args.seq_len))
    total_steps = args.warmup_steps + adapt_steps
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"{args.tag}_budget{args.budget_tokens}_seed{seed}")

    start_step = _resume(ckpt_dir, total_steps, adapters, optimizer)
    if start_step >= total_steps:
        log.info("already complete (step %d >= %d); skipping", start_step, total_steps)
        return 0

    wiki_ids = wikitext_ids("train", tok)
    pub_ids = pubmed_train_ids(tok)
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
            current_step[0], adapters, optimizer, {"tag": args.tag},
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
            loss.backward()
            if (step - start_step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            mean_abs = sum(
                float(ad.A.detach().abs().mean()) + float(ad.B.detach().abs().mean())
                for ad in adapters
            ) / (2 * len(adapters))
            train_metrics.append({
                "step": step,
                "loss": float(loss.item()),
                "modulator_M": 1.0,  # no surprise signal for backprop LoRA
                "mean_abs_delta_b": 0.0,
                "mean_abs_b": mean_abs,
                "tokens_seen": (step + 1) * ids.numel(),
            })
            if (step + 1) % max(args.grad_accum * 100, 1) == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"brain_ckpt_step{step + 1}.pt"),
                    step + 1, adapters, optimizer, {"tag": args.tag},
                )
            if step % 10 == 0 or step == total_steps - 1:
                m = train_metrics[-1]
                log.info("step %d/%d loss=%.4f |w|=%.3e",
                         step, total_steps, m["loss"], m["mean_abs_b"])
    finally:
        signal.signal(signal.SIGINT, prev_handlers[0])
        signal.signal(signal.SIGTERM, prev_handlers[1])

    _save_checkpoint(
        os.path.join(ckpt_dir, f"brain_ckpt_step{total_steps}.pt"),
        total_steps, adapters, optimizer, {"tag": args.tag},
    )
    log.info("learning done in %.1f s (%d steps)", time.time() - t0, len(train_metrics))

    # ── evaluation ─────────────────────────────────────────────────
    wiki_test = wikitext_ids("test", tok)
    pub_test = pubmed_eval_ids(tok, max_tokens=args.eval_pubmed_tokens)

    cache_path = os.path.join(args.frozen_cache_dir, f"frozen_{mshort}_wikitext2.json")
    with open(cache_path) as fh:
        src_frozen = json.load(fh)
    cache_path = os.path.join(args.frozen_cache_dir, f"frozen_{mshort}_pubmed.json")
    with open(cache_path) as fh:
        tgt_frozen = json.load(fh)

    src_plastic = plastic_eval(model, wiki_test, args.eval_window, args.eval_stride)
    tgt_plastic = plastic_eval(model, pub_test, args.eval_window, args.eval_stride)
    src_stats = block_paired_stats(
        src_frozen["per_block"]["nll"], src_plastic["per_block"]["nll"],
        src_frozen["per_block"]["tokens"],
    )
    tgt_stats = block_paired_stats(
        tgt_frozen["per_block"]["nll"], tgt_plastic["per_block"]["nll"],
        tgt_frozen["per_block"]["tokens"],
    )

    source_ppl_delta = src_plastic["ppl"] - src_frozen["ppl"]
    target_ppl_delta = tgt_frozen["ppl"] - tgt_plastic["ppl"]
    forgetting_pct = (src_plastic["ppl"] / src_frozen["ppl"] - 1.0) * 100.0

    allw = torch.cat([ad.A.detach().flatten() for ad in adapters]
                     + [ad.B.detach().flatten() for ad in adapters])
    result = {
        "experiment": "e032_capacity_gain",
        "step": "1.2",
        "tag": args.tag,
        "method": "lora",
        "plasticity": "lora",
        "rank": args.rank,
        "model": args.model,
        "model_short": mshort,
        "modulator": {"mode": "none_backprop", "alpha": None, "s0": None,
                      "k": None, "M_max": None},
        "source_domain": "wikitext2",
        "target_domain": "pubmed",
        "adaptation_tokens": args.budget_tokens,
        "adaptation_steps": adapt_steps,
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
        },
        "plastic_weights": {
            "count": n_params,
            "bytes": n_params * 4,
            "mean_magnitude": float(allw.abs().mean()),
            "max_magnitude": float(allw.abs().max()),
        },
        "n_target_blocks": tgt_stats["n_blocks"],
        "n_source_blocks": src_stats["n_blocks"],
        "train_metrics": train_metrics,
    }

    os.makedirs(output_dir, exist_ok=True)
    budget_tag = f"{args.budget_tokens // 1000}k"
    out_path = os.path.join(
        output_dir, f"{mshort}_pubmed_{budget_tag}_{args.tag}_seed{seed}.json"
    )
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log.info(
        "RESULT -> %s | target Δppl=%+.3f (ci %s) | source forgetting=%+.3f%%",
        out_path, target_ppl_delta, [f"{x:.3f}" for x in tgt_stats["delta_ppl_ci95"]],
        forgetting_pct,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
