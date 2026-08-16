#!/usr/bin/env python3
"""E036 — Consolidation Mechanism runner (Step 2.3).

The last brain mechanism: **sleep-like consolidation** — transfer important
plastic changes from a short-term (ST) store to a long-term (LT) store. The
stack is the proven **T-C ternary STE gated LoRA** adapter (E035: 86 KB each,
surprise-gated, selective). With per-domain adapters + selectivity, forgetting
is already near zero — so the scientific question is what consolidation *adds*:

1. **Forward transfer** — an LT store that accumulates cross-domain knowledge;
   each new domain warm-starts from it and should adapt faster / better.
2. **Storage management** — one long-term adapter + small domain deltas vs N
   full adapters.

Sequence (3 domains, 100K each): WikiText-2 (source, warmup M=0) → PubMed
(D1) → CNN/DailyMail (D2) → **C4** (D3, Common Crawl web, odc-by; frozen ppl
13.57 — harder than CNN 11.97 so the surprise gate opens at the D3 boundary;
legal corpora were all easier → gate closed → vacuous, hence C4).

Three conditions (same protocol, SmolLM2-1.7B, 3 seeds 42/43/44, T-C ternary
STE gated LoRA, rank-1 344,064-param budget, surprise lr η·M_t η=1e-3,
eval window 512 / stride 256):

* ``--condition b1`` — independent per-domain adapters, no consolidation
  (the current-best baseline). One adapter per domain, each with its own
  WikiText warmup; BT = 0 by construction. ``--b1-domain pubmed|cnn|c4``
  selects which domain this sub-cell runs (3 sub-cells per seed).
* ``--condition b2`` — interference floor: one single continuing adapter
  across all 3 domains (EMA continuous, no resets).
* ``--condition c`` — consolidation: LT store + per-domain ST. After each
  domain: transfer the top-K% (by |ΔW| magnitude, **K = 10%**, pre-registered)
  of the ST's latent-score changes into LT (**add rule**); reset ST;
  next domain warm-starts ST from LT. **No LT decay.** LT's scales stay at
  the canonical init (injection magnitude ~0.01); only latent-score signs
  transfer.

Pre-registered success criteria (see 10-e036-consolidation.md §7):
* C's D3 Δppl ≥ B1's D3 Δppl (forward transfer non-negative, ideally better)
  where C's D3 = the warm-started full ST after D3 vs B1's fresh D3 adapter.
* C's BT on D1/D2 < 0.1, where BT_C(Dd) = ppl(LT_final on Dd) − ppl(LT_after_dd
  on Dd) — pure later-domain interference on the consolidated store.
* C's total storage ≤ B1's total storage (deployed LT-only and LT+sparse
  deltas; both reported).
* Report C/B1 on D3 Δppl and on storage; adaptation speed (steps to plateau)
  via 50K-token C4 probe evals every 10 steps in the D3 phase.

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e036_consolidation \\
        --condition c --tag c --budget-tokens 100000 --seed 42

Output: ``results/brain/e036/{mshort}_{condition}{_domain}_{budget}_{tag}_seed{s}.json``
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
    c4_eval_ids,
    c4_probe_ids,
    c4_train_ids,
    cnn_dailymail_eval_ids,
    cnn_dailymail_train_ids,
    make_combined_batch_iter,
    make_four_domain_batch_iter,
    pubmed_eval_ids,
    pubmed_train_ids,
    wikitext_ids,
)
from ph_neuro.brain.lora import (  # noqa: E402
    TernaryLoRAAdapter,
    add_delta_to_lt,
    build_ternary_lora_adapters,
    latent_change_topk,
    n_lora_params,
    pack_ternary_adapters,
    sparse_delta_storage,
    tc_latent_state,
    ternary_storage_report,
    warm_start_st_from_lt,
    zero_lt_state,
)
from ph_neuro.brain.modulator import SurpriseModulator  # noqa: E402
from ph_neuro.brain.stats import block_paired_stats, cross_seed_summary  # noqa: E402
from ph_neuro.examples.run_e034_lora import (  # noqa: E402
    compute_effective_lr,
    default_min_free_gb,
    disable_triton_bmm,
    frozen_eval,
    load_model,
    model_short,
    plastic_eval,
)

log = logging.getLogger("e036")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 1024

# E031 locked surprise defaults (identical to E034/E035).
DEFAULT_S0 = 0.05
DEFAULT_K_SIG = 60.0
DEFAULT_M_MAX = 1.0
DEFAULT_ALPHA = 0.99

# E036 consolidation defaults (pre-registered, see 10-e036-consolidation.md).
DEFAULT_CONSOLIDATE_K = 0.10  # top 10% of |ΔW| latent-score changes
DEFAULT_LT_DECAY = 0.0        # no LT decay (persistent store)
DEFAULT_PROBE_TOKENS = 50_000
DEFAULT_PROBE_EVERY = 10

CHECKPOINT_FORMAT = "ph_neuro_e036_consolidation_checkpoint"

CONDITIONS = ("b1", "b2", "c")
B1_DOMAINS = ("pubmed", "cnn", "c4")

# Domain order in the sequence.
SEQUENCE = ("pubmed", "cnn", "c4")

# (domain key → (train loader, eval loader, probe loader or None))
DOMAIN_LOADERS = {
    "pubmed": (pubmed_train_ids, pubmed_eval_ids, None),
    "cnn": (cnn_dailymail_train_ids, cnn_dailymail_eval_ids, None),
    "c4": (c4_train_ids, c4_eval_ids, c4_probe_ids),
}

# Frozen-domain name used for the frozen eval cache (must match E034/E035).
FROZEN_DOMAIN_NAME = {
    "wikitext2": "wikitext2",
    "pubmed": "pubmed",
    "cnn": "cnn_dailymail",
    "c4": "c4",
}


def setup_logging(tag: str, budget: int, seed: int, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"e036_{tag}_budget{budget}_seed{seed}.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return path


# ── E036 checkpointing (format-tagged; mirrors E035 + LT/boundary state) ──


def _save_checkpoint(path: str, step: int, adapters, optimizer, modulator,
                     lt_states, boundaries_done, eval_cache, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "plastic": {i: ad.state_dict() for i, ad in enumerate(adapters)},
        "optimizer": optimizer.state_dict(),
        "modulator": modulator.state_dict() if modulator is not None else None,
        "lt_states": lt_states,
        "boundaries_done": list(boundaries_done),
        "eval_cache": eval_cache,
        "rng_state": torch.get_rng_state().cpu(),
        "config": config,
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _resume(checkpoint_dir: str, steps: int, adapters, optimizer,
            modulator, lt_states, boundaries_done, eval_cache) -> int:
    """Restore the latest checkpoint < ``steps``; return the step to resume from.

    Restores plastic + optimizer + EMA + LT states + boundary flags + the
    computed-eval cache (+ RNG). "Already complete" is decided by the caller
    via the result JSON's existence, NOT by a stale final checkpoint (E035
    lesson).
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
    if ckpt.get("lt_states") is not None:
        lt_states.clear()
        lt_states.extend(ckpt["lt_states"])
    boundaries_done.clear()
    boundaries_done.update(ckpt.get("boundaries_done", []))
    eval_cache.clear()
    eval_cache.update(ckpt.get("eval_cache", {}))
    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"].to(torch.uint8).cpu())
    log.info("resumed from step %d (%s) [lt=%d boundaries=%s eval_cache=%d]",
             best_step, best_path, len(lt_states), sorted(boundaries_done),
             len(eval_cache))
    return best_step


# ── eval helpers ───────────────────────────────────────────────────


def _compact_stats(frozen: dict, plastic: dict, domain: str, artifact: str,
                   phase: str, window: int, stride: int) -> dict:
    """Compact eval summary (no per-block lists) for the eval cache / JSON."""
    stats = block_paired_stats(
        frozen["per_block"]["nll"],
        plastic["per_block"]["nll"],
        frozen["per_block"]["tokens"],
    )
    return {
        "domain": domain,
        "artifact": artifact,      # "full" (live adapter) | "lt" | "st"
        "phase": phase,            # "after_d1" | "after_d2" | "after_d3"
        "frozen_ppl": float(frozen["ppl"]),
        "plastic_ppl": float(plastic["ppl"]),
        "delta_ppl": float(frozen["ppl"] - plastic["ppl"]),  # + = improved
        "delta_ppl_ci95": stats.get("delta_ppl_ci95"),
        "paired_t": stats.get("paired_t"),
        "paired_p": stats.get("paired_p"),
        "cohens_d": stats.get("cohens_d"),
        "n_blocks": int(frozen["n_blocks"]),
        "n_tokens": int(frozen["n_tokens"]),
        "plastic_n_tokens": int(plastic["n_tokens"]),
    }


def eval_artifacts(model, tok, domain_ids, domain: str, window: int, stride: int,
                   frozen_cache_dir: str, *, live_adapters=None,
                   lt_states=None, artifact: str, phase: str) -> dict:
    """Evaluate a domain with a given adapter artifact.

    ``live_adapters`` = the attached ST adapter list (evaluated as-is, e.g.
    the full short-term after its domain). ``lt_states`` = a long-term latent
    state (evaluated via a freshly built T-C adapter; the live hooks are
    temporarily disabled). Exactly one of the two must be provided.
    """
    frozen = frozen_eval(model, tok, domain_ids, FROZEN_DOMAIN_NAME[domain],
                         frozen_cache_dir, window, stride, live_adapters)
    if lt_states is not None:
        # Build fresh T-C adapters from the LT state; disable live hooks so the
        # eval is purely the LT injection.
        if live_adapters is not None:
            for ad in live_adapters:
                ad.set_enabled(False)
        tmp = build_ternary_lora_adapters(model, 1, "cuda" if torch.cuda.is_available() else "cpu", mode="tc")
        for ad, st in zip(tmp, lt_states):
            ad.load_state_dict({
                "A_latent": st["A_latent"].to(torch.float32),
                "B_latent": st["B_latent"].to(torch.float32),
                "A_scale": st["A_scale"].to(torch.float32),
                "B_scale": st["B_scale"].to(torch.float32),
            })
        try:
            plastic = plastic_eval(model, domain_ids, window, stride)
        finally:
            for ad in tmp:
                ad.remove()
            if live_adapters is not None:
                for ad in live_adapters:
                    ad.set_enabled(True)
        out = _compact_stats(frozen, plastic, domain, artifact, phase, window, stride)
        log.info("eval %s [%s, %s] ppl %.4f -> %.4f (Δ%+.4f)",
                 domain, artifact, phase, frozen["ppl"], plastic["ppl"],
                 out["delta_ppl"])
        return out
    plastic = plastic_eval(model, domain_ids, window, stride)
    out = _compact_stats(frozen, plastic, domain, artifact, phase, window, stride)
    log.info("eval %s [%s, %s] ppl %.4f -> %.4f (Δ%+.4f)",
             domain, artifact, phase, frozen["ppl"], plastic["ppl"], out["delta_ppl"])
    return out


def eval_probe(model, tok, probe_ids, window: int, stride: int) -> float:
    """Fast in-training probe ppl (no frozen cache needed for the plateau shape)."""
    return float(plastic_eval(model, probe_ids, window, stride)["ppl"])


# ── gated-lr training loop with boundary callbacks ─────────────────


def train_gated(
    args, model, tok, adapters, optimizer, modulator,
    data_iter, start_step, total_steps, warmup_steps, ckpt_dir,
    *, boundaries: dict[int, str], on_boundary,
    probe_ids=None, probe_start=None, probe_every=DEFAULT_PROBE_EVERY,
    lt_states=None, boundaries_done=None, eval_cache=None,
    checkpoint_every: int = 100,
) -> list[dict]:
    """Run the E034 gated-lr loop (mirrors E035) with E036 boundary hooks.

    ``boundaries`` maps a **finished step** (the last step of a phase) to a
    phase name; ``on_boundary(step, phase)`` is invoked once (idempotent via
    ``boundaries_done``) at each boundary. ``probe_ids`` triggers a probe eval
    every ``probe_every`` steps from ``probe_start`` (adaptation-speed curve).
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
            current_step[0], adapters, optimizer, modulator, lt_states,
            boundaries_done, eval_cache, {"tag": args.tag},
        )
        os._exit(130)  # noqa: PLR1722

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    train_metrics: list[dict] = []
    probe_metrics: list[dict] = []
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

            surprise_s, M = 0.0, 1.0
            if modulator is not None:
                surprise_s, M = modulator.update(loss)
            in_warmup = step < warmup_steps
            effective_lr, do_update = compute_effective_lr(
                "surprise", args.lr, M, None, in_warmup,
            )
            if do_update:
                optimizer.param_groups[0]["lr"] = effective_lr
                loss.backward()
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
                "step_time_s": float(time.time() - t_step0),
                "tokens_seen": (step + 1) * ids.numel(),
            })

            # Probe (adaptation-speed curve) — every N steps from probe_start
            # (plus the final step, so the plateau reference is the end point).
            if probe_ids is not None and probe_start is not None and \
                    step >= probe_start and \
                    ((step - probe_start) % probe_every == 0 or step == total_steps - 1):
                pp = eval_probe(model, tok, probe_ids, args.eval_window,
                                args.eval_stride)
                probe_metrics.append({"adapt_step": step, "ppl": pp})
                log.info("probe @ adapt step %d: ppl=%.4f", step, pp)

            # Boundary action (idempotent).
            if step in boundaries and boundaries[step] not in boundaries_done:
                phase = boundaries[step]
                on_boundary(step, phase)
                boundaries_done.add(phase)
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"brain_ckpt_step{step + 1}.pt"),
                    step + 1, adapters, optimizer, modulator, lt_states,
                    boundaries_done, eval_cache, {"tag": args.tag},
                )

            if (step + 1) % checkpoint_every == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"brain_ckpt_step{step + 1}.pt"),
                    step + 1, adapters, optimizer, modulator, lt_states,
                    boundaries_done, eval_cache, {"tag": args.tag},
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
        total_steps, adapters, optimizer, modulator, lt_states,
        boundaries_done, eval_cache, {"tag": args.tag},
    )
    log.info("learning done in %.1f s (%d steps)", time.time() - t0, len(train_metrics))
    return train_metrics, probe_metrics


# ── storage helpers ────────────────────────────────────────────────


def _adapter_weights(adapters) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for ad in adapters:
        A_tern, B_tern, _, _ = ad.ternary_snapshot()
        parts.append(A_tern.float().flatten())
        parts.append(B_tern.float().flatten())
    return torch.cat(parts)


def storage_report(adapters, output_dir: str, mshort: str, tag: str, seed: int) -> dict:
    """2-bit packed storage for a T-C adapter set (same as E035)."""
    packed = pack_ternary_adapters(adapters)
    report = ternary_storage_report(adapters, packed)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{mshort}_{tag}_seed{seed}.ternary2bit")
    with open(path, "wb") as fh:
        fh.write(packed.cpu().numpy().tobytes())
    disk_bytes = os.path.getsize(path)
    return {**report, "packed": True, "disk_bytes": int(disk_bytes),
            "packed_path": path}


def consolidation_storage_report(adapters, n_deltas: int, k: float) -> dict:
    """C's storage: the deployed LT + the per-domain sparse deltas."""
    packed = pack_ternary_adapters(adapters)
    lt = ternary_storage_report(adapters, packed)
    n_params = lt["n_params"]
    n_kept = int(round(n_params * k))
    per_delta = sparse_delta_storage(n_params, n_kept)
    deltas_bytes = n_deltas * per_delta["total_bytes"]
    idx_deltas_bytes = n_deltas * per_delta["int32_index_variant_bytes"]
    return {
        "lt_packed_bytes": lt["packed_bytes"],
        "lt_scale_bytes": lt["scale_bytes"],
        "lt_total_bytes": lt["packed_bytes"] + lt["scale_bytes"],
        "k": k,
        "n_deltas": n_deltas,
        "per_delta_bytes": per_delta["total_bytes"],
        "deltas_total_bytes": deltas_bytes,
        "c_total_bytes": lt["packed_bytes"] + lt["scale_bytes"] + deltas_bytes,
        "c_total_index_variant_bytes": lt["packed_bytes"] + lt["scale_bytes"] + idx_deltas_bytes,
        "per_delta_index_variant_bytes": per_delta["int32_index_variant_bytes"],
    }


# ── condition runner factories ─────────────────────────────────────


def run_b1(args, model, tok, adapters, optimizer, modulator, mshort) -> dict:
    """Independent per-domain adapter (WikiText warmup → one domain)."""
    domain = args.b1_domain
    adapt_steps = math.ceil(args.budget_tokens / (args.batch_size * args.seq_len))
    total_steps = args.warmup_steps + adapt_steps
    ckpt_dir = os.path.join(
        args.output_dir, "checkpoints",
        f"b1_{domain}_{args.tag}_budget{args.budget_tokens}_seed{args.seed}",
    )
    boundaries: dict[int, str] = {}
    boundaries_done: set[str] = set()
    eval_cache: dict = {}

    train_fn, eval_fn, probe_fn = DOMAIN_LOADERS[domain]
    wiki_ids = wikitext_ids("train", tok)
    domain_train = train_fn(tok)
    data_iter = make_combined_batch_iter(
        wiki_ids, domain_train, args.warmup_steps, args.batch_size,
        args.seq_len, args.seed,
    )
    start_step = _resume(ckpt_dir, total_steps, adapters, optimizer, modulator,
                         [], boundaries_done, eval_cache)
    for _ in range(start_step):
        next(data_iter)

    probe_ids = None
    probe_start = None
    if domain == "c4" and probe_fn is not None:
        probe_ids = probe_fn(tok)
        probe_start = args.warmup_steps

    def on_boundary(step, phase):
        pass  # B1: no mid-run boundary actions; final eval happens after training.

    train_metrics, probe_metrics = train_gated(
        args, model, tok, adapters, optimizer, modulator, data_iter,
        start_step, total_steps, args.warmup_steps, ckpt_dir,
        boundaries=boundaries, on_boundary=on_boundary,
        probe_ids=probe_ids, probe_start=probe_start,
        lt_states=[], boundaries_done=boundaries_done, eval_cache=eval_cache,
        checkpoint_every=max(args.grad_accum * 100, 1),
    )

    # Final eval of the domain adapter.
    domain_eval_ids = eval_fn(tok, max_tokens=args.eval_max_tokens)
    eval_cache[f"{domain}_final"] = eval_artifacts(
        model, tok, domain_eval_ids, domain, args.eval_window, args.eval_stride,
        args.frozen_cache_dir, live_adapters=adapters,
        artifact="full", phase="final",
    )
    # Source eval (WikiText-2) for forgetting.
    wiki_test = wikitext_ids("test", tok)
    eval_cache["wikitext2_final"] = eval_artifacts(
        model, tok, wiki_test, "wikitext2", args.eval_window, args.eval_stride,
        args.frozen_cache_dir, live_adapters=adapters,
        artifact="full", phase="final",
    )

    # Gate statistics.
    ms = [m["modulator_M"] for m in train_metrics]
    adapt_ms = ms[args.warmup_steps:] if len(ms) > args.warmup_steps else ms
    gate_stats = {
        "mean_surprise_M": float(sum(ms) / len(ms)) if ms else 0.0,
        "mean_M_adapt": float(sum(adapt_ms) / max(len(adapt_ms), 1)),
        "pct_steps_M_gt_05": float(sum(1.0 for m in ms if m > 0.5) / len(ms)) if ms else 0.0,
        "final_M": float(ms[-1]) if ms else 0.0,
        "effective_mean_lr": float(sum(m["effective_lr"] for m in train_metrics) /
                                  len(train_metrics)) if train_metrics else 0.0,
        "n_steps": len(ms),
        "mean_step_time_s": float(sum(m["step_time_s"] for m in train_metrics) /
                                  len(train_metrics)) if train_metrics else 0.0,
    }

    storage = storage_report(adapters, args.output_dir, mshort,
                             f"b1_{domain}", args.seed)
    plastic_weights = {
        "count": n_lora_params(adapters),
        "mean_magnitude": float(_adapter_weights(adapters).abs().mean()),
        "sparsity": float((_adapter_weights(adapters).abs() < 1e-8).float().mean()),
    }

    result = {
        "experiment": "e036_consolidation",
        "step": "2.3",
        "condition": "b1",
        "b1_domain": domain,
        "tag": args.tag,
        "adapter": "tc",
        "seed": args.seed,
        "model": args.model,
        "model_short": mshort,
        "warmup_steps": args.warmup_steps,
        "adaptation_tokens": args.budget_tokens,
        "adaptation_steps": adapt_steps,
        "consolidation": None,
        "eval_cache": eval_cache,
        "metrics": {
            "domain": domain,
            "domain_ppl_frozen": eval_cache[f"{domain}_final"]["frozen_ppl"],
            "domain_ppl_plastic": eval_cache[f"{domain}_final"]["plastic_ppl"],
            "domain_ppl_delta": eval_cache[f"{domain}_final"]["delta_ppl"],
            "domain_ppl_delta_ci95": eval_cache[f"{domain}_final"]["delta_ppl_ci95"],
            "source_ppl_frozen": eval_cache["wikitext2_final"]["frozen_ppl"],
            "source_ppl_plastic": eval_cache["wikitext2_final"]["plastic_ppl"],
            "source_ppl_delta": eval_cache["wikitext2_final"]["plastic_ppl"] -
                               eval_cache["wikitext2_final"]["frozen_ppl"],
            "forgetting_pct": (eval_cache["wikitext2_final"]["plastic_ppl"] /
                               eval_cache["wikitext2_final"]["frozen_ppl"] - 1.0) * 100.0,
            **gate_stats,
        },
        "probe_metrics": probe_metrics,
        "storage": storage,
        "plastic_weights": plastic_weights,
        "train_metrics": train_metrics,
    }
    return result


def run_sequence(args, model, tok, adapters, optimizer, modulator, mshort) -> dict:
    """B2 (continuing) or C (consolidation) — the 3-domain sequence."""
    condition = args.condition
    adapt_steps = math.ceil(args.budget_tokens / (args.batch_size * args.seq_len))
    total_steps = args.warmup_steps + 3 * adapt_steps
    phase1_end = args.warmup_steps + adapt_steps          # end of D1 (PubMed)
    phase2_end = phase1_end + adapt_steps                 # end of D2 (CNN)
    boundaries = {
        phase1_end: "d1_pubmed",
        phase2_end: "d2_cnn",
        total_steps - 1: "d3_c4",
    }
    ckpt_dir = os.path.join(
        args.output_dir, "checkpoints",
        f"{condition}_{args.tag}_budget{args.budget_tokens}_seed{args.seed}",
    )
    boundaries_done: set[str] = set()
    eval_cache: dict = {}

    lt_states: list[OrderedDict] = []
    if condition == "c":
        lt_states = zero_lt_state(adapters)

    wiki_ids = wikitext_ids("train", tok)
    pub_ids = pubmed_train_ids(tok)
    cnn_ids = cnn_dailymail_train_ids(tok)
    c4_ids_train = c4_train_ids(tok)
    data_iter = make_four_domain_batch_iter(
        wiki_ids, pub_ids, cnn_ids, c4_ids_train,
        args.warmup_steps, adapt_steps, adapt_steps,
        args.batch_size, args.seq_len, args.seed,
    )
    start_step = _resume(ckpt_dir, total_steps, adapters, optimizer, modulator,
                         lt_states, boundaries_done, eval_cache)
    for _ in range(start_step):
        next(data_iter)

    # Eval corpora.
    pub_test = pubmed_eval_ids(tok, max_tokens=args.eval_max_tokens)
    cnn_test = cnn_dailymail_eval_ids(tok, max_tokens=args.eval_max_tokens)
    c4_test = c4_eval_ids(tok, max_tokens=args.eval_max_tokens)
    c4_probe = c4_probe_ids(tok, max_tokens=args.probe_tokens)
    log.info("eval tokens: pubmed=%d cnn=%d c4=%d probe=%d",
             pub_test.numel(), cnn_test.numel(), c4_test.numel(), c4_probe.numel())

    def eval_domain_cache(key, domain, ids, artifact, phase):
        if key in eval_cache:
            log.info("eval %s reused from checkpoint", key)
            return eval_cache[key]
        out = eval_artifacts(
            model, tok, ids, domain, args.eval_window, args.eval_stride,
            args.frozen_cache_dir, live_adapters=adapters,
            lt_states=(lt_states if artifact == "lt" else None),
            artifact=artifact, phase=phase,
        )
        eval_cache[key] = out
        return out

    def on_boundary(step, phase):
        if condition == "b2":
            # B2: continuing adapter — eval at boundaries for the BT trace.
            if phase == "d1_pubmed":
                eval_domain_cache("pubmed_after_d1", "pubmed", pub_test,
                                  "full", "after_d1")
            elif phase == "d2_cnn":
                eval_domain_cache("pubmed_after_d2", "pubmed", pub_test,
                                  "full", "after_d2")
                eval_domain_cache("cnn_after_d2", "cnn", cnn_test,
                                  "full", "after_d2")
            return
        # condition == "c": capture ST final, compute the top-K delta from the
        # ST's warm-start (= the current LT; D1's LT is the zero store, so
        # delta_1 is the full D1 change), add to LT, warm-start ST from the
        # new LT, and eval the consolidated LT store.
        st_final = [tc_latent_state(ad) for ad in adapters]
        delta = latent_change_topk(lt_states, st_final, args.consolidate_k)
        add_delta_to_lt(lt_states, delta)
        warm_start_st_from_lt(adapters, lt_states)
        # "Reset short-term": the ST's AdamW momentum/velocity from the
        # previous domain is stale for the newly warm-started parameters —
        # clear it so the next domain starts with a fresh optimizer (a clean
        # ST reset; the surprise EMA stays continuous so the boundary gate
        # still opens).
        optimizer.state.clear()
        log.info("consolidated %s: top-K(%.0f%%) n_kept=%d threshold=%.3e "
                 "-> LT + %d adapters, optimizer reset", phase,
                 100 * args.consolidate_k, delta["n_kept"], delta["threshold"],
                 len(lt_states))
        if phase == "d1_pubmed":
            eval_domain_cache("pubmed_after_d1", "pubmed", pub_test,
                              "lt", "after_d1")
        elif phase == "d2_cnn":
            eval_domain_cache("pubmed_after_d2", "pubmed", pub_test,
                              "lt", "after_d2")
            eval_domain_cache("cnn_after_d2", "cnn", cnn_test,
                              "lt", "after_d2")
        # phase == "d3_c4": the final D3 evals happen after training returns.

    train_metrics, probe_metrics = train_gated(
        args, model, tok, adapters, optimizer, modulator, data_iter,
        start_step, total_steps, args.warmup_steps, ckpt_dir,
        boundaries=boundaries, on_boundary=on_boundary,
        probe_ids=c4_probe, probe_start=phase2_end + 1,
        lt_states=lt_states, boundaries_done=boundaries_done,
        eval_cache=eval_cache, checkpoint_every=max(args.grad_accum * 100, 1),
    )

    # ── final evals after the full sequence ─────────────────────────
    if condition == "b2":
        eval_domain_cache("pubmed_after_d3", "pubmed", pub_test, "full", "after_d3")
        eval_domain_cache("cnn_after_d3", "cnn", cnn_test, "full", "after_d3")
        eval_domain_cache("c4_after_d3", "c4", c4_test, "full", "after_d3")
        metrics = {
            "d1_domain": "pubmed",
            "d1_ppl_after_d1": eval_cache["pubmed_after_d1"]["plastic_ppl"],
            "d1_ppl_after_d3": eval_cache["pubmed_after_d3"]["plastic_ppl"],
            "d1_ppl_delta_after_d3": eval_cache["pubmed_after_d3"]["delta_ppl"],
            "bt_d1": eval_cache["pubmed_after_d3"]["plastic_ppl"] -
                     eval_cache["pubmed_after_d1"]["plastic_ppl"],
            "d2_domain": "cnn",
            "d2_ppl_after_d2": eval_cache["cnn_after_d2"]["plastic_ppl"],
            "d2_ppl_after_d3": eval_cache["cnn_after_d3"]["plastic_ppl"],
            "d2_ppl_delta_after_d3": eval_cache["cnn_after_d3"]["delta_ppl"],
            "bt_d2": eval_cache["cnn_after_d3"]["plastic_ppl"] -
                     eval_cache["cnn_after_d2"]["plastic_ppl"],
            "d3_domain": "c4",
            "d3_ppl_after_d3": eval_cache["c4_after_d3"]["plastic_ppl"],
            "d3_ppl_delta_after_d3": eval_cache["c4_after_d3"]["delta_ppl"],
            "d3_ppl_delta_ci95": eval_cache["c4_after_d3"]["delta_ppl_ci95"],
        }
    else:  # condition == "c"
        # Capture ST_3 (full warm-started D3 short-term) BEFORE the transfer,
        # then transfer into LT and eval the consolidated store on all domains.
        st3_final = [tc_latent_state(ad) for ad in adapters]
        delta3 = latent_change_topk(lt_states, st3_final, args.consolidate_k)
        add_delta_to_lt(lt_states, delta3)
        eval_domain_cache("c4_st_after_d3", "c4", c4_test, "st", "after_d3")
        eval_domain_cache("pubmed_after_d3", "pubmed", pub_test, "lt", "after_d3")
        eval_domain_cache("cnn_after_d3", "cnn", cnn_test, "lt", "after_d3")
        eval_domain_cache("c4_after_d3", "c4", c4_test, "lt", "after_d3")
        log.info("final consolidation: LT now holds delta_1+delta_2+delta_3 "
                 "(n_kept=%d)", delta3["n_kept"])
        metrics = {
            "d1_domain": "pubmed",
            "d1_ppl_after_d1": eval_cache["pubmed_after_d1"]["plastic_ppl"],
            "d1_ppl_after_d3": eval_cache["pubmed_after_d3"]["plastic_ppl"],
            "d1_ppl_delta_after_d3": eval_cache["pubmed_after_d3"]["delta_ppl"],
            "bt_d1": eval_cache["pubmed_after_d3"]["plastic_ppl"] -
                     eval_cache["pubmed_after_d1"]["plastic_ppl"],
            "d2_domain": "cnn",
            "d2_ppl_after_d2": eval_cache["cnn_after_d2"]["plastic_ppl"],
            "d2_ppl_after_d3": eval_cache["cnn_after_d3"]["plastic_ppl"],
            "d2_ppl_delta_after_d3": eval_cache["cnn_after_d3"]["delta_ppl"],
            "bt_d2": eval_cache["cnn_after_d3"]["plastic_ppl"] -
                     eval_cache["cnn_after_d2"]["plastic_ppl"],
            "d3_domain": "c4",
            "d3_ppl_st_after_d3": eval_cache["c4_st_after_d3"]["plastic_ppl"],
            "d3_ppl_st_delta_after_d3": eval_cache["c4_st_after_d3"]["delta_ppl"],
            "d3_ppl_st_delta_ci95": eval_cache["c4_st_after_d3"]["delta_ppl_ci95"],
            "d3_ppl_lt_after_d3": eval_cache["c4_after_d3"]["plastic_ppl"],
            "d3_ppl_lt_delta_after_d3": eval_cache["c4_after_d3"]["delta_ppl"],
            "d3_ppl_lt_delta_ci95": eval_cache["c4_after_d3"]["delta_ppl_ci95"],
        }

    # Gate statistics.
    ms = [m["modulator_M"] for m in train_metrics]
    adapt_ms = ms[args.warmup_steps:] if len(ms) > args.warmup_steps else ms
    gate_stats = {
        "mean_surprise_M": float(sum(ms) / len(ms)) if ms else 0.0,
        "mean_M_adapt": float(sum(adapt_ms) / max(len(adapt_ms), 1)),
        "pct_steps_M_gt_05": float(sum(1.0 for m in ms if m > 0.5) / len(ms)) if ms else 0.0,
        "final_M": float(ms[-1]) if ms else 0.0,
        "effective_mean_lr": float(sum(m["effective_lr"] for m in train_metrics) /
                                  len(train_metrics)) if train_metrics else 0.0,
        "n_steps": len(ms),
        "mean_step_time_s": float(sum(m["step_time_s"] for m in train_metrics) /
                                  len(train_metrics)) if train_metrics else 0.0,
    }

    if condition == "b2":
        storage = storage_report(adapters, args.output_dir, mshort,
                                 args.tag, args.seed)
    else:
        storage = consolidation_storage_report(adapters, n_deltas=3,
                                               k=args.consolidate_k)
        # Also write the LT packed artifact for the storage accounting.
        packed = pack_ternary_adapters(adapters)
        lt_report = ternary_storage_report(adapters, packed)
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir,
                            f"{mshort}_{args.tag}_lt_seed{args.seed}.ternary2bit")
        with open(path, "wb") as fh:
            fh.write(packed.cpu().numpy().tobytes())
        storage["lt_disk_bytes"] = int(os.path.getsize(path))
        storage["lt_report"] = lt_report

    plastic_weights = {
        "count": n_lora_params(adapters),
        "mean_magnitude": float(_adapter_weights(adapters).abs().mean()),
        "sparsity": float((_adapter_weights(adapters).abs() < 1e-8).float().mean()),
    }

    consolidation = None
    if condition == "c":
        consolidation = {
            "k": args.consolidate_k,
            "transfer_rule": "add",
            "lt_decay": args.lt_decay,
            "warm_start": "copy_lt_into_st",
            "n_deltas": 3,
            "lt_n_params": n_lora_params(adapters),
        }

    result = {
        "experiment": "e036_consolidation",
        "step": "2.3",
        "condition": condition,
        "b1_domain": None,
        "tag": args.tag,
        "adapter": "tc",
        "seed": args.seed,
        "model": args.model,
        "model_short": mshort,
        "warmup_steps": args.warmup_steps,
        "adaptation_tokens": args.budget_tokens,
        "adaptation_steps": adapt_steps,
        "sequence": list(SEQUENCE),
        "consolidation": consolidation,
        "eval_cache": eval_cache,
        "metrics": metrics,
        "probe_metrics": probe_metrics,
        "storage": storage,
        "plastic_weights": plastic_weights,
        "train_metrics": train_metrics,
    }
    return result


# ── arg parser ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--condition", choices=CONDITIONS, default="c")
    p.add_argument("--b1-domain", choices=B1_DOMAINS, default=None,
                   help="B1: which independent per-domain adapter to run")
    p.add_argument("--tag", required=True)
    p.add_argument("--budget-tokens", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-3)  # η (E034 base)
    p.add_argument("--consolidate-k", type=float, default=DEFAULT_CONSOLIDATE_K)
    p.add_argument("--lt-decay", type=float, default=DEFAULT_LT_DECAY)
    p.add_argument("--s0", type=float, default=DEFAULT_S0)
    p.add_argument("--k", type=float, default=DEFAULT_K_SIG)
    p.add_argument("--m-max", type=float, default=DEFAULT_M_MAX)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--no-checkpoint", action="store_true",
                   help="disable grad checkpointing")
    p.add_argument("--output-dir", default="results/brain/e036")
    p.add_argument("--log-dir", default="logs/brain/e036")
    p.add_argument("--frozen-cache-dir", default="results/brain/e036/cache")
    p.add_argument("--gpu-policy", choices=("exit", "wait", "warn"), default="exit")
    p.add_argument("--device", default=None)
    p.add_argument("--min-free-gb", type=float, default=None)
    p.add_argument("--eval-window", type=int, default=512)
    p.add_argument("--eval-stride", type=int, default=256)
    p.add_argument("--eval-max-tokens", type=int, default=500_000,
                   help="eval corpus size per domain (500K = locked protocol; "
                        "smaller only for smoke runs with a separate cache dir)")
    p.add_argument("--probe-tokens", type=int, default=DEFAULT_PROBE_TOKENS)
    p.add_argument("--no-deregister", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.condition == "b1" and args.b1_domain is None:
        raise SystemExit("--condition b1 requires --b1-domain pubmed|cnn|c4")
    if args.condition != "b1" and args.b1_domain is not None:
        raise SystemExit("--b1-domain only applies to --condition b1")

    mshort = model_short(args.model)
    seed = args.seed
    output_dir = os.path.abspath(args.output_dir)
    log_path = setup_logging(args.tag, args.budget_tokens, seed,
                             os.path.abspath(args.log_dir))
    b1_note = f" b1_domain={args.b1_domain}" if args.b1_domain else ""
    log.info("E036 start condition=%s%s tag=%s budget=%d seed=%d model=%s "
             "consolidate_k=%.2f (%s)",
             args.condition, b1_note, args.tag, args.budget_tokens, seed,
             args.model, args.consolidate_k, log_path)
    torch.manual_seed(seed)

    adapt_steps = math.ceil(args.budget_tokens / (args.batch_size * args.seq_len))
    if args.condition == "b1":
        total_steps = args.warmup_steps + adapt_steps
    else:
        total_steps = args.warmup_steps + 3 * adapt_steps
    log.info("adapt steps/domain=%d total=%d", adapt_steps, total_steps)

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

    adapters = build_ternary_lora_adapters(model, args.rank, device, mode="tc")
    n_params = n_lora_params(adapters)
    log.info("%d T-C ternary adapters, %d trainable params (%.1f KB fp32) — "
             "budget match to E034/E035 (rank %d)",
             len(adapters), n_params, n_params * 4 / 1024, args.rank)

    params = [p for ad in adapters for p in ad.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    modulator = SurpriseModulator(
        mode="surprise_ema", alpha=args.alpha, s0=args.s0, k=args.k, M_max=args.m_max,
    )

    # Result JSON path — computed early; "already complete" = its existence.
    budget_tag = f"{args.budget_tokens // 1000}k"
    if args.condition == "b1":
        fname = f"{mshort}_b1_{args.b1_domain}_{budget_tag}_{args.tag}_seed{seed}.json"
    else:
        fname = f"{mshort}_{args.condition}_{budget_tag}_{args.tag}_seed{seed}.json"
    out_path = os.path.join(output_dir, fname)
    if os.path.exists(out_path):
        log.info("already complete (result JSON exists): %s", out_path)
        return 0

    if args.condition == "b1":
        result = run_b1(args, model, tok, adapters, optimizer, modulator, mshort)
    else:
        result = run_sequence(args, model, tok, adapters, optimizer, modulator, mshort)

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    m = result["metrics"]
    if args.condition == "b1":
        log.info("RESULT -> %s | b1 domain=%s Δppl=%+.3f | forgetting=%+.3f%%",
                 out_path, args.b1_domain, m["domain_ppl_delta"],
                 m["forgetting_pct"])
    elif args.condition == "b2":
        log.info("RESULT -> %s | b2 D3 Δppl=%+.3f | BT_D1=%+.4f BT_D2=%+.4f",
                 out_path, m["d3_ppl_delta_after_d3"], m["bt_d1"], m["bt_d2"])
    else:
        log.info("RESULT -> %s | c D3(st) Δppl=%+.3f D3(lt)=%+.3f | "
                 "BT_D1=%+.4f BT_D2=%+.4f",
                 out_path, m["d3_ppl_st_delta_after_d3"],
                 m["d3_ppl_lt_delta_after_d3"], m["bt_d1"], m["bt_d2"])
    log.info("E036 complete condition=%s tag=%s seed=%d",
             args.condition, args.tag, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
