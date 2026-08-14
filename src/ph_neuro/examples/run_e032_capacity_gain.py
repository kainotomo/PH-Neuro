#!/usr/bin/env python3
"""E032 — Capacity & Gain experiment runner (Phase 1.2, local plasticity).

Frozen SmolLM2-1.7B + **low-rank plastic matrices** (``W_plastic = B@A``)
updated by a local surprise-modulated Hebbian rule (no backprop), evaluated
against the LOCKED Step 0.5 protocol (WikiText-2 source → PubMed target).
One process runs one (config, budget, seed) cell and writes one
protocol-schema result JSON to ``results/brain/e032/``.

A. Rank sweep:       ``--plasticity low_rank --rank {1,2,4}`` (E031 defaults
                     for η and the surprise signal — isolates capacity).
B. Gain sweep:       ``--lr {1e-3,3e-3,1e-2}`` then ``--s0/--k`` then
                     ``--m-max`` at the best rank.
C. Decay ablation:   ``--decay {1e-5,1e-4}`` at the best A+B config.
E. 1M anneal:        ``--budget-tokens 1000000`` at the best local config.

The low-rank local update (derivation in ``docs/brain/06-e032-capacity-gain.md``):

    ΔW = η·M·mean_t(pre ⊗ post)            (3-factor Hebbian, as E031)
    ΔA = η·M·mean_t((Bᵀ·post_t) ⊗ pre_t)   (project ΔW onto A via Bᵀ)
    ΔB = η·M·mean_t(post_t ⊗ (A·pre_t))    (project ΔW onto B via A)

Init: A ~ N(0, 1/d_in), B = 0 (scaled random projection — deadlock-break,
stable; see docs/brain/06-e032-capacity-gain.md §3.3).

Env notes: identical to E031 (no C compiler → disable the Triton ``bmm``
path; ``accelerate`` absent; GPU shared with a game → gated before every run).

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e032_capacity_gain \\
        --plasticity low_rank --rank 4 --tag lrr4 --budget-tokens 100000 \\
        --seed 42

Output: ``results/brain/e032/smolllm2_1p7b_pubmed_{budget}_{tag}_seed{seed}.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── quiet HF cache chatter ─────────────────────────────────────────
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from ph_neuro.brain import BrainWrapper  # noqa: E402
from ph_neuro.brain.brain_wrapper import check_gpu_free, gpu_free_mb  # noqa: E402
from ph_neuro.brain.datasets import (  # noqa: E402
    make_combined_batch_iter,
    pubmed_eval_ids,
    pubmed_train_ids,
    wikitext_ids,
)
from ph_neuro.brain.stats import block_paired_stats  # noqa: E402

log = logging.getLogger("e032")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 1024

# E031 locked surprise defaults (kept unless overridden by the sweep).
DEFAULT_LR = 1e-3
DEFAULT_S0 = 0.05
DEFAULT_K = 60.0
DEFAULT_M_MAX = 1.0
DEFAULT_DECAY = 0.0


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
    path = os.path.join(log_dir, f"e032_{tag}_budget{budget}_seed{seed}.log")
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handlers = [
        logging.FileHandler(path, mode="a"),
        logging.StreamHandler(sys.stdout),
    ]
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


def frozen_eval(brain: BrainWrapper, domain_ids: torch.Tensor, domain: str,
                cache_dir: str, window: int, stride: int) -> dict:
    """Frozen eval for a domain; cached to disk (seed-independent, reused).

    If ``cache_dir`` holds an E031-produced cache (results/brain/e031/cache),
    it is reused so frozen ppl is identical across experiments (they measure
    the same seed-independent baseline).
    """
    cache_path = _frozen_cache_path(cache_dir, model_short(
        brain.model.config._name_or_path or brain.model.config.model_type
    ), domain)
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            cached = json.load(fh)
        log.info("frozen %s eval reused from cache (%d blocks)", domain, cached["n_blocks"])
        return cached
    log.info("computing frozen %s eval (seed-independent baseline)", domain)
    res = brain.evaluate(ids=domain_ids, window=window, stride=stride, mode="frozen")
    summary = res["frozen"]
    payload = {
        "domain": domain,
        "ppl": summary["ppl"],
        "mean_nll": summary["mean_nll"],
        "n_tokens": summary["n_tokens"],
        "n_blocks": len(summary["per_block"]["nll"]),
        "per_block": summary["per_block"],
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, cache_path)
    return payload


# ── main ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--plasticity", choices=("low_rank", "vector_bias"), default="low_rank")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--tag", required=True, help="experiment cell tag (e.g. lrr4)")
    p.add_argument("--budget-tokens", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    p.add_argument("--s0", type=float, default=DEFAULT_S0)
    p.add_argument("--k", type=float, default=DEFAULT_K)
    p.add_argument("--m-max", type=float, default=DEFAULT_M_MAX)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
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
    p.add_argument("--keep-checkpoints", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tag = args.tag
    mshort = model_short(args.model)
    seed = args.seed
    output_dir = os.path.abspath(args.output_dir)
    log_path = setup_logging(tag, args.budget_tokens, seed, os.path.abspath(args.log_dir))
    log.info(
        "E032 start tag=%s plasticity=%s rank=%d budget=%d seed=%d model=%s (%s)",
        tag, args.plasticity, args.rank, args.budget_tokens, seed, args.model, log_path,
    )
    torch.manual_seed(seed)

    adapt_steps = math.ceil(args.budget_tokens / (args.batch_size * args.seq_len))
    total_steps = args.warmup_steps + adapt_steps
    log.info("adapt steps=%d total steps=%d", adapt_steps, total_steps)

    if not args.no_deregister:
        disable_triton_bmm()

    min_free_gb = args.min_free_gb or default_min_free_gb(args.model)
    log.info("GPU pre-check: need >= %.1f GiB free", min_free_gb)
    check_gpu_free(min_free_gb, args.gpu_policy, log)
    free_mb = gpu_free_mb()
    log.info("GPU free: %s MiB", free_mb if free_mb is not None else "n/a")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.model, device)

    ckpt_dir = os.path.join(output_dir, "checkpoints", f"{tag}_budget{args.budget_tokens}_seed{seed}")
    if args.keep_checkpoints:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Surprise modulator config from the sweep params (all explicit).
    modulator_cfg = {
        "mode": "surprise_ema",
        "alpha": 0.99,
        "s0": args.s0,
        "k": args.k,
        "M_max": args.m_max,
    }

    brain = BrainWrapper(
        model,
        plasticity=args.plasticity,
        rank=args.rank,
        modulator_cfg=modulator_cfg,
        lr=args.lr,
        decay_rate=args.decay,
        tokenizer=tok,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=100,
        min_free_gb=min_free_gb,
        log=log,
    )
    log.info("brain summary: %s", json.dumps(brain.summary()))

    # ── learn (warmup on wiki M=0 → adapt on pubmed) ───────────────
    log.info("building learn stream: warmup=%d steps (wiki) + adapt=%d steps (pubmed)",
             args.warmup_steps, adapt_steps)
    wiki_ids = wikitext_ids("train", tok)
    pub_ids = pubmed_train_ids(tok)
    batch_iter = make_combined_batch_iter(
        wiki_ids, pub_ids, args.warmup_steps, args.batch_size, args.seq_len, seed
    )
    t0 = time.time()
    train_metrics = brain.learn(
        batch_iter,
        steps=total_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gpu_policy=args.gpu_policy,
        warmup_steps=args.warmup_steps,
        seed=seed,
    )
    log.info("learning done in %.1f s (%d steps)", time.time() - t0, len(train_metrics))

    # ── evaluation ─────────────────────────────────────────────────
    log.info("tokenizing eval corpora…")
    wiki_test = wikitext_ids("test", tok)
    pub_test = pubmed_eval_ids(tok, max_tokens=args.eval_pubmed_tokens)
    log.info("eval tokens: wiki_test=%d, pubmed_eval=%d",
             wiki_test.numel(), pub_test.numel())

    def eval_domain(domain_ids: torch.Tensor, domain: str):
        frozen = frozen_eval(brain, domain_ids, domain, args.frozen_cache_dir,
                             args.eval_window, args.eval_stride)
        plastic_res = brain.evaluate(
            ids=domain_ids, window=args.eval_window, stride=args.eval_stride,
            mode="plastic",
        )
        plastic = plastic_res["plastic"]
        stats = block_paired_stats(
            frozen["per_block"]["nll"],
            plastic["per_block"]["nll"],
            frozen["per_block"]["tokens"],
        )
        return frozen, plastic, stats

    src_frozen, src_plastic, src_stats = eval_domain(wiki_test, "wikitext2")
    tgt_frozen, tgt_plastic, tgt_stats = eval_domain(pub_test, "pubmed")

    # ── metrics ────────────────────────────────────────────────────
    source_ppl_delta = src_plastic["ppl"] - src_frozen["ppl"]  # + = forgetting
    target_ppl_delta = tgt_frozen["ppl"] - tgt_plastic["ppl"]  # + = improved
    forgetting_pct = (src_plastic["ppl"] / src_frozen["ppl"] - 1.0) * 100.0

    surprise_stats: dict = {}
    if train_metrics:
        ms = [m["modulator_M"] for m in train_metrics]
        surprise_stats = {
            "mean_surprise_M": float(sum(ms) / len(ms)),
            "pct_steps_M_gt_05": float(sum(1.0 for m in ms if m > 0.5) / len(ms)),
            "final_M": float(ms[-1]),
            "n_steps": len(ms),
        }

    # Plastic-magnitude diagnostics over A and B (low-rank) or bias (vector).
    mags = []
    for ip in brain._injection_points:  # noqa: SLF001 - deliberate introspection
        if ip.A is not None:
            mags.append(ip.A.detach().flatten())
            mags.append(ip.B.detach().flatten())
        else:
            mags.append(ip.bias.detach().flatten())
    allw = torch.cat(mags)
    plastic_weights = {
        "count": brain.plastic_parameter_count(),
        "bytes": brain.plastic_memory_bytes(),
        "mean_magnitude": float(allw.abs().mean()),
        "max_magnitude": float(allw.abs().max()),
        "sparsity": float((allw.abs() < 1e-8).float().mean()),
        "mean_abs_B": float(
            torch.cat([ip.B.detach().flatten() for ip in brain._injection_points
                       if ip.B is not None]).abs().mean()
        ),
        "mean_abs_A": float(
            torch.cat([ip.A.detach().flatten() for ip in brain._injection_points
                       if ip.A is not None]).abs().mean()
        ),
    }

    result = {
        "experiment": "e032_capacity_gain",
        "step": "1.2",
        "tag": tag,
        "method": "lowrank" if args.plasticity == "low_rank" else "vector_bias",
        "plasticity": args.plasticity,
        "rank": args.rank,
        "model": args.model,
        "model_short": mshort,
        "modulator": {
            "mode": "surprise_ema",
            "alpha": 0.99,
            "s0": args.s0,
            "k": args.k,
            "M_max": args.m_max,
        },
        "source_domain": "wikitext2",
        "target_domain": "pubmed",
        "adaptation_tokens": args.budget_tokens,
        "adaptation_steps": adapt_steps,
        "warmup_steps": args.warmup_steps,
        "seed": seed,
        "lr": args.lr,
        "decay_rate": args.decay,
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
            **surprise_stats,
        },
        "plastic_weights": plastic_weights,
        "n_target_blocks": tgt_stats["n_blocks"],
        "n_source_blocks": src_stats["n_blocks"],
        "train_metrics": train_metrics,
    }

    os.makedirs(output_dir, exist_ok=True)
    budget_tag = f"{args.budget_tokens // 1000}k"
    out_path = os.path.join(
        output_dir, f"{mshort}_pubmed_{budget_tag}_{tag}_seed{seed}.json"
    )
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log.info(
        "RESULT -> %s | target Δppl=%+.3f (ci %s) | source forgetting=%+.3f%%",
        out_path, target_ppl_delta, [f"{x:.3f}" for x in tgt_stats["delta_ppl_ci95"]],
        forgetting_pct,
    )
    log.info("E032 complete tag=%s budget=%d seed=%d", tag, args.budget_tokens, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
