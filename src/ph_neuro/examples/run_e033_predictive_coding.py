#!/usr/bin/env python3
"""E033 — Predictive Coding experiment runner (Phase 1.3, re-scoped).

Error-based local rule (the LAST local-rule experiment): frozen
SmolLM2-1.7B + rank-1 plastic matrices ``W_plastic = B@A`` updated by a
**local, no-backprop predictive-coding rule** — the per-injection-site
reconstruction-error (PC-ERR) formulation — evaluated against the LOCKED
Step 0.5 protocol (WikiText-2 source → PubMed target). One process runs one
(config, budget, seed) cell and writes one protocol-schema result JSON to
``results/brain/e033/``.

Mechanism (derivation in ``docs/brain/07-e033-predictive-coding.md``):

    At each of the 48 injection sites (24 o_proj + 24 down_proj):
        x      = pre      (projection input,  captured)
        post   = frozen projection output     (captured, pre-injection)
        x̂      = W_inv·post = U@(V@post)      (low-rank linear inverse, rank 8)
        ε      = x − x̂                        (signed per-dim reconstruction error)
        inverse update (local recirculation, no backprop):
            ΔV = η_inv·mean((Uᵀ·ε) ⊗ post)/(rms(Uᵀε)·rms(post))
            ΔU = η_inv·mean(ε ⊗ (V·post))/(rms(ε)·rms(V·post))
        plastic update (error-driven; surprise-gated; ONLY pre → ε differs
            from the E032 Hebbian):
            ΔA = η·M·mean((Bᵀ·post) ⊗ ε)/(rms(Bᵀpost)·rms(ε))
            ΔB = η·M·mean(post ⊗ (A·ε))/(rms(post)·rms(A·ε))

Budget match: rank-1 A/B = 344,064 params — EXACTLY the E032 Part D LoRA
budget/init (A ~ N(0, 1/d_in), B = 0); only the update rule differs. The
inverse U/V are auxiliary learning machinery (like AdamW states), reported
separately, NOT counted in the matched budget.

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e033_predictive_coding \\
        --plasticity predictive_coding --rank 1 --tag pc --budget-tokens 100000 \\
        --seed 42

Output: ``results/brain/e033/smolllm2_1p7b_pubmed_{budget}_{tag}_seed{seed}.json``
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

log = logging.getLogger("e033")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 1024

# E031 locked surprise defaults (kept — only the update direction changes).
DEFAULT_LR = 1e-3
DEFAULT_S0 = 0.05
DEFAULT_K = 60.0
DEFAULT_M_MAX = 1.0
DEFAULT_DECAY = 0.0
DEFAULT_INV_RANK = 8
DEFAULT_INV_LR = 1e-3
DEFAULT_INV_DECAY = 1e-4


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
    path = os.path.join(log_dir, f"e033_{tag}_budget{budget}_seed{seed}.log")
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

    Reuses the E031/E032-produced cache when present so frozen ppl is
    bit-identical across experiments (same seed-independent baseline).
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
    p.add_argument("--plasticity", choices=("predictive_coding", "low_rank", "vector_bias"),
                   default="predictive_coding")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--inv-rank", type=int, default=DEFAULT_INV_RANK)
    p.add_argument("--inv-lr", type=float, default=DEFAULT_INV_LR)
    p.add_argument("--inv-decay", type=float, default=DEFAULT_INV_DECAY)
    p.add_argument("--tag", required=True, help="experiment cell tag (e.g. pc)")
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
    p.add_argument("--output-dir", default="results/brain/e033")
    p.add_argument("--log-dir", default="logs/brain/e033")
    p.add_argument("--frozen-cache-dir", default="results/brain/e033/cache")
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
        "E033 start tag=%s plasticity=%s rank=%d inv_rank=%d budget=%d seed=%d model=%s (%s)",
        tag, args.plasticity, args.rank, args.inv_rank, args.budget_tokens, seed,
        args.model, log_path,
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

    # Surprise modulator config (locked E031 defaults; only the update
    # direction changes in E033).
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
        inv_rank=args.inv_rank,
        inv_lr=args.inv_lr,
        inv_decay=args.inv_decay,
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

    # Plastic-magnitude diagnostics over A/B (and the auxiliary U/V inverse).
    mags, umags, vmags = [], [], []
    for ip in brain._injection_points:  # noqa: SLF001 - deliberate introspection
        if ip.A is not None:
            mags.append(ip.A.detach().flatten())
            mags.append(ip.B.detach().flatten())
        else:
            mags.append(ip.bias.detach().flatten())
        if ip.U is not None:
            umags.append(ip.U.detach().flatten())
            vmags.append(ip.V.detach().flatten())
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
    inverse_weights = {
        "count": brain.inverse_parameter_count(),
        "bytes": brain.inverse_parameter_count() * 4,
        "mean_abs_U": float(torch.cat(umags).abs().mean()) if umags else 0.0,
        "mean_abs_V": float(torch.cat(vmags).abs().mean()) if vmags else 0.0,
        "mean_abs_error": brain.mean_inverse_error(pub_test),
    }

    result = {
        "experiment": "e033_predictive_coding",
        "step": "1.3",
        "tag": tag,
        "method": "predictive_coding" if args.plasticity == "predictive_coding"
        else ("lowrank" if args.plasticity == "low_rank" else "vector_bias"),
        "plasticity": args.plasticity,
        "rank": args.rank,
        "inv_rank": args.inv_rank,
        "inv_lr": args.inv_lr,
        "inv_decay": args.inv_decay,
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
        "inverse_weights": inverse_weights,
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
        "RESULT -> %s | target Δppl=%+.3f (ci %s) | source forgetting=%+.3f%% "
        "| mean|ε|=%.4f",
        out_path, target_ppl_delta, [f"{x:.3f}" for x in tgt_stats["delta_ppl_ci95"]],
        forgetting_pct, inverse_weights["mean_abs_error"],
    )
    log.info("E033 complete tag=%s budget=%d seed=%d", tag, args.budget_tokens, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
