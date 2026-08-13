#!/usr/bin/env python3
"""E031 — Minimal Viable Brain-Wrapper experiment (Phase 1.1).

Frozen SmolLM2-1.7B + surprise-modulated vector-bias Hebbian plasticity on
PubMed, evaluated against the LOCKED Step 0.5 protocol (WikiText-2 source →
PubMed target). One process runs one (baseline, budget, seed) cell and writes
one protocol-schema result JSON to ``results/brain/e031/``.

Baselines (protocol §4)::

    frozen    — zero plasticity (eval with without_plasticity()).
    random    — plastic biases init to randn(0, 0.01) on the seed; no learning.
    constM    — Hebbian with constant M = 1.0 (no surprise). Key ablation.
    surprise  — surprise-modulated EMA/sigmoid M (the method). Locked defaults.

Budgets (protocol §3): 10k = mechanism go/no-go; **100k = primary surprise
test point** (EMA τ ≈ 102K tokens). 1k = micro sanity only.

Env notes (verified 2026-08-12): this machine has no C compiler, so torch
2.13's Triton fused ``bmm`` (RoPE path) must be disabled via
``torch.backends.python_native.disable_operations("bmm")`` plus
``attn_implementation="eager"`` when loading the model. ``accelerate`` is not
installed. GPU is shared with a game — free memory is gated before every run.

Usage::

    .venv/bin/python -m ph_neuro.examples.run_e031_minimal_viable \\
        --baseline surprise --budget-tokens 100000 --seed 42

Output: ``results/brain/e031/{model}_{target}_{budget}_{baseline}_seed{seed}.json``
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
    make_combined_batch_iter,  # noqa: E402
    pubmed_eval_ids,
    pubmed_train_ids,
    wikitext_ids,
)
from ph_neuro.brain.stats import block_paired_stats  # noqa: E402

log = logging.getLogger("e031")

BATCH_SIZE = 4
SEQ_LEN = 256
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 1024

BASELINES = ("frozen", "random", "constM", "surprise")


def model_short(model_id: str) -> str:
    if "SmolLM2-1.7B" in model_id:
        return "smolllm2_1p7b"
    if "gpt2" in model_id:
        return "gpt2_124m"
    return model_id.replace("/", "__")


def default_min_free_gb(model_id: str) -> float:
    if "SmolLM2" in model_id:
        return 6.0  # protocol §Operational Design (RTX 4060 8 GB, shared GPU)
    if "gpt2" in model_id:
        return 2.0
    return 4.0


def setup_logging(baseline: str, budget: int, seed: int, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(
        log_dir, f"e031_{baseline}_budget{budget}_seed{seed}.log"
    )
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


def _frozen_cache_path(output_dir: str, mshort: str, domain: str) -> str:
    return os.path.join(output_dir, "cache", f"frozen_{mshort}_{domain}.json")


def frozen_eval(brain: BrainWrapper, domain_ids: torch.Tensor, domain: str,
                cache_path: str, window: int, stride: int) -> dict:
    """Frozen eval for a domain; cached to disk (seed-independent, reused)."""
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            cached = json.load(fh)
        log.info("frozen %s eval reused from cache (%d blocks)", domain, cached["n_blocks"])
        return cached
    log.info("computing frozen %s eval (this is the seed-independent baseline)", domain)
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
    p.add_argument("--baseline", choices=BASELINES, required=True)
    p.add_argument("--budget-tokens", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--output-dir", default="results/brain/e031")
    p.add_argument("--log-dir", default="logs/brain/e031")
    p.add_argument("--gpu-policy", choices=("exit", "wait", "warn"), default="exit")
    p.add_argument("--device", default=None)
    p.add_argument("--min-free-gb", type=float, default=None)
    p.add_argument("--eval-window", type=int, default=512)
    p.add_argument("--eval-stride", type=int, default=256)
    p.add_argument("--eval-pubmed-tokens", type=int, default=500_000)
    p.add_argument("--no-deregister", action="store_true", help="skip Triton bmm workaround")
    p.add_argument("--keep-checkpoints", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    baseline, budget = args.baseline, args.budget_tokens
    mshort = model_short(args.model)
    seed = args.seed
    output_dir = os.path.abspath(args.output_dir)
    log_path = setup_logging(baseline, budget, seed, os.path.abspath(args.log_dir))
    log.info(
        "E031 start baseline=%s budget=%d seed=%d model=%s (%s)",
        baseline, budget, seed, args.model, log_path,
    )
    torch.manual_seed(seed)

    if baseline in ("frozen", "random"):
        budget = 0  # no adaptation for these baselines
    adapt_steps = math.ceil(budget / (args.batch_size * args.seq_len))
    total_steps = args.warmup_steps + adapt_steps if baseline == "surprise" else adapt_steps
    warmup_steps = args.warmup_steps if baseline == "surprise" else 0

    if not args.no_deregister:
        disable_triton_bmm()

    min_free_gb = args.min_free_gb or default_min_free_gb(args.model)
    log.info("GPU pre-check: need >= %.1f GiB free", min_free_gb)
    check_gpu_free(min_free_gb, args.gpu_policy, log)
    free_mb = gpu_free_mb()
    log.info("GPU free: %s MiB", free_mb if free_mb is not None else "n/a")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.model, device)

    ckpt_dir = None
    if baseline in ("constM", "surprise"):
        ckpt_dir = os.path.join(output_dir, "checkpoints", f"{baseline}_{budget}_seed{seed}")
        if args.keep_checkpoints:
            os.makedirs(ckpt_dir, exist_ok=True)

    modulator_cfg: dict | None = None
    if baseline == "constM":
        modulator_cfg = {"mode": "constant", "M": 1.0}
    # surprise uses the locked defaults (no cfg needed)

    brain = BrainWrapper(
        model,
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

    # ── baseline-specific plastic init / learning ──────────────────
    train_metrics: list[dict] = []
    if baseline == "random":
        # Generator must match the bias device (CUDA) — a CPU generator
        # raises "Expected a 'cuda' device type for generator".
        g = torch.Generator(device=brain.device).manual_seed(seed)
        for ip in brain._injection_points:  # noqa: SLF001 - deliberate baseline init
            ip.bias.normal_(0.0, 0.01, generator=g)
        log.info("random baseline: plastic biases ~ N(0, 0.01) on seed %d", seed)

    if baseline in ("constM", "surprise"):
        log.info(
            "building learn stream: warmup=%d steps (wiki) + adapt=%d steps (pubmed)",
            warmup_steps, adapt_steps,
        )
        wiki_ids = wikitext_ids("train", tok)
        pub_ids = pubmed_train_ids(tok)
        batch_iter = make_combined_batch_iter(
            wiki_ids, pub_ids, warmup_steps, args.batch_size, args.seq_len, seed
        )
        t0 = time.time()
        train_metrics = brain.learn(
            batch_iter,
            steps=total_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            gpu_policy=args.gpu_policy,
            warmup_steps=warmup_steps,
            seed=seed,
        )
        log.info("learning done in %.1f s (%d steps)", time.time() - t0, len(train_metrics))

    # ── evaluation ─────────────────────────────────────────────────
    log.info("tokenizing eval corpora…")
    wiki_test = wikitext_ids("test", tok)
    pub_test = pubmed_eval_ids(tok, max_tokens=args.eval_pubmed_tokens)
    log.info(
        "eval tokens: wiki_test=%d, pubmed_eval=%d",
        wiki_test.numel(), pub_test.numel(),
    )

    def eval_domain(domain_ids: torch.Tensor, domain: str):
        cache_path = _frozen_cache_path(output_dir, mshort, domain)
        frozen = frozen_eval(brain, domain_ids, domain, cache_path,
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

    biases = torch.cat([ip.bias.detach().flatten() for ip in brain._injection_points])
    plastic_weights = {
        "count": brain.plastic_parameter_count(),
        "bytes": brain.plastic_memory_bytes(),
        "mean_magnitude": float(biases.abs().mean()),
        "sparsity": float((biases.abs() < 1e-8).float().mean()),
        "max_magnitude": float(biases.abs().max()),
    }

    result = {
        "experiment": "e031_minimal_viable",
        "baseline": baseline,  # frozen | random | constM | surprise
        "model": args.model,
        "model_short": mshort,
        "plasticity": "vector_bias",
        "modulator": (
            "surprise_ema" if baseline == "surprise"
            else "constant" if baseline == "constM" else "none"
        ),
        "source_domain": "wikitext2",
        "target_domain": "pubmed",
        "adaptation_tokens": budget,
        "adaptation_steps": adapt_steps,
        "warmup_steps": warmup_steps,
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
    budget_tag = f"{budget // 1000}k" if baseline in ("constM", "surprise") else "na"
    out_path = os.path.join(
        output_dir, f"{mshort}_pubmed_{budget_tag}_{baseline}_seed{seed}.json"
    )
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log.info(
        "RESULT -> %s | target Δppl=%+.3f (ci %s) | source forgetting=%+.3f%%",
        out_path, target_ppl_delta, [f"{x:.3f}" for x in tgt_stats["delta_ppl_ci95"]],
        forgetting_pct,
    )
    log.info("E031 complete baseline=%s budget=%d seed=%d", baseline, budget, seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
