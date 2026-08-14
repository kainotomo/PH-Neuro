#!/usr/bin/env python3
"""E034 — verify the second domain (CNN/DailyMail) + compute its frozen baseline.

Documents the E034 second-domain choice and produces the seed-independent
frozen-eval cache for CNN/DailyMail (and, for completeness, records the frozen
ppl of all three domains on the same protocol window/stride). The CNN frozen
cache is reused by every E034 cell (skip-if-exists), so this runs once.

Verified 2026-08-14: ``abisee/cnn_dailymail`` config ``3.0.0``, license
**apache-2.0**, splits train 287,113 / val 13,368 / test 11,490, fields
``article``/``highlights``/``id`` (document = ``article``). Frozen ppl on a
deterministic 500,000-token test subsample (seed 42): **11.971** — a moderate
+12.3% shift over WikiText-2 (10.664) and +4.5% over PubMed (11.457), i.e. in
the surprise sigmoid's sensitive range (the design's requirement).

Usage::

    .venv/bin/python research/scripts/verify_e034_second_domain.py \\
        --cache-dir results/brain/e034/cache

Writes ``results/brain/e034/cache/frozen_smolllm2_1p7b_cnn_dailymail.json``
and ``results/brain/e034/cache/e034_second_domain_verify.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()

from ph_neuro.brain.brain_wrapper import check_gpu_free  # noqa: E402
from ph_neuro.brain.datasets import (  # noqa: E402
    cnn_dailymail_eval_ids,
    pubmed_eval_ids,
    wikitext_ids,
)

WINDOW = 512
STRIDE = 256
EVAL_TOKENS = 500_000

DOMAINS = ("wikitext2", "pubmed", "cnn_dailymail")


def _eval_frozen(model, ids: torch.Tensor) -> dict:
    ids = ids.to(next(model.parameters()).device)
    n = ids.numel()
    blocks: list[tuple[float, int]] = []
    with torch.no_grad():
        for begin in range(0, n, STRIDE):
            end = min(begin + WINDOW, n)
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
        "n_blocks": len(blocks),
        "per_block": {"nll": [b[0] for b in blocks], "tokens": [b[1] for b in blocks]},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--cache-dir", default="results/brain/e034/cache")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    mshort = "smolllm2_1p7b" if "SmolLM2-1.7B" in args.model else args.model.replace("/", "__")
    os.makedirs(args.cache_dir, exist_ok=True)
    check_gpu_free(args.min_free_gb, "exit")

    try:
        torch.backends.python_native.disable_operations("bmm")
    except Exception:  # noqa: BLE001 - API may differ across torch builds
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    summaries: dict[str, dict] = {}
    for domain in DOMAINS:
        cache_path = os.path.join(args.cache_dir, f"frozen_{mshort}_{domain}.json")
        if domain == "wikitext2":
            ids = wikitext_ids("test", tok)
        elif domain == "pubmed":
            ids = pubmed_eval_ids(tok, max_tokens=EVAL_TOKENS)
        else:
            ids = cnn_dailymail_eval_ids(tok, max_tokens=EVAL_TOKENS)
        # Reuse an existing frozen cache (wiki/pubmed come from E031/E032 —
        # seed-independent, bit-identical) instead of recomputing.
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                summary = json.load(fh)
        else:
            summary = _eval_frozen(model, ids)
            tmp = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump({"domain": domain, **summary}, fh)
            os.replace(tmp, cache_path)
        summaries[domain] = {
            "ppl": summary["ppl"],
            "n_tokens": summary["n_tokens"],
            "n_blocks": summary["n_blocks"],
            "cache": os.path.exists(cache_path),
        }
        print(f"frozen {domain}: ppl={summary['ppl']:.3f} "
              f"(tokens={summary['n_tokens']}, blocks={summary['n_blocks']})")

    base = summaries["wikitext2"]["ppl"]
    table = {
        "domain": {d: s["ppl"] for d, s in summaries.items()},
        "shift_vs_wikitext_pct": {
            d: round((s["ppl"] / base - 1.0) * 100.0, 2) for d, s in summaries.items()
        },
    }
    verify = {
        "second_domain": "cnn_dailymail",
        "hf_id": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "license": "apache-2.0",
        "document_field": "article",
        "eval_tokens": EVAL_TOKENS,
        "eval_seed": 42,
        "window": WINDOW,
        "stride": STRIDE,
        "frozen_ppl_table": table,
        "caches": {d: s["cache"] for d, s in summaries.items()},
        "verified": "2026-08-14",
    }
    out_path = os.path.join(args.cache_dir, "e034_second_domain_verify.json")
    with open(out_path, "w") as fh:
        json.dump(verify, fh, indent=2)
    print(f"verify JSON -> {out_path}")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
