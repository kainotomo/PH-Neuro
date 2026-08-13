"""E031 dataset pipeline — WikiText-2 (source) + PubMed (target).

Implements the LOCKED Step 0.5 evaluation-protocol data requirements:

* **Source eval**: WikiText-2 ``test`` split in full (301,948 SmolLM2 tokens).
* **Target eval**: a deterministic **500,000-token** subsample of the PubMed
  ``test`` split (document permutation with ``random.Random(42)``, accumulate
  until ≥ 500K tokens) — bit-identical across every seed/baseline.
* **Learn**: warmup on WikiText-2 ``train`` (M=0) → adapt on PubMed ``train``.
  Only the first ~300K tokens of each train split are tokenized (the learn
  loop cycles a block-shuffled buffer, so this is plenty).

Token streams are cached to ``data/brain/*.pt`` (gitignored via the repo-wide
``*.pt`` rule) so re-runs skip tokenization entirely.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from ph_neuro.brain.brain_wrapper import cyclic_batch_iter, tokenize_list

WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
PUBMED_ID = "ccdv/pubmed-summarization"

# Cached token streams live here (the repo-level ``*.pt`` gitignore rule
# keeps them out of git).
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "brain"

# How many train tokens to tokenize for the learn buffers. ~300K tokens is
# 3× the 100K primary budget — plenty of variety across seeds after the
# deterministic block shuffle.
TRAIN_BUFFER_TOKENS = 300_000
# Locked target eval subsample size (protocol §2).
PUBMED_EVAL_TOKENS = 500_000
PUBMED_EVAL_SEED = 42


# ── HF loading ─────────────────────────────────────────────────────


def load_wikitext(split: str = "train") -> list[str]:
    """Load a WikiText-2 split as a list of raw text strings."""
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT_ID, WIKITEXT_CONFIG, split=split)
    return [row["text"] for row in ds]


def load_pubmed(split: str = "train") -> list[str]:
    """Load a PubMed split as documents ``abstract + " " + article``."""
    from datasets import load_dataset

    ds = load_dataset(PUBMED_ID, split=split)
    return [f"{row['abstract']} {row['article']}" for row in ds]


# ── tokenize + cache ───────────────────────────────────────────────


def _cache_path(tag: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{tag}.pt"


def _cache_load(path: Path) -> torch.Tensor | None:
    if path.exists():
        return torch.load(path, weights_only=True)
    return None


def tokenized_cache(
    tag: str,
    tokenizer,
    texts: list[str] | None = None,
    *,
    max_tokens: int | None = None,
    cache: bool = True,
) -> torch.Tensor:
    """Return a cached flat LongTensor for ``tag``, tokenizing if needed.

    ``max_tokens`` truncates **incrementally** (docs are tokenized until the
    budget is reached, then cut) so we never materialize a huge intermediate
    token stream just to keep a small head of it. The result is identical to
    tokenizing everything and slicing off the first ``max_tokens`` ids.
    """
    path = _cache_path(tag)
    if path.exists():
        return torch.load(path, weights_only=True)
    if texts is None:
        raise ValueError(f"no cache {path} and no texts provided")
    if max_tokens is not None:
        ids: list[int] = []
        total = 0
        for text in texts:
            if not text:
                continue
            toks = tokenizer(text, add_special_tokens=False).input_ids
            ids.extend(toks)
            total += len(toks)
            if total >= max_tokens:
                break
        ids = ids[:max_tokens]
        out = torch.tensor(ids, dtype=torch.long)
        if cache:
            torch.save(out, path)
        return out
    ids = tokenize_list(texts, tokenizer)
    if cache:
        torch.save(ids, path)
    return ids


def wikitext_ids(split: str, tokenizer) -> torch.Tensor:
    """Flat WikiText-2 token ids for a split (cached, full split)."""
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_wikitext2_{split}"
    return tokenized_cache(tag, tokenizer, texts=load_wikitext(split))


def pubmed_train_ids(tokenizer, *, max_tokens: int = TRAIN_BUFFER_TOKENS) -> torch.Tensor:
    """Flat PubMed train token ids (cached, truncated to ``max_tokens``)."""
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_pubmed_train_max{max_tokens}"
    return tokenized_cache(
        tag, tokenizer, texts=load_pubmed("train"), max_tokens=max_tokens
    )


def pubmed_eval_ids(
    tokenizer, *, max_tokens: int = PUBMED_EVAL_TOKENS, seed: int = PUBMED_EVAL_SEED
) -> torch.Tensor:
    """Flat PubMed eval token ids — deterministic 500K subsample of test.

    Document permutation uses ``random.Random(seed)`` (protocol §2: seed 42),
    accumulating documents until ≥ ``max_tokens``.
    """
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_pubmed_test_sub{max_tokens}_s{seed}"
    path = _cache_path(tag)
    if path.exists():
        return torch.load(path, weights_only=True)
    texts = load_pubmed("test")
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    collected: list[int] = []
    total = 0
    for i in order:
        toks = tokenizer(texts[i], add_special_tokens=False).input_ids
        collected.extend(toks)
        total += len(toks)
        if total >= max_tokens:
            break
    ids = torch.tensor(collected, dtype=torch.long)
    torch.save(ids, path)
    return ids


# ── learn batch stream ─────────────────────────────────────────────


def make_combined_batch_iter(
    wiki_train_ids: torch.Tensor,
    pubmed_train_ids: torch.Tensor,
    warmup_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int | None,
):
    """Yield learn batches: ``warmup_steps`` WikiText batches, then PubMed.

    Deterministic given ``(seed, tokens)`` — regenerating the iterator and
    skipping ``start_step`` items reproduces the same stream, so checkpoints
    resume onto the identical data sequence.
    """
    wiki_iter = cyclic_batch_iter(wiki_train_ids, batch_size, seq_len, seed)
    pub_iter = cyclic_batch_iter(pubmed_train_ids, batch_size, seq_len, seed)
    for _ in range(warmup_steps):
        yield next(wiki_iter)
    while True:
        yield next(pub_iter)
