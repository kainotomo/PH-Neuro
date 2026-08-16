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


# ── E034: second domain — CNN/DailyMail (news, apache-2.0) ─────────


def load_cnn_dailymail(split: str = "train") -> list[str]:
    """Load a CNN/DailyMail split as a list of news ``article`` strings.

    E034 second domain (verified 2026-08-14): ``abisee/cnn_dailymail``
    config ``3.0.0``, license **apache-2.0** (permissive — product-path
    compatible; unlike ``pile-of-law``'s cc-by-nc-sa-4.0 or
    ``codeparrot/github-code``'s ``other``). Frozen ppl on the 500K test
    subsample: **11.971** — a moderate (+12.3% vs WikiText-2, +4.5% vs
    PubMed) shift that keeps the surprise sigmoid in its sensitive range.
    """
    from datasets import load_dataset

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
    return [row["article"] for row in ds]


def cnn_dailymail_train_ids(
    tokenizer, *, max_tokens: int = TRAIN_BUFFER_TOKENS
) -> torch.Tensor:
    """Flat CNN/DailyMail train token ids (cached, truncated incrementally)."""
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_cnn_train_max{max_tokens}"
    return tokenized_cache(
        tag, tokenizer, texts=load_cnn_dailymail("train"), max_tokens=max_tokens
    )


def cnn_dailymail_eval_ids(
    tokenizer, *, max_tokens: int = PUBMED_EVAL_TOKENS, seed: int = PUBMED_EVAL_SEED
) -> torch.Tensor:
    """Flat CNN/DailyMail eval token ids — deterministic 500K subsample of test.

    Document permutation uses ``random.Random(seed)`` (protocol §2 convention:
    seed 42), accumulating articles until ≥ ``max_tokens`` — bit-identical
    across seeds/configs, so paired statistics are valid.
    """
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_cnn_test_sub{max_tokens}_s{seed}"
    path = _cache_path(tag)
    if path.exists():
        return torch.load(path, weights_only=True)
    texts = load_cnn_dailymail("test")
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


def make_three_domain_batch_iter(
    wiki_train_ids: torch.Tensor,
    pubmed_train_ids: torch.Tensor,
    cnn_train_ids: torch.Tensor,
    warmup_steps: int,
    phase1_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int | None,
):
    """Yield the E034 two-domain learn stream.

    ``warmup_steps`` WikiText batches (M=0 warmup), then ``phase1_steps``
    PubMed batches (domain 1), then CNN/DailyMail batches forever (domain 2).
    Deterministic given ``(seed, tokens)`` — regenerating the iterator and
    skipping ``start_step`` items reproduces the same stream, so checkpoints
    resume onto the identical data sequence.
    """
    wiki_iter = cyclic_batch_iter(wiki_train_ids, batch_size, seq_len, seed)
    pub_iter = cyclic_batch_iter(pubmed_train_ids, batch_size, seq_len, seed)
    cnn_iter = cyclic_batch_iter(cnn_train_ids, batch_size, seq_len, seed)
    for _ in range(warmup_steps):
        yield next(wiki_iter)
    for _ in range(phase1_steps):
        yield next(pub_iter)
    while True:
        yield next(cnn_iter)


# ── E036: third domain — C4 (Common Crawl web text, odc-by) ─────────
#
# D3 for the E036 consolidation sequence (WikiText-2 → PubMed → CNN/DailyMail
# → C4). Verified 2026-08-15:
# * ``allenai/c4`` config ``en`` — a colossal cleaned version of Common
#   Crawl's web corpus (Google's C4; Rafel et al. 2020), **license ODC-BY**
#   (Open Data Commons Attribution 1.0 — attribution-only, no non-commercial
#   / no share-alike restriction → product-path compatible; consistent with
#   the project's permissive-license rule that rejects only NC/SA-restricted
#   licenses like pile-of-law's cc-by-nc-sa).
# * **Frozen ppl (500K eval subsample) = 13.568** — harder than WikiText-2
#   (10.66), PubMed (11.46), CNN/DailyMail (11.97) → the surprise gate opens
#   at the D3 boundary (loss *rises* past the EMA). Legal corpora (SCOTUS
#   8.41, LEDGAR 8.46, EUR-LEX 6.51) were all *easier* than the source —
#   the gate would have stayed closed (loss falls) → D3 adaptation ~0 → a
#   vacuous forward-transfer test. C4's difficulty gradient (10.66 → 11.46 →
#   11.97 → 13.57) keeps the gate open at every boundary and makes D3 the
#   strongest forward-transfer probe.
# * ``text`` field (raw web document); train ~364M docs / validation ~364K
#   docs. Splits load natively via parquet/streaming (verified).
#
# Determinism: the eval/probe corpora are fixed 500K/50K-token subsamples of
# the **validation** split (doc permutation seed 42 / 43 over a fixed 4000-doc
# stream head); the train buffer is a fixed 300K-token slice (permuted seed
# over a fixed 3000-doc head). All bit-identical across seeds/configs.

C4_EVAL_HEAD_DOCS = 4_000
C4_TRAIN_HEAD_DOCS = 3_000


def _load_c4_head(split: str, max_docs: int) -> list[str]:
    """Fetch a deterministic head of the C4 ``en`` split as raw web documents.

    Iterates the streaming dataset in shard order (deterministic file order)
    and takes the first ``max_docs`` ``text`` fields.
    """
    import itertools

    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    return [row["text"] for row in itertools.islice(ds, max_docs)]


def _permuted_token_subsample(
    tokenizer, texts: list[str], *, max_tokens: int, seed: int
) -> torch.Tensor:
    """Deterministic document-permuted token subsample (protocol §2 convention)."""
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
    return torch.tensor(collected[:max_tokens], dtype=torch.long)


def c4_train_ids(
    tokenizer, *, max_tokens: int = TRAIN_BUFFER_TOKENS
) -> torch.Tensor:
    """Flat C4 train token ids (cached, deterministic 300K-token slice)."""
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_c4_train_max{max_tokens}"
    path = _cache_path(tag)
    if path.exists():
        return torch.load(path, weights_only=True)
    texts = _load_c4_head("train", C4_TRAIN_HEAD_DOCS)
    ids = _permuted_token_subsample(tokenizer, texts, max_tokens=max_tokens, seed=42)
    torch.save(ids, path)
    return ids


def c4_eval_ids(
    tokenizer, *, max_tokens: int = PUBMED_EVAL_TOKENS, seed: int = PUBMED_EVAL_SEED
) -> torch.Tensor:
    """Flat C4 eval token ids — deterministic 500K subsample of validation.

    Doc permutation seed 42 over a fixed 4000-doc stream head (protocol §2
    convention) — bit-identical across seeds/configs, so paired statistics
    are valid. Frozen ppl on this corpus: **13.568** (measured 2026-08-15).
    """
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_c4_test_sub{max_tokens}_s{seed}"
    path = _cache_path(tag)
    if path.exists():
        return torch.load(path, weights_only=True)
    texts = _load_c4_head("validation", C4_EVAL_HEAD_DOCS)
    ids = _permuted_token_subsample(tokenizer, texts, max_tokens=max_tokens, seed=seed)
    torch.save(ids, path)
    return ids


def c4_probe_ids(
    tokenizer, *, max_tokens: int = 50_000, seed: int = 43
) -> torch.Tensor:
    """A small deterministic C4 probe for in-training adaptation-speed evals.

    A fixed 50,000-token subsample of the C4 validation split (doc permutation
    seed 43 — distinct from the 500K locked eval corpus seed 42, so the probe
    is an independent quick signal). Bit-identical across seeds/configs.
    """
    tag = f"{tokenizer.name_or_path.replace('/', '__')}_c4_test_probe{max_tokens}_s{seed}"
    path = _cache_path(tag)
    if path.exists():
        return torch.load(path, weights_only=True)
    texts = _load_c4_head("validation", C4_EVAL_HEAD_DOCS)
    ids = _permuted_token_subsample(tokenizer, texts, max_tokens=max_tokens, seed=seed)
    torch.save(ids, path)
    return ids


def make_four_domain_batch_iter(
    wiki_train_ids: torch.Tensor,
    pubmed_train_ids: torch.Tensor,
    cnn_train_ids: torch.Tensor,
    c4_train_ids: torch.Tensor,
    warmup_steps: int,
    phase1_steps: int,
    phase2_steps: int,
    batch_size: int,
    seq_len: int,
    seed: int | None,
):
    """Yield the E036 three-domain learn stream (WikiText → PubMed → CNN → C4).

    ``warmup_steps`` WikiText batches (M=0 warmup), then ``phase1_steps``
    PubMed batches (domain 1), then ``phase2_steps`` CNN/DailyMail batches
    (domain 2), then C4 (web) batches forever (domain 3). Deterministic given
    ``(seed, tokens)`` — regenerating the iterator and skipping ``start_step``
    items reproduces the same stream, so checkpoints resume onto the identical
    data sequence.
    """
    wiki_iter = cyclic_batch_iter(wiki_train_ids, batch_size, seq_len, seed)
    pub_iter = cyclic_batch_iter(pubmed_train_ids, batch_size, seq_len, seed)
    cnn_iter = cyclic_batch_iter(cnn_train_ids, batch_size, seq_len, seed)
    c4_iter = cyclic_batch_iter(c4_train_ids, batch_size, seq_len, seed)
    for _ in range(warmup_steps):
        yield next(wiki_iter)
    for _ in range(phase1_steps):
        yield next(pub_iter)
    for _ in range(phase2_steps):
        yield next(cnn_iter)
    while True:
        yield next(c4_iter)
