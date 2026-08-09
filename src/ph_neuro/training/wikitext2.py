"""WikiText-2 dataset loader for Milestone M2.2.

Loads the WikiText-2 dataset (HuggingFace ``Salesforce/wikitext-2-raw-v1``),
tokenizes it with the GPT-2 BPE tokenizer (``tiktoken`` — identical to M2.1),
**concatenates** the raw text lines and packs them into fixed-length causal
language-modeling sequences of ``seq_len`` tokens, and returns ``DataLoader``
objects that yield ``(input_ids, targets)`` with ``targets[t] == input_ids[t+1]``
(shifted by one token).

Design (mirrors ``ph_neuro.training.tinystories`` so M2.1 infrastructure
reuses 100%):

    - WikiText-2 is *small* (~2.3M train / ~270K validation tokens after
      GPT-2 BPE), so the whole dataset downloads + tokenizes in seconds and
      the token cache is cached to disk (``data_dir``) as ``.pt`` tensors —
      re-runs skip the network entirely.
    - Unlike TinyStories, WikiText-2 on HuggingFace has NO train/val split in
      the raw config, so we use the explicit HuggingFace splits:
      ``wikitext['train']`` for training and ``wikitext['validation']`` for
      validation (the ``-raw`` config splits the raw text into train /
      validation / test articles).
    - ``make_synthetic_*`` helpers generate small local corpora so tests and
      smoke runs never need network access.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

DATASET_NAME = "Salesforce/wikitext"
# The raw (un-tokenized) WikiText-2 config: raw text articles, no <unk>.
CONFIG_NAME = "wikitext-2-raw-v1"
TOKENIZER_NAME = "gpt2"

# Default sequence length (M2.2 uses 256, same as M2.1).
DEFAULT_SEQ_LEN = 256


# ── Tokenizer ──────────────────────────────────────────────────────


def make_gpt2_tokenizer():
    """Return the GPT-2 BPE tokenizer (``tiktoken`` encoding).

    Re-exported here so the M2.2 runner has one import surface; delegates to
    the canonical implementation in :mod:`ph_neuro.training.tinystories`.

    Raises:
        ImportError: If ``tiktoken`` is not installed.
    """
    from ph_neuro.training.tinystories import make_gpt2_tokenizer as _make  # noqa: PLC0415

    return _make()


# ── Tokenize + pack ────────────────────────────────────────────────


def tokenize_texts(texts, tokenizer) -> torch.Tensor:
    """Tokenize a list of strings into a single flat int32 tensor.

    Args:
        texts: Iterable of strings.
        tokenizer: A ``tiktoken`` encoding (has an ``encode`` method).

    Returns:
        Flat ``torch.int32`` tensor of all tokens concatenated.
    """
    from ph_neuro.training.tinystories import tokenize_texts as _tokenize  # noqa: PLC0415

    return _tokenize(texts, tokenizer)


def pack_sequences(
    tokens: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a flat token stream into overlapping LM windows.

    Splits ``tokens`` into non-overlapping blocks of ``seq_len`` tokens
    where ``input_ids`` is block ``i`` and ``targets`` is block ``i``
    shifted right by one token (i.e. next-token prediction targets).

    Args:
        tokens: Flat ``(N,)`` integer tensor.
        seq_len: Sequence length of each window.

    Returns:
        Tuple ``(input_ids, targets)``, both shape ``(n_windows, seq_len)``
        of dtype ``int64``. ``targets[n, t] == input_ids[n, t+1]`` and
        ``targets[n, -1]`` predicts the first token of the next window.
    """
    from ph_neuro.training.tinystories import pack_sequences as _pack  # noqa: PLC0415

    return _pack(tokens, seq_len)


# ── HF download + cache ────────────────────────────────────────────


def _hf_available_splits() -> list[str]:
    """Return available splits of the WikiText-2 dataset."""
    try:
        from datasets import get_dataset_split_names  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The 'datasets' package is required to download WikiText-2. "
            "Install it with: .venv/bin/pip install datasets"
        ) from exc
    try:
        return list(get_dataset_split_names(DATASET_NAME, CONFIG_NAME))
    except Exception:  # noqa: BLE001 - remote datasets can be flaky
        return ["train", "validation", "test"]


def _download_and_tokenize(
    split: str,
    tokenizer,
    max_samples: int | None,
    data_dir: str,
) -> torch.Tensor:
    """Download a split, tokenize, save the flat token cache.

    WikiText-2 articles are stored as a ``text`` column (one line per row,
    some blank). We concatenate ALL rows into one flat token stream (the
    M2.2 brief: "concatenation + chunking").

    Returns:
        Flat int32 token tensor.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(
        data_dir,
        f"wikitext2_{TOKENIZER_NAME}_{split}_max{max_samples or 'all'}.pt",
    )
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The 'datasets' package is required to download WikiText-2. "
            "Install it with: .venv/bin/pip install datasets"
        ) from exc

    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split=split)
    texts = ds["text"]
    if max_samples is not None:
        texts = texts[:max_samples]

    pieces: list[torch.Tensor] = []
    for text in texts:
        ids = tokenizer.encode(text)
        pieces.append(torch.tensor(ids, dtype=torch.int32))
    if not pieces:
        raise RuntimeError(f"No text downloaded from split '{split}'")
    tokens = torch.cat(pieces)
    torch.save(tokens, cache_path)
    return tokens


# ── Public loader ──────────────────────────────────────────────────


def get_wikitext2_data(
    data_dir: str = "data/wikitext2",
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = 8,
    max_samples: int | None = None,
    force_download: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Download (or load cached) WikiText-2 and build train/val/test loaders.

    Args:
        data_dir: Directory for the dataset cache (tokens + HF cache).
        seq_len: LM sequence length (chunk size).
        batch_size: DataLoader batch size.
        max_samples: Cap on the number of text rows read per split (``None``
            = the whole dataset). WikiText-2 is small; default keeps all.
        force_download: Re-download even if a token cache exists.
        num_workers: DataLoader workers (0 = main process, safest under GPU
            contention — B2 lesson).
        seed: Seed for shuffling the training loader.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader, meta)`` where
        ``meta`` holds tokenizer info, sequence counts and the model vocab
        size.
    """
    tokenizer = make_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab

    os.makedirs(data_dir, exist_ok=True)
    seq_cache = os.path.join(
        data_dir,
        f"train_val_test_seq{seq_len}_max{max_samples or 'all'}.pt",
    )
    if os.path.exists(seq_cache) and not force_download:
        train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets = (
            torch.load(seq_cache, weights_only=True)
        )
    else:
        train_tokens = _download_and_tokenize(
            "train", tokenizer, max_samples, data_dir
        )
        val_tokens = _download_and_tokenize(
            "validation", tokenizer, max_samples, data_dir
        )
        test_tokens = _download_and_tokenize(
            "test", tokenizer, max_samples, data_dir
        )

        train_inputs, train_targets = pack_sequences(train_tokens, seq_len)
        val_inputs, val_targets = pack_sequences(val_tokens, seq_len)
        test_inputs, test_targets = pack_sequences(test_tokens, seq_len)
        torch.save(
            (train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets),
            seq_cache,
        )

    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    meta = {
        "dataset": DATASET_NAME,
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "max_samples": max_samples,
        "n_train_seqs": len(train_dataset),
        "n_val_seqs": len(val_dataset),
        "n_test_seqs": len(test_dataset),
        "n_train_tokens": int(train_inputs.numel()),
        "n_val_tokens": int(val_inputs.numel()),
        "n_test_tokens": int(test_inputs.numel()),
        "n_train_steps_per_epoch": len(train_loader),
        "n_val_steps_per_epoch": len(val_loader),
    }
    return train_loader, val_loader, test_loader, meta
