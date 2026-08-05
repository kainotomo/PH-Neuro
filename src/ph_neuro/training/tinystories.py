"""TinyStories dataset loader for Milestone M2.1.

Loads the TinyStories dataset (HuggingFace ``roneneldan/TinyStories``),
tokenizes it with the GPT-2 BPE tokenizer (``tiktoken``), packs tokens
into fixed-length causal language-modeling sequences, and returns
``DataLoader`` objects that yield ``(input_ids, targets)`` with
``targets[t] == input_ids[t+1]`` (shifted by one token).

Design:
    - Download + tokenization is cached to disk (``data_dir``) as
      ``.pt`` tensors, so re-runs skip the network entirely.
    - Streaming tokenization keeps memory bounded even for the full
      ~2 GB dataset (a ``--max-samples`` cap limits how much is read).
    - ``make_synthetic_*`` helpers generate small local corpora so tests
      and smoke runs never need network access.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "gpt2"

# 1M tokens ≈ 4 MB of text — a safe default cap for smoke runs. Set to
# ``None`` for the full dataset.
_DEFAULT_MAX_SAMPLES = 50_000


# ── Tokenizer ──────────────────────────────────────────────────────


def make_gpt2_tokenizer():
    """Return the GPT-2 BPE tokenizer (``tiktoken`` encoding).

    Raises:
        ImportError: If ``tiktoken`` is not installed.
    """
    try:
        import tiktoken  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "tiktoken is required for the GPT-2 tokenizer. "
            "Install it with: .venv/bin/pip install tiktoken"
        ) from exc
    return tiktoken.get_encoding(TOKENIZER_NAME)


# ── Tokenize + pack ────────────────────────────────────────────────


def tokenize_texts(texts, tokenizer) -> torch.Tensor:
    """Tokenize a list of strings into a single flat int32 tensor.

    Args:
        texts: Iterable of strings.
        tokenizer: A ``tiktoken`` encoding (has an ``encode`` method).

    Returns:
        Flat ``torch.int32`` tensor of all tokens concatenated.
    """
    pieces: list[torch.Tensor] = []
    for text in texts:
        ids = tokenizer.encode(text)
        pieces.append(torch.tensor(ids, dtype=torch.int32))
    if not pieces:
        return torch.empty(0, dtype=torch.int32)
    return torch.cat(pieces)


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
    tokens = tokens.long()
    n_full = (tokens.numel() - 1) // seq_len
    if n_full < 1:
        raise ValueError(
            f"Not enough tokens ({tokens.numel()}) for seq_len={seq_len}"
        )
    tokens = tokens[: n_full * seq_len + 1]
    input_ids = tokens[:-1].view(n_full, seq_len)
    targets = tokens[1:].view(n_full, seq_len)
    return input_ids, targets


def build_lm_dataset(
    tokens: torch.Tensor, seq_len: int
) -> tuple[Dataset, dict]:
    """Build a ``TensorDataset`` of ``(input_ids, targets)`` from tokens.

    Args:
        tokens: Flat token tensor.
        seq_len: Sequence length.

    Returns:
        Tuple ``(dataset, meta)`` where ``meta`` holds ``n_seqs``.
    """
    input_ids, targets = pack_sequences(tokens, seq_len)
    dataset = TensorDataset(input_ids, targets)
    return dataset, {"n_seqs": input_ids.shape[0], "seq_len": seq_len}


# ── HF download + cache ────────────────────────────────────────────


def _hf_available_splits() -> list[str]:
    """Return available splits of the TinyStories dataset."""
    try:
        from datasets import get_dataset_split_names  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The 'datasets' package is required to download TinyStories. "
            "Install it with: .venv/bin/pip install datasets"
        ) from exc
    try:
        return list(get_dataset_split_names(DATASET_NAME))
    except Exception:  # noqa: BLE001 - remote datasets can be flaky
        return ["train"]


def _stream_stories(split: str, max_samples: int | None):
    """Yield ``text`` strings from the TinyStories split (streaming)."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    for i, example in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        yield example["text"]


def _download_and_tokenize(
    split: str,
    tokenizer,
    seq_len: int,
    max_samples: int | None,
    data_dir: str,
) -> torch.Tensor:
    """Download a split, tokenize streaming, save the flat token cache.

    Returns:
        Flat int32 token tensor.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(
        data_dir,
        f"tinystories_{TOKENIZER_NAME}_{split}_max{max_samples or 'all'}.pt",
    )
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    pieces: list[torch.Tensor] = []
    for _n_stories, text in enumerate(_stream_stories(split, max_samples), start=1):
        ids = tokenizer.encode(text)
        pieces.append(torch.tensor(ids, dtype=torch.int32))
    if not pieces:
        raise RuntimeError(f"No stories downloaded from split '{split}'")
    tokens = torch.cat(pieces)
    torch.save(tokens, cache_path)
    return tokens


# ── Public loader ──────────────────────────────────────────────────


def get_tinystories_data(
    data_dir: str = "data/tinystories",
    seq_len: int = 256,
    batch_size: int = 8,
    val_ratio: float = 0.01,
    max_samples: int | None = _DEFAULT_MAX_SAMPLES,
    force_download: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict]:
    """Download (or load cached) TinyStories and build train/val loaders.

    Args:
        data_dir: Directory for the dataset cache (tokens + HF cache).
        seq_len: LM sequence length.
        batch_size: DataLoader batch size.
        val_ratio: Fraction of tokens held out for validation when the
            dataset has no dedicated validation split.
        max_samples: Cap on the number of stories to download (``None``
            = everything). Defaults to 50k stories for a reasonable
            training set.
        force_download: Re-download even if a token cache exists.
        num_workers: DataLoader workers (0 = main process, safest under
            GPU contention).
        seed: Seed for shuffling.

    Returns:
        Tuple ``(train_loader, val_loader, meta)`` where ``meta`` holds
        tokenizer info, sequence counts and the model vocab size.
    """
    tokenizer = make_gpt2_tokenizer()
    vocab_size = tokenizer.n_vocab

    # Cache key embeds max_samples + seq_len so re-loads are consistent.
    os.makedirs(data_dir, exist_ok=True)
    seq_cache = os.path.join(
        data_dir,
        f"train_val_seq{seq_len}_max{max_samples or 'all'}.pt",
    )
    if os.path.exists(seq_cache) and not force_download:
        train_inputs, train_targets, val_inputs, val_targets = torch.load(
            seq_cache, weights_only=True
        )
    else:
        splits = _hf_available_splits()
        has_val = "validation" in splits
        train_tokens = _download_and_tokenize(
            "train", tokenizer, seq_len, max_samples, data_dir
        )
        if has_val:
            val_tokens = _download_and_tokenize(
                "validation", tokenizer, seq_len, max_samples, data_dir
            )
        else:
            n_val = max(1, int(train_tokens.numel() * val_ratio))
            train_tokens, val_tokens = (
                train_tokens[:-n_val],
                train_tokens[-n_val:],
            )

        train_inputs, train_targets = pack_sequences(train_tokens, seq_len)
        val_inputs, val_targets = pack_sequences(val_tokens, seq_len)
        torch.save(
            (train_inputs, train_targets, val_inputs, val_targets), seq_cache
        )

    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

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

    meta = {
        "dataset": DATASET_NAME,
        "tokenizer": TOKENIZER_NAME,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "max_samples": max_samples,
        "n_train_seqs": len(train_dataset),
        "n_val_seqs": len(val_dataset),
        "n_train_tokens": int(train_inputs.numel()),
        "n_val_tokens": int(val_inputs.numel()),
    }
    return train_loader, val_loader, meta


# ── Synthetic data (tests / smoke — no network) ────────────────────


def make_synthetic_token_sequences(
    n_seqs: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate deterministic, learnable token sequences.

    Uses a linear congruential recurrence ``t_{k+1} = (3 * t_k + 5) % V``
    so next-token prediction has real structure a small model can learn
    (used by tests and smoke runs instead of downloading TinyStories).

    Returns:
        ``(n_seqs, seq_len)`` int64 tensor.
    """
    g = torch.Generator().manual_seed(seed)
    seqs = torch.empty(n_seqs, seq_len, dtype=torch.long)
    seqs[:, 0] = torch.randint(0, vocab_size, (n_seqs,), generator=g)
    a, c = 3, 5
    for t in range(1, seq_len):
        seqs[:, t] = (a * seqs[:, t - 1] + c) % vocab_size
    return seqs


def make_synthetic_lm_loader(
    vocab_size: int = 64,
    seq_len: int = 32,
    batch_size: int = 4,
    n_batches: int = 8,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader from synthetic learnable sequences (tests/smoke).

    Returns a loader of ``(input_ids, targets)`` pairs; ``targets`` is
    ``input_ids`` shifted by one token.

    Args:
        vocab_size: Small vocabulary for the synthetic corpus.
        seq_len: Sequence length.
        batch_size: Batch size.
        n_batches: Number of batches (determines dataset size).
        seed: Random seed.

    Returns:
        ``DataLoader`` yielding ``(input_ids, targets)``.
    """
    n_seqs = n_batches * batch_size
    seqs = make_synthetic_token_sequences(n_seqs, seq_len + 1, vocab_size, seed)
    dataset = TensorDataset(seqs[:, :-1], seqs[:, 1:])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def make_synthetic_stories(n_samples: int = 100, seed: int = 0) -> list[str]:
    """Generate tiny fake "stories" (short simple sentences) for tests.

    Useful for exercising the text -> tokens -> sequences pipeline without
    downloading TinyStories. Each story is a deterministic pseudo-random
    sentence over a tiny alphabet.

    Args:
        n_samples: Number of fake stories.
        seed: Random seed.

    Returns:
        List of short strings.
    """
    words = [
        "the", "cat", "dog", "ran", "and", "saw", "a", "big", "red",
        "ball", "in", "the", "park", "one", "day", "it", "was", "happy",
    ]
    g = torch.Generator().manual_seed(seed)
    stories: list[str] = []
    for _ in range(n_samples):
        n_words = int(torch.randint(4, 10, (1,), generator=g).item())
        idx = torch.randint(0, len(words), (n_words,), generator=g)
        stories.append(" ".join(words[i] for i in idx.tolist()) + ".")
    return stories
