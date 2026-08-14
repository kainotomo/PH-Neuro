"""Per-architecture adapters: which projection submodules to hook.

The Brain Wrapper is model-agnostic; all architecture-specific knowledge
lives here (a thin ~20-line adapter per architecture, per Step 0.4). Two
architectures are supported in Phase 1.1:

* **SmolLM2** (``config.model_type == "llama"``) — injects at
  ``self_attn.o_proj`` and ``mlp.down_proj`` (both ``nn.Linear``).
* **GPT-2** (``config.model_type == "gpt2"``) — injects at ``attn.c_proj``
  and ``mlp.c_proj`` (both ``Conv1D``).

The detection key is ``config.model_type``, NOT the Python class (SmolLM2 is
a ``LlamaForCausalLM``), so the whole SmolLM2 scaling ladder (135M/360M/1.7B)
is covered by one adapter.

**Verified (2026-08-12, transformers 5.15.0 + cached weights):** GPT-2's
``Conv1D(nf, nx)`` stores its weight as ``(nx, nf)`` — the transpose of an
``nn.Linear`` — and has **no** ``.out_features``/``.in_features`` attributes;
the out-dimension is read from ``module.nf``. See
:func:`_get_out_features`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

# ── InjectionPoint ──────────────────────────────────────────────────


@dataclass
class InjectionPoint:
    """One plastic weight injected at one frozen projection module.

    The plastic tensors (``bias`` for vector-bias mode; ``A`` + ``B`` for
    low-rank mode) are **not** ``nn.Parameters`` and are never part of the
    frozen model's ``state_dict()`` — they live in the wrapper, so the frozen
    model's state stays clean and the identity guarantee holds by
    construction.

    Low-rank mode (E032): ``W_plastic = B @ A`` with ``A: (r, d_in)``,
    ``B: (d_out, r)``, injected as ``output + B @ (A @ x)``. When ``rank > 0``
    the point carries ``A``/``B`` and ``in_features``; ``bias`` is kept as a
    zero tensor for state-schema compatibility but is unused by low-rank
    injection/update.
    """

    name: str  # stable id, e.g. "L03.o_proj" or "L07.mlp_c_proj"
    module: nn.Module  # frozen projection (nn.Linear | Conv1D)
    bias: torch.Tensor  # plastic bias, shape (out_features,), float32, zeros-init
    out_features: int | None = None  # d_out, read at construction
    pre_handle: object | None = None  # registered forward_pre_hook
    post_handle: object | None = None  # registered forward_hook (inject + capture)
    full_path: str = ""  # block-relative dotted path, e.g. "self_attn.o_proj"
    # low-rank mode (E032): A: (r, d_in), B: (d_out, r), float32, zeros-init
    A: torch.Tensor | None = None
    B: torch.Tensor | None = None
    in_features: int | None = None  # d_in, read at construction
    rank: int = 0  # 0 = vector-bias mode; > 0 = low-rank mode


# ── Out/in-features helpers ────────────────────────────────────────


def _get_out_features(module: nn.Module) -> int:
    """Robustly read the output dimension of a projection module.

    Handles ``nn.Linear`` (``.out_features``) and GPT-2 ``Conv1D`` (``.nf``;
    it has no ``out_features``/``in_features`` attributes — verified).
    """
    out = getattr(module, "out_features", None)
    if out is not None:
        return int(out)
    nf = getattr(module, "nf", None)
    if nf is not None:
        return int(nf)
    raise AttributeError(
        f"cannot determine out_features for {type(module).__name__}; "
        "expected nn.Linear (.out_features) or Conv1D (.nf)"
    )


def _get_in_features(module: nn.Module) -> int:
    """Robustly read the input dimension of a projection module.

    Mirrors :func:`_get_out_features`: ``nn.Linear`` (``.in_features``) and
    GPT-2 ``Conv1D`` (``.nx`` — verified: ``Conv1D(nf, nx)`` stores its weight
    as ``(nx, nf)`` and exposes ``.nx``/``.nf``, not ``in_features``).
    """
    inn = getattr(module, "in_features", None)
    if inn is not None:
        return int(inn)
    nx = getattr(module, "nx", None)
    if nx is not None:
        return int(nx)
    raise AttributeError(
        f"cannot determine in_features for {type(module).__name__}; "
        "expected nn.Linear (.in_features) or Conv1D (.nx)"
    )


def _resolve(block: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a dotted submodule path relative to a block, e.g. ``self_attn.o_proj``."""
    mod = block
    for part in dotted_path.split("."):
        mod = getattr(mod, part)
    return mod


# ── BlockWrapper protocol ──────────────────────────────────────────


class BlockWrapper(Protocol):
    """Per-architecture adapter contract (Step 0.4 spec)."""

    block_paths: tuple[str, ...]

    def get_injection_points(
        self, block: nn.Module, layer_idx: int
    ) -> list[InjectionPoint]: ...


class SmolLM2BlockWrapper:
    """Adapter for LLaMA-style blocks (SmolLM2 135M/360M/1.7B).

    Injects at ``o_proj`` (attention output) and ``down_proj`` (MLP output),
    both ``nn.Linear``. ``o_proj``: 2048→2048; ``down_proj``: 8192→2048
    (verified 2026-08-12, transformers 5.15.0).
    """

    block_paths: tuple[str, ...] = ("self_attn.o_proj", "mlp.down_proj")

    def get_injection_points(
        self, block: nn.Module, layer_idx: int, *, rank: int = 0
    ) -> list[InjectionPoint]:
        return [
            self._make(block, layer_idx, "self_attn.o_proj", "o_proj", rank),
            self._make(block, layer_idx, "mlp.down_proj", "down_proj", rank),
        ]

    @staticmethod
    def _make(block, layer_idx, path, suffix, rank: int = 0):
        mod = _resolve(block, path)
        out = _get_out_features(mod)
        inn = _get_in_features(mod)
        A = B = None
        if rank > 0:
            A = torch.zeros(rank, inn, dtype=torch.float32)
            B = torch.zeros(out, rank, dtype=torch.float32)
        return InjectionPoint(
            name=f"L{layer_idx:02d}.{suffix}",
            module=mod,
            bias=torch.zeros(out, dtype=torch.float32),
            out_features=out,
            full_path=path,
            A=A,
            B=B,
            in_features=inn,
            rank=rank,
        )


class GPT2BlockWrapper:
    """Adapter for classic pre-norm GPT-2 blocks (openai-community/gpt2).

    Injects at ``attn.c_proj`` (attention output) and ``mlp.c_proj`` (MLP
    output), both ``Conv1D``. ``attn.c_proj``: 768→768; ``mlp.c_proj``:
    3072→768 (verified 2026-08-12).
    """

    block_paths: tuple[str, ...] = ("attn.c_proj", "mlp.c_proj")

    def get_injection_points(
        self, block: nn.Module, layer_idx: int, *, rank: int = 0
    ) -> list[InjectionPoint]:
        return [
            self._make(block, layer_idx, "attn.c_proj", "attn_c_proj", rank),
            self._make(block, layer_idx, "mlp.c_proj", "mlp_c_proj", rank),
        ]

    @staticmethod
    def _make(block, layer_idx, path, suffix, rank: int = 0):
        mod = _resolve(block, path)
        out = _get_out_features(mod)
        inn = _get_in_features(mod)
        A = B = None
        if rank > 0:
            A = torch.zeros(rank, inn, dtype=torch.float32)
            B = torch.zeros(out, rank, dtype=torch.float32)
        return InjectionPoint(
            name=f"L{layer_idx:02d}.{suffix}",
            module=mod,
            bias=torch.zeros(out, dtype=torch.float32),
            out_features=out,
            full_path=path,
            A=A,
            B=B,
            in_features=inn,
            rank=rank,
        )


# ── Factories ──────────────────────────────────────────────────────


def get_block_wrapper(model) -> type[BlockWrapper] | BlockWrapper:
    """Return the BlockWrapper adapter for ``model`` (keyed on config.model_type)."""
    t = model.config.model_type
    if t == "llama":
        return SmolLM2BlockWrapper()
    if t == "gpt2":
        return GPT2BlockWrapper()
    raise NotImplementedError(
        f"model_type={t!r} not supported (supported: llama, gpt2)"
    )


def get_block_container(model) -> nn.ModuleList:
    """Return the module list of decoder blocks for ``model``."""
    t = model.config.model_type
    if t == "llama":
        return model.model.layers
    if t == "gpt2":
        return model.transformer.h
    raise NotImplementedError(
        f"model_type={t!r} not supported (supported: llama, gpt2)"
    )
