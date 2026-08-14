"""Minimal manual LoRA adapters (E032 Part D / E034).

Backprop LoRA at the **same parameter budget** as the BrainWrapper low-rank
local modes: rank-1 ``A:(r, d_in)``/``B:(d_out, r)`` pairs injected at every
``o_proj``/``down_proj`` (LLaMA/SmolLM2) or ``attn.c_proj``/``mlp.c_proj``
(GPT-2) via forward hooks. ``peft`` is not a dependency, so this is the
project's own minimal implementation (~50 lines), identical in structure and
init to the local low-rank plastic representation — only the update rule
differs (AdamW backprop here vs local Hebbian/predictive-coding in the Brain
Wrapper).

``LoRAAdapter`` is testable in isolation (used by ``run_e032_lora.py`` and
``run_e034_lora.py``) and by ``tests/brain/test_e034_lora.py``.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from ph_neuro.brain.block_wrappers import (
    _get_in_features,
    _get_out_features,
    get_block_container,
    get_block_wrapper,
)


class LoRAAdapter:
    """One trainable LoRA pair injected at one frozen projection module.

    ``output + (B @ (A @ x))`` via a forward hook; A/B are real
    ``nn.Parameters`` so gradients flow back through the hook. Init follows
    the E032 convention: ``A ~ N(0, 1/sqrt(d_in))``, ``B = 0`` (the injection
    is exactly zero at construction → the frozen model is unchanged; identical
    to the local low-rank mode's init, so the comparison isolates the update
    rule, not the init).
    """

    def __init__(self, module, rank: int, device, dtype=torch.float32):
        self.name = module.__class__.__name__
        self.out_features = _get_out_features(module)
        self.in_features = _get_in_features(module)
        self.rank = int(rank)
        self.device = torch.device(device)
        # Scaled random projection init (matches the local low-rank mode).
        self.A = torch.randn(
            self.rank, self.in_features, dtype=dtype, device=self.device
        ) * (1.0 / math.sqrt(self.in_features))
        self.B = torch.zeros(self.out_features, self.rank, dtype=dtype, device=self.device)
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)
        # Frozen-baseline eval disables the injection (hooks pass through).
        self.enabled = True
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if not self.enabled:
            return output
        x = args[0]
        t = torch.einsum("ri,bsi->bsr", self.A.to(output.dtype), x)
        return output + torch.einsum("or,bsr->bso", self.B.to(output.dtype), t)

    def set_enabled(self, enabled: bool) -> None:
        """Temporarily disable/enable the injection (frozen-baseline eval)."""
        self.enabled = bool(enabled)

    def parameters(self):
        yield self.A
        yield self.B

    def remove(self):
        self.handle.remove()

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            [("A", self.A.detach().clone().to(torch.float32)),
             ("B", self.B.detach().clone().to(torch.float32))]
        )

    def load_state_dict(self, state) -> None:
        self.A.data.copy_(state["A"].to(self.device))
        self.B.data.copy_(state["B"].to(self.device))

    def n_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def mean_abs(self) -> float:
        """Mean |A| + |B| (a cheap plastic-weight diagnostic, E032 convention)."""
        return float(self.A.detach().abs().mean()) + float(
            self.B.detach().abs().mean()
        )


def build_lora_adapters(model, rank: int, device) -> list[LoRAAdapter]:
    """Attach a LoRA adapter at every o_proj/down_proj (llama) / c_proj (gpt2)."""
    container = get_block_container(model)
    wrapper = get_block_wrapper(model)
    adapters: list[LoRAAdapter] = []
    for i, block in enumerate(container):
        for path in wrapper.block_paths:
            mod = block
            for part in path.split("."):
                mod = getattr(mod, part)
            adapters.append(LoRAAdapter(mod, rank, device))
    return adapters


def n_lora_params(adapters: list[LoRAAdapter]) -> int:
    return sum(ad.n_params() for ad in adapters)


def all_lora_weights(adapters: list[LoRAAdapter]) -> torch.Tensor:
    """Concatenated A+B flattened weights (for magnitude diagnostics)."""
    return torch.cat(
        [ad.A.detach().flatten() for ad in adapters]
        + [ad.B.detach().flatten() for ad in adapters]
    )
