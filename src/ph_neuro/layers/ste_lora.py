"""LoRA (Low-Rank Adaptation) layer for frozen ternary STE backbones.

Implements the B2 experiment (QLoRA + Frozen Ternary Backbone): a ternary
``TernarySTELinear``-style layer whose **latent scores are frozen** after
pre-training, augmented with trainable **low-rank adapters** (Hu et al.,
2021, "LoRA: Low-Rank Adaptation of Large Language Models").

Forward pass
============

.. math::

    \\text{out} = x \\cdot W_\\text{tern}^\\top + \\frac{\\alpha}{r} \\cdot
    (x \\cdot A^\\top) \\cdot B^\\top + b

where:

- :math:`W_\\text{tern} = \\text{sign}(\\text{latent\\_scores})` is the frozen
  ternary weight matrix (values in \\{-1, 0, +1\\}).
- :math:`A` (shape ``(r, in_features)``) and :math:`B` (shape
  ``(out_features, r)``) are the LoRA matrices.
- :math:`r` is the rank and :math:`\\alpha` the scaling constant
  (default ``alpha = r``, matching the standard LoRA convention that the
  ``alpha``/``r`` ratio controls the effective step size).
- :math:`b` is the (frozen) bias.

``B`` is initialized to zero, so the LoRA branch contributes nothing at
initialization — the model behaves exactly like the frozen backbone until
LoRA is trained. ``A`` is initialized with Kaiming uniform.

Why this matters for continual learning
=======================================

Because the ternary backbone (``latent_scores`` + ``bias``) is **frozen**,
it can never change across tasks, giving **zero forgetting by design**.
Each task trains its own LoRA adapter pair; the adapters are stored
separately and swapped in for inference.

Usage::

    layer = TernarySTELoRALinear(784, 256, r=8)
    layer.freeze_backbone()                 # latent_scores + bias frozen

    # Only LoRA params get gradients
    optimizer = torch.optim.AdamW(layer.lora_parameters(), lr=0.001)
    for x, y in loader:
        out = layer(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    state = layer.get_lora_state()          # {'lora_A': ..., 'lora_B': ...}
    layer.load_lora_state(state)            # restore a saved adapter

.. seealso::
    :class:`TernarySTELinear` in :mod:`ph_neuro.layers.ste_linear` for the
    non-LoRA variant used to pre-train the backbone.
"""

from __future__ import annotations

from collections.abc import Iterator

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_linear import ste_sign

# ── LoRA helper functions (model-level) ─────────────────────────────


def iter_lora_layers(
    model: nn.Module,
) -> Iterator[tuple[str, "TernarySTELoRALinear"]]:
    """Iterate over all :class:`TernarySTELoRALinear` modules in ``model``.

    Yields ``(name, layer)`` pairs where ``name`` is the module's path
    within the model (e.g. ``"0"``, ``"3"`` for an ``nn.Sequential``).
    """
    for name, module in model.named_modules():
        if isinstance(module, TernarySTELoRALinear):
            yield name, module


def freeze_backbone(model: nn.Module) -> None:
    """Freeze the ternary backbone of every LoRA layer in ``model``.

    Sets ``requires_grad = False`` on each LoRA layer's ``latent_scores``
    and ``bias``, **and** on any BatchNorm affine parameters, so only the
    LoRA ``A``/``B`` matrices remain trainable. BatchNorm running stats
    are handled by the experiment runner (the model runs in ``eval()``
    mode during LoRA fine-tuning).
    """
    for _, layer in iter_lora_layers(model):
        layer.freeze_backbone()

    # BatchNorm affine weights are part of the frozen backbone.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)


def get_model_lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect LoRA states for every LoRA layer in ``model``.

    Returns:
        Flat dict: ``{f"{name}.lora_A": Tensor, f"{name}.lora_B": Tensor}``.
        Suitable for ``torch.save`` / ``torch.load``.
    """
    state: dict[str, torch.Tensor] = {}
    for name, layer in iter_lora_layers(model):
        layer_state = layer.get_lora_state()
        for key, value in layer_state.items():
            state[f"{name}.{key}"] = value
    return state


def load_model_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a flat LoRA state dict (from :func:`get_model_lora_state`).

    Args:
        model: Model containing :class:`TernarySTELoRALinear` layers.
        state: Flat dict keyed by ``"{name}.lora_A"`` / ``"{name}.lora_B"``.
    """
    for name, layer in iter_lora_layers(model):
        layer_state = {
            key.split(".", 1)[1]: value
            for key, value in state.items()
            if key.startswith(f"{name}.")
        }
        layer.load_lora_state(layer_state)


def reset_lora(model: nn.Module) -> None:
    """Re-initialize every LoRA adapter in ``model`` to zero contribution.

    Useful to guarantee each task starts from the exact frozen backbone.
    """
    for _, layer in iter_lora_layers(model):
        layer.reset_lora()


def count_lora_parameters(model: nn.Module) -> int:
    """Total number of trainable LoRA parameters across the model."""
    return sum(layer.count_lora_parameters() for _, layer in iter_lora_layers(model))


# ── LoRA STE Linear Layer ───────────────────────────────────────────


class TernarySTELoRALinear(nn.Module):
    """Linear layer with a frozen ternary backbone and trainable LoRA.

    Stores float latent scores (ternary backbone) plus low-rank LoRA
    matrices ``A`` and ``B``. The latent scores are meant to be frozen
    after pre-training (see :meth:`freeze_backbone`); the LoRA matrices
    are the only trainable parameters during continual-learning fine-tuning.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        r: LoRA rank (number of low-rank dimensions).
        alpha: LoRA scaling constant. Defaults to ``r``.
        bias: If ``True``, keeps a (frozen) learnable bias.
        device: Torch device.
        dtype: Torch dtype for all parameters.

    Attributes:
        latent_scores: Frozen ternary backbone parameter.
        lora_A: LoRA matrix ``(r, in_features)``.
        lora_B: LoRA matrix ``(out_features, r)``.
        scaling: ``alpha / r``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank r must be >= 1, got {r}")
        self._in_features = in_features
        self._out_features = out_features
        self._r = int(r)
        self._alpha = float(alpha if alpha is not None else r)
        self.scaling = self._alpha / self._r

        # ── Frozen ternary backbone ─────────────────────────────
        self.latent_scores = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.normal_(self.latent_scores, mean=0.0, std=0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # ── LoRA adapters ────────────────────────────────────────
        # A: (r, in), B: (out, r). B starts at zero so delta = 0 initially.
        self.lora_A = nn.Parameter(
            torch.empty(self._r, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, self._r, device=device, dtype=dtype)
        )
        # Kaiming uniform on A (like nn.Linear), zeros on B.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # By default the backbone stays trainable until freeze_backbone()
        # is called (this matches a plain TernarySTELinear's behaviour).

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def rank(self) -> int:
        return self._r

    @property
    def alpha(self) -> float:
        return self._alpha

    def freeze_backbone(self) -> None:
        """Freeze ``latent_scores`` and ``bias`` (only LoRA stays trainable)."""
        self.latent_scores.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Make ``latent_scores`` and ``bias`` trainable again."""
        self.latent_scores.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the LoRA parameters (``lora_A`` and ``lora_B``)."""
        yield self.lora_A
        yield self.lora_B

    def ternary_weight(self) -> torch.Tensor:
        """The (frozen) ternary weight matrix {-1, 0, +1}.

        Returns:
            int8 tensor of shape ``(out_features, in_features)``.
        """
        return self.latent_scores.sign().to(torch.int8)

    def get_lora_state(self) -> dict[str, torch.Tensor]:
        """Snapshot of the LoRA adapter (A and B).

        Returns:
            Dict: ``{"lora_A": Tensor, "lora_B": Tensor}``.
        """
        return {
            "lora_A": self.lora_A.detach().clone(),
            "lora_B": self.lora_B.detach().clone(),
        }

    def load_lora_state(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the LoRA adapter from a snapshot (see :meth:`get_lora_state`).

        Args:
            state: Dict with ``lora_A`` and ``lora_B`` tensors.
        """
        with torch.no_grad():
            self.lora_A.copy_(state["lora_A"])
            self.lora_B.copy_(state["lora_B"])

    def reset_lora(self) -> None:
        """Re-initialize the LoRA adapter to zero contribution.

        ``B`` is reset to zero and ``A`` to Kaiming uniform, so the LoRA
        branch once again contributes nothing (identical to frozen backbone).
        """
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def count_lora_parameters(self) -> int:
        """Number of LoRA parameters in this layer (``r * (in + out)``)."""
        return self._r * (self._in_features + self._out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen ternary contribution + LoRA contribution.

        Args:
            x: Input tensor, shape ``(batch, *, in_features)``.

        Returns:
            Output tensor, shape ``(batch, *, out_features)``.
        """
        # Frozen ternary backbone
        w_tern = ste_sign(self.latent_scores)
        out = F.linear(x, w_tern)

        # LoRA branch: (x @ Aᵀ) @ Bᵀ, scaled by alpha / r
        h = F.linear(x, self.lora_A)
        delta = F.linear(h, self.lora_B)
        out = out + self.scaling * delta

        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"r={self._r}, alpha={self._alpha:g}, "
            f"bias={self.bias is not None}"
        )
