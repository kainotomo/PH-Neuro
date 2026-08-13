"""Surprise modulator — the global float32 signal that gates plasticity.

Implements the locked Step 0.3 design:

    L_t  = mean over sequence of −log P(token_t | context)   # float32
    L̂_t  ← α·L̂_{t−1} + (1−α)·L_t                            # EMA, α = 0.99
    s_t  = (L_t − L̂_t) / L̂_t                                # relative deviation
    M_t  = M_max / (1 + exp(−k·(s_t − s₀)))                 # sigmoid, float32

``M`` is one global float32 scalar per update, broadcast to all layers. All
math is float32 end-to-end (bf16 would underflow ``M ≈ 1e-3`` to zero).

Defaults (locked): α=0.99, s₀=0.05, k=60, M_max=1.0.

Also supports a ``"constant"`` mode (``M`` fixed at ``constant_M``) used for
the B3 constant-M ablation baseline, where no EMA state is needed.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class SurpriseModulator:
    """EMA-of-loss surprise signal mapped through a sigmoid.

    Modes:
        * ``"surprise_ema"`` (default): ``M = sigmoid(k·(s − s₀))`` where
          ``s`` is the relative loss deviation from a running EMA.
        * ``"constant"``: ``M`` is a fixed constant (B3 ablation).
    """

    _MODES = ("surprise_ema", "constant")
    _ALLOWED_KEYS = {
        "mode",
        "alpha",
        "s0",
        "k",
        "M_max",
        "constant_M",
        "M",  # protocol alias for constant_M (B3: {"mode": "constant", "M": 1.0})
    }

    def __init__(
        self,
        *,
        mode: str = "surprise_ema",
        alpha: float = 0.99,
        s0: float = 0.05,
        k: float = 60.0,
        M_max: float = 1.0,  # noqa: N803 - locked spec notation
        constant_M: float = 1.0,  # noqa: N803 - locked spec notation
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode={mode!r} not in {self._MODES}")
        self.mode = mode
        self.alpha = float(alpha)
        self.s0 = float(s0)
        self.k = float(k)
        self.M_max = float(M_max)
        self.constant_M = float(constant_M)
        self.dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
        # EMA state
        self.ema_loss: torch.Tensor | None = None  # float32 scalar or None (unset)
        self.initialized: bool = False

    # ── core ────────────────────────────────────────────────────────

    def update(self, loss: float | torch.Tensor) -> tuple[float, float]:
        """Advance the EMA and return ``(s, M)``.

        Args:
            loss: the current sequence-level cross-entropy loss (a float or
                a scalar tensor; converted to a float32 scalar internally).

        Returns:
            ``(s, M)`` as Python floats. ``s`` is the relative loss deviation
            ``(L − L̂)/L̂``; ``M`` is the sigmoid-modulated surprise in
            ``[0, M_max]``. In ``"constant"`` mode returns ``(0.0, M)``.
        """
        if self.mode == "constant":
            return 0.0, float(self.constant_M)

        L = torch.as_tensor(float(loss), dtype=self.dtype)  # noqa: N806
        if not self.initialized or self.ema_loss is None:
            L_hat = L.detach().clone()  # noqa: N806
        else:
            L_hat = self.alpha * self.ema_loss + (1.0 - self.alpha) * L  # noqa: N806
        self.ema_loss = L_hat.detach().clone()
        self.initialized = True

        s = (L - L_hat) / L_hat
        M = self.M_max / (1.0 + math.exp(-self.k * (float(s) - self.s0)))  # noqa: N806
        return float(s), float(M)

    def reset(self) -> None:
        """Clear EMA state (unset sentinel)."""
        self.ema_loss = None
        self.initialized = False

    # ── serialization ───────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ema_loss": float(self.ema_loss) if self.ema_loss is not None else None,
            "initialized": self.initialized,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ema = state.get("ema_loss")
        self.ema_loss = torch.as_tensor(ema, dtype=self.dtype) if ema is not None else None
        self.initialized = bool(state.get("initialized", ema is not None))

    # ── helpers ─────────────────────────────────────────────────────

    @classmethod
    def validate_config(cls, cfg: dict[str, Any]) -> None:
        """Raise ``ValueError`` on unknown modulator config keys (per spec)."""
        bad = set(cfg) - cls._ALLOWED_KEYS
        if bad:
            raise ValueError(f"unknown modulator_cfg keys: {sorted(bad)}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> SurpriseModulator:
        """Build a modulator from a ``modulator_cfg`` dict (validated)."""
        cfg = dict(cfg or {})
        cls.validate_config(cfg)
        if "M" in cfg and "constant_M" not in cfg:
            cfg["constant_M"] = cfg.pop("M")
        return cls(**cfg)
