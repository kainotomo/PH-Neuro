"""8-bit optimizer helpers for the Phase 2.5 memory sprint.

``make_adamw`` returns a bitsandbytes 8-bit AdamW when available (optimizer
states 8→2 B/param, **-75%**) and falls back to plain fp32 AdamW otherwise.

8-bit AdamW is 100% compatible with DQT's custom autograd (``_DQTGradFn``),
validated by ``tests/ad_hoc/test_8bit_adam_dqt.py`` (OPT-1): accuracy matches
the fp32 baseline and its ``state_dict()`` round-trips exactly through
``torch.save``/``torch.load``, so the M2.x pause/resume checkpointing keeps
working.

Opt out with the env var ``PH_NEURO_NO_8BIT=1`` (e.g. for an fp32 A/B run)::

    PH_NEURO_NO_8BIT=1 .venv/bin/python -m ph_neuro.examples.run_m2_1_dqt_transformer ...
"""

from __future__ import annotations

import os

import torch


def _use_8bit(flag: bool | None) -> bool:
    """Resolve 8-bit preference: explicit flag beats the env var."""
    if flag is not None:
        return flag
    return os.environ.get("PH_NEURO_NO_8BIT", "0") != "1"


def make_adamw(
    params,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    use_8bit: bool | None = None,
) -> torch.optim.Optimizer:
    """AdamW optimizer — 8-bit (bitsandbytes) when available, else fp32.

    Args:
        params: Iterable of parameters or a list of param-group dicts
            (e.g. M2.3's slow-router param groups).
        lr: Learning rate.
        betas: AdamW betas.
        eps: AdamW epsilon.
        weight_decay: Weight decay.
        amsgrad: Passed through to fp32 AdamW (8-bit ignores it).
        use_8bit: Force on/off; ``None`` = auto (env ``PH_NEURO_NO_8BIT=1``
            disables 8-bit, useful for A/B testing).

    Returns:
        ``bnb.optim.AdamW8bit`` when available, else ``torch.optim.AdamW``.
    """
    if _use_8bit(use_8bit):
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        except ImportError:
            print(
                "⚠️  bitsandbytes not installed — falling back to fp32 AdamW "
                "(`.venv/bin/pip install bitsandbytes` for 8-bit)."
            )
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
