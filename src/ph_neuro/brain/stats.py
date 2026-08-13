"""Statistical helpers for E031 (protocol §6) — no scipy dependency.

Implements:
* Paired block-level t-test (n = sliding windows, paired frozen ↔ plastic).
* Bootstrap 95% percentile CI on aggregate Δppl (resampling blocks with
  replacement, 10,000 iterations, paired structure preserved).
* Cohen's d (mean / SD of per-block NLL differences).
* Cross-seed paired t-test on per-seed Δppl values.

The two-sided t-distribution p-value uses a regularized incomplete beta
function (continued-fraction method) so we need no scipy.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

# ── incomplete beta / t-distribution (pure Python) ─────────────────


def _beta_cont_frac(a: float, b: float, x: float, max_iter: int = 300) -> float:
    """Continued-fraction evaluation of the incomplete beta (Lentz)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_pre = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(ln_pre) / a
    cf = _beta_cont_frac(a, b, x)
    return front * cf


def paired_t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value for a paired t-statistic with ``df`` degrees of freedom.

    ``p = I_{df/(df+t²)}(df/2, 1/2)``.
    """
    x = df / (df + t * t)
    return betainc(df / 2.0, 0.5, x)


# ── block-level paired stats ───────────────────────────────────────


def block_paired_stats(
    frozen_nll: Sequence[float],
    plastic_nll: Sequence[float],
    frozen_tokens: Sequence[int] | None = None,
    *,
    n_boot: int = 10_000,
    boot_seed: int = 0,
) -> dict:
    """Paired frozen↔plastic statistics over per-window mean NLLs.

    Args:
        frozen_nll: per-window **sum** NLL (nats over the window's tokens) for
            the frozen model — the unit produced by ``BrainWrapper.evaluate()``.
        plastic_nll: per-window sum NLL for the plastic model (same order).
        frozen_tokens: per-window token counts (same order as ``frozen_nll``;
            also used for the plastic side — windows are identical).

    Returns:
        A dict with block-level paired t/p/d (on **per-token mean NLL** per
        window, so variable-length windows are compared fairly), aggregate
        Δppl (token-weighted), and a bootstrap 95% percentile CI on Δppl.
    """
    if len(frozen_nll) != len(plastic_nll):
        raise ValueError("frozen and plastic block lists must be equal length")
    n = len(frozen_nll)
    if n < 2:
        return {"error": "too few blocks"}
    if frozen_tokens is None:
        frozen_tokens = [1.0] * n
    elif len(frozen_tokens) != n:
        raise ValueError("frozen_tokens must match the block lists' length")

    # Per-token mean NLL per window (fair across variable-length windows).
    f_mean = [f / max(w, 1) for f, w in zip(frozen_nll, frozen_tokens, strict=True)]
    p_mean = [p / max(w, 1) for p, w in zip(plastic_nll, frozen_tokens, strict=True)]

    diffs = [fm - pm for fm, pm in zip(f_mean, p_mean, strict=True)]  # + = plastic better
    mean_d = sum(diffs) / n
    var = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t_stat = mean_d / se if se > 0 else 0.0
    p_val = paired_t_pvalue(t_stat, n - 1) if se > 0 else 1.0
    cohens_d = mean_d / sd if sd > 0 else 0.0

    # Aggregate Δppl = ppl(frozen) − ppl(plastic) over all tokens (sum-NLL
    # inputs are already totals, so no extra per-token weighting).
    tot_f = sum(f for f in frozen_nll)
    tot_p = sum(p for p in plastic_nll)
    tot_w = sum(frozen_tokens)
    ppl_f = math.exp(tot_f / tot_w)
    ppl_p = math.exp(tot_p / tot_w)
    delta_ppl = ppl_f - ppl_p

    # Bootstrap percentile CI on Δppl (paired, block resampling; each
    # resample recomputes aggregate ppl from per-token mean NLLs).
    rng = random.Random(boot_seed)
    indices = list(range(n))
    deltas = []
    for _ in range(n_boot):
        sample = [rng.choice(indices) for _ in range(n)]
        w = sum(frozen_tokens[i] for i in sample)
        nf = sum(frozen_nll[i] for i in sample) / w
        np_ = sum(plastic_nll[i] for i in sample) / w
        deltas.append(math.exp(nf) - math.exp(np_))
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]

    return {
        "n_blocks": n,
        "mean_delta_nll": mean_d,  # nats, + = plastic better
        "std_delta_nll": sd,
        "paired_t": t_stat,
        "paired_p": p_val,
        "cohens_d": cohens_d,
        "delta_ppl": delta_ppl,
        "delta_ppl_ci95": [lo, hi],
        "ppl_frozen": ppl_f,
        "ppl_plastic": ppl_p,
    }


# ── cross-seed summary ─────────────────────────────────────────────


def cross_seed_summary(per_seed: list[dict], key: str = "target_ppl_delta") -> dict:
    """Mean ± SD of a per-seed metric plus a one-sample paired t-test.

    Args:
        per_seed: list of per-seed metric dicts (all must contain ``key``).
        key: the metric to summarize (e.g. ``"target_ppl_delta"``).

    Returns:
        ``{"n": n, "mean": ..., "sd": ..., "t": ..., "p": ..., "cohens_d": ...}``
        testing mean != 0.
    """
    vals = [float(d[key]) for d in per_seed]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else 0.0
    p = paired_t_pvalue(t, n - 1) if se > 0 and n > 1 else 1.0
    d = mean / sd if sd > 0 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": p, "cohens_d": d}
