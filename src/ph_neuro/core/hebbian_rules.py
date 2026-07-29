"""Hebbian plasticity rules for ternary weight updates.

All rules operate on latent scores (fp16) paired with ternary weights.
The core Hebbian rule is:

    Δscore = lr × pre_activation × post_activation

Since pre/post are ternary {-1, 0, +1}, the update per synapse is
always one of: ``+lr``, ``-lr``, or ``0``.
"""

from __future__ import annotations

import torch


def hebbian_update(
    scores: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Basic Hebbian update: ``Δscore = lr × postᵀ @ pre / batch_size``.

    "Neurons that fire together, wire together" — correlated pre/post
    activity strengthens the synapse.

    Args:
        scores: Latent score tensor, shape ``(out_features, in_features)``.
        pre: Pre-activations, shape ``(batch, in_features)``, values in {-1, 0, +1}.
        post: Post-activations, shape ``(batch, out_features)``, values in {-1, 0, +1}.
        lr: Learning rate.

    Returns:
        Updated score tensor (same shape as ``scores``).
    """
    delta = lr * (post.T.to(scores.dtype) @ pre.to(scores.dtype))
    return scores + delta / pre.shape[0]


def anti_hebbian_update(
    scores: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Anti-Hebbian update: ``Δscore = -lr × postᵀ @ pre / batch_size``.

    Anti-correlated activity weakens the synapse. Used for wrong-class
    output neurons in supervised Hebbian learning.

    Args:
        scores: Latent score tensor, shape ``(out_features, in_features)``.
        pre: Pre-activations, shape ``(batch, in_features)``, values in {-1, 0, +1}.
        post: Post-activations, shape ``(batch, out_features)``, values in {-1, 0, +1}.
        lr: Learning rate.

    Returns:
        Updated score tensor.
    """
    delta = -lr * (post.T.to(scores.dtype) @ pre.to(scores.dtype))
    return scores + delta / pre.shape[0]


def oja_update(
    scores: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Oja's rule: ``Δscore = lr × (postᵀ @ pre - postᵀ @ (score × pre))``.

    Oja's rule adds a weight-normalization term that prevents unbounded
    growth. Useful for unsupervised Hebbian learning where all synapses
    would otherwise strengthen indefinitely.

    Args:
        scores: Latent score tensor, shape ``(out_features, in_features)``.
        pre: Pre-activations, shape ``(batch, in_features)``, values in {-1, 0, +1}.
        post: Post-activations, shape ``(batch, out_features)``, values in {-1, 0, +1}.
        lr: Learning rate.

    Returns:
        Updated score tensor.
    """
    post_f = post.to(scores.dtype)
    pre_f = pre.to(scores.dtype)
    delta = lr * (post_f.T @ pre_f)
    norm = lr * (post_f.pow(2).T @ pre_f)
    return scores + delta / pre.shape[0] - scores * norm.mean(dim=1, keepdim=True)


def bcm_update(
    scores: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    lr: float,
    theta_m: float = 0.0,
) -> torch.Tensor:
    """BCM (Bienenstock-Cooper-Munro) rule.

    BCM introduces a sliding threshold: if post-activation exceeds theta_m,
    LTP (long-term potentiation) occurs; below theta_m, LTD (long-term
    depression) occurs. This creates competition between neurons.

    For ternary post-activations, theta_m controls whether a firing neuron
    (+1) strengthens or weakens its inputs.

    Args:
        scores: Latent score tensor, shape ``(out_features, in_features)``.
        pre: Pre-activations, shape ``(batch, in_features)``, values in {-1, 0, +1}.
        post: Post-activations, shape ``(batch, out_features)``, values in {-1, 0, +1}.
        lr: Learning rate.
        theta_m: Modification threshold. Default 0.0.

    Returns:
        Updated score tensor.
    """
    # BCM: Δw ∝ post × (post - theta_m)ᵀ @ pre
    post_f = post.to(scores.dtype)
    modulated = post_f * (post_f - theta_m)  # shape (batch, out_features)
    delta = lr * (modulated.T @ pre.to(scores.dtype))
    return scores + delta / pre.shape[0]


def neuromodulated_update(
    scores: torch.Tensor,
    pre: torch.Tensor,
    modulator: torch.Tensor,
    lr: float,
    post: torch.Tensor | None = None,
) -> torch.Tensor:
    """Three-factor neuromodulated Hebbian update.

    The core three-factor rule (Frémaux & Gerstner, 2016):

        Δscore = η · M · pre · post

    where M is a neuromodulator ∈ {-1, 0, +1} (or continuous) that
    controls whether the pre×post correlation is strengthened,
    ignored, or weakened.

    When ``post`` is provided, the modulator is applied element-wise:
    ``Δ = lr × (modulator ⊙ post)ᵀ @ pre / batch_size``.

    When ``post`` is ``None``, the modulator directly replaces post:
    ``Δ = lr × modulatorᵀ @ pre / batch_size``. This is the simpler
    form used by ``NeuromodulatedHebbianClassifier`` where the
    modulator already encodes both target activity and sign.

    Args:
        scores: Latent score tensor, shape ``(out_features, in_features)``.
        pre: Pre-activations, shape ``(batch, in_features)``, values in {-1, 0, +1}.
        modulator: Neuromodulator, shape ``(batch, out_features)`` or
            ``(batch,)``. Values in {-1, 0, +1} or continuous.
        lr: Learning rate.
        post: Optional post-activations, shape ``(batch, out_features)``.
            If provided, modulator is applied element-wise to post before
            the matmul. If ``None``, modulator is used directly as the
            left operand.

    Returns:
        Updated score tensor (same shape as ``scores``).
    """
    mod_f = modulator.to(scores.dtype)
    pre_f = pre.to(scores.dtype)

    if post is not None:
        # Three-factor: Δ = lr × (M ⊙ post)ᵀ @ pre
        post_f = post.to(scores.dtype)
        combined = mod_f * post_f  # element-wise modulator × post
        delta = lr * (combined.T @ pre_f)
    else:
        # Direct modulator: Δ = lr × Mᵀ @ pre (modulator encodes both
        # "which neuron" and "strengthen/weaken")
        if mod_f.dim() == 1:
            # Per-sample modulator: unsqueeze to (batch, 1) for broadcast
            mod_f = mod_f.unsqueeze(-1)
        delta = lr * (mod_f.T @ pre_f)

    return scores + delta / pre.shape[0]
