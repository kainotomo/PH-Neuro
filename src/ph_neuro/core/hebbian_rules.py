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
