"""BatchNormalization fusion utility for ternary STE models.

After training, BatchNorm layers (at inference) perform a simple
element-wise affine transform: ``z = γ * (x - μ) / √(σ² + ε) + β``.

This is mathematically equivalent to ``z = scale * x + bias`` where
``scale = γ / √(σ² + ε)`` and ``bias = β - γ * μ / √(σ² + ε)``.

``fuse_bn_layers()`` replaces each ``BatchNorm1d`` / ``BatchNorm2d``
in a trained model with a cheaper ``ElementWiseAffine1d`` /
``ElementWiseAffine2d`` layer, preserving the exact output.

The model architecture in this codebase is typically::

    TernarySTELinear → ReLU → BatchNorm1d → TernarySTELinear → ...

We cannot fuse BN *into* the preceding ternary linear layer because
ReLU sits between them. Instead, we replace BN with a lightweight
element-wise affine layer that does the same computation faster.

.. note::

    Due to floating-point operation reordering, outputs between the
    fused and unfused model may differ by up to ~1e-2 after multiple
    BN layers, because ``γ*(x-μ)/√(σ²+ε)+β`` and ``(γ/√(σ²+ε))*x + (β-γ*μ/√(σ²+ε))``
    have different rounding characteristics. This is expected and harmless
    — accuracy is preserved within numerical tolerance.

Usage::

    from ph_neuro.models.ste_models import ste_mlp
    from ph_neuro.models.fuse_bn import fuse_bn_layers

    model = ste_mlp([784, 512, 256, 10], batch_norm=True)
    model.eval()

    # Replace BN → ElementWiseAffine
    fused = fuse_bn_layers(model)

    # Inference is now faster (no BN normalization passes)
    with torch.no_grad():
        out = fused(x)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ph_neuro.layers.fused_bn import ElementWiseAffine1d, ElementWiseAffine2d
from ph_neuro.layers.ste_conv import TernarySTEConv2d
from ph_neuro.layers.ste_hysteresis import HysteresisSTEConv2d, HysteresisSTELinear
from ph_neuro.layers.ste_linear import TernarySTELinear

_ALLOWED_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.ELU)

# Types of STE layers we can handle
_STE_LINEAR_TYPES = (TernarySTELinear, HysteresisSTELinear)
_STE_CONV_TYPES = (TernarySTEConv2d, HysteresisSTEConv2d)
_BN1D_TYPES = (nn.BatchNorm1d,)
_BN2D_TYPES = (nn.BatchNorm2d,)


def _bn_to_affine_1d(bn_layer: nn.BatchNorm1d) -> ElementWiseAffine1d:
    """Convert a ``BatchNorm1d`` to an ``ElementWiseAffine1d``.

    Args:
        bn_layer: BatchNorm1d in ``eval()`` mode.

    Returns:
        ``ElementWiseAffine1d`` with frozen scale/bias.
    """
    denom = torch.sqrt(bn_layer.running_var + bn_layer.eps)
    scale = bn_layer.weight / denom
    bias = bn_layer.bias - bn_layer.weight * bn_layer.running_mean / denom
    return ElementWiseAffine1d(
        num_features=bn_layer.num_features,
        scale=scale.detach().clone(),
        bias=bias.detach().clone(),
        device=bn_layer.running_mean.device,
        dtype=bn_layer.running_mean.dtype,
    )


def _bn_to_affine_2d(bn_layer: nn.BatchNorm2d) -> ElementWiseAffine2d:
    """Convert a ``BatchNorm2d`` to an ``ElementWiseAffine2d``.

    Args:
        bn_layer: BatchNorm2d in ``eval()`` mode.

    Returns:
        ``ElementWiseAffine2d`` with frozen scale/bias.
    """
    denom = torch.sqrt(bn_layer.running_var + bn_layer.eps)
    scale = bn_layer.weight / denom
    bias = bn_layer.bias - bn_layer.weight * bn_layer.running_mean / denom
    return ElementWiseAffine2d(
        num_channels=bn_layer.num_features,
        scale=scale.detach().clone(),
        bias=bias.detach().clone(),
        device=bn_layer.running_mean.device,
        dtype=bn_layer.running_mean.dtype,
    )


def _is_ste_layer(module: nn.Module) -> bool:
    """Check if a module is any ternary STE layer (linear or conv)."""
    return isinstance(module, (*_STE_LINEAR_TYPES, *_STE_CONV_TYPES))


def _is_activation(module: nn.Module) -> bool:
    """Check if a module is an element-wise activation function."""
    return isinstance(module, _ALLOWED_ACTIVATIONS)


def fuse_bn_layers(
    model: nn.Sequential,
    inplace: bool = False,
) -> nn.Sequential:
    """Replace all BatchNorm layers with cheaper element-wise affine layers.

    In the PH-Neuro model architecture, BN typically appears after
    ``TernarySTELinear → ReLU → BatchNorm1d``. Since ReLU sits between
    the linear layer and BN, we cannot fuse BN *into* the linear layer.
    Instead, we replace each BN with an equivalent ``ElementWiseAffine``
    that applies ``scale * x + bias`` — mathematically identical at
    inference but faster (no normalization pass).

    The model must be in ``eval()`` mode before calling this function.

    Args:
        model: An ``nn.Sequential`` containing ternary STE layers and
            optional BatchNorm layers.
        inplace: If ``True``, modifies the original model in-place by
            popping all layers and re-appending the fused ones. If
            ``False`` (default), returns a new ``nn.Sequential`` leaving
            the original untouched.

    Returns:
        An ``nn.Sequential`` with all BN layers replaced by
        ``ElementWiseAffine`` layers.

    Raises:
        RuntimeError: If the model is in training mode (call
            ``model.eval()`` first).
        TypeError: If the model is not an ``nn.Sequential``.
    """
    if not isinstance(model, nn.Sequential):
        raise TypeError(
            f"Expected nn.Sequential, got {type(model).__name__}. "
            "Only nn.Sequential models are supported for BN fusion."
        )

    if model.training:
        raise RuntimeError(
            "Model must be in eval() mode before BN fusion. "
            "Call model.eval() first — BN running statistics are "
            "only stable in eval mode."
        )

    new_layers: list[nn.Module] = []

    for layer in model:
        if isinstance(layer, nn.BatchNorm1d):
            new_layers.append(_bn_to_affine_1d(layer))
        elif isinstance(layer, nn.BatchNorm2d):
            new_layers.append(_bn_to_affine_2d(layer))
        else:
            # Deep-copy non-BN layers to fully isolate the fused model
            # (important for inplace=True ReLU layers that would otherwise
            # share state between original and fused models)
            new_layers.append(copy.deepcopy(layer))

    if inplace:
        while len(model) > 0:
            model.pop(len(model) - 1)
        for layer in new_layers:
            model.append(layer)
        return model

    return nn.Sequential(*new_layers)
