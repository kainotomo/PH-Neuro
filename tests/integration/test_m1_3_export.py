"""Integration tests for Milestone M1.3 — Model export ONNX/C (<100MB, RPi).

Verifies the full export pipeline for DQT models:
    1. DQT conv layer  → nn.Conv2d with identical output
    2. DQT linear layer → nn.Linear with identical output
    3. Full dqt_cnn()   → inference model with identical output
    4. ONNX export → onnxruntime → identical output (roundtrip)
    5. ONNX model size  < 100MB
    6. 2-bit packed ternary export → load → identical weights
    7. All inference weights frozen (requires_grad=False)
    8. BN fusion works on the inference model

All tests run on CPU and use small synthetic inputs (no datasets needed).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
from ph_neuro.models.dqt_models import dqt_cnn
from ph_neuro.models.export import (
    dqt_to_inference_model,
    estimate_packed_size,
    export_packed_ternary,
    export_to_onnx,
    get_model_params_count,
    get_model_size_mb,
    load_packed_ternary,
    verify_onnx,
)
from ph_neuro.models.fuse_bn import fuse_bn_layers
from ph_neuro.utils.packing import pack_ternary, unpack_ternary

DEVICE = torch.device("cpu")


# ── 1. Conv layer conversion ───────────────────────────────────────


def test_dqt_to_inference_conv() -> None:
    """TernaryDQTConv2d → nn.Conv2d with identical output."""
    torch.manual_seed(0)
    dqt_layer = TernaryDQTConv2d(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )
    dqt_model = nn.Sequential(dqt_layer)
    dqt_model.eval()

    inference = dqt_to_inference_model(dqt_model)

    # Conversion produced an nn.Conv2d with the frozen ternary weights
    assert isinstance(inference[0], nn.Conv2d)
    assert inference[0].kernel_size == (3, 3)
    assert inference[0].stride == (2, 2)
    assert inference[0].padding == (1, 1)

    x = torch.randn(4, 3, 16, 16)
    with torch.no_grad():
        ref = dqt_model(x)
        out = inference(x)

    assert out.shape == ref.shape
    assert torch.equal(out, ref), "inference conv output must match DQT conv exactly"


# ── 2. Linear layer conversion ─────────────────────────────────────


def test_dqt_to_inference_linear() -> None:
    """TernaryDQTLinear → nn.Linear with identical output."""
    torch.manual_seed(0)
    dqt_layer = TernaryDQTLinear(in_features=128, out_features=16, bias=True)
    dqt_model = nn.Sequential(dqt_layer)
    dqt_model.eval()

    inference = dqt_to_inference_model(dqt_model)

    assert isinstance(inference[0], nn.Linear)
    assert inference[0].in_features == 128
    assert inference[0].out_features == 16
    assert inference[0].bias is not None

    x = torch.randn(5, 128)
    with torch.no_grad():
        ref = dqt_model(x)
        out = inference(x)

    assert out.shape == ref.shape
    assert torch.equal(out, ref), "inference linear output must match DQT linear exactly"


# ── 3. Full dqt_cnn() conversion ───────────────────────────────────


def test_dqt_cnn_to_inference() -> None:
    """The whole dqt_cnn() model converts to an identical inference model."""
    torch.manual_seed(0)
    model = dqt_cnn()
    model.eval()

    inference = dqt_to_inference_model(model)

    # Same number of layers
    assert len(inference) == len(model)

    # No DQT layers remain — all standard PyTorch
    for module in inference:
        assert not isinstance(
            module, (TernaryDQTConv2d, TernaryDQTLinear)
        ), "inference model must contain no DQT layers"

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = model(x)
        out = inference(x)

    assert out.shape == ref.shape == (2, 10)
    assert torch.equal(out, ref), "inference dqt_cnn output must match DQT model exactly"


# ── 4. ONNX export roundtrip ───────────────────────────────────────


def test_onnx_export_roundtrip(tmp_path) -> None:
    """Export → onnxruntime → identical output (± 1e-5)."""
    torch.manual_seed(0)
    model = dqt_cnn()
    model.eval()

    inference = dqt_to_inference_model(model)
    onnx_path = tmp_path / "dqt_cnn.onnx"
    export_to_onnx(inference, (1, 3, 32, 32), str(onnx_path))

    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0

    # Verify with a different (dynamic) batch size than the export dummy
    dummy = torch.randn(3, 3, 32, 32)
    assert verify_onnx(inference, str(onnx_path), dummy, rtol=1e-3, atol=1e-5)


# ── 5. ONNX model size < 100MB ─────────────────────────────────────


def test_onnx_model_size(tmp_path) -> None:
    """Exported ONNX model must be < 100MB (trivially — models are KBs)."""
    torch.manual_seed(0)
    model = dqt_cnn()
    model.eval()

    inference = dqt_to_inference_model(model)
    onnx_path = tmp_path / "dqt_cnn_size.onnx"
    export_to_onnx(inference, (1, 3, 32, 32), str(onnx_path))

    size_mb = get_model_size_mb(onnx_path)
    assert size_mb < 100.0, f"ONNX model too large: {size_mb:.2f} MB"
    # Sanity: ONNX stores fp32 weights, so it should be ≈ fp32 weight size
    # plus a small graph/overhead margin (not 10x larger).
    n_params = get_model_params_count(model)
    fp32_mb = n_params * 4 / (1024 * 1024)
    assert size_mb < fp32_mb + 5.0, (
        f"ONNX size {size_mb:.2f} MB exceeds expected fp32 weight size "
        f"{fp32_mb:.2f} MB + margin"
    )


# ── 6. Packed ternary export ───────────────────────────────────────


def test_packed_export(tmp_path) -> None:
    """pack_ternary → save → load → unpack_ternary → identical weights."""
    torch.manual_seed(0)
    model = dqt_cnn()
    model.eval()

    packed_path = tmp_path / "dqt_cnn.ternary"
    export_packed_ternary(model, str(packed_path))

    assert packed_path.exists()

    # 2-bit packing: file must be well under half the raw int8 weight bytes
    # (raw int8 would be n_ternary bytes; 2-bit packing is ~n_ternary / 4
    # plus a small per-layer metadata header).
    n_ternary = get_model_params_count(model)
    expected_packed = estimate_packed_size(model)
    assert expected_packed <= (n_ternary + 3) // 4
    assert packed_path.stat().st_size < n_ternary / 2, "packed file should be ~4x smaller than raw int8"

    # Load and compare each layer's weights with the original
    loaded = load_packed_ternary(str(packed_path))

    original_layers = [
        m for m in model.children()
        if isinstance(m, (TernaryDQTConv2d, TernaryDQTLinear))
    ]
    assert len(loaded) == len(original_layers)

    for (name, shape, weights), orig in zip(loaded, original_layers):
        expected = orig.weight_ternary
        assert tuple(weights.shape) == tuple(expected.shape)
        assert torch.equal(weights, expected), f"layer {name} weights changed after pack/unpack"

    # Independent check: pack_ternary/unpack_ternary roundtrip on one layer
    w = original_layers[0].weight_ternary
    packed = pack_ternary(w)
    restored = unpack_ternary(packed, w.shape)
    assert torch.equal(restored, w)


# ── 7. All inference weights frozen ────────────────────────────────


def test_inference_model_no_grad() -> None:
    """Every parameter in the inference model must have requires_grad=False."""
    torch.manual_seed(0)
    model = dqt_cnn()
    model.eval()

    inference = dqt_to_inference_model(model)

    params = list(inference.parameters())
    assert params, "inference model should have parameters"

    for p in params:
        assert p.requires_grad is False, f"parameter {p.shape} is not frozen"

    # Model is in eval() mode
    assert not inference.training

    # Buffers (BN running stats) also frozen
    for name, buf in inference.named_buffers():
        assert not buf.requires_grad


# ── 8. BN fusion before export ─────────────────────────────────────


def test_bn_fusion_before_export() -> None:
    """fuse_bn_layers() works on the inference model."""
    torch.manual_seed(0)
    model = dqt_cnn()
    model.eval()

    inference = dqt_to_inference_model(model)

    # Inference model still has BatchNorm layers (fused later)
    n_bn = sum(
        1 for m in inference if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
    )
    assert n_bn > 0, "dqt_cnn should contain BatchNorm layers"

    fused = fuse_bn_layers(inference, inplace=False)
    fused.eval()

    n_bn_after = sum(
        1 for m in fused if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
    )
    assert n_bn_after == 0, "all BatchNorm layers should be fused away"

    # Output preserved within numerical tolerance (per L5/E011 finding)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = inference(x)
        out = fused(x)
    np.testing.assert_allclose(ref.numpy(), out.numpy(), rtol=1e-3, atol=1e-2)
