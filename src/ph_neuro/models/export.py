"""Model export utilities for DQT ternary models (Milestone M1.3).

Converts a trained DQT *training* model into an inference-only model built
from standard ``nn.Conv2d`` / ``nn.Linear`` layers with frozen ternary
weights, fuses BatchNorm into element-wise affine layers, and exports the
result to ONNX for edge deployment on CPU (Raspberry Pi, <100 MB).

Why a separate inference model is required
-----------------------------------------
DQT layers use custom autograd Functions (``_DQTGradFn``,
``_DQTConvGradFn``) which ``torch.onnx.export`` cannot trace. At inference
time, however, the DQT forward pass is trivial::

    conv2d(weight_ternary.float(), x)      # TernaryDQTConv2d
    linear(weight_ternary.float(), x)      # TernaryDQTLinear
    ReLU / BatchNorm / MaxPool / Flatten

This module rebuilds that graph using standard layers so the trained
ternary weights can ship as a small, CPU-runnable ONNX model. The
training-only float buffers (``weight_float``) are dropped; only the
frozen int8 ternary weights survive — a ~4-16x size reduction vs the FP32
PyTorch checkpoint.

The converter also accepts classic STE layers (``TernarySTELinear``,
``TernarySTEConv2d``) whose ternary weights are derived from latent scores
via ``sign()`` — enabling export of ``ste_mlp`` models too.
"""

from __future__ import annotations

import contextlib
import os
import struct
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn as nn

from ph_neuro.layers.ste_conv import TernarySTEConv2d
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_conv import TernaryDQTConv2d
from ph_neuro.layers.ste_linear import TernarySTELinear
from ph_neuro.models.fuse_bn import fuse_bn_layers
from ph_neuro.utils.packing import pack_ternary, unpack_ternary

# Weight-bearing ternary layers that can be converted to standard layers.
_DQT_CONV_TYPES = (TernaryDQTConv2d, TernarySTEConv2d)
_DQT_LINEAR_TYPES = (TernaryDQTLinear, TernarySTELinear)
_TERNARY_LAYER_TYPES = (*_DQT_CONV_TYPES, *_DQT_LINEAR_TYPES)

# Layers that pass through unchanged (no trainable weights to convert).
_PASS_THROUGH_TYPES = (
    nn.ReLU,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.MaxPool2d,
    nn.Flatten,
    nn.Dropout,
)


@contextlib.contextmanager
def _full_float32_precision() -> Iterator[None]:
    """Temporarily disable TF32 so CUDA matches CPU/ONNX float32 semantics.

    On Ampere+ GPUs PyTorch defaults ``torch.backends.cudnn.allow_tf32`` to
    ``True``, which runs convolutions with reduced-precision TF32 (~10-bit
    mantissa). TF32 introduces ~1e-3 relative error that can be amplified
    into large *absolute* logit differences on deep models — falsely failing
    ONNX verification against the CPU (full float32) onnxruntime output.
    Disabling it gives an apples-to-apples full-float32 comparison.
    """
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.backends.cuda.matmul.allow_tf32 = old_matmul

# 2-bit packed ternary file format constants (see export_packed_ternary).
_PACKED_MAGIC = b"PHN3"
_PACKED_VERSION = 1


# ── Helpers ────────────────────────────────────────────────────────


def _get_ternary_weight(module: nn.Module) -> torch.Tensor:
    """Return the int8 ternary weight tensor of a ternary layer.

    DQT layers store ternary weights directly in ``weight_ternary`` (int8
    buffer); STE layers derive them from latent scores via ``sign()``.
    """
    if hasattr(module, "weight_ternary"):
        return module.weight_ternary
    return module.ternary_weight()


def is_ternary_layer(module: nn.Module) -> bool:
    """Whether a module is a DQT/STE weight-bearing ternary layer."""
    return isinstance(module, _TERNARY_LAYER_TYPES)


def get_model_params_count(model: nn.Module) -> int:
    """Count the number of ternary weights in a model.

    Counts the actual int8 ternary weights (``weight_ternary`` buffers for
    DQT layers, ``sign(latent_scores)`` for STE layers) — the weights that
    survive into the exported ONNX / packed file.

    Args:
        model: A DQT/STE model (or inference model).

    Returns:
        Number of ternary weights across all ternary layers.
    """
    total = 0
    for module in model.modules():
        if is_ternary_layer(module):
            total += _get_ternary_weight(module).numel()
    return total


def estimate_packed_size(model: nn.Module) -> int:
    """Estimate the packed size of a model's ternary weights, in bytes.

    Ternary weights are packed 2-bit (4 weights per byte), so the packed
    size is ``ceil(n_weights / 4)`` bytes — 4x smaller than int8 and 16x
    smaller than FP32.

    Args:
        model: A DQT/STE model (or inference model).

    Returns:
        Packed size in bytes.
    """
    return (get_model_params_count(model) + 3) // 4


def get_model_size_mb(model_or_path: nn.Module | str | os.PathLike) -> float:
    """Return the size of a model (or model file) in MB.

    - If given a file path (ONNX / checkpoint / packed file): the on-disk
      file size in MB.
    - If given a PyTorch module: the total bytes of all ``state_dict``
      tensors (parameters + buffers) in MB.

    Args:
        model_or_path: A ``nn.Module`` or a path to a model file.

    Returns:
        Size in megabytes.
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        return os.path.getsize(str(model_or_path)) / (1024 * 1024)
    total_bytes = 0
    for tensor in model_or_path.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)


# ── Inference Model Conversion ──────────────────────────────────────


def dqt_to_inference_model(dqt_model: nn.Sequential) -> nn.Sequential:
    """Convert a DQT training model to an inference-only model.

    Replaces ``TernaryDQTConv2d`` → ``nn.Conv2d`` and
    ``TernaryDQTLinear`` → ``nn.Linear`` using the frozen int8 ternary
    weights cast to float. ``BatchNorm``, ``ReLU``, ``MaxPool2d``,
    ``Flatten`` and ``Dropout`` pass through unchanged (BatchNorm is fused
    later by :func:`export_to_onnx`).

    All converted weights are frozen (``requires_grad=False``) and the
    resulting model is placed in ``eval()`` mode.

    Args:
        dqt_model: An ``nn.Sequential`` containing DQT/STE layers and
            standard activations / pooling / BN / flatten layers.

    Returns:
        ``nn.Sequential`` with only standard PyTorch layers, ready for
        ONNX export.

    Raises:
        TypeError: If a module type is not supported (neither a ternary
            weight-bearing layer nor a supported pass-through layer).
    """
    layers: list[nn.Module] = []

    for module in dqt_model.children():
        if isinstance(module, _DQT_CONV_TYPES):
            conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module._stride,
                padding=module._padding,
                dilation=module._dilation,
                bias=module.bias is not None,
            )
            # Set frozen ternary weight
            conv.weight.data = _get_ternary_weight(module).float().clone()
            conv.weight.requires_grad = False
            if module.bias is not None:
                conv.bias.data = module.bias.data.clone()
                conv.bias.requires_grad = False
            layers.append(conv)

        elif isinstance(module, _DQT_LINEAR_TYPES):
            linear = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )
            linear.weight.data = _get_ternary_weight(module).float().clone()
            linear.weight.requires_grad = False
            if module.bias is not None:
                linear.bias.data = module.bias.data.clone()
                linear.bias.requires_grad = False
            layers.append(linear)

        elif isinstance(module, _PASS_THROUGH_TYPES):
            layers.append(module)  # Pass through

        else:
            raise TypeError(
                f"Unsupported module type: {type(module).__name__}. "
                f"Only DQT/STE layers, BN, ReLU, MaxPool, Flatten, Dropout "
                f"are supported."
            )

    model = nn.Sequential(*layers)

    # Export is a CPU-only operation (ONNX tracing uses CPU tensors), so the
    # inference model must live on CPU even if the source DQT model was
    # trained on CUDA.
    model = model.to("cpu")
    model.eval()

    # Freeze EVERY parameter — including BatchNorm's scale/shift — so the
    # inference model is fully frozen (nothing can be trained or updated).
    for param in model.parameters():
        param.requires_grad = False

    return model


# ── ONNX Export ─────────────────────────────────────────────────────


def export_to_onnx(
    inference_model: nn.Sequential,
    input_shape: tuple[int, ...],
    output_path: str | os.PathLike,
    opset_version: int = 18,
) -> None:
    """Export an inference model to ONNX format.

    BatchNorm is fused to element-wise affine layers first (via
    :func:`ph_neuro.models.fuse_bn.fuse_bn_layers`), producing a graph that
    is both faster on CPU and fully ONNX-compatible. The export uses a
    dynamic batch axis so the resulting model accepts any batch size.

    Weights are embedded **inline** in the ``.onnx`` file (``external_data=
    False``) so the result is a single self-contained artifact that can be
    copied to a Raspberry Pi without a companion ``.onnx.data`` file.

    Args:
        inference_model: The output of :func:`dqt_to_inference_model`.
        input_shape: Input tensor shape, e.g. ``(1, 3, 32, 32)``.
        output_path: Where to write the ``.onnx`` file.
        opset_version: ONNX opset version (default 18 — the opset that the
            current exporter implements natively; onnxruntime on ARM /
            Raspberry Pi supports it).
    """
    from ph_neuro.models.fuse_bn import fuse_bn_layers

    model = fuse_bn_layers(inference_model, inplace=False)
    model.eval()

    output_path = str(output_path)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        external_data=False,
    )


def verify_onnx(
    inference_model: nn.Sequential,
    onnx_path: str | os.PathLike,
    dummy_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Verify that an exported ONNX model matches its PyTorch source.

    Runs the same fused graph (BatchNorm → affine) through onnxruntime and
    asserts the outputs match the PyTorch reference within tolerance.

    Args:
        inference_model: The same model that was passed to
            :func:`export_to_onnx` (BatchNorm is fused internally).
        onnx_path: Path to the exported ``.onnx`` file.
        dummy_input: Input tensor used for comparison (any batch size).
        rtol: Relative tolerance for ``assert_allclose``.
        atol: Absolute tolerance for ``assert_allclose``.

    Returns:
        ``True`` if the ONNX output matches the PyTorch output.

    Raises:
        AssertionError: If the outputs differ beyond tolerance.
        ImportError: If ``onnxruntime`` is not installed.
    """
    import onnxruntime as ort

    model = fuse_bn_layers(inference_model, inplace=False)
    model.eval()

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]

    np.testing.assert_allclose(torch_out, onnx_out, rtol=rtol, atol=atol)
    return True


# ── 2-bit Packed Ternary Export (companion .ternary file) ───────────


def _packed_layer_payload(module: nn.Module) -> bytes:
    """Pack a single ternary layer's weights into 2-bit bytes."""
    w = _get_ternary_weight(module).detach().cpu()
    packed = pack_ternary(w)
    return packed.numpy().tobytes()


def export_packed_ternary(model: nn.Sequential, output_path: str | os.PathLike) -> None:
    """Write all ternary weights of a model to a compact 2-bit ``.ternary`` file.

    Each ternary weight {-1, 0, +1} is stored in 2 bits (4 weights/byte) —
    4x smaller than int8, 16x smaller than FP32. The file records, in
    module order, the layer class name, weight shape, and packed payload so
    weights can be restored losslessly with :func:`load_packed_ternary`.

    Binary layout::

        magic      b"PHN3"        4 bytes
        version    uint8          1
        n_layers   uint32
        per layer:
            name_len  uint16
            name      UTF-8
            rank      uint8
            shape     int32 × rank
            nbytes    int64
            payload   packed 2-bit bytes

    Args:
        model: A DQT/STE model (or inference model) with ternary layers.
        output_path: Destination ``.ternary`` file path.
    """
    output_path = str(output_path)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    entries = []
    for module in model.children():
        if is_ternary_layer(module):
            w = _get_ternary_weight(module)
            entries.append(
                (module.__class__.__name__, tuple(w.shape), _packed_layer_payload(module))
            )

    with open(output_path, "wb") as f:
        f.write(_PACKED_MAGIC)
        f.write(struct.pack("<B", _PACKED_VERSION))
        f.write(struct.pack("<I", len(entries)))
        for name, shape, payload in entries:
            name_b = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<B", len(shape)))
            for dim in shape:
                f.write(struct.pack("<i", dim))
            f.write(struct.pack("<q", len(payload)))
            f.write(payload)


def load_packed_ternary(path: str | os.PathLike) -> list[tuple[str, tuple[int, ...], torch.Tensor]]:
    """Read a ``.ternary`` file back into per-layer ternary weights.

    Args:
        path: Path to a file written by :func:`export_packed_ternary`.

    Returns:
        List of ``(class_name, shape, weights)`` tuples in the original
        module order, where ``weights`` is an int8 tensor of values in
        {-1, 0, +1} with the recorded shape.
    """
    entries: list[tuple[str, tuple[int, ...], torch.Tensor]] = []
    with open(path, "rb") as f:
        magic = f.read(len(_PACKED_MAGIC))
        if magic != _PACKED_MAGIC:
            raise ValueError(f"Not a PH-Neuro packed ternary file: {path}")
        (version,) = struct.unpack("<B", f.read(1))
        if version != _PACKED_VERSION:
            raise ValueError(f"Unsupported packed file version: {version}")
        (n_layers,) = struct.unpack("<I", f.read(4))

        for _ in range(n_layers):
            (name_len,) = struct.unpack("<H", f.read(2))
            name = f.read(name_len).decode("utf-8")
            (rank,) = struct.unpack("<B", f.read(1))
            shape = struct.unpack(f"<{'i' * rank}", f.read(4 * rank))
            (nbytes,) = struct.unpack("<q", f.read(8))
            payload = f.read(nbytes)

            packed = np.frombuffer(payload, dtype=np.int8)
            packed_t = torch.from_numpy(packed)
            weights = unpack_ternary(packed_t, shape)
            entries.append((name, shape, weights))

    return entries


def export_model_to_onnx(
    dqt_model: nn.Sequential,
    input_shape: tuple[int, ...],
    output_path: str | os.PathLike,
    opset_version: int = 17,
    verify: bool = True,
    packed_path: str | os.PathLike | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """One-call helper: convert + export + (optionally) verify + pack.

    Convenience wrapper used by the M1.3 CLI. Runs
    :func:`dqt_to_inference_model`, :func:`export_to_onnx`, optionally
    :func:`verify_onnx` and :func:`export_packed_ternary`, and returns a
    summary dict.

    Args:
        dqt_model: The trained DQT/STE training model.
        input_shape: Input tensor shape, e.g. ``(1, 3, 32, 32)``.
        output_path: Destination ``.onnx`` path.
        opset_version: ONNX opset version.
        verify: If ``True``, run onnxruntime verification after export.
        packed_path: Optional path to also write a 2-bit ``.ternary`` file.
        device: Device used for the PyTorch reference (default ``"cpu"`` —
            ONNX runs on CPU anyway).

    Returns:
        Summary dict with ``onnx_path``, ``onnx_size_mb``, ``packed_path``,
        ``packed_size_mb``, ``n_ternary_weights``, ``packed_bytes``,
        ``verified`` and ``max_abs_diff`` (if verified).
    """
    from ph_neuro.models.fuse_bn import fuse_bn_layers

    dqt_model = dqt_model.to(device)
    dqt_model.eval()

    inference_model = dqt_to_inference_model(dqt_model)
    export_to_onnx(inference_model, input_shape, output_path, opset_version=opset_version)

    summary: dict = {
        "onnx_path": str(output_path),
        "onnx_size_mb": get_model_size_mb(output_path),
        "n_ternary_weights": get_model_params_count(dqt_model),
        "packed_bytes": estimate_packed_size(dqt_model),
        "packed_path": str(packed_path) if packed_path is not None else None,
        "packed_size_mb": None,
        "verified": None,
        "max_abs_diff": None,
    }

    if packed_path is not None:
        export_packed_ternary(dqt_model, packed_path)
        summary["packed_size_mb"] = get_model_size_mb(packed_path)

    if verify:
        dummy = torch.randn(1, *input_shape[1:], device=device)
        # Reference must use the exact fused graph that was exported. Run it
        # with full float32 precision (no TF32) so the CUDA reference matches
        # the CPU onnxruntime output within float noise.
        with _full_float32_precision():
            fused = fuse_bn_layers(inference_model, inplace=False).to(device)
            with torch.no_grad():
                ref = fused(dummy).cpu().numpy()

        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path))
        onnx_out = session.run(None, {"input": dummy.cpu().numpy()})[0]
        max_abs_diff = float(np.max(np.abs(ref - onnx_out)))
        summary["verified"] = bool(
            np.allclose(ref, onnx_out, rtol=1e-3, atol=1e-5)
        )
        summary["max_abs_diff"] = max_abs_diff

    return summary
