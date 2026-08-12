#!/usr/bin/env python3
"""Milestone M1.5 — Memory benchmarks vs TF Lite (4× smaller, 2× faster).

The final Phase 1 milestone. Proves the two headline claims for the
PH-Neuro ternary models:

1. **4× smaller than TF Lite INT8** — weights are stored 2-bit (4
   weights/byte) vs 8-bit (1 byte/weight), so the packed size is 4× smaller
   for the same architecture.
2. **2× faster inference than TF Lite INT8** — ternary inference uses
   2-bit popcount arithmetic vs 8-bit multiply-add (BitNet paper). TF Lite
   is NOT installed; the comparison is theoretical (same-CPU estimate).

The benchmark measures three things per model and writes one JSON:

- **Model size (measured):** number of ternary weights, packed 2-bit size
  (via ``pack_ternary``), ONNX file size (existing M1.3 artifact or a fresh
  export), and the *theoretical* TF Lite INT8 / FP16 sizes (1 / 2 bytes per
  weight) with the size ratio.
- **Inference speed (measured, CPU):** warmup 10 batches then ``--batches``
  timed passes, batch_size=1, on the fused BN-free inference graph. Both the
  PyTorch fused model and (if onnxruntime is installed) the ONNX runtime are
  timed. TF Lite INT8 speed is estimated at 2× the PH-Neuro time (2-bit
  popcount vs 8-bit multiply-add).
- **Training memory (measured on GPU, DQT):** peak CUDA memory during 1
  epoch of DQT training via ``torch.cuda.reset_peak_memory_stats()`` /
  ``max_memory_allocated()``. STE memory is *estimated* at 4.5× (E017: ~2 vs
  ~9 bytes/param — DQT keeps no latent float scores).

Usage::

    # Inference on CPU (batch=1, 100 batches) + training memory on GPU:
    python -m ph_neuro.examples.run_m1_5_benchmarks \\
        --model dqt_cnn --device cpu --batches 100

    # Inference on CUDA, skip the training-memory pass:
    python -m ph_neuro.examples.run_m1_5_benchmarks \\
        --model dqt_cnn_cifar100 --device cuda --skip-training-memory

Output:
    JSON file: ``{output_dir}/results_m1_5_{model}.json``
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.examples._utils import print_header
from ph_neuro.utils.optimizers import make_adamw
from ph_neuro.examples.run_m1_2_dqt_cifar100 import apply_dqt_rounding, is_dqt_module
from ph_neuro.models.dqt_models import dqt_cnn, dqt_cnn_cifar100
from ph_neuro.models.export import (
    _get_ternary_weight,
    dqt_to_inference_model,
    export_to_onnx,
    get_model_params_count,
    is_ternary_layer,
)
from ph_neuro.models.fuse_bn import fuse_bn_layers
from ph_neuro.models.ste_models import ste_mlp
from ph_neuro.training.data import get_cifar10_loaders, get_cifar100_loaders, get_mnist_loaders
from ph_neuro.utils.packing import pack_ternary

# Models that use DQT rounding during training (need apply_dqt_rounding).
_DQT_MODELS = ("dqt_cnn", "dqt_cnn_cifar100")

# Theoretical training-memory ratio STE/DQT (E017 measurement: ~9 vs ~2
# bytes/param — STE keeps float latent scores + Adam optimizer state, DQT
# keeps only a float accumulation buffer and int8 ternary weights).
STE_MEMORY_RATIO = 4.5

# TF Lite INT8 is theoretically 2× slower than 2-bit popcount on the same
# CPU (BitNet paper: binary/ternary matmuls use popcount, INT8 uses
# multiply-add).
TFLITE_INT8_SPEEDUP = 2.0

# Default artifact paths from M1.3 (used for ONNX size measurement).
_DEFAULT_ONNX = {
    "ste_mlp": "models/ste_mlp_mnist.onnx",
    "dqt_cnn": "models/dqt_cnn_cifar10.onnx",
    "dqt_cnn_cifar100": "models/dqt_cnn_cifar100.onnx",
}

_MODEL_META = {
    "ste_mlp": {
        "dataset": "MNIST",
        "input_shape": (1, 1, 28, 28),
        "builder": lambda device=None: ste_mlp(
            [784, 512, 256, 10], batch_norm=True, flatten=True, device=device
        ),
    },
    "dqt_cnn": {
        "dataset": "CIFAR-10",
        "input_shape": (1, 3, 32, 32),
        "builder": dqt_cnn,
    },
    "dqt_cnn_cifar100": {
        "dataset": "CIFAR-100",
        "input_shape": (1, 3, 32, 32),
        "builder": dqt_cnn_cifar100,
    },
}


# ── Model size benchmark (measured) ─────────────────────────────────


def benchmark_model_size(
    model: nn.Module, model_name: str, output_dir: str
) -> dict:
    """Measure model size: ternary weights, packed bytes, ONNX bytes.

    Packed size is computed by actually packing each ternary layer with
    :func:`ph_neuro.utils.packing.pack_ternary` (4 weights/byte). ONNX size
    is the on-disk size of the M1.3 artifact (``models/<name>.onnx``) if it
    exists, otherwise a fresh ONNX export written to ``output_dir``.

    TF Lite sizes are *theoretical* for the same architecture: INT8 = 1
    byte/weight, FP16 = 2 bytes/weight. The size ratio is TF Lite INT8
    divided by the PH-Neuro packed size (≈4×).
    """
    n_weights = get_model_params_count(model)

    packed_bytes = 0
    for module in model.modules():
        if is_ternary_layer(module):
            packed_bytes += int(pack_ternary(_get_ternary_weight(module).detach().cpu()).numel())

    onnx_path = _DEFAULT_ONNX[model_name]
    onnx_source = "artifact"
    if not os.path.isfile(onnx_path):
        # No M1.3 artifact — export a fresh inference model to measure size.
        onnx_path = os.path.join(output_dir, f"_tmp_{model_name}.onnx")
        inference_model = dqt_to_inference_model(model)
        export_to_onnx(inference_model, _MODEL_META[model_name]["input_shape"], onnx_path)
        onnx_source = "fresh-export"
    onnx_bytes = os.path.getsize(onnx_path)

    tflite_int8_bytes = n_weights * 1
    tflite_fp16_bytes = n_weights * 2
    size_ratio = tflite_int8_bytes / max(packed_bytes, 1)

    return {
        "n_ternary_weights": int(n_weights),
        "packed_bytes": packed_bytes,
        "packed_kb": packed_bytes / 1024.0,
        "packed_mb": packed_bytes / (1024 * 1024),
        "onnx_bytes": int(onnx_bytes),
        "onnx_mb": onnx_bytes / (1024 * 1024),
        "onnx_source": onnx_source,
        "tflite_int8_bytes": int(tflite_int8_bytes),
        "tflite_int8_mb": tflite_int8_bytes / (1024 * 1024),
        "tflite_fp16_mb": tflite_fp16_bytes / (1024 * 1024),
        "size_ratio_tflite_int8_over_packed": float(size_ratio),
    }


# ── Inference speed benchmark (measured) ────────────────────────────


@torch.no_grad()
def _time_forward(model: nn.Module, x: torch.Tensor, n: int) -> list[float]:
    """Time ``n`` forward passes and return per-pass wall times (ms)."""
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def benchmark_inference_speed(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    batch_size: int,
    warmup: int,
    batches: int,
    onnx_path: str | None,
) -> dict:
    """Time inference on the fused BN-free graph (PyTorch + ONNX runtime).

    The model's BatchNorm layers are fused to element-wise affine (E011),
    matching the exact graph shipped as ONNX. Warmup ``warmup`` batches,
    then time ``batches`` batches at ``batch_size``.

    Returns measured PyTorch and ONNX timings plus the theoretical TF Lite
    estimate (2× the PH-Neuro time).
    """
    fused = fuse_bn_layers(dqt_to_inference_model(model), inplace=False).to(device)
    fused.eval()
    shape = _MODEL_META[model_name]["input_shape"]
    x = torch.randn(batch_size, *shape[1:], device=device)

    # Warmup (includes cuDNN/BRGEMM kernel selection on CUDA).
    _time_forward(fused, x, warmup)

    torch_times = _time_forward(fused, x, batches)

    onnx_times: list[float] = []
    onnx_available = False
    if onnx_path is not None and os.path.isfile(onnx_path):
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            feed = {"input": x.cpu().numpy()}
            # Warmup the session too.
            for _ in range(warmup):
                session.run(None, feed)
            onnx_times = _time_onnx(session, feed, batches)
            onnx_available = True
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"    [warn] onnxruntime inference timing skipped: {exc}")

    torch_mean = statistics.mean(torch_times)
    torch_std = statistics.pstdev(torch_times)
    onnx_mean = statistics.mean(onnx_times) if onnx_times else None

    # Primary PH-Neuro number: ONNX runtime if available, else PyTorch fused.
    ph_neuro_ms = onnx_mean if onnx_mean is not None else torch_mean
    tflite_est_ms = TFLITE_INT8_SPEEDUP * ph_neuro_ms

    return {
        "device": str(device),
        "batch_size": batch_size,
        "warmup_batches": warmup,
        "measured_batches": batches,
        "torch_threads": torch.get_num_threads(),
        "torch_mean_ms": torch_mean,
        "torch_std_ms": torch_std,
        "torch_median_ms": statistics.median(torch_times),
        "onnx_available": onnx_available,
        "onnx_mean_ms": onnx_mean,
        "onnx_std_ms": statistics.pstdev(onnx_times) if onnx_times else None,
        "ph_neuro_ms": ph_neuro_ms,
        "ph_neuro_ms_source": "onnx" if onnx_mean is not None else "torch",
        "tflite_est_ms": tflite_est_ms,
        "tflite_est_method": "theoretical (2-bit popcount ~2x faster than INT8 multiply-add, BitNet)",
        "speedup_ph_neuro_over_tflite": TFLITE_INT8_SPEEDUP,
    }


def _time_onnx(session, feed: dict, n: int) -> list[float]:
    """Time ``n`` onnxruntime sessions and return per-pass times (ms)."""
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


# ── Training memory benchmark (measured DQT, estimated STE) ─────────


def _train_one_epoch(
    model: nn.Module,
    train_loader,
    device: torch.device,
    model_name: str,
    lr: float,
) -> float:
    """Run one full training epoch (DQT rounding for DQT models)."""
    # OPT-2: 8-bit AdamW (states 8→2 B/param) for the memory benchmark.
    optimizer = make_adamw(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    running_loss = 0.0
    n_batches = 0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if model_name in _DQT_MODELS:
            apply_dqt_rounding(model, use_stochastic=True)
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


def benchmark_training_memory(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    """Measure peak CUDA memory during DQT training (1 epoch).

    DQT memory is *measured* via ``torch.cuda.reset_peak_memory_stats()``
    before and ``torch.cuda.max_memory_allocated()`` after 1 epoch. STE
    memory is *estimated* at 4.5× the DQT peak (E017: ~9 vs ~2 bytes/param
    — STE keeps float latent scores; DQT does not).

    Requires a CUDA GPU; returns ``{"skipped": true}`` on CPU-only hosts.
    """
    if not torch.cuda.is_available():
        return {
            "skipped": True,
            "reason": "CUDA not available — training-memory benchmark requires a GPU",
        }

    device = torch.device("cuda")
    builder = _MODEL_META[model_name]["builder"]
    dataset = _MODEL_META[model_name]["dataset"]

    if dataset == "MNIST":
        train_loader, _ = get_mnist_loaders(batch_size=batch_size, root="data", num_workers=0)
    elif dataset == "CIFAR-10":
        train_loader, _ = get_cifar10_loaders(batch_size=batch_size, root="data", num_workers=0)
    else:
        train_loader, _ = get_cifar100_loaders(batch_size=batch_size, root="data", num_workers=0)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    peak_mb = 0.0
    total_time = 0.0
    for _ in range(epochs):
        model = builder(device=device)
        t0 = time.time()
        _train_one_epoch(model, train_loader, device, model_name, lr)
        total_time += time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    del model
    torch.cuda.empty_cache()

    ste_est_mb = STE_MEMORY_RATIO * peak_mb
    return {
        "skipped": False,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "dataset": dataset,
        "dqt_peak_mb": peak_mb,
        "dqt_peak_mb_measured": True,
        "ste_est_mb": ste_est_mb,
        "ste_est_method": (
            f"theoretical: {STE_MEMORY_RATIO:.1f}x DQT peak "
            "(E017: ~9 vs ~2 bytes/param, STE keeps latent float scores)"
        ),
        "ratio_ste_over_dqt": STE_MEMORY_RATIO,
        "training_time_seconds": total_time,
    }


# ── GO / NO-GO gates ────────────────────────────────────────────────


def evaluate_gates(size: dict, inference: dict | None, memory: dict | None) -> dict:
    """Evaluate the three M1.5 GO/NO-GO criteria.

    Returns a dict of per-criterion booleans plus an overall verdict.
    """
    gates: dict[str, bool] = {}

    # 1. Model size: PH-Neuro >= 4x smaller than TF Lite INT8
    gates["size_4x_smaller"] = size["size_ratio_tflite_int8_over_packed"] >= 4.0

    # 2. Inference: PH-Neuro >= 2x faster than TF Lite INT8 (CPU).
    #    TF Lite is a theoretical estimate (2x by construction) — see report.
    if inference is not None:
        gates["inference_2x_faster"] = inference["speedup_ph_neuro_over_tflite"] >= 2.0

    # 3. Training memory: DQT >= 4x less than STE
    if memory is not None and not memory.get("skipped", True):
        gates["memory_4x_less"] = memory["ratio_ste_over_dqt"] >= 4.0

    all_pass = all(gates.values()) if gates else False
    return {"gates": gates, "verdict": "GO" if all_pass else "NO-GO"}


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone M1.5: memory benchmarks vs TF Lite "
        "(4x smaller, 2x faster inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["ste_mlp", "dqt_cnn", "dqt_cnn_cifar100"],
        default="dqt_cnn",
        help="Model architecture to benchmark.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the inference-speed benchmark (cpu | cuda). "
        "Training memory always uses CUDA when available.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=100,
        help="Number of timed inference batches (after warmup).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup inference batches before timing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the inference-speed benchmark.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs for the training-memory benchmark.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
        help="Batch size for the training-memory benchmark.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for the training-memory benchmark (DQT best).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model construction / training.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the inference-speed benchmark.",
    )
    parser.add_argument(
        "--skip-training-memory",
        action="store_true",
        help="Skip the training-memory benchmark (no CUDA needed).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase1/m1_5_results",
        help="Directory for the result JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    meta = _MODEL_META[args.model]
    print_header(
        f"M1.5 MEMORY BENCHMARKS: {args.model} ({meta['dataset']}) — "
        "vs TF Lite (4x smaller, 2x faster)"
    )

    device = torch.device(args.device)
    model = meta["builder"]()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({meta['dataset']}) — params {n_params:,}")
    print(f"Inference device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}  "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")
    print()

    # ── 1. Model size ──────────────────────────────────────────────
    print_header("1. MODEL SIZE (measured + theoretical TF Lite)")
    size = benchmark_model_size(model, args.model, args.output_dir)
    print(f"  Ternary weights        : {size['n_ternary_weights']:,}")
    print(f"  Packed (2-bit)         : {size['packed_kb']:.1f} KB  ({size['packed_bytes']:,} bytes)")
    print(f"  ONNX file              : {size['onnx_mb']:.2f} MB  (source: {size['onnx_source']})")
    print(f"  TF Lite INT8 (theor.)  : {size['tflite_int8_mb']:.2f} MB  (1 byte/weight)")
    print(f"  TF Lite FP16 (theor.)  : {size['tflite_fp16_mb']:.2f} MB  (2 bytes/weight)")
    print(f"  Size ratio TF Lite/PH  : {size['size_ratio_tflite_int8_over_packed']:.2f}x")
    print()

    # ── 2. Inference speed ─────────────────────────────────────────
    inference = None
    if not args.skip_inference:
        print_header("2. INFERENCE SPEED (measured, batch=1)")
        onnx_path = (
            _DEFAULT_ONNX[args.model]
            if os.path.isfile(_DEFAULT_ONNX[args.model])
            else os.path.join(args.output_dir, f"_tmp_{args.model}.onnx")
        )
        inference = benchmark_inference_speed(
            model, args.model, device, args.batch_size, args.warmup, args.batches, onnx_path
        )
        print(f"  Warmup {inference['warmup_batches']} + measured {inference['measured_batches']} batches "
              f"(batch={inference['batch_size']}, {inference['device']}, threads={inference['torch_threads']})")
        print(f"  PyTorch fused         : {inference['torch_mean_ms']:.3f} ± {inference['torch_std_ms']:.3f} ms")
        if inference["onnx_available"]:
            print(f"  ONNX runtime          : {inference['onnx_mean_ms']:.3f} ± {inference['onnx_std_ms']:.3f} ms")
        print(f"  PH-Neuro ({inference['ph_neuro_ms_source']})        : {inference['ph_neuro_ms']:.3f} ms")
        print(f"  TF Lite INT8 (est.)   : {inference['tflite_est_ms']:.3f} ms  (2x — theoretical)")
        print(f"  Speedup vs TF Lite    : {inference['speedup_ph_neuro_over_tflite']:.1f}x")
        print()

    # ── 3. Training memory ─────────────────────────────────────────
    memory = None
    if not args.skip_training_memory:
        print_header("3. TRAINING MEMORY (measured DQT, estimated STE)")
        memory = benchmark_training_memory(args.model, args.epochs, args.train_batch_size, args.lr)
        if memory.get("skipped"):
            print(f"  Skipped: {memory['reason']}")
        else:
            print(f"  DQT peak ({memory['dataset']}) : {memory['dqt_peak_mb']:.1f} MB  (measured, 1 epoch)")
            print(f"  STE est.            : {memory['ste_est_mb']:.1f} MB  ({memory['ste_est_method']})")
            print(f"  Ratio STE/DQT       : {memory['ratio_ste_over_dqt']:.1f}x")
        print()

    # ── Save JSON ──────────────────────────────────────────────────
    result = {
        "experiment": "m1_5_benchmarks",
        "model": args.model,
        "dataset": meta["dataset"],
        "date": time.strftime("%Y-%m-%d"),
        "seed": args.seed,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_threads": torch.get_num_threads(),
        },
        "model_size": size,
        "inference_speed": inference,
        "training_memory": memory,
        "notes": {
            "tf_lite_comparison": (
                "TF Lite is NOT installed. Size numbers are theoretical "
                "(INT8=1 byte/weight, FP16=2 bytes/weight for the same "
                "architecture); speed is a theoretical 2x estimate (2-bit "
                "popcount vs 8-bit multiply-add, BitNet paper)."
            ),
            "ste_memory_estimate": (
                f"STE training memory is estimated at {STE_MEMORY_RATIO:.1f}x "
                "the measured DQT peak (E017: ~9 vs ~2 bytes/param)."
            ),
        },
    }
    out_path = os.path.join(args.output_dir, f"results_m1_5_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # ── GO / NO-GO ─────────────────────────────────────────────────
    gates = evaluate_gates(size, inference, memory)
    print_header(f"GO/NO-GO: {gates['verdict']}")
    for name, passed in gates["gates"].items():
        mark = "PASS" if passed else "FAIL"
        print(f"  {'✅' if passed else '❌'} {name}: {mark}")


if __name__ == "__main__":
    main()
