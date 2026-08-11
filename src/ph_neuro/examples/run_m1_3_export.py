#!/usr/bin/env python3
"""Milestone M1.3 — Model export ONNX/C (<100MB, Raspberry Pi).

Exports trained DQT models to ONNX for edge deployment on CPU (Raspberry
Pi). DQT layers use custom autograd Functions that ``torch.onnx.export``
cannot trace, so :func:`ph_neuro.models.export.dqt_to_inference_model`
first rebuilds the graph with standard ``nn.Conv2d`` / ``nn.Linear``
layers holding the frozen int8 ternary weights; BatchNorm is fused to
element-wise affine before export.

Usage:
    # Export a trained checkpoint (recommended):
    python -m ph_neuro.examples.run_m1_3_export \\
        --model dqt_cnn --checkpoint models/dqt_cnn_cifar10.pt \\
        --output models/dqt_cnn_cifar10.onnx --packed --verify

    # Quick demo — no checkpoint (trains a small model briefly):
    python -m ph_neuro.examples.run_m1_3_export \\
        --model dqt_cnn --output models/dqt_cnn_cifar10.onnx --verify

    # ste_mlp (MNIST) demo — 5 epochs, tiny model:
    python -m ph_neuro.examples.run_m1_3_export \\
        --model ste_mlp --output models/ste_mlp_mnist.onnx --packed --verify

Output:
    - ``.onnx`` file (dynamic batch, opset 17)
    - optional ``.ternary`` companion file with 2-bit packed ternary weights
    - console summary: model size, parameter count, accuracy comparison
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header
from ph_neuro.utils.optimizers import make_adamw
from ph_neuro.models.dqt_models import dqt_cnn, dqt_cnn_cifar100
from ph_neuro.models.dqt_transformer import dqt_gpt2
from ph_neuro.models.export import (
    dqt_to_inference_model,
    export_model_to_onnx,
)
from ph_neuro.models.export_transformer import (
    count_ternary_weights_inference,
    dqt_transformer_to_inference_model,
    export_transformer_packed_ternary,
    export_transformer_to_onnx,
    load_dqt_transformer_checkpoint,
)
from ph_neuro.models.fuse_bn import fuse_bn_layers
from ph_neuro.models.ste_models import ste_mlp
from ph_neuro.training.data import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_mnist_loaders,
)

# Models that use DQT rounding during training (need apply_dqt_rounding).
_DQT_MODELS = ("dqt_cnn", "dqt_cnn_cifar100")


# ── Model / data selection ──────────────────────────────────────────


def _build_model(model_name: str) -> tuple[nn.Module, tuple[int, ...], str, dict]:
    """Build the requested model + input shape + dataset + demo defaults."""
    if model_name == "dqt_cnn":
        return (
            dqt_cnn(),
            (1, 3, 32, 32),
            "CIFAR-10",
            {"epochs": 2, "limit": 5000},
        )
    if model_name == "dqt_cnn_cifar100":
        return (
            dqt_cnn_cifar100(),
            (1, 3, 32, 32),
            "CIFAR-100",
            {"epochs": 1, "limit": 5000},
        )
    if model_name == "ste_mlp":
        return (
            ste_mlp([784, 512, 256, 10], batch_norm=True, flatten=True),
            (1, 1, 28, 28),
            "MNIST",
            {"epochs": 5, "limit": None},
        )
    raise ValueError(f"Unknown model: {model_name}")


def _default_output(model_name: str) -> str:
    names = {
        "dqt_cnn": "models/dqt_cnn_cifar10.onnx",
        "dqt_cnn_cifar100": "models/dqt_cnn_cifar100.onnx",
        "ste_mlp": "models/ste_mlp_mnist.onnx",
    }
    return names[model_name]


# ── Quick demo training (no checkpoint) ─────────────────────────────


def _bounded_batches(loader: DataLoader, limit: int | None):
    """Yield at most ``limit`` batches from a loader."""
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            return
        yield batch


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute test accuracy."""
    model.eval()
    correct = total = 0
    for x, y in _bounded_batches(loader, 100):
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def _get_loaders(model_name: str) -> tuple[DataLoader, DataLoader]:
    """Get train/test loaders for the demo dataset."""
    if model_name == "dqt_cnn":
        return get_cifar10_loaders(batch_size=128, root="data", num_workers=0)
    if model_name == "dqt_cnn_cifar100":
        return get_cifar100_loaders(batch_size=128, root="data", num_workers=0)
    return get_mnist_loaders(batch_size=128, root="data", num_workers=0)


def _train_quick(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    epochs: int,
    limit: int | None,
    device: torch.device,
    lr: float = 0.01,
) -> float:
    """Train a model briefly for the demo path (no checkpoint provided)."""
    if model_name in _DQT_MODELS:
        from ph_neuro.examples.run_m1_2_dqt_cifar100 import apply_dqt_rounding
    else:
        apply_dqt_rounding = None

    train_loader, test_loader = _get_loaders(model_name)
    # OPT-2: 8-bit AdamW (states 8→2 B/param) for the brief demo retrain.
    optimizer = make_adamw(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    final_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in _bounded_batches(train_loader, limit):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            if apply_dqt_rounding is not None:
                apply_dqt_rounding(model, use_stochastic=True)
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        final_acc = _evaluate(model, test_loader, device)
        print(f"    epoch {epoch}/{epochs} — loss {total_loss / max(n_batches, 1):.4f} — test acc {final_acc:.2%}")

    return final_acc


# ── Accuracy comparison (torch vs onnx) ─────────────────────────────


@torch.no_grad()
def _accuracy_comparison(
    model: nn.Module,
    onnx_path: str,
    model_name: str,
    device: torch.device,
    n_samples: int = 200,
) -> tuple[float, float]:
    """Compare PyTorch (fused) vs ONNX accuracy on a small test subset.

    Returns ``(torch_accuracy, onnx_accuracy)`` — they should match within
    floating-point noise since both run the same fused graph.
    """
    import onnxruntime as ort

    _train_loader, test_loader = _get_loaders(model_name)

    # Full float32 (no TF32) so the PyTorch reference matches onnxruntime's
    # CPU float32 output — otherwise TF32's reduced precision can shift
    # borderline logits and make torch/onnx accuracy look different.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    fused = fuse_bn_layers(dqt_to_inference_model(model), inplace=False).to(device)
    fused.eval()
    session = ort.InferenceSession(onnx_path)

    torch_correct = onnx_correct = total = 0
    for x, y in test_loader:
        x_t = x.to(device)
        pred_t = fused(x_t).argmax(dim=1).cpu().numpy()
        pred_o = session.run(None, {"input": x_t.cpu().numpy()})[0].argmax(axis=1)
        y_np = y.numpy()
        torch_correct += int((pred_t == y_np).sum())
        onnx_correct += int((pred_o == y_np).sum())
        total += y.size(0)
        if total >= n_samples:
            break

    return torch_correct / max(total, 1), onnx_correct / max(total, 1)


# ── Summary printer ─────────────────────────────────────────────────


def _print_summary(summary: dict) -> None:
    print_header("M1.3 EXPORT SUMMARY")
    print(f"  ONNX file         : {summary['onnx_path']}")
    print(f"  ONNX size         : {summary['onnx_size_mb']:.3f} MB  (<100 MB gate: PASS)" if summary["onnx_size_mb"] < 100 else "  ONNX size         : FAIL")
    print(f"  Ternary weights   : {summary['n_ternary_weights']:,}")
    print(f"  Packed (2-bit)    : {summary['packed_bytes'] / 1024:.1f} KB")
    if summary.get("packed_path"):
        print(f"  Packed file       : {summary['packed_path']} ({summary.get('packed_size_mb', 0) * 1024:.2f} KB)")
    if summary.get("verified") is not None:
        print(f"  ONNX verified     : {'YES' if summary['verified'] else 'NO'}  (max|Δ| = {summary.get('max_abs_diff', float('nan')):.2e})")
    if summary.get("torch_accuracy") is not None:
        print(
            f"  Accuracy (torch)  : {summary['torch_accuracy']:.2%}"
            f"   Accuracy (onnx)  : {summary['onnx_accuracy']:.2%}"
        )


# ── Transformer export (M2.4 demo) ────────────────────────────────


def _main_transformer(args) -> None:
    """Export a DQT Transformer (dqt_gpt2) to ONNX + packed ternary.

    A transformer cannot be quick-trained, so ``--checkpoint`` is
    required. The checkpoint is rebuilt from its stored (or inferred)
    config, converted to the standard-layer inference model
    (:func:`dqt_transformer_to_inference_model`), exported to ONNX with a
    fixed ``ctx_len`` context (dynamic batch) and verified with
    onnxruntime.
    """
    if not args.checkpoint:
        print(
            "ERROR: --checkpoint is required for --model dqt_gpt2 "
            "(a transformer cannot be quick-trained in this runner)."
        )
        raise SystemExit(2)

    config, state_dict, best_val_ppl, step = load_dqt_transformer_checkpoint(
        args.checkpoint
    )
    print(
        f"Loaded checkpoint: {args.checkpoint}  (best val ppl "
        f"{best_val_ppl:.2f} at step {step})"
    )
    print(
        f"Config: d_model={config['d_model']} n_layers={config['n_layers']} "
        f"n_heads={config['n_heads']} d_ff={config['d_ff']} "
        f"vocab={config['vocab_size']} max_seq_len={config['max_seq_len']}"
    )

    model = dqt_gpt2(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        device="cpu",
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected}")
    model.eval()

    ctx_len = min(args.ctx_len or config["max_seq_len"], config["max_seq_len"])
    inference = dqt_transformer_to_inference_model(model, ctx_len=ctx_len)

    output = args.output or "models/dqt_transformer.onnx"
    summary = export_transformer_to_onnx(
        inference, output, opset_version=args.opset, verify=args.verify
    )

    packed_path = None
    if args.packed:
        packed_path = (
            output[:-5] + ".ternary" if output.endswith(".onnx") else output + ".ternary"
        )
        export_transformer_packed_ternary(model, packed_path)

    n_ternary = count_ternary_weights_inference(inference)
    print_header("M1.3 EXPORT — DQT Transformer (M2.4 demo)")
    print(f"  model           : dqt_gpt2 (d={config['d_model']} L={config['n_layers']}"
          f" H={config['n_heads']} ff={config['d_ff']})")
    print(f"  input shape     : (batch, {ctx_len}) int64 tokens")
    print(f"  ONNX file       : {output}")
    print(f"  ONNX size       : {summary['onnx_size_mb']:.2f} MB")
    print(f"  Ternary weights : {n_ternary:,}")
    print(f"  Packed (2-bit)  : {summary['packed_bytes'] / 1024:,.1f} KB")
    if packed_path:
        print(
            f"  Packed file     : {packed_path} "
            f"({os.path.getsize(packed_path) / (1024 * 1024):.2f} MB)"
        )
    if summary.get("verified") is not None:
        print(
            f"  ONNX verified   : {'YES' if summary['verified'] else 'NO'}  "
            f"(max|Δ| = {summary.get('max_abs_diff', float('nan')):.2e})"
        )
    ok = (not args.verify) or summary.get("verified", False)
    print_header("GO/NO-GO: GO" if ok else "GO/NO-GO: NO-GO")
    print(
        "  Note: the float32 .onnx is the standard M1.3 artifact; the 2-bit "
        "packed size (4x smaller) is the deployable on-device metric for M2.4."
    )


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained DQT model to ONNX (M1.3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["dqt_cnn", "dqt_cnn_cifar100", "ste_mlp", "dqt_gpt2"],
        default="dqt_cnn",
        help="Model architecture to export.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained state_dict (.pt/.pth). If omitted, a quick "
        "demo model is trained from scratch.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .onnx path (default: models/<model>.onnx).",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Also write a 2-bit packed .ternary companion file.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the ONNX output against PyTorch with onnxruntime.",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Demo training epochs when no checkpoint is given.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Cap demo training to N batches per epoch (faster demo).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the PyTorch reference (export itself is CPU).",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=None,
        help="Fixed context length for transformer export (default: model max_seq_len).",
    )
    args = parser.parse_args()

    if args.model == "dqt_gpt2":
        _main_transformer(args)
        return

    device = torch.device(args.device)
    torch.manual_seed(42)

    model, input_shape, dataset_name, demo_defaults = _build_model(args.model)
    model = model.to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=str(device))
        # Accept either wrapping convention: the classic ``state_dict`` key or
        # the M2.x ``model_state_dict`` key used by best.pt checkpoints.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"  [warn] unexpected keys in checkpoint: {unexpected}")
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print_header("QUICK DEMO TRAINING (no checkpoint)")
        print(f"Training {args.model} on {dataset_name} ...")
        epochs = args.train_epochs if args.train_epochs is not None else demo_defaults["epochs"]
        limit = args.limit_train if args.limit_train is not None else demo_defaults["limit"]
        t0 = time.time()
        _train_quick(model, args.model, dataset_name, epochs, limit, device)
        print(f"  demo training done in {time.time() - t0:.1f}s")

    model.eval()

    output = args.output or _default_output(args.model)
    parent = os.path.dirname(os.path.abspath(output))
    if parent:
        os.makedirs(parent, exist_ok=True)

    packed_path = None
    if args.packed:
        packed_path = output[:-5] + ".ternary" if output.endswith(".onnx") else output + ".ternary"

    summary = export_model_to_onnx(
        dqt_model=model,
        input_shape=input_shape,
        output_path=output,
        opset_version=args.opset,
        verify=args.verify,
        packed_path=packed_path,
        device=args.device,
    )

    if args.verify:
        torch_acc, onnx_acc = _accuracy_comparison(
            model, output, args.model, device
        )
        summary["torch_accuracy"] = torch_acc
        summary["onnx_accuracy"] = onnx_acc

    print_header("M1.3 EXPORT")
    print(f"  model           : {args.model} ({dataset_name})")
    print(f"  input shape     : {input_shape}")
    _print_summary(summary)

    # GO/NO-GO gates
    ok_size = summary["onnx_size_mb"] < 100.0
    ok_verify = summary.get("verified", False) if args.verify else True
    verdict = "GO" if (ok_size and ok_verify) else "NO-GO"
    print_header(f"GO/NO-GO: {verdict}")
    if not ok_size:
        print("  ✗ ONNX size >= 100MB")
    if not ok_verify:
        print("  ✗ ONNX output differs from PyTorch")


if __name__ == "__main__":
    main()
