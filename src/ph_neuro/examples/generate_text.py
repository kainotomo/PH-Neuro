#!/usr/bin/env python3
"""Milestone M2.4 — On-device inference demo: DQT Transformer text generation.

Autoregressive text generation with a trained DQT Transformer, running on
CPU only (smartphone simulation — no GPU, no DQT custom autograd
functions). The trained DQT model is first converted to the standard-layer
inference model (:func:`ph_neuro.models.export_transformer.dqt_transformer_to_inference_model`),
which bakes the int8 ternary weights + the DQT ``1/sqrt(in_features)``
output scaling into plain ``nn.Linear`` weights, so generation runs with
``torch.no_grad()`` on a graph identical to the one exported to ONNX.

Generation is standard GPT-style autoregressive sampling:
    forward pass → logits → temperature scaling → top-k filtering →
    softmax → sample next token → append → repeat.

The context is a FIXED length (``--ctx-len``, default = model ``max_seq_len``):
the growing prompt+generated token sequence is right-padded to ``ctx_len``
each step and the logits are read at the last real token's position. Right-
padding is safe because the standard causal mask means real tokens never
attend to pad tokens.

Two inference backends:
    - PyTorch (CPU, ``torch.no_grad()``) — default
    - ONNX Runtime (``--onnx PATH``) — the deployed-artifact path

``--compare`` runs BOTH backends and prints a speed comparison
(PyTorch CPU vs ONNX CPU).

Usage::

    # PyTorch CPU generation
    python -m ph_neuro.examples.generate_text \\
        --checkpoint m2_4_demo/checkpoints/seed42/best.pt \\
        --prompt "Once upon a time" \\
        --max-tokens 100 --temperature 0.8 --top-k 50

    # ONNX Runtime generation (requires the exported .onnx)
    python -m ph_neuro.examples.generate_text \\
        --checkpoint m2_4_demo/checkpoints/seed42/best.pt \\
        --onnx models/dqt_transformer_demo.onnx \\
        --prompt "Once upon a time" --max-tokens 100

    # Export ONNX + benchmark both backends
    python -m ph_neuro.examples.generate_text \\
        --checkpoint m2_4_demo/checkpoints/seed42/best.pt \\
        --export-onnx models/dqt_transformer_demo.onnx --compare

Output:
    - Generated text (stdout)
    - tokens/sec, total time, model size (ONNX / checkpoint / packed 2-bit)
    - optional JSON: ``{output_dir}/results_generate_{tag}.json``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.examples._utils import print_header
from ph_neuro.models.dqt_transformer import dqt_gpt2
from ph_neuro.models.export_transformer import (
    count_ternary_weights_inference,
    dqt_transformer_to_inference_model,
    export_transformer_to_onnx,
    load_dqt_transformer_checkpoint,
)
from ph_neuro.training.tinystories import make_gpt2_tokenizer

# GPT-2 BPE special tokens.
EOT_ID = 50256  # <|endoftext|>
PAD_ID = 0      # right-padding id (real tokens never attend to pads)

MB = 1024 * 1024


# ── Model construction ──────────────────────────────────────────────


def build_inference_model(
    checkpoint: str,
    ctx_len: int | None = None,
    device: str = "cpu",
) -> tuple[dict, torch.nn.Module, float, int]:
    """Load a DQT Transformer checkpoint and convert it for CPU inference.

    Args:
        checkpoint: Path to ``best.pt`` / ``ckpt_step*.pt`` or a bare
            state_dict.
        ctx_len: Fixed context length (None = model ``max_seq_len``).
        device: Device for the inference model (default ``"cpu"``).

    Returns:
        ``(config, inference_model, best_val_ppl, step)``.
    """
    config, state_dict, best_val_ppl, step = load_dqt_transformer_checkpoint(
        checkpoint
    )
    print(
        f"Checkpoint: {checkpoint}  (best val ppl {best_val_ppl:.2f} at "
        f"step {step})" if best_val_ppl == best_val_ppl else f"Checkpoint: {checkpoint}"
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
        dropout=0.0,
        device="cpu",
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected}")

    if ctx_len is None:
        ctx_len = config["max_seq_len"]
    ctx_len = min(int(ctx_len), int(config["max_seq_len"]))

    inference = dqt_transformer_to_inference_model(model, ctx_len=ctx_len)
    inference = inference.to(device)
    inference.eval()
    print(f"Inference model on {device} (ctx_len={ctx_len})")
    return config, inference, best_val_ppl, step


# ── Generation ──────────────────────────────────────────────────────


def _predict_row(
    backend, ids: torch.Tensor, last_pos: int, temperature: float
) -> torch.Tensor:
    """Run one forward pass and return the scaled logits of the last token.

    ``backend`` is a PyTorch module (returns logits tensor) or an
    onnxruntime session (returns numpy). Only the single row of logits for
    the last real token is materialized into torch — for ONNX this avoids
    converting the full ``(1, ctx, vocab)`` output from numpy on every
    step (a ~50 MB copy), keeping the benchmark focused on the actual
    forward-pass cost.

    Args:
        backend: PyTorch module or onnxruntime session.
        ids: Token ids, shape ``(1, ctx_len)``, int64.
        last_pos: Index of the last real token in ``ids``.
        temperature: Softmax temperature (``>0``; 1.0 = no scaling).

    Returns:
        Scaled logits for the next-token prediction, shape ``(vocab,)``.
    """
    if isinstance(backend, torch.nn.Module):
        logits = backend(ids)[0, last_pos, :]
    else:
        out = backend.run(None, {"input": ids.numpy()})[0]
        logits = torch.from_numpy(np.asarray(out[0, last_pos, :]))
    return logits / max(temperature, 1e-6)


@torch.no_grad()
def generate(
    backend,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    ctx_len: int,
    seed: int = 42,
    eot: int = EOT_ID,
    pad_id: int = PAD_ID,
) -> dict:
    """Autoregressively generate text with temperature + top-k sampling.

    Args:
        backend: PyTorch inference model or onnxruntime session.
        tokenizer: GPT-2 tiktoken encoding.
        prompt: Prompt string.
        max_tokens: Max tokens to generate.
        temperature: Softmax temperature (``>0``; 1.0 = no scaling).
        top_k: Keep only the top-k logits before softmax (0 = off).
        ctx_len: Fixed context length (right-padding).
        seed: Sampling seed.
        eot: End-of-text token id to stop on.
        pad_id: Padding token id.

    Returns:
        Dict: ``text``, ``prompt``, ``prompt_tokens``, ``n_generated``,
        ``stop_reason``, ``elapsed_s``, ``tokens_per_sec``.
    """
    torch.manual_seed(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    prompt_ids = tokenizer.encode(prompt)
    seq: list[int] = list(prompt_ids)

    # Warmup pass (exclude from timing) — ensures fair PyTorch vs ONNX
    # comparison (the first call pays one-time init/thread-pool cost).
    warm = torch.tensor(
        [seq[-ctx_len:] + [pad_id] * (ctx_len - min(len(seq), ctx_len))],
        dtype=torch.long,
    )
    _predict_row(backend, warm, min(len(seq), ctx_len) - 1, max(temperature, 1e-6))

    start = time.time()
    n_generated = 0
    stop_reason = "max_tokens"
    for _ in range(max_tokens):
        window = seq[-ctx_len:]
        pad = ctx_len - len(window)
        ids = torch.tensor([window + [pad_id] * pad], dtype=torch.long)
        last_pos = len(window) - 1

        logits = _predict_row(backend, ids, last_pos, temperature)

        if top_k and top_k > 0:
            k = min(int(top_k), logits.numel())
            topk_vals, _ = logits.topk(k)
            logits = logits.masked_fill(logits < topk_vals[-1], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1, generator=rng).item())
        seq.append(next_id)
        n_generated += 1
        if next_id == eot:
            stop_reason = "eot"
            break

    elapsed = time.time() - start
    text = tokenizer.decode(seq)
    return {
        "text": text,
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "n_generated": n_generated,
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "tokens_per_sec": n_generated / elapsed if elapsed > 0 else 0.0,
    }


# ── Sizes / summary ─────────────────────────────────────────────────


def packed_size_bytes(model: torch.nn.Module) -> int:
    """2-bit packed size of the ternary weights (4 weights per byte)."""
    n = count_ternary_weights_inference(model)
    return (n + 3) // 4


def model_state_mb(model: torch.nn.Module) -> float:
    """In-memory FP32 state size of the inference model (MB)."""
    total = 0
    for tensor in model.state_dict().values():
        total += tensor.numel() * tensor.element_size()
    return total / MB


def _print_generation(result: dict) -> None:
    print_header("GENERATED TEXT")
    print(result["text"])
    print()
    print(
        f"  {result['n_generated']} tokens generated in {result['elapsed_s']:.2f}s "
        f"→ {result['tokens_per_sec']:.1f} tokens/sec  "
        f"(stop: {result['stop_reason']})"
    )


def _print_sizes(config, inference, checkpoint_path, onnx_path) -> dict:
    n_ternary = count_ternary_weights_inference(inference)
    packed = packed_size_bytes(inference)
    sizes = {
        "n_ternary_weights": n_ternary,
        "n_float_embedding": config["vocab_size"] * config["d_model"],
        "packed_bytes": packed,
        "packed_mb": packed / MB,
        "state_mb": model_state_mb(inference),
        "checkpoint_mb": os.path.getsize(checkpoint_path) / MB
        if os.path.exists(checkpoint_path)
        else None,
        "onnx_mb": os.path.getsize(onnx_path) / MB
        if onnx_path and os.path.exists(onnx_path)
        else None,
    }
    print_header("MODEL SIZE")
    print(f"  Ternary weights : {n_ternary:,}")
    print(f"  Packed (2-bit)  : {packed / 1024:,.1f} KB  ({sizes['packed_mb']:.3f} MB)")
    print(f"  Inference FP32  : {sizes['state_mb']:.2f} MB (in-memory state)")
    if sizes["checkpoint_mb"] is not None:
        print(f"  Checkpoint      : {sizes['checkpoint_mb']:.2f} MB")
    if sizes["onnx_mb"] is not None:
        print(f"  ONNX file       : {sizes['onnx_mb']:.2f} MB")
    return sizes


def _save_json(output_dir: str, tag: str, payload: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"results_generate_{tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ── CLI ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="M2.4 — DQT Transformer text generation (CPU inference demo).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a DQT transformer checkpoint (best.pt or ckpt_step*.pt).")
    parser.add_argument("--prompt", default="Once upon a time",
                        help="Prompt text.")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k filtering (0 = off).")
    parser.add_argument("--ctx-len", type=int, default=None,
                        help="Fixed context length (default: model max_seq_len).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampling seed.")
    parser.add_argument("--onnx", default=None,
                        help="Path to an exported .onnx — use ONNX Runtime "
                             "instead of PyTorch for inference.")
    parser.add_argument("--export-onnx", default=None,
                        help="Export the checkpoint to this .onnx path first, "
                             "then use it for inference.")
    parser.add_argument("--compare", action="store_true",
                        help="Run both PyTorch CPU and ONNX CPU and print a "
                             "speed comparison.")
    parser.add_argument("--device", default="cpu",
                        help="Device for the PyTorch path (cpu = smartphone sim).")
    parser.add_argument("--output-dir", default=None,
                        help="Write a results JSON to this directory.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (``None`` = ``sys.argv``)."""
    return build_parser().parse_args(argv)


def main() -> None:
    args = parse_args()

    config, inference, best_val_ppl, step = build_inference_model(
        args.checkpoint, ctx_len=args.ctx_len, device=args.device
    )
    tokenizer = make_gpt2_tokenizer()

    # Optional: export ONNX from this checkpoint.
    onnx_path = args.onnx
    if args.export_onnx:
        print()
        print_header("ONNX EXPORT")
        summary = export_transformer_to_onnx(
            inference.to("cpu"), args.export_onnx, verify=True
        )
        print(
            f"  Exported: {args.export_onnx} "
            f"({summary['onnx_size_mb']:.2f} MB, verified={summary['verified']}, "
            f"max|Δ|={summary['max_abs_diff']:.2e})"
        )
        onnx_path = args.export_onnx

    sizes = _print_sizes(config, inference, args.checkpoint, onnx_path)

    results: dict = {
        "experiment": "m2_4_generate_text",
        "checkpoint": args.checkpoint,
        "config": config,
        "best_val_ppl": best_val_ppl,
        "step": step,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "ctx_len": inference.ctx_len,
        "seed": args.seed,
        "sizes": sizes,
        "runs": {},
    }

    # ── PyTorch CPU run ────────────────────────────────────────────
    print()
    print_header(f"GENERATION — PyTorch CPU ({args.device})")
    res_torch = generate(
        inference,
        tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        inference.ctx_len,
        seed=args.seed,
    )
    _print_generation(res_torch)
    results["runs"]["pytorch_cpu"] = res_torch

    # ── ONNX CPU run ───────────────────────────────────────────────
    if onnx_path and os.path.exists(onnx_path):
        import onnxruntime as ort

        print()
        print_header("GENERATION — ONNX CPU")
        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        res_onnx = generate(
            session,
            tokenizer,
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_k,
            inference.ctx_len,
            seed=args.seed,
        )
        _print_generation(res_onnx)
        results["runs"]["onnx_cpu"] = res_onnx

        if args.compare:
            print()
            print_header("SPEED COMPARISON — PyTorch CPU vs ONNX CPU")
            t = res_torch["tokens_per_sec"]
            o = res_onnx["tokens_per_sec"]
            print(f"  PyTorch CPU : {t:8.1f} tokens/sec  ({res_torch['elapsed_s']:.2f}s)")
            print(f"  ONNX CPU    : {o:8.1f} tokens/sec  ({res_onnx['elapsed_s']:.2f}s)")
            if o > 0:
                print(f"  Ratio       : ONNX is {t / o:.2f}× the PyTorch rate "
                      f"({'faster' if o > t else 'slower'})")
    else:
        print()
        print("  (no ONNX model given — skipping ONNX run; pass --onnx PATH "
              "or --export-onnx PATH)")

    if args.output_dir:
        path = _save_json(args.output_dir, f"ctx{inference.ctx_len}_t{args.temperature}", results)
        print()
        print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
