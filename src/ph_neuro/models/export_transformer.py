"""Transformer (GPT-2) inference conversion + ONNX export for DQT models.

Milestone M2.4 — on-device inference demo. The M1.3 vision pipeline
(:func:`dqt_to_inference_model`) rebuilds ``nn.Sequential`` models from
standard layers so ``torch.onnx.export`` can trace them. A DQT Transformer
is NOT an ``nn.Sequential`` (it has RMSNorm, RoPE, causal multi-head
attention, GELU feed-forward blocks), so it needs its own inference
reconstruction.

This module builds :class:`DQTTransformerInference` — the same forward
graph as :class:`ph_neuro.models.dqt_transformer.DQTTransformer` but with
standard, ONNX-traceable layers. The key trick, identical to M1.3, is that
each DQT ternary projection is replaced by a plain ``nn.Linear`` whose
weight is the frozen int8 ternary weight multiplied by the DQT output
scale (``1/sqrt(in_features)``)::

    (x @ W_ternary^T) * scale  ==  x @ (W_ternary * scale)^T

so the trained ternary weights + the BitNet-style output scaling that DQT
transformers require are baked into one standard float ``nn.Linear``.
RoPE tables are copied as buffers; RMSNorm stays a float element-wise
module. The result is a single self-contained CPU model with the exact
same forward semantics, ready for ``torch.onnx.export`` and for
autoregressive text generation on CPU (smartphone simulation).

All exported/exported-onnx artifacts:
    - ``.onnx`` — full float32 graph (weights are float copies of the
      ternary values, so the file is ~4x the packed size).
    - ``.ternary`` — 2-bit packed ternary weights (4x smaller than the
      ONNX float copy, 16x smaller than a raw FP32 checkpoint) — the
      deployable on-device artifact.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.models.dqt_transformer import DQTTransformer
from ph_neuro.models.export import _PACKED_MAGIC, _PACKED_VERSION, get_model_size_mb
from ph_neuro.utils.packing import pack_ternary

__all__ = [
    "RMSNormLayer",
    "TransformerInferenceAttention",
    "TransformerInferenceBlock",
    "DQTTransformerInference",
    "infer_transformer_config_from_state_dict",
    "load_dqt_transformer_checkpoint",
    "dqt_transformer_to_inference_model",
    "count_ternary_weights_inference",
    "export_transformer_to_onnx",
    "export_transformer_packed_ternary",
]


# ── Exportable float RMSNorm (matches TernaryDQTRMSNorm.forward) ────


class RMSNormLayer(nn.Module):
    """Float RMSNorm (x / sqrt(mean(x^2) + eps) * weight), ONNX-traceable."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE rotate-half (matches ph_neuro.layers.ste_dqt_transformer)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ── Inference attention ─────────────────────────────────────────────


class TransformerInferenceAttention(nn.Module):
    """Causal self-attention with standard layers + RoPE (fixed ctx_len).

    Mirrors :class:`ph_neuro.layers.ste_dqt_transformer.TernaryDQTMultiheadAttention`
    with the DQT projections replaced by ``nn.Linear`` whose weights have
    the ternary weights AND the DQT ``1/sqrt(in_features)`` output scale
    baked in. Sequence length is FIXED at ``ctx_len`` so the RoPE slice and
    the causal mask are static constants — this is what makes the ONNX
    graph fully static in the time axis (only the batch is dynamic).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ctx_len: int,
        theta_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ctx_len = ctx_len
        assert self.d_head % 2 == 0, "d_head must be even for RoPE"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_head)

        # RoPE tables (buffers) — sliced to ctx_len
        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head)
        )
        t = torch.arange(ctx_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (ctx_len, d_head)
        self.register_buffer("rope_cos", emb.cos())
        self.register_buffer("rope_sin", emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE (fixed ctx_len -> static slice)
        cos = self.rope_cos[:seq].unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:seq].unsqueeze(0).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        attn = q @ k.transpose(-2, -1) * self.scale  # (B, H, T, T)
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn = attn.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq, self.d_model)
        return self.o_proj(out)


# ── Inference FFN + block ───────────────────────────────────────────


class TransformerInferenceBlock(nn.Module):
    """Pre-norm block: attn(RMSNorm) + ffn(RMSNorm) with residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, ctx_len: int):
        super().__init__()
        self.attn_norm = RMSNormLayer(d_model)
        self.ffn_norm = RMSNormLayer(d_model)
        self.attention = TransformerInferenceAttention(d_model, n_heads, ctx_len)
        self.fc_in = nn.Linear(d_model, d_ff, bias=False)
        self.fc_out = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.fc_out(F.gelu(self.fc_in(self.ffn_norm(x))))
        return x


# ── Full inference transformer ──────────────────────────────────────


class DQTTransformerInference(nn.Module):
    """Inference-only GPT-2-style transformer (standard, ONNX-traceable).

    Same forward as :class:`DQTTransformer` but every DQT ternary
    projection is a plain ``nn.Linear`` with ``W = W_ternary * scale``
    baked in. Sequence length is fixed at ``ctx_len``; the model expects
    right-padded token id sequences of exactly ``ctx_len`` tokens and
    returns logits ``(batch, ctx_len, vocab_size)``. Right-padding is safe
    because the causal mask means real tokens never attend to pad tokens.

    Args:
        vocab_size: Tokenizer vocabulary size.
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of blocks.
        d_ff: Feed-forward hidden dimension.
        ctx_len: Fixed context length for export/generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        ctx_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ctx_len = ctx_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerInferenceBlock(d_model, n_heads, d_ff, ctx_len)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNormLayer(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Language-model forward.

        Args:
            tokens: Token ids, shape ``(batch, ctx_len)`` (int64).

        Returns:
            Logits, shape ``(batch, ctx_len, vocab_size)``.
        """
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ── Config inference ────────────────────────────────────────────────


def infer_transformer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer a DQT Transformer architecture config from its state_dict.

    Every dimension is recoverable from the saved tensors (the RoPE
    buffers encode ``max_seq_len`` and ``d_head``, from which ``n_heads``
    follows). Used as a fallback when a checkpoint does not store an
    explicit ``config``.

    Args:
        state_dict: A ``DQTTransformer.state_dict()``.

    Returns:
        Config dict with ``vocab_size``, ``d_model``, ``n_heads``,
        ``n_layers``, ``d_ff`` and ``max_seq_len``.
    """
    d_model = state_dict["token_embedding.weight"].shape[1]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    n_layers = 0
    while f"blocks.{n_layers}.attn_norm.weight" in state_dict:
        n_layers += 1
    d_ff = state_dict["blocks.0.feed_forward.fc_in.linear.weight_ternary"].shape[0]
    d_head = state_dict["blocks.0.attention.rope_cos"].shape[1]
    max_seq_len = state_dict["blocks.0.attention.rope_cos"].shape[0]
    assert d_model % d_head == 0
    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "n_heads": int(d_model // d_head),
        "n_layers": int(n_layers),
        "d_ff": int(d_ff),
        "max_seq_len": int(max_seq_len),
    }


def load_dqt_transformer_checkpoint(
    checkpoint_path: str,
) -> tuple[dict, dict, float, int]:
    """Load a DQT Transformer checkpoint into ``(config, state_dict, ...)``.

    Accepts either a ``best.pt``/``ckpt_step*.pt`` training checkpoint (a
    dict with ``model_state_dict`` and optional ``config``) or a bare
    ``state_dict``. Returns the architecture config (stored ``config`` or
    inferred from shapes), the model ``state_dict``, the best validation
    perplexity (or NaN) and the step (or 0).

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        ``(config, state_dict, best_val_ppl, step)``.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        config = ckpt.get("config") or infer_transformer_config_from_state_dict(state)
        best_val_ppl = float(ckpt.get("best_val_ppl", float("nan")))
        step = int(ckpt.get("step", 0))
        return dict(config), state, best_val_ppl, step
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        return (
            infer_transformer_config_from_state_dict(state),
            state,
            float("nan"),
            0,
        )
    # Bare state_dict
    return infer_transformer_config_from_state_dict(ckpt), ckpt, float("nan"), 0


# ── Conversion ──────────────────────────────────────────────────────


def count_ternary_weights_inference(model: nn.Module) -> int:
    """Count the ternary-origin weights in an inference transformer.

    The DQT ternary weights live in the model's standard ``nn.Linear``
    layers (the float token embedding and the tiny RMSNorm scales are NOT
    ternary). Returns the total number of ternary weights, which is what
    the 2-bit packed size is computed from.
    """
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.weight.numel()
    return total


def _baked_linear(source: nn.Module, out_features: int, in_features: int) -> nn.Linear:
    """Build an ``nn.Linear`` with the DQT ternary weight + scale baked in.

    ``(x @ W_ternary^T) * scale == x @ (W_ternary * scale)^T``, so the
    DQT output scaling folds into the weight — the inference layer is one
    standard matmul, no separate scale multiply.
    """
    linear = nn.Linear(in_features, out_features, bias=False)
    w = source.weight_ternary.float().detach().cpu()
    scale = float(getattr(source, "output_scale", 1.0))
    linear.weight.data = (w * scale).clone()
    linear.weight.requires_grad = False
    return linear


def dqt_transformer_to_inference_model(
    dqt_model: DQTTransformer, ctx_len: int | None = None
) -> DQTTransformerInference:
    """Convert a trained DQT Transformer into the inference-only model.

    Args:
        dqt_model: A ``DQTTransformer`` (training model) on any device.
        ctx_len: Fixed context length for the inference model. Defaults to
            ``dqt_model.max_seq_len`` (the RoPE table size).

    Returns:
        A frozen :class:`DQTTransformerInference` on CPU.
    """
    if ctx_len is None:
        ctx_len = int(dqt_model.max_seq_len)
    ctx_len = min(int(ctx_len), int(dqt_model.max_seq_len))

    model = DQTTransformerInference(
        vocab_size=dqt_model.vocab_size,
        d_model=dqt_model.d_model,
        n_heads=dqt_model.n_heads,
        n_layers=dqt_model.n_layers,
        d_ff=dqt_model.d_ff,
        ctx_len=ctx_len,
    )

    # Float embedding
    model.token_embedding.weight.data = (
        dqt_model.token_embedding.weight.detach().cpu().clone()
    )
    model.token_embedding.weight.requires_grad = False

    for src, dst in zip(dqt_model.blocks, model.blocks, strict=True):
        dst.attn_norm.weight.data = src.attn_norm.weight.detach().cpu().clone()
        dst.ffn_norm.weight.data = src.ffn_norm.weight.detach().cpu().clone()
        attn = src.attention
        dst.attention.q_proj = _baked_linear(
            attn.q_proj, attn.q_proj.out_features, attn.q_proj.in_features
        )
        dst.attention.k_proj = _baked_linear(
            attn.k_proj, attn.k_proj.out_features, attn.k_proj.in_features
        )
        dst.attention.v_proj = _baked_linear(
            attn.v_proj, attn.v_proj.out_features, attn.v_proj.in_features
        )
        dst.attention.o_proj = _baked_linear(
            attn.o_proj, attn.o_proj.out_features, attn.o_proj.in_features
        )
        # RoPE tables: slice the source tables to ctx_len
        dst.attention.rope_cos = attn.rope_cos.detach().cpu()[:ctx_len].clone()
        dst.attention.rope_sin = attn.rope_sin.detach().cpu()[:ctx_len].clone()
        dst.fc_in = _baked_linear(
            src.feed_forward.fc_in,
            src.feed_forward.fc_in.out_features,
            src.feed_forward.fc_in.in_features,
        )
        dst.fc_out = _baked_linear(
            src.feed_forward.fc_out,
            src.feed_forward.fc_out.out_features,
            src.feed_forward.fc_out.in_features,
        )

    model.final_norm.weight.data = dqt_model.final_norm.weight.detach().cpu().clone()
    model.lm_head = _baked_linear(
        dqt_model.lm_head, dqt_model.lm_head.out_features, dqt_model.lm_head.in_features
    )

    model = model.to("cpu")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ── ONNX export ────────────────────────────────────────────────────


def export_transformer_to_onnx(
    inference_model: DQTTransformerInference,
    output_path: str,
    opset_version: int = 18,
    verify: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> dict:
    """Export an inference transformer to ONNX and (optionally) verify.

    The input is a fixed-length int64 token-id tensor ``(batch, ctx_len)``
    with a dynamic batch axis; the output is ``(batch, ctx_len, vocab)``.
    Verification runs the same graph through onnxruntime on CPU and
    compares to the PyTorch reference.

    Args:
        inference_model: Output of :func:`dqt_transformer_to_inference_model`.
        output_path: Destination ``.onnx`` path.
        opset_version: ONNX opset (default 18).
        verify: If True, verify with onnxruntime.
        rtol / atol: Tolerances for verification.

    Returns:
        Summary dict: ``onnx_path``, ``onnx_size_mb``, ``ctx_len``,
        ``n_ternary_weights``, ``packed_bytes``, ``verified``,
        ``max_abs_diff``.
    """
    output_path = str(output_path)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    inference_model.eval()
    ctx_len = inference_model.ctx_len
    dummy = torch.zeros(1, ctx_len, dtype=torch.long)

    torch.onnx.export(
        inference_model,
        dummy,
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

    summary: dict = {
        "onnx_path": output_path,
        "onnx_size_mb": get_model_size_mb(output_path),
        "ctx_len": int(ctx_len),
        "n_ternary_weights": count_ternary_weights_inference(inference_model),
        "packed_bytes": (count_ternary_weights_inference(inference_model) + 3) // 4,
        "verified": None,
        "max_abs_diff": None,
    }

    if verify:
        import onnxruntime as ort

        with torch.no_grad():
            ref = inference_model(dummy).numpy()
        session = ort.InferenceSession(output_path)
        onnx_out = session.run(None, {"input": dummy.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(ref - onnx_out)))
        summary["verified"] = bool(
            np.allclose(ref, onnx_out, rtol=rtol, atol=atol)
        )
        summary["max_abs_diff"] = max_abs_diff

    return summary


# ── 2-bit packed ternary companion file ─────────────────────────────


def export_transformer_packed_ternary(
    dqt_model: DQTTransformer, output_path: str
) -> str:
    """Write the ternary weights of a DQT Transformer to a 2-bit ``.ternary`` file.

    Uses the same PHN3 format as
    :func:`ph_neuro.models.export.export_packed_ternary` (so
    :func:`ph_neuro.models.export.load_packed_ternary` can read it back),
    but walks the transformer recursively (``named_modules``) because the
    DQT projections live inside the attention/FFN sub-modules.

    Args:
        dqt_model: A ``DQTTransformer`` (training model).
        output_path: Destination ``.ternary`` path.

    Returns:
        The output path.
    """
    import struct  # noqa: PLC0415

    output_path = str(output_path)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    entries = []
    for name, module in dqt_model.named_modules():
        # Pack only the modules that actually OWN the ternary weights as a
        # registered buffer. ``TernaryDQTLinear3D`` (the transformer wrapper)
        # exposes ``weight_ternary`` as a *property* that delegates to its
        # inner ``TernaryDQTLinear`` — using ``getattr`` here would pack every
        # weight TWICE (wrapper + inner). Checking ``_buffers`` keeps exactly
        # one entry per ternary weight.
        if "weight_ternary" not in module._buffers:
            continue
        w = module.weight_ternary.detach().cpu()
        payload = pack_ternary(w).numpy().tobytes()
        entries.append((name, tuple(w.shape), payload))

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
    return output_path
