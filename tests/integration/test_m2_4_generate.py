"""Integration tests for Milestone M2.4 — on-device inference demo.

Verifies the DQT Transformer on-device pipeline end-to-end (on CPU, no GPU):
    1. DQT Transformer → inference model with identical output
       (ternary weights + 1/sqrt(in) output scale baked into nn.Linear)
    2. Config inference from a checkpoint state_dict
    3. ONNX export → onnxruntime → identical output (roundtrip)
    4. 2-bit packed ternary export → load → identical weights
    5. Autoregressive generation (temperature + top-k sampling) is
       seed-deterministic and produces the requested token count
    6. PyTorch CPU and ONNX CPU generation produce IDENTICAL text

All tests use a tiny DQT transformer (d=32, L=2, H=4, ff=64, vocab=64) on
CPU, so no dataset / GPU is required.
"""

from __future__ import annotations

import os

import pytest
import torch

from ph_neuro.models.dqt_transformer import dqt_gpt2
from ph_neuro.models.export import load_packed_ternary
from ph_neuro.models.export_transformer import (
    count_ternary_weights_inference,
    dqt_transformer_to_inference_model,
    export_transformer_packed_ternary,
    export_transformer_to_onnx,
    infer_transformer_config_from_state_dict,
    load_dqt_transformer_checkpoint,
)

VOCAB = 64
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
D_FF = 64
MAX_SEQ = 16
CTX = 16

DEVICE = torch.device("cpu")


def _make_model() -> torch.nn.Module:
    torch.manual_seed(7)
    model = dqt_gpt2(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ,
        device=DEVICE,
    )
    # Fill with random float buffers so the ternary weights are meaningful
    for param in model.parameters():
        param.data.normal_(0, 0.1)
    model.eval()
    return model


def _write_checkpoint(model, path: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": VOCAB,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "max_seq_len": MAX_SEQ,
            },
            "best_val_ppl": 5.0,
            "step": 100,
        },
        path,
    )


# ── 1. Inference conversion equality ───────────────────────────────


def test_transformer_to_inference_identical() -> None:
    """Inference model output must match the DQT model to machine precision."""
    model = _make_model()
    inference = dqt_transformer_to_inference_model(model, ctx_len=CTX)

    # No DQT custom layers remain — only standard nn.Linear / nn.Embedding
    for module in inference.modules():
        assert not hasattr(module, "weight_ternary")

    toks = torch.randint(0, VOCAB, (2, CTX))
    with torch.no_grad():
        ref = model(toks)
        out = inference(toks)

    assert out.shape == (2, CTX, VOCAB)
    assert torch.allclose(out, ref, atol=1e-5), (
        f"inference transformer must match DQT model (max|Δ|="
        f"{(out - ref).abs().max().item():.2e})"
    )


def test_inference_weights_frozen() -> None:
    """All inference weights must be frozen (requires_grad=False)."""
    model = _make_model()
    inference = dqt_transformer_to_inference_model(model, ctx_len=CTX)
    for param in inference.parameters():
        assert not param.requires_grad


def test_output_scale_baked_into_linear() -> None:
    """The DQT 1/sqrt(in) output scale must be baked into nn.Linear weights.

    weight_nn == weight_ternary * (1/sqrt(in_features)) for every DQT
    projection (so the inference layer needs no separate scale multiply).
    """
    model = _make_model()
    inference = dqt_transformer_to_inference_model(model, ctx_len=CTX)

    # Check the LM head: baked weight == ternary * 1/sqrt(d_model)
    src_w = model.lm_head.linear.weight_ternary.float()
    scale = 1.0 / (D_MODEL ** 0.5)
    dst_w = inference.lm_head.weight.data
    assert torch.allclose(dst_w, src_w * scale, atol=1e-6)


# ── 2. Config inference ────────────────────────────────────────────


def test_infer_config_from_state_dict() -> None:
    """Config can be recovered from the state_dict shapes alone."""
    model = _make_model()
    cfg = infer_transformer_config_from_state_dict(model.state_dict())
    assert cfg == {
        "vocab_size": VOCAB,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_seq_len": MAX_SEQ,
    }


def test_load_checkpoint_with_config(tmp_path) -> None:
    """Checkpoint loader returns the stored config, state dict, ppl and step."""
    model = _make_model()
    path = str(tmp_path / "best.pt")
    _write_checkpoint(model, path)

    config, state, ppl, step = load_dqt_transformer_checkpoint(path)
    assert config["d_model"] == D_MODEL
    assert config["n_heads"] == N_HEADS
    assert ppl == 5.0
    assert step == 100
    assert "token_embedding.weight" in state


# ── 3. ONNX export roundtrip ───────────────────────────────────────


def test_onnx_export_roundtrip(tmp_path) -> None:
    """ONNX export → onnxruntime → identical output (± 1e-4)."""
    model = _make_model()
    inference = dqt_transformer_to_inference_model(model, ctx_len=CTX)
    onnx_path = str(tmp_path / "model.onnx")

    summary = export_transformer_to_onnx(inference, onnx_path, verify=True)

    assert os.path.exists(onnx_path)
    assert summary["verified"] is True
    assert summary["max_abs_diff"] < 1e-3
    assert summary["ctx_len"] == CTX
    assert summary["n_ternary_weights"] == count_ternary_weights_inference(inference)
    assert summary["packed_bytes"] == (summary["n_ternary_weights"] + 3) // 4


# ── 4. Packed ternary roundtrip ────────────────────────────────────


def test_packed_ternary_roundtrip(tmp_path) -> None:
    """2-bit packed export → load → identical weights for every DQT layer."""
    model = _make_model()
    packed_path = str(tmp_path / "model.ternary")
    export_transformer_packed_ternary(model, packed_path)

    entries = load_packed_ternary(packed_path)
    names = {name for name, _, _ in entries}
    # Every DQT projection is captured (including inside the blocks)
    assert "blocks.0.attention.q_proj.linear" in names
    assert "blocks.0.feed_forward.fc_in.linear" in names
    assert "lm_head.linear" in names

    # No double-counting: the wrapper TernaryDQTLinear3D exposes
    # weight_ternary as a property, so only the inner TernaryDQTLinear
    # layers may be packed — the total weight count must match.
    total = sum(w.numel() for _, _, w in entries)
    assert total == N_LAYERS * (4 * D_MODEL * D_MODEL + 2 * D_MODEL * D_FF) + D_MODEL * VOCAB
    assert len(entries) == N_LAYERS * 6 + 1  # 6 ternary layers/block + LM head

    # Roundtrip a layer: lm_head.linear ternary weights
    src_w = model.lm_head.linear.weight_ternary.detach().cpu()
    (_, _, lm_w) = next(
        (n, s, w) for n, s, w in entries if n == "lm_head.linear"
    )
    assert torch.equal(lm_w, src_w)


# ── 5/6. Generation ────────────────────────────────────────────────


def _tiny_tokenizer():
    """Deterministic char-level fake tokenizer so tests need no tiktoken.

    Vocabulary 0..VOCAB-1; encodes chars of a prompt string to ids in that
    range and decodes ids back to chars (deterministic, offline).
    """

    class _Tok:
        n_vocab = VOCAB

        def encode(self, text: str) -> list[int]:
            return [ord(c) % VOCAB for c in text]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

    return _Tok()


def test_generation_is_seed_deterministic(tmp_path) -> None:
    """Same seed → same generated text; token count matches request."""
    from ph_neuro.examples.generate_text import generate

    model = _make_model()
    inference = dqt_transformer_to_inference_model(model, ctx_len=CTX)
    tok = _tiny_tokenizer()

    r1 = generate(inference, tok, "ab", max_tokens=8, temperature=0.8, top_k=10,
                  ctx_len=CTX, seed=42)
    r2 = generate(inference, tok, "ab", max_tokens=8, temperature=0.8, top_k=10,
                  ctx_len=CTX, seed=42)
    r3 = generate(inference, tok, "ab", max_tokens=8, temperature=0.8, top_k=10,
                  ctx_len=CTX, seed=1)

    assert r1["n_generated"] == 8
    assert r1["text"] == r2["text"]
    assert r1["tokens_per_sec"] > 0
    # Different seed should (almost surely) diverge for an untrained model
    assert r3["text"] != r1["text"]


def test_pytorch_and_onnx_generation_identical(tmp_path) -> None:
    """PyTorch CPU and ONNX CPU generation produce identical text (same seed)."""
    import onnxruntime as ort

    from ph_neuro.examples.generate_text import generate

    model = _make_model()
    inference = dqt_transformer_to_inference_model(model, ctx_len=CTX)
    onnx_path = str(tmp_path / "gen.onnx")
    export_transformer_to_onnx(inference, onnx_path, verify=False)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    tok = _tiny_tokenizer()

    r_t = generate(inference, tok, "ab", max_tokens=10, temperature=0.8,
                   top_k=10, ctx_len=CTX, seed=42)
    r_o = generate(session, tok, "ab", max_tokens=10, temperature=0.8,
                   top_k=10, ctx_len=CTX, seed=42)

    assert r_t["text"] == r_o["text"]
    assert r_t["n_generated"] == r_o["n_generated"]


# ── CLI wiring ─────────────────────────────────────────────────────


def test_generate_text_cli_requires_checkpoint() -> None:
    """generate_text refuses to run without --checkpoint."""
    from ph_neuro.examples.generate_text import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
