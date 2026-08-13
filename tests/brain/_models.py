"""Tiny real-architecture models for BrainWrapper tests (no network/GPU).

Builds genuinely-structured SmolLM2-style (LLaMA) and GPT-2-style causal LMs
from HF configs so injection-point discovery and hooks run against the real
module paths (``model.model.layers[i].self_attn.o_proj`` etc. / GPT-2's
``Conv1D``).
"""

from __future__ import annotations

import torch


def tiny_llama(vocab: int = 128, hidden: int = 32, layers: int = 2,
               device: str = "cpu"):
    """A tiny LLaMA-architecture CausalLM (SmolLM2 uses model_type 'llama')."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    if device != "cpu":
        model.to(device)
    return model


def tiny_gpt2(vocab: int = 128, hidden: int = 32, layers: int = 2,
              device: str = "cpu"):
    """A tiny GPT-2 CausalLM (classic pre-norm, Conv1D projections)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab,
        n_embd=hidden,
        n_layer=layers,
        n_head=4,
        n_positions=128,
        n_ctx=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        scale_attn_weights=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    if device != "cpu":
        model.to(device)
    return model


def random_token_ids(batch: int = 2, seq: int = 32, vocab: int = 128,
                     device: str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, vocab, (batch, seq), generator=g, device=device)
