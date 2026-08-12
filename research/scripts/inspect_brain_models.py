"""PH-Neuro Brain Phase 0 / Step 0.1 — Phase B technical inspection.

Loads a decoder-only causal LM from HuggingFace (CPU-only) and reports:
  * full module tree + block structure
  * parameter count and top-level breakdown
  * HF-cache disk size
  * peak CPU RSS during load and single-batch inference
  * throughput (tok/s) on a sample text
  * activation interception: forward_pre_hook / forward_hook on every
    transformer block, plus output_hidden_states=True verification

Usage (venv):
  .venv/bin/python research/scripts/inspect_brain_models.py <model_id> \
      [--seq-len 256] [--iters 5]
"""
from __future__ import annotations

import argparse
import os
import resource
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def rss_mb() -> float:
    """Peak resident set size in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def find_blocks(model):
    cands = [
        ("model.model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("model.transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("model.gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
    ]
    for name, obj in cands:
        if obj is not None and len(obj) > 0:
            return name, obj
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    print(f"=== Model: {args.model_id} ===")
    base = rss_mb()
    print(f"[memory] baseline peak RSS: {base:.0f} MB")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s (tokenizer+weights); peak RSS: {rss_mb():.0f} MB")

    # Disk size from the HF cache
    safe = args.model_id.replace("/", "--")
    cache = os.path.expanduser(f"~/.cache/huggingface/hub/models--{safe}")
    disk = 0
    if os.path.isdir(cache):
        for root, _dirs, files in os.walk(cache):
            for f in files:
                try:
                    disk += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    print(f"[disk] HF cache size: {human(disk)}")

    # Parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] total: {total:,} ({total/1e6:.1f}M)")
    top = {}
    for name, p in model.named_parameters():
        k = name.split(".")[0]
        top[k] = top.get(k, 0) + p.numel()
    print("[params] by top-level module:", {k: f"{v/1e6:.1f}M" for k, v in sorted(top.items(), key=lambda x: -x[1])})

    # Architecture overview
    cfg = model.config
    print("[arch] model_type:", cfg.model_type, "| architectures:", cfg.architectures)
    print("[arch] config:", {
        "n_layer": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None)),
        "n_head": getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None)),
        "n_kv_head": getattr(cfg, "num_key_value_heads", None),
        "n_embd": getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None)),
        "d_ff": getattr(cfg, "intermediate_size", getattr(cfg, "n_inner", None)),
        "vocab": getattr(cfg, "vocab_size", None),
        "max_len": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "tie_word_emb": getattr(cfg, "tie_word_embeddings", None),
    })

    # Module tree (3 levels deep)
    print("[arch] module tree (3 levels):")
    for name, m in model.named_modules():
        if name.count(".") <= 2:
            print(f"  {name or '<root>':38s} {type(m).__name__:30s} {sum(p.numel() for p in m.parameters())/1e6:8.1f}M")

    # Transformer blocks
    blk_path, blocks = find_blocks(model)
    n_blocks = len(blocks) if blocks is not None else 0
    print(f"[blocks] path={blk_path} count={n_blocks}")
    if blocks is not None:
        print("[blocks] block[0] submodules:")
        for n, m in blocks[0].named_children():
            print(f"    {n}: {type(m).__name__}")

    # Throughput
    sample = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    enc = tokenizer(sample, return_tensors="pt", max_length=args.seq_len, truncation=True)
    input_ids = enc["input_ids"]
    n_tok = input_ids.shape[1]
    print(f"[sample] {n_tok} tokens, batch=1, dtype=bf16")

    with torch.no_grad():
        _ = model(input_ids)  # warmup
        t0 = time.time()
        for _ in range(args.iters):
            _ = model(input_ids)
        dt = time.time() - t0
    print(f"[throughput] {n_tok*args.iters} tok in {dt:.2f}s -> {n_tok*args.iters/dt:.1f} tok/s (CPU)")
    print(f"[memory] peak RSS after inference: {rss_mb():.0f} MB (delta {rss_mb()-base:.0f} MB)")

    # Hook interception verification
    if blocks is not None:
        pre, post = {}, {}
        handles = []
        for i, blk in enumerate(blocks):
            def mk_pre(idx):
                def h(_mod, inp):
                    x = inp[0]
                    pre[idx] = (tuple(x.shape), str(x.dtype))
                return h

            def mk_post(idx):
                def h(_mod, _inp, out):
                    x = out[0] if isinstance(out, tuple) else out
                    post[idx] = (tuple(x.shape), str(x.dtype))
                return h

            handles.append(blk.register_forward_pre_hook(mk_pre(i)))
            handles.append(blk.register_forward_hook(mk_post(i)))

        with torch.no_grad():
            out_hid = model(input_ids, output_hidden_states=True)

        print(f"[hooks] pre fired: {len(pre)}/{n_blocks}, post fired: {len(post)}/{n_blocks}")
        if n_blocks:
            print(f"[hooks] block[0] pre={pre.get(0)} post={post.get(0)}")
            print(f"[hooks] block[{n_blocks-1}] pre={pre.get(n_blocks-1)} post={post.get(n_blocks-1)}")

        hs = out_hid.hidden_states
        print(f"[hidden_states] len={len(hs)}; hs[0]={tuple(hs[0].shape)} hs[1]={tuple(hs[1].shape)} "
              f"... hs[-1]={tuple(hs[-1].shape)}")
        ok = True
        for i in range(n_blocks):
            if i in pre:
                ok = ok and pre[i][0] == tuple(hs[i].shape)
            if i in post:
                ok = ok and post[i][0] == tuple(hs[i + 1].shape)
        print(f"[hooks] pre[i]==hs[i] and post[i]==hs[i+1] for all blocks: {ok}")

        # Architectural quirks
        attn = getattr(blocks[0], "self_attn", None) or getattr(blocks[0], "attn", None)
        mlp = getattr(blocks[0], "mlp", None)
        print(f"[arch] block[0].attn type: {type(attn).__name__}")
        print(f"[arch] block[0].mlp   type: {type(mlp).__name__}")
        for h in handles:
            h.remove()

    print("[done]")


if __name__ == "__main__":
    main()
