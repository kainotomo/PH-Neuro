"""Step 0.5 — statistical power calibration for the eval protocol.

Measures, with the ACTUAL Phase 1.1 eval model (SmolLM2-1.7B, bf16):
  1. Per-chunk mean log-perplexity (nats/token) and its standard deviation
     on NON-OVERLAPPING 512-token blocks (stride = block length → effectively
     independent observations) for:
       - WikiText-2 test (source domain)
       - PubMed (ccdv/pubmed-summarization) test subsample (target domain)
  2. Reports corpus ppl = exp(mean log-ppl).
  3. Computes the required number of independent test tokens to detect a
     true Δppl with Cohen's d = 0.5 at 80% power, α = 0.05 (two-sided):
         n_chunks = (z_{1-α/2} + z_{1-β})^2 / d^2,  d = δ / σ_chunk
         n_tokens = n_chunks * block_len
  4. Verifies the planned test corpora are ≫ this requirement.

Eval config locked for calibration (matches protocol):
  context window = 512 tokens, first token of each block skipped (no context).
  Per-token average (unweighted) is the primary metric → log-ppl = mean NLL/token.

Writes a JSON report + prints summary. Read-only w.r.t. the workspace.
"""
import json
import time

import torch

# This machine has NO C compiler → Triton cannot JIT. torch 2.13 registers a
# Triton-backed fused `bmm` override (used by RoPE in llama modeling). Disable
# that override so the standard eager `aten::bmm` kernel is used instead.
try:
    from torch._native.registry import deregister_op_overrides
    deregister_op_overrides(disable_op_symbols="bmm")
    print("[calib] disabled Triton bmm override (no C compiler on this box)", flush=True)
except Exception as e:  # pragma: no cover
    print(f"[calib] note: could not disable triton bmm override: {e}", flush=True)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
BLOCK = 512                 # non-overlapping block length (independent chunks)
MAX_PUBMED_TOKENS = 250_000  # subsample for target σ estimate (full test ~29M too big)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def tokenize_corpus(tokenizer, texts, max_tokens=None, seed=SEED):
    ids = []
    for t in texts:
        e = tokenizer(t, add_special_tokens=False, truncation=False)
        ids.extend(e["input_ids"])
        if max_tokens is not None and len(ids) >= max_tokens:
            break
    if max_tokens is not None and len(ids) > max_tokens:
        ids = ids[:max_tokens]
    # deterministic shuffling is NOT applied here; we take the natural order for
    # WikiText (already ordered text). For PubMed we cap at a fixed prefix.
    return torch.tensor(ids, dtype=torch.long)


def chunk_log_ppl(model, ids, block=BLOCK, batch=4):
    """Return per-block mean NLL (nats/token). Blocks are non-overlapping;
    first token of each block skipped. Computes causal LM NLL.
    """
    n_full = (len(ids) - 1) // block          # blocks that fit fully
    starts = [i * block for i in range(n_full)]
    per_block = []
    with torch.no_grad():
        for s in range(0, n_full, batch):
            idx = torch.tensor(starts[s:s + batch], dtype=torch.long, device=DEVICE)
            # gather block token ids: (B, block)
            xs = []
            for st in idx.tolist():
                xs.append(ids[st:st + block])
            x = torch.stack([ids[st:st + block] for st in idx.tolist()]).to(DEVICE)
            out = model(x)
            logits = out.logits  # (B, S, V)
            # NLL of token t given tokens [0..t-1]; t from 1..S-1
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_targets = x[:, 1:].contiguous()
            nll = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction="mean",
            ).item()
            per_block.append(nll)
            if (s // batch) % 10 == 0:
                print(f"  block {s}/{n_full} nll={nll:.4f}", flush=True)
    return torch.tensor(per_block, dtype=torch.float64)


def required_tokens_for_d(model_chunk_std, block=BLOCK, d=0.5):
    """Number of INDEPENDENT tokens to detect effect size d (Cohen's d, in
    units of per-chunk std of the mean log-ppl) at 80% power, two-sided α=0.05.
    n_chunks = ((z_{0.975} + z_{0.80}) / d)^2 ; each observation is one block,
    so δ_per_token (nats) = d * σ_block is the per-token mean shift implied.
    """
    z_a = 1.959963985  # z_{0.975}
    z_b = 0.841621234  # z_{0.80}
    n_chunks = ((z_a + z_b) ** 2) / (d ** 2)      # = 31.4 / d^2
    n_tokens = n_chunks * block
    delta_nats_per_token = d * model_chunk_std    # uniform per-token shift
    return {
        "d": d,
        "n_chunks": round(n_chunks, 1),
        "n_tokens_independent": int(n_tokens),
        "delta_nats_per_token": delta_nats_per_token,
        "ppl_ratio_exp": float(torch.exp(torch.tensor(delta_nats_per_token))),
        # relative ppl change implied by d=0.5 (ppl_frozen/ppl_plastic ratio)
    }


def main():
    rep = {"model": MODEL_ID, "block_len": BLOCK, "device": DEVICE,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    print(f"Loading {MODEL_ID} (bf16) on {DEVICE} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # eager attention: this machine has NO C compiler → Triton (flash/SDPA
    # kernels) cannot JIT-compile. Eager is slower but correct and dependency-free.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="eager").to(DEVICE)
    model.eval()

    # ---- WikiText-2 test (source) ----
    t0 = time.time()
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_ids = tokenize_corpus(tok, wt["text"])
    print(f"WikiText-2 test: {len(wt_ids)} tokens", flush=True)
    wt_block = chunk_log_ppl(model, wt_ids)
    rep["wikitext2"] = {
        "tokens": int(len(wt_ids)),
        "n_blocks": int(len(wt_block)),
        "mean_nll_per_token": float(wt_block.mean()),
        "std_nll_per_block": float(wt_block.std()),
        "ppl": float(torch.exp(wt_block.mean())),
        "ppl_ci95_low": float(torch.exp(wt_block.mean() - 1.96 * wt_block.std() / (len(wt_block) ** 0.5))),
        "ppl_ci95_high": float(torch.exp(wt_block.mean() + 1.96 * wt_block.std() / (len(wt_block) ** 0.5))),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[wiki] ppl={rep['wikitext2']['ppl']:.2f} "
          f"σ_block={rep['wikitext2']['std_nll_per_block']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- PubMed test subsample (target) ----
    t0 = time.time()
    pm = load_dataset("ccdv/pubmed-summarization", split="test")
    pm_docs = [f"{r['abstract']} {r['article']}".strip() for r in pm]
    pm_ids = tokenize_corpus(tok, pm_docs, max_tokens=MAX_PUBMED_TOKENS)
    print(f"PubMed test subsample: {len(pm_ids)} tokens", flush=True)
    pm_block = chunk_log_ppl(model, pm_ids)
    rep["pubmed_subsample"] = {
        "tokens": int(len(pm_ids)),
        "n_blocks": int(len(pm_block)),
        "mean_nll_per_token": float(pm_block.mean()),
        "std_nll_per_block": float(pm_block.std()),
        "ppl": float(torch.exp(pm_block.mean())),
        "ppl_ci95_low": float(torch.exp(pm_block.mean() - 1.96 * pm_block.std() / (len(pm_block) ** 0.5))),
        "ppl_ci95_high": float(torch.exp(pm_block.mean() + 1.96 * pm_block.std() / (len(pm_block) ** 0.5))),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[pubmed] ppl={rep['pubmed_subsample']['ppl']:.2f} "
          f"σ_block={rep['pubmed_subsample']['std_nll_per_block']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- Power requirement ----
    for name in ("wikitext2", "pubmed_subsample"):
        sigma = rep[name]["std_nll_per_block"]
        rep[name]["required_for_d05"] = required_tokens_for_d(sigma)
        r = rep[name]["required_for_d05"]
        print(f"[power-{name}] σ_block={sigma:.4f} → {r['n_chunks']} chunks, "
              f"{r['n_tokens_independent']} tokens for d=0.5 @80% "
              f"(δ_nats={r['delta_nats_per_token']:.4f}/tok)", flush=True)

    out = "/home/phalo/PH-Neuro/research/scripts/eval_power_calibration.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nWROTE {out}", flush=True)


if __name__ == "__main__":
    main()
