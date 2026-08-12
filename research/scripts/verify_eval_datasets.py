"""Step 0.5 verification — eval datasets.

Verifies, on-disk:
1. WikiText-2 (Salesforce/wikitext, config wikitext-2-raw-v1): availability,
   split sizes (examples + words), and token counts under BOTH candidate tokenizers
   (SmolLM2-1.7B and GPT-2) for the test split.
2. ccdv/pubmed-summarization (PubMed abstract corpus): availability, split
   sizes (test abstracts), token counts under both tokenizers for the test split.
3. Exact number of training steps per adaptation budget given the locked
   batch_size=4, seq_len=256 (from Step 0.4 BrainWrapper.learn defaults).

This is a read-only verification; it writes a JSON report for the eval protocol doc.
"""
import json
import os
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_count(tokenizer, texts, max_examples=None):
    """Tokenize and return (n_tokens_total, n_seq_len256_tokens)."""
    total = 0
    for i, t in enumerate(texts):
        if max_examples is not None and i >= max_examples:
            break
        enc = tokenizer(t, add_special_tokens=False)
        total += len(enc["input_ids"])
    return total


def main():
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}

    tok_smollm = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tok_gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # ---------- 1. WikiText-2 ----------
    t0 = time.time()
    wt2 = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    report["checks"]["wikitext2_raw_v1"] = {
        "load_seconds": round(time.time() - t0, 1),
        "splits": {k: v.num_rows for k, v in wt2.items()},
    }
    test_texts = [r["text"] for r in wt2["test"]]
    n_words = sum(len(t.split()) for t in test_texts)
    report["checks"]["wikitext2_raw_v1"]["test_words"] = n_words

    n_tok_smollm = tokenize_count(tok_smollm, test_texts)
    n_tok_gpt2 = tokenize_count(tok_gpt2, test_texts)
    report["checks"]["wikitext2_raw_v1"]["test_tokens_smollm"] = n_tok_smollm
    report["checks"]["wikitext2_raw_v1"]["test_tokens_gpt2"] = n_tok_gpt2
    print(f"[wikitext2] test rows={wt2['test'].num_rows} words={n_words} "
          f"tok_smollm={n_tok_smollm} tok_gpt2={n_tok_gpt2} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---------- 2. PubMed (ccdv/pubmed-summarization) ----------
    t0 = time.time()
    pm = load_dataset("ccdv/pubmed-summarization")
    report["checks"]["pubmed-summarization"] = {
        "load_seconds": round(time.time() - t0, 1),
        "splits": {k: v.num_rows for k, v in pm.items()},
    }
    # concatenate article abstract sections into a document per example
    def doc_of(r):
        art = r.get("article", "")
        abstract = r.get("abstract", "")
        return f"{abstract} {art}".strip()

    test_docs = [doc_of(r) for r in pm["test"]]
    n_words_pm = sum(len(d.split()) for d in test_docs)
    report["checks"]["pubmed-summarization"]["test_words"] = n_words_pm

    n_tok_smollm_pm = tokenize_count(tok_smollm, test_docs)
    n_tok_gpt2_pm = tokenize_count(tok_gpt2, test_docs)
    report["checks"]["pubmed-summarization"]["test_tokens_smollm"] = n_tok_smollm_pm
    report["checks"]["pubmed-summarization"]["test_tokens_gpt2"] = n_tok_gpt2_pm
    print(f"[pubmed] test rows={pm['test'].num_rows} words={n_words_pm} "
          f"tok_smollm={n_tok_smollm_pm} tok_gpt2={n_tok_gpt2_pm} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---------- 3. Training steps per budget ----------
    BATCH, SEQ = 4, 256
    tok_per_step = BATCH * SEQ
    budgets = {"micro": 1_000, "small": 10_000, "medium": 100_000, "large": 1_000_000}
    steps = {}
    for name, tok in budgets.items():
        steps[name] = {
            "tokens": tok,
            "sequences_at_seq256": tok // SEQ,
            "steps_bs4": tok / tok_per_step,
            "steps_bs4_int": -(-tok // tok_per_step),  # ceil
        }
    report["checks"]["adaptation_budgets"] = {
        "batch_size": BATCH,
        "seq_len": SEQ,
        "tokens_per_step": tok_per_step,
        "budgets": steps,
    }
    for name, s in steps.items():
        print(f"[budget] {name}: {s['tokens']} tok = {s['sequences_at_seq256']} seq "
              f"= {s['steps_bs4']:.1f} steps (bs4)", flush=True)

    out = "/home/phalo/PH-Neuro/research/scripts/eval_dataset_verify.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWROTE {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
