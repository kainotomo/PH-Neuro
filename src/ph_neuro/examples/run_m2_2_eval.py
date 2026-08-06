#!/usr/bin/env python3
"""Milestone M2.2 — evaluate / test a trained 250M DQT transformer on WikiText-2.

Loads a trained M2_2_CONFIG checkpoint into a :class:`DQTTransformer` and lets
you (a) measure validation + held-out TEST perplexity on WikiText-2 and (b)
generate a short text continuation with greedy decoding (GPT-2 tokenizer).

The checkpoints saved during training carry optimizer state (3.57 GB) for
pause/resume; this loader keeps only ``model_state_dict`` so the eval model
uses the same weights but no optimizer.

Usage::

    # Evaluate the final checkpoint of seed 43 on the WikiText-2 TEST split:
    .venv/bin/python -m ph_neuro.examples.run_m2_2_eval --seed 43

    # Use a specific checkpoint:
    .venv/bin/python -m ph_neuro.examples.run_m2_2_eval \\
        --checkpoint m2_2_results/checkpoints/seed43/ckpt_step7000.pt

    # Generate a continuation from a prompt:
    .venv/bin/python -m ph_neuro.examples.run_m2_2_eval --seed 43 \\
        --prompt "The history of Greece is"

Output:
    Val PPL, Test PPL (held-out), and an optional generated sample.
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from ph_neuro.examples._utils import print_header
from ph_neuro.examples.run_m2_2_dqt_wikitext2 import evaluate_perplexity
from ph_neuro.models.dqt_transformer import M2_2_CONFIG, dqt_gpt2
from ph_neuro.training.wikitext2 import get_wikitext2_data, make_gpt2_tokenizer

CHECKPOINT_DIR = "m2_2_results/checkpoints"


def find_final_checkpoint(seed: int) -> str:
    """Highest-step checkpoint for ``seed`` (the final trained model)."""
    matches = glob.glob(os.path.join(CHECKPOINT_DIR, f"seed{seed}", "ckpt_step*.pt"))

    def _step(path: str) -> int:
        try:
            return int(os.path.basename(path).replace("ckpt_step", "").replace(".pt", ""))
        except ValueError:
            return -1

    if not matches:
        raise FileNotFoundError(
            f"No checkpoints found for seed {seed} under {CHECKPOINT_DIR}/seed{seed}/"
        )
    return max(matches, key=_step)


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Build an M2_2_CONFIG DQT transformer and load the checkpoint weights."""
    cfg = M2_2_CONFIG
    model = dqt_gpt2(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        use_grad_checkpointing=False,  # eval only — no checkpoint overhead
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {os.path.basename(ckpt_path)} "
          f"(step {ckpt.get('step', '?')}, best_val_ppl {ckpt.get('best_val_ppl', '?'):.2f})")
    return model


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Greedy (argmax) autoregressive generation from a prompt.

    Note: at ppl ~480 the model is too under-trained (tiny data budget) to
    produce coherent text — the sample is a pipeline check, not quality text.
    """
    model.eval()
    ids = tokenizer.encode(prompt) if prompt else []
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        if seq.shape[1] >= model.max_seq_len:
            seq = seq[:, -model.max_seq_len :]
        logits = model(seq)  # (1, T, V)
        next_id = logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1)
        seq = torch.cat([seq, next_id], dim=1)
    return prompt + tokenizer.decode(seq[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2.2: evaluate / test a trained 250M DQT transformer on WikiText-2"
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint path (default: final/highest for --seed)")
    parser.add_argument("--prompt", default=None,
                        help="Optional prompt for greedy text generation")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = args.checkpoint or find_final_checkpoint(args.seed)
    model = load_model(ckpt_path, device)

    # WikiText-2 val + held-out TEST splits (cached — no re-download)
    _train_loader, val_loader, test_loader, meta = get_wikitext2_data(
        data_dir="data/wikitext2",
        seq_len=256,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print()
    print_header("M2.2 Model Evaluation (WikiText-2)")
    val_ppl = evaluate_perplexity(model, val_loader, device)
    test_ppl = evaluate_perplexity(model, test_loader, device)
    print(f"  Validation PPL:  {val_ppl:.2f}")
    print(f"  Test PPL:        {test_ppl:.2f}   (held-out split)")
    print(f"  Model:           {meta['dataset']} | {ckpt_path}")
    print()

    if args.prompt is not None:
        tokenizer = make_gpt2_tokenizer()
        sample = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
        print_header("Sample generation (greedy)")
        print(f"  {sample}")
        print()


if __name__ == "__main__":
    main()
