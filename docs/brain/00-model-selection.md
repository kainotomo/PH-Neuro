# Step 0.1 — Model Selection

> **Status:** 🔬 In Progress
> **Goal:** Select the best pre-trained open-source model to serve as the "born brain" for the first Brain Wrapper experiment.

---

## Selection Principle

We select models for research fit, not recency or SOTA. A model's age is
irrelevant — a well-established, older model is often preferable (most-studied,
best-documented, stable baselines). We do **not** chase the newest or most
advanced models. Choosing the model(s) is the user's decision; the assistant's
role is limited to stating the criteria, not proposing candidates.

---

## Selection Criteria

The ideal candidate model must satisfy all of the following:

1. **Permissive license** — Apache 2.0, MIT, or equivalent. No restrictions on modification, redistribution, or commercial use.
2. **Accessible size** — 100M–3B parameters. Must load and run inference comfortably on an RTX 4060 (8 GB VRAM) with room for plastic weights and batch processing.
3. **Clean architecture** — Standard transformer (GPT-2 style or LLaMA style). Activation interception (pre/post at each transformer block) must be straightforward via PyTorch forward hooks or `output_hidden_states`.
4. **Known baseline** — Well-documented perplexity on standard benchmarks (WikiText-2, PTB). A meaningful improvement from plasticity must be measurable above noise.
5. **Tokenizer quality** — Subword tokenizer (BPE, SentencePiece). Must handle diverse text domains (scientific, legal, code).

### Nice-to-Have

- Multiple size variants (125M, 355M, 1.1B) for scaling experiments
- Transformer implementation follows a standard pattern (embed → N× block → head)
- Active community / well-maintained on HuggingFace

---

## Candidate Selection (by the user)

> **No models are proposed here — by design.** Today is 2026-08-12, and the
> pre-trained model landscape (availability, licenses, quality) changes
> frequently. Naming specific models with parameter counts, licenses, or
> perplexity figures here would risk being stale or wrong. The assistant
> provides the criteria above; choosing the actual model(s) to investigate is
> the user's call, made from **current** sources (e.g. Hugging Face).

Candidate(s) selected by the user — fill in against the criteria above:

| Model | License | Size | Baseline ppl (Wiki-2/PTB) | Hook ease | Variants? | Role (primary / gen-test / scaling) |
|:------|:-------:|:----:|:-------------------------:|:---------:|:---------:|:-----------------------------------:|
|       |         |      |                           |           |           |                                     |

---

## Recommendation (Preliminary)

Deferred to the user: choose the primary, generalization-test, and scaling
candidates against the criteria above, using current information (e.g. from
Hugging Face). No recommendation is made here.

---

## Next Steps

- [ ] User selects the model(s) to investigate (from current sources)
- [ ] Load the selected model and inspect the forward pass structure
- [ ] Verify that activation interception works for pre/post at each transformer block
- [ ] Measure inference memory and throughput on RTX 4060
- [ ] Run baseline perplexity on WikiText-2 and candidate target domains (PubMed)
- [ ] Finalize decision and document rationale
