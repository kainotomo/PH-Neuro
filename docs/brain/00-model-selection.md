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
3. **Clean architecture** — Standard decoder-only transformer (classic pre-norm or RoPE/SwiGLU style). Activation interception (pre/post at each transformer block) must be straightforward via PyTorch forward hooks or `output_hidden_states`.
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

---

## Workflow for This Step

The assistant (other chat) follows this sequence:

### Phase A — Survey
1. Browse HuggingFace for decoder-only causal LM models matching the criteria above (permissive license, 100M–3B params). Use current information — do NOT rely on any pre-existing list from this document.
2. Present the findings to the user in a table with: model name, HuggingFace ID, license, parameter count, architecture family (GPT-2-classic / LLaMA-modern / other), number of size variants, and any notable caveats.
3. **Stop here.** Wait for the user to choose primary, generalization-test, and scaling candidates. The assistant MUST NOT make this choice.

### Phase B — Technical Inspection (only after user selection)
4. For each selected model, write and run a small Python script that loads the model from HuggingFace and inspects:
   - Full architecture structure (embedding, transformer blocks, attention/MLP sublayers, final head)
   - Named module paths to verify activation interception is possible at every transformer block
   - Parameter count, disk size, peak CPU/GPU memory during loading and single-batch inference
   - Basic throughput (tokens/sec) on a sample text
5. Verify activation interception for the primary and generalization-test candidates: confirm that `output_hidden_states=True` or forward hooks can capture pre- and post-activations at each block. Note any architectural quirks.
6. Fill in the "Candidate Selection" table above with measured values for the user-chosen models.

### Phase C — Document
7. Write a reasoned final recommendation in the "Recommendation" section based on the measured data and criteria.
8. Update `docs/brain/BRAIN.md` — mark Step 0.1 as ✅ Complete.
9. Update `ROADMAP.md` — mark Step 0.1 as ✅ Complete in the Brain Phase 0 table.

**Operational rules:**
- Use venv: `/home/phalo/PH-Neuro/.venv/bin/python`
- No GPU needed for model loading and inspection (CPU-only)
- If a model requires authentication or fails to load, document it and skip
- All findings go into THIS file (`docs/brain/00-model-selection.md`)
