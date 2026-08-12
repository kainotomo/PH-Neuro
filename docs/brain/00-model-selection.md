# Step 0.1 — Model Selection

> **Status:** 🔬 In Progress
> **Goal:** Select the best pre-trained open-source model to serve as the "born brain" for the first Brain Wrapper experiment.

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

## Candidate Models

### GPT-2 Family (OpenAI, 2019)
| Variant | Params | Layers | Hidden | Heads | License |
|:--------|:------:|:------:|:------:|:-----:|:-------|
| GPT-2 Small | 124M | 12 | 768 | 12 | MIT |
| GPT-2 Medium | 355M | 24 | 1024 | 16 | MIT |
| GPT-2 Large | 774M | 36 | 1280 | 20 | MIT |

- **Architecture:** Classic decoder-only transformer, learned position embeddings, GELU activation, LayerNorm
- **WikiText-2 ppl:** ~29.4 (Small), ~22.8 (Medium)
- **Activation hooking:** `output_hidden_states=True` returns all block outputs. Individual block internals accessible via named modules.
- **Pros:** MIT license, widely studied, multiple sizes, simple architecture. The standard reference.
- **Cons:** Trained on web text only (2019), limited to 1024 context. Not the best perplexity per parameter.

### SmolLM2 Family (HuggingFace, 2024–2025)
| Variant | Params | Layers | Hidden | Heads | License |
|:--------|:------:|:------:|:------:|:-----:|:-------|
| SmolLM2-135M | 135M | 30 | 576 | 9 | Apache 2.0 |
| SmolLM2-360M | 360M | 32 | 960 | 15 | Apache 2.0 |
| SmolLM2-1.7B | 1.7B | 32 | 2048 | 32 | Apache 2.0 |

- **Architecture:** LLaMA-style, RoPE, SwiGLU, RMSNorm, grouped-query attention
- **Perplexity:** Better than GPT-2 at comparable size (modern training recipes, better data)
- **Activation hooking:** Similar to LLaMA — blocks accessible via `model.model.layers[i]`. Slightly more complex due to SwiGLU gating.
- **Pros:** Apache 2.0, modern architecture, better quality, multiple sizes. Good test of architectural generalization.
- **Cons:** Less studied than GPT-2. SwiGLU adds complexity for activation interception.

### TinyLlama (2024)
| Variant | Params | Layers | Hidden | Heads | License |
|:--------|:------:|:------:|:------:|:-----:|:-------|
| TinyLlama-1.1B | 1.1B | 22 | 2048 | 32 | Apache 2.0 |

- **Architecture:** LLaMA-style, RoPE, SwiGLU, RMSNorm
- **Perplexity:** ~15–18 on WikiText-2 (significantly better than GPT-2 Small)
- **Pros:** Apache 2.0, good quality, 2048 context. Good scaling target.
- **Cons:** 1.1B is at the upper end of what's comfortable on 8 GB for inference + plasticity. No smaller variant.

### Other Candidates (for completeness)

| Model | Params | License | Notes |
|:------|:------:|:-------:|:------|
| Qwen2.5-0.5B | 494M | Apache 2.0 | LLaMA-style, strong quality, multilingual. Good option. |
| Qwen2.5-1.5B | 1.5B | Apache 2.0 | Larger variant. |
| Phi-3-mini | 3.8B | MIT | Upper bound of size. Very high quality per parameter. |
| Gemma-2-2B | 2.6B | Gemma (permissive) | Google. Good quality but license has usage restrictions. |
| MobileLLM-125M | 125M | ? | Meta. Optimized for mobile. |
| OPT-125M | 125M | MIT | Meta. GPT-2 clone, but lower quality. |
| Pythia-160M | 160M | Apache 2.0 | EleutherAI. Well-documented training dynamics. Good for studying learning. |

---

## Decision Matrix (to be completed)

| Model | License | Size | PPL (Wiki-2) | Hook Ease | Variants | Verdict |
|:------|:------:|:----:|:-----------:|:---------:|:--------:|:-------:|
| GPT-2 Small | MIT ✅ | 124M | ~29.4 | Easy | 3 sizes | **Primary candidate** |
| SmolLM2-135M | Apache 2.0 ✅ | 135M | ~22-24 | Medium | 3 sizes | **Generalization test** |
| TinyLlama | Apache 2.0 ✅ | 1.1B | ~15-18 | Medium | 1 size | Scaling target |
| Qwen2.5-0.5B | Apache 2.0 ✅ | 494M | ~20-22 | Medium | 2 sizes | Strong alternative |
| Pythia-160M | Apache 2.0 ✅ | 160M | ~26-28 | Easy | 5 sizes | Training dynamics study |

---

## Recommendation (Preliminary)

**Primary:** GPT-2 Small (124M) — MIT license, simplest architecture, most studied, easy hooking. The standard reference. If Brain Wrapper works on GPT-2, it's immediately credible.

**Generalization test:** SmolLM2-135M — Apache 2.0, different architecture (LLaMA-style), better perplexity. Tests whether the method is architecture-agnostic.

**Scaling target:** GPT-2 Medium (355M) or SmolLM2-360M — tests whether larger models benefit more from plasticity.

---

## Next Steps

- [ ] Load each shortlisted model and inspect the forward pass structure
- [ ] Verify that activation interception works for pre/post at each transformer block
- [ ] Measure inference memory and throughput for each on RTX 4060
- [ ] Run baseline perplexity on WikiText-2 and candidate target domains (PubMed)
- [ ] Finalize decision and document rationale
