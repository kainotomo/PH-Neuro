# Step 0.1 — Model Selection

> **Status:** ✅ Complete (2026-08-12)
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

## Candidate Selection (surveyed 2026-08-12; chosen by the user)

**Phase A survey** — current HuggingFace data. All candidates below satisfied
the criteria (permissive license, 100M–3B, decoder-only causal LM):

| Family | HF ID | License | In-range sizes | Arch family | Notes |
|:-------|:------|:-------:|:--------------:|:------------|:------|
| GPT-2 | `openai-community/gpt2` | MIT | 124M/355M/774M/1.5B | GPT-2 classic | Most-studied baseline |
| GPT-Neo | `EleutherAI/gpt-neo-125m` | MIT | 125M/1.3B/2.7B | GPT-2-style (local attn) | |
| Pythia | `EleutherAI/pythia-160m` | Apache-2.0 | 160M–2.8B | GPT-NeoX (parallel attn) | Intermediate checkpoints |
| SmolLM2 | `HuggingFaceTB/SmolLM2-135M` | Apache-2.0 | 135M/360M/1.7B | LLaMA-modern | Active community |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B` | Apache-2.0 (0.5/1.5B); 3B = `other` ⚠️ | 0.5B/1.5B/3B | LLaMA-modern | 3B license caveat |
| Qwen3 | `Qwen/Qwen3-0.6B` | Apache-2.0 | 0.6B/1.7B | LLaMA-modern | Newest line |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache-2.0 | 1.1B | LLaMA-modern | |
| OLMo | `allenai/OLMo-1B` | Apache-2.0 | 1B | LLaMA-style | Open training data |
| H2O Danube | `h2oai/h2o-danube-1.8b-base` | Apache-2.0 | 1.8B | Mistral-style | |
| Granite | `ibm-granite/granite-3.0-2b-base` | Apache-2.0 | 2B | LLaMA-modern | |
| Phi-2 | `microsoft/phi-2` | MIT | 2.7B | LLaMA-style | |
| OPT | `facebook/opt-125m` | `other` ⚠️ | 125M–175B | GPT-2-style | License caveat |
| MPT-1B | `mosaicml/mpt-1b-redpajama-200b` | Apache-2.0 (base) ⚠️ | 1B | MPT (ALiBi) | Availability ⚠️ |
| **BitNet b1.58 2B4T** | `microsoft/bitnet-b1.58-2B-4T` | MIT | 2B | BitNet (native ternary) | Bench candidate |

**Excluded (restricted license or size):** Llama-3.2-1B, Gemma-3 270M/1B
(`llama3.2`/`gemma` — gated), StableLM-3B (`cc-by-sa-4.0`), OpenELM
(`apple-amlr`), Falcon (7B+/Mamba), Cerebras-GPT (originals not surfacing),
GPT-OSS / GLM-5.2 / DeepSeek (20B+).

**Chosen by the user (2026-08-12):** primary = **SmolLM2-1.7B** ·
generalization-test = **GPT-2 (124M)** · scaling = **SmolLM2 ladder (135M→1.7B)** ·
bench = **BitNet b1.58 2B4T** (surveyed only).

### Measured results (Phase B — CPU-only, bf16, seq 256)

| Model | License | Size (measured) | Baseline ppl | Hook ease (measured) | Variants? | Role |
|:------|:-------:|:----:|:----:|:----:|:----:|:----:|
| SmolLM2-1.7B | Apache-2.0 | **1,711M** (1.71B) | See tech report (arXiv:2502.02737) | ✅ 24/24 blocks; `pre[i]==hs[i]`, `post[i]==hs[i+1]` | 135M/360M/1.7B | **Primary** |
| GPT-2 (124M) | MIT | **124.4M** | ~29 (WikiText-2, documented) | ✅ 12/12 blocks; exact match | 124M–1.5B | **Gen-test** |
| SmolLM2-135M | Apache-2.0 | **134.5M** | — | ✅ 30/30 blocks | 135M/360M/1.7B | Scaling (small) |
| SmolLM2-360M | Apache-2.0 | **361.8M** | — | ✅ 32/32 blocks | 135M/360M/1.7B | Scaling (mid) |
| BitNet b1.58 2B4T | MIT | ~2B (survey only) | arXiv:2504.12285 | ⚠️ deferred — custom code + pinned transformers fork | 2B only | Bench (Phase 2) |

**Measured detail (2026-08-12, CPU):**

| Model | Params | Layers | Heads (kv) | d_model | d_ff | Vocab | Max ctx | Disk (HF cache) | Throughput (CPU) | Peak RSS |
|:------|-------:|:------:|:----------:|:-------:|:----:|:------:|:-------:|:---------------:|:----------------:|:--------:|
| SmolLM2-1.7B | 1,711.4M | 24 | 32 (32) | 2048 | 8192 | 49152 | 8192 | 6.4 GB | 54.9 tok/s | 4.1 GB |
| GPT-2 124M | 124.4M | 12 | 12 (−) | 768 | 3072 | 50257 | 1024 | 1.0 GB | 23.6 tok/s | 1.4 GB |
| SmolLM2-360M | 361.8M | 32 | 15 (5) | 960 | 2560 | 49152 | 8192 | 1.4 GB | 216.6 tok/s | 1.5 GB |
| SmolLM2-135M | 134.5M | 30 | 9 (3) | 576 | 1536 | 49152 | 8192 | 0.5 GB | 503.4 tok/s | 1.0 GB |

**Activation-interception verification (primary + gen-test):** forward
`pre_hook`/`hook` on every transformer block **and** `output_hidden_states=True`
both succeed. For all blocks `pre[i]` (block input) equals `hidden_states[i]`
and `post[i]` (block output) equals `hidden_states[i+1]`. Block paths:
`model.model.layers` (SmolLM2/LLaMA) and `model.transformer.h` (GPT-2). No
architectural quirks block interception. All three SmolLM2 tiers share one
architecture (`LlamaForCausalLM`) → a single Brain Wrapper implementation
generalizes across the scaling ladder.

**Inspection tool:** `research/scripts/inspect_brain_models.py` — reusable
CPU-only inspector used for all measurements above. Loads a model and reports
module tree, parameter count, HF-cache disk size, peak RSS during load +
inference, throughput (tok/s), and pre/post hook interception at every
transformer block. Usage: `.venv/bin/python research/scripts/inspect_brain_models.py <model_id> [--seq-len 256] [--iters 5]`.

---

## Recommendation (Final — 2026-08-12)

**Primary — SmolLM2-1.7B** (`HuggingFaceTB/SmolLM2-1.7B`, Apache-2.0).
Measured: 1,711M params / 24 LLaMA blocks / RoPE + GQA + SwiGLU; hooks verified
24/24 with exact pre/post capture. Best capability-in-range of the inspected
candidates, and matches the project's existing DQT transformer conventions
(RMSNorm, RoPE, GELU). Its 3-tier size ladder (135M/360M/1.7B) shares one
architecture, so the same Brain Wrapper code scales without re-plumbing.

**Generalization-test — GPT-2 124M** (`openai-community/gpt2`, MIT).
Measured: 124.4M params / 12 classic pre-norm blocks (learned positional
embeddings, no RoPE, no GQA); hooks verified 12/12. Maximally contrasts with
the primary (GPT-2-classic vs LLaMA-modern) → the strongest "does the
mechanism transfer?" test, plus the best-documented perplexity baseline
(WikiText-2 ~29).

**Scaling — SmolLM2 ladder (135M → 360M → 1.7B).** All three tiers inspected
and clean (30/32/24 blocks; 503/217/55 tok/s CPU). Enables the Phase 3
scaling-law question ("do larger models benefit more from plasticity?") with
constant architecture.

**Bench — BitNet b1.58 2B4T** (`microsoft/bitnet-b1.58-2B-4T`, MIT). Surveyed
only: a **native ternary {−1,0,+1}** backbone (~2B) — philosophically ideal
for PH-Neuro. Deferred to Phase 2 because it needs `trust_remote_code` + a
pinned transformers fork (not stock-loadable yet) and has no size ladder.
Re-inspect if it becomes the Phase 2/3 subject.

**Decision rationale vs criteria:** license ✅ (Apache-2.0/MIT) · size ✅ (all
load in <4.2 GB RAM on CPU; trivially fit the RTX 4060's 8 GB VRAM with room
for plastic weights + batch) · clean arch ✅ (standard embed → N× block → head;
hooks exact on every block) · known baseline ✅ (GPT-2 ~29 Wiki-2; SmolLM2 tech
report) · tokenizer ✅ (BPE; SmolLM2 vocab 49,152 / GPT-2 50,257 —
diverse-text capable).

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
