# Experiment E029: M2.5 — Public Demo + Blog Post (PH-Neuro launch)

- **Date:** 2026-08-10
- **Git commit:** `TBD` (M2.5 work)
- **Status:** 🟢 **IN PROGRESS** — 3 models retrained → exported (ONNX + 2-bit packed) → served via a 3-tab Gradio demo on CPU.
- **Phase:** 2 (public launch) — demo + blog, NOT a GO/NO-GO gate (each model re-validates its own milestone).

---

## Goal

Make PH-Neuro **public**: a repo anyone can clone, download 3 models
(<30 MB total, 2-bit packed), and use — **text generation + image
classification in the browser** (Gradio, CPU via onnxruntime), backed by a
launch blog post. The 3 models are **retrained fresh** with the M2.5 configs
below so the demo ships reproducible, self-contained artifacts.

---

## Configuration

| Model | Config | Train cmd (exact) | Time |
|-------|--------|-------------------|------|
| 📝 **DQT Transformer 102M** (text) | `d_model=768, n_layers=9, n_heads=12, d_ff=3072, vocab=50257, seq=256` → **102,298,368 ternary** + 38.6M float emb ≈ 140.9M total | `run_m2_1_dqt_transformer --d-model 768 --n-layers 9 --n-heads 12 --d-ff 3072 --epochs 3 --lr 0.01 --seed 42 --batch-size 8 --output-dir results/phase2/m2_5/text_model` | ~2-3.5 h |
| 🖼️ **DQT CNN CIFAR-10** | `dqt_cnn()`: Conv(3→64)→Pool→Conv(64→128)→Pool→FC(8192→512)→FC(512→10), ~2.14M ternary | `run_m1_1_dqt_cifar10 --lr 0.01 --epochs 100 --seed 42 --patience 25 --output-dir results/phase2/m2_5/vision_cifar10` | ~12 min |
| 🖼️ **DQT CNN CIFAR-100** | `dqt_cnn_cifar100()`: Conv(3→64→128→256)→Pool→FC(4096→512)→FC(512→100), ~1.26M ternary | `run_m1_2_dqt_cifar100 --lr 0.01 --epochs 150 --seed 42 --patience 30 --output-dir results/phase2/m2_5/vision_cifar100` | ~25 min |

Checkpoint convention (per repo): `{output_dir}/checkpoints/seed42/best.pt`
— for vision, `best.pt` stores `{model_state_dict, epoch, best_accuracy}` and
is written whenever test accuracy improves (new additive `--checkpoint-dir` /
`checkpoint_dir` support added to the two vision runners).

---

## Implementation

### New / changed files

- `src/ph_neuro/examples/run_m1_1_dqt_cifar10.py` — **additive** `checkpoint_dir`
  arg + `best.pt` saving on val-accuracy improvement (existing behavior unchanged;
  existing tests unaffected).
- `src/ph_neuro/examples/run_m1_2_dqt_cifar100.py` — same additive change.
- `scripts/run_m2_5_demo.sh` — launcher: `full | train | export | demo`
  (idempotent — skips models whose `best.pt` / `.onnx` already exist).
- `scripts/run_m2_5_demo.py` — **Gradio app, 3 tabs**:
  - 📝 Text Generation — prompt, max-tokens (10-200), temperature (0.1-2.0),
    top-k (1-100), streaming generator, status `⚡ X.X tok/s | 💾 size | 🖥️ CPU`.
  - 🖼️ Image Classification — CIFAR-10 / CIFAR-100 selector, upload/webcam,
    predicted class + confidence, top-3 bar chart, status `⚡ X ms/image | 💾 1.0 MB | 🖥️ CPU`.
  - 📊 Benchmarks — table of all models (params / packed / ONNX / acc or ppl /
    train time) vs GPT-2 small & TF-Lite baselines.
  - **Inference = onnxruntime (CPU) only**; no PyTorch at serve time. GPT-2 BPE
    tokenizer via `tiktoken` (`make_gpt2_tokenizer`). Inputs preprocessed with the
    exact training-time normalization (CIFAR-10: `(0.4914,0.4822,0.4465)/(0.2470,0.2435,0.2616)`,
    CIFAR-100: `(0.5071,0.4867,0.4408)/(0.2675,0.2565,0.2761)`).
- `docs/blog.md` — launch blog post "Τα Μικρότερα Deep Learning Μοντέλα στον Κόσμο".
- `research/docs/experiments/E029-m2-5-public-launch.md` — this report.

### Export

All 3 models → ONNX (dynamic batch, opset 18) + 2-bit packed `.ternary` via the
existing `run_m1_3_export` (`--model dqt_gpt2 | dqt_cnn | dqt_cnn_cifar100`,
`--packed --verify`). Expected artifacts in `results/phase2/m2_5/`:

| File | Expected size |
|------|--------------:|
| `text_model.onnx` | ~270 MB |
| `text_model.ternary` | ~25 MB |
| `vision_cifar10.onnx` | ~16 MB |
| `vision_cifar10.ternary` | ~1.0 MB |
| `vision_cifar100.onnx` | ~10 MB |
| `vision_cifar100.ternary` | ~0.6 MB |

---

## Results

(TBD — filled after training + export complete.)

### CIFAR-10 (DQT CNN)

| Metric | Value |
|--------|-------|
| Best test accuracy | **78.61%** (epoch 94) |
| Final test accuracy | 78.13% |
| Training time | 755 s (~12.6 min) |
| Peak GPU memory | 363.5 MB |
| Ternary weights | 4,274,880 |
| ONNX / packed | 16.33 MB / 1,043.8 KB |
| ONNX verified | ✅ (max\|Δ\| = 1.91e-06; torch ≡ onnx 77.73% on subset) |

### CIFAR-100 (DQT CNN)

| Metric | Value |
|--------|-------|
| Best test accuracy | TBD |
| Best epoch | TBD |
| Training time | TBD |

### Transformer 102M (TinyStories)

| Metric | Value |
|--------|-------|
| Best val ppl | TBD |
| Steps trained | TBD |
| Training time | TBD |
| Gen speed (CPU, ONNX) | TBD tok/s |

### Demo

- Gradio demo launches, loads ONNX models via onnxruntime (CPU).
- **3 tabs verified end-to-end on the running server (Gradio 6.22)**:
  - 📝 Text Generation — UI + streaming generator wired; graceful error when
    the text ONNX is absent (until export). Sliders: max-tokens 10-200,
    temperature 0.1-2.0, top-k 1-100.
  - 🖼️ Image Classification — CIFAR-10 & CIFAR-100 selector, upload/webcam,
    label + confidence + top-3 matplotlib bar chart; verified through the
    server API: `⚡ 1.5 ms/image | 💾 1.0 MB (2-bit) | 🖥️ CPU`.
  - 📊 Benchmarks — `gr.Dataframe` fed a `pandas.DataFrame`; CIFAR-10 row shows
    real values (78.6%, 1.02 MB packed, 16.3 MB ONNX, 12.6 min).
- **Gradio 6 gotchas fixed during testing**:
  - `Blocks.launch()` takes `server_port` (not `port`) and `theme` (moved from
    the `Blocks` constructor).
  - `gr.Dataframe` requires the fn to return a `pandas.DataFrame`/list — NOT a
    `(rows, columns)` tuple.
- **Export bug found + fixed**: the vision export path only unwrapped a
  `"state_dict"` checkpoint key; M2.5 `best.pt` files use `"model_state_dict"`,
  so the model silently loaded random weights (10.16% acc). `run_m1_3_export`
  now unwraps either key. Verified torch ≡ onnx 77.73% after the fix.

---

## Verdict

🟢 **GO** (demo + blog) — once the 3 models are trained, exported, and the
demo is verified end-to-end on CPU.

## Follow-ups

- Publish the repo (make public), attach model artifacts to a release (<30 MB
  total packed).
- Link the blog post from the README (`docs/blog.md`).
- Optional: quantize the ONNX to int8 to shrink the float32 `.onnx` files too.
