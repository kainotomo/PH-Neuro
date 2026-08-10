#!/usr/bin/env python3
"""PH-Neuro M2.5 — Public demo: DQT ternary models in the browser (Gradio).

The PH-Neuro launch demo. Loads the three M2.5 models (DQT Transformer 102M,
DQT CNN CIFAR-10, DQT CNN CIFAR-100) as **ONNX + onnxruntime** (no PyTorch
needed for inference) and runs entirely on CPU — a smartphone-class deploy.

Three tabs:

    Tab 1 — 📝 Text Generation     GPT-2 BPE tokenizer + DQT Transformer 102M
    Tab 2 — 🖼️ Image Classification CIFAR-10 / CIFAR-100 DQT CNNs (upload/webcam)
    Tab 3 — 📊 Benchmarks           params / size / accuracy / ppl vs GPT-2 & TF Lite

Usage::

    .venv/bin/python scripts/run_m2_5_demo.py
    .venv/bin/python scripts/run_m2_5_demo.py --onnx-dir results/phase2/m2_5 --port 7860

The ``--onnx-dir`` defaults to ``results/phase2/m2_5`` where the M2.5 export
pipeline writes ``text_model.onnx``, ``vision_cifar10.onnx`` and
``vision_cifar100.onnx`` (plus their 2-bit ``.ternary`` companions, used for
the size shown in the status bars).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_ONNX_DIR = "results/phase2/m2_5"
DEFAULT_PORT = 7860
DEFAULT_SERVER_NAME = "0.0.0.0"

TEXT_ONNX = "text_model.onnx"
TEXT_PACKED = "text_model.ternary"
CIFAR10_ONNX = "vision_cifar10.onnx"
CIFAR10_PACKED = "vision_cifar10.ternary"
CIFAR100_ONNX = "vision_cifar100.onnx"
CIFAR100_PACKED = "vision_cifar100.ternary"

# GPT-2 BPE special tokens (matches ph_neuro generate_text / tinystories).
EOT_ID = 50256  # <|endoftext|>
PAD_ID = 0  # right-padding id (real tokens never attend to pads)

# CIFAR-10 classes (torchvision order).
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-100 fine label names (torchvision order, alphabetical).
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

# Per-class normalization (torchvision mean/std used at training time).
_NORM = {
    "cifar10": (
        np.array([0.4914, 0.4822, 0.4465], dtype=np.float32),
        np.array([0.2470, 0.2435, 0.2616], dtype=np.float32),
    ),
    "cifar100": (
        np.array([0.5071, 0.4867, 0.4408], dtype=np.float32),
        np.array([0.2675, 0.2565, 0.2761], dtype=np.float32),
    ),
}

MB = 1024 * 1024

# ── ONNX session cache ──────────────────────────────────────────────

_sessions: dict[str, object] = {}
_tokenizer = None


def _require(onnx_dir: str, fname: str) -> str:
    """Resolve ``fname`` under ``onnx_dir`` or raise a helpful error."""
    path = os.path.join(onnx_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model file: {path}\n\n"
            "Run the M2.5 export pipeline first, e.g.:\n"
            "  bash scripts/run_m2_5_demo.sh export\n"
            "  # or export the 3 models manually (see docs/export_guide.md)"
        )
    return path


def _get_session(onnx_dir: str, fname: str) -> object:
    """Return a lazily-created onnxruntime session (CPU)."""
    key = os.path.join(onnx_dir, fname)
    if key not in _sessions:
        path = _require(onnx_dir, fname)
        _sessions[key] = _inference_session(path)
    return _sessions[key]


def _inference_session(path: str):
    """Build an onnxruntime session preferring the CPUExecutionProvider."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    # Prefer a good per-session thread count on big machines; keep the
    # default (all cores) — onnxruntime already picks a sensible value.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _file_mb(onnx_dir: str, fname: str) -> float:
    """File size in MB, or 0.0 if the file is absent."""
    path = os.path.join(onnx_dir, fname)
    if os.path.exists(path):
        return os.path.getsize(path) / MB
    return 0.0


def _packed_mb_label(onnx_dir: str, fname: str, onnx_fname: str) -> str:
    """Human status-bar size label: packed (2-bit) size, fallback to ONNX."""
    packed = _file_mb(onnx_dir, fname)
    onnx = _file_mb(onnx_dir, onnx_fname)
    if packed > 0:
        return f"{packed:.1f} MB (2-bit)"
    return f"{onnx:.1f} MB"


# ── GPT-2 tokenizer ────────────────────────────────────────────────


def _get_tokenizer():
    """Lazily build the GPT-2 BPE tokenizer (tiktoken)."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from ph_neuro.training.tinystories import make_gpt2_tokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ph_neuro is not importable — run `pip install -e .` first"
            ) from exc
        _tokenizer = make_gpt2_tokenizer()
    return _tokenizer


# ── Tab 1: Text generation ─────────────────────────────────────────


def _predict_row(session, ids: np.ndarray, last_pos: int, temperature: float) -> np.ndarray:
    """One forward pass, returning the logits row of the last real token."""
    logits = session.run(None, {"input": ids})[0]  # (1, ctx, vocab)
    row = logits[0, last_pos, :].astype(np.float64)
    return row / max(temperature, 1e-6)


def _top_k_filter(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Mask all but the top-k logits to -inf (in place on a copy)."""
    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[0])
        cutoff = np.partition(logits, -k)[-k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    return logits


def generate_text_stream(
    onnx_dir: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
):
    """Generate text with the DQT Transformer (ONNX, CPU), streaming tokens.

    Yields ``(partial_text, status)`` tuples so the demo streams live.
    """
    tokenizer = _get_tokenizer()
    try:
        session = _get_session(onnx_dir, TEXT_ONNX)
    except FileNotFoundError as exc:
        yield (
            f"❌ {exc}\n\n"
            "The text model is not built yet. Run:\n"
            "  bash scripts/run_m2_5_demo.sh export\n"
            "(needs the trained checkpoint, ~2 h on an RTX 4060).",
            "🖥️ CPU | ❌ text model missing",
        )
        return
    ctx_len = session.get_inputs()[0].shape[1]

    prompt = (prompt or "").strip() or "Once upon a time, a little"
    prompt_ids = tokenizer.encode(prompt)
    seq: list[int] = list(prompt_ids)

    # Warmup (one-time init cost excluded from the timing).
    warm = np.array(
        [seq[-ctx_len:] + [PAD_ID] * (ctx_len - min(len(seq), ctx_len))],
        dtype=np.int64,
    )
    _predict_row(session, warm, min(len(seq), ctx_len) - 1, temperature)

    start = time.time()
    n_generated = 0
    status_fmt = "⚡ {tok_s:.1f} tok/s | 💾 {size} | 🖥️ CPU"
    size_label = _packed_mb_label(onnx_dir, TEXT_PACKED, TEXT_ONNX)
    yield tokenizer.decode(seq), status_fmt.format(tok_s=0.0, size=size_label)

    for _ in range(max_tokens):
        window = seq[-ctx_len:]
        pad = ctx_len - len(window)
        ids = np.array([window + [PAD_ID] * pad], dtype=np.int64)
        last_pos = len(window) - 1

        logits = _predict_row(session, ids, last_pos, temperature)
        logits = _top_k_filter(logits, top_k)

        # Softmax + sample.
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        next_id = int(np.random.choice(probs.shape[0], p=probs))
        seq.append(next_id)
        n_generated += 1

        elapsed = time.time() - start
        tok_s = n_generated / elapsed if elapsed > 0 else 0.0
        yield (
            tokenizer.decode(seq),
            status_fmt.format(tok_s=tok_s, size=size_label),
        )
        if next_id == EOT_ID:
            break


def generate_text_full(
    onnx_dir: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
):
    """Non-streaming wrapper (returns the final text + status + info)."""
    last_text = ""
    last_status = ""
    for text, status in generate_text_stream(onnx_dir, prompt, max_tokens, temperature, top_k):
        last_text, last_status = text, status
    return last_text, last_status


# ── Tab 2: Image classification ────────────────────────────────────


def _preprocess_image(image: np.ndarray, dataset: str) -> np.ndarray:
    """Resize a webcam/uploaded image to 32x32 and apply training-time norm."""
    from PIL import Image

    img = Image.fromarray(image).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean, std = _NORM[dataset]
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]  # (1, 3, 32, 32)


def classify_image(onnx_dir: str, image: np.ndarray, dataset: str):
    """Classify an image and return (label_html, top3_figure, status)."""
    if image is None:
        return "Upload an image first.", None, "🖥️ CPU | ⏳ waiting"
    if dataset not in ("cifar10", "cifar100"):
        dataset = "cifar10"

    classes = CIFAR10_CLASSES if dataset == "cifar10" else CIFAR100_CLASSES
    onnx_fname = CIFAR10_ONNX if dataset == "cifar10" else CIFAR100_ONNX
    packed_fname = CIFAR10_PACKED if dataset == "cifar10" else CIFAR100_PACKED

    session = _get_session(onnx_dir, onnx_fname)
    x = _preprocess_image(image, dataset)

    start = time.time()
    logits = session.run(None, {"input": x})[0][0]
    elapsed_ms = (time.time() - start) * 1000.0

    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    top3 = np.argsort(probs)[::-1][:3]

    pred_idx = int(top3[0])
    conf = float(probs[pred_idx])
    label_html = (
        f"## 🏆 {classes[pred_idx].replace('_', ' ').title()}\n\n"
        f"Confidence: **{conf:.1%}**  (model: {dataset.upper()})"
    )

    figure = _top3_barplot(
        [(classes[int(i)].replace("_", " ").title(), float(probs[int(i)])) for i in top3]
    )

    size_label = _packed_mb_label(onnx_dir, packed_fname, onnx_fname)
    status = f"⚡ {elapsed_ms:.1f} ms/image | 💾 {size_label} | 🖥️ CPU"
    return label_html, figure, status


def _top3_barplot(items):
    """Build a horizontal top-3 bar chart with matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [it[0] for it in items][::-1]
    vals = [it[1] for it in items][::-1]

    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=110)
    colors = ["#3b82f6", "#60a5fa", "#93c5fd"]
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("Top-3 predictions")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ── Tab 3: Benchmarks ──────────────────────────────────────────────


def _read_result_json(onnx_dir: str, fname: str) -> dict:
    """Best-effort read of a result JSON (empty dict if missing)."""
    import json

    path = os.path.join(onnx_dir, fname)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def build_benchmarks(onnx_dir: str):
    """Build the benchmark table (models + comparison baselines).

    Returns a ``pandas.DataFrame`` (the type ``gr.Dataframe`` expects).
    """
    import pandas as pd

    columns = ["Model", "Params", "Packed 2-bit", "ONNX", "Accuracy / PPL", "Train time"]

    text_res = _read_result_json(
        onnx_dir, "text_model/results_m2_1_dqt_transformer_lr0.01_seed42.json"
    )
    c10_res = _read_result_json(
        onnx_dir, "vision_cifar10/results_dqt_cifar10_lr0.01_seed42.json"
    )
    c100_res = _read_result_json(
        onnx_dir, "vision_cifar100/results_dqt_cifar100_lr0.01_seed42.json"
    )

    text_ppl = text_res.get("best_val_ppl")
    text_acc = f"ppl {text_ppl:.1f}" if isinstance(text_ppl, (int, float)) else "ppl —"
    # The transformer run used pause/resume, so the JSON's
    # `training_time_seconds` only covers the final resume segment. Report the
    # steady-state full-run estimate instead: 48,708 steps at ~0.145 s/step on
    # an RTX 4060 (measured in the M2.1 full run) ≈ 2 h.
    text_steps = text_res.get("steps_trained")
    if isinstance(text_steps, (int, float)) and text_steps > 0:
        text_time_s = f"{text_steps * 0.145 / 3600:.1f} h"
    else:
        text_time_s = "~2 h"

    c10_acc = c10_res.get("best_accuracy")
    c10_acc_s = f"{100 * c10_acc:.1f}%" if isinstance(c10_acc, (int, float)) else "79%"
    c10_time = c10_res.get("training_time_seconds")
    c10_time_s = f"{c10_time / 60:.1f} min" if isinstance(c10_time, (int, float)) else "~10 min"

    c100_acc = c100_res.get("best_accuracy")
    c100_acc_s = f"{100 * c100_acc:.1f}%" if isinstance(c100_acc, (int, float)) else "54%"
    c100_time = c100_res.get("training_time_seconds")
    c100_time_s = f"{c100_time / 60:.1f} min" if isinstance(c100_time, (int, float)) else "~20 min"

    rows = [
        [
            "🟢 DQT Transformer 102M (PH-Neuro)",
            "102.3M ternary",
            f"{_file_mb(onnx_dir, TEXT_PACKED):.1f} MB",
            f"{_file_mb(onnx_dir, TEXT_ONNX):.0f} MB",
            text_acc,
            text_time_s,
        ],
        [
            "🟢 DQT CNN CIFAR-10 (PH-Neuro)",
            "4.3M ternary",
            f"{_file_mb(onnx_dir, CIFAR10_PACKED):.2f} MB",
            f"{_file_mb(onnx_dir, CIFAR10_ONNX):.1f} MB",
            c10_acc_s,
            c10_time_s,
        ],
        [
            "🟢 DQT CNN CIFAR-100 (PH-Neuro)",
            "2.5M ternary",
            f"{_file_mb(onnx_dir, CIFAR100_PACKED):.2f} MB",
            f"{_file_mb(onnx_dir, CIFAR100_ONNX):.1f} MB",
            c100_acc_s,
            c100_time_s,
        ],
        [
            "⚪ GPT-2 small (reference)",
            "124M float",
            "n/a",
            "~500 MB",
            "ppl 29 (1.5B tokens)",
            "days",
        ],
        [
            "⚪ MobileNetV2 / TF-Lite (reference)",
            "3.4M float",
            "n/a",
            "~14 MB",
            "CIFAR-10 ~91%",
            "hours",
        ],
    ]
    return pd.DataFrame(rows, columns=columns)


# ── App ────────────────────────────────────────────────────────────


def build_app(onnx_dir: str):
    """Build the 3-tab Gradio app."""
    import gradio as gr

    onnx_dir = os.path.abspath(onnx_dir)

    # Header (shared).
    with gr.Blocks(
        title="PH-Neuro — The World's Smallest Deep Learning Models",
    ) as demo:
        gr.Markdown(
            "# 🧠 PH-Neuro\n"
            "### The world's smallest deep learning models — ternary weights, "
            "2-bit packed, running 100% on your CPU.\n"
            f"*Models: `{onnx_dir}` · Inference: ONNX Runtime · Backend: CPU*\n"
            "---"
        )

        with gr.Tab("📝 Text Generation"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="Once upon a time, a little",
                        lines=2,
                        placeholder="Start a story…",
                    )
                    with gr.Row():
                        max_tokens = gr.Slider(10, 200, value=80, step=10, label="Max tokens")
                        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                        top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                    generate_btn = gr.Button("✨ Generate", variant="primary")
                    status_text = gr.Markdown("⚡ -- tok/s | 💾 -- | 🖥️ CPU")
                with gr.Column(scale=4):
                    output_text = gr.Textbox(label="Generated text", lines=16, interactive=False)

            generate_btn.click(
                fn=generate_text_stream,
                inputs=[gr.State(onnx_dir), prompt, max_tokens, temperature, top_k],
                outputs=[output_text, status_text],
            )

        with gr.Tab("🖼️ Image Classification"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_choice = gr.Radio(
                        ["cifar10", "cifar100"],
                        value="cifar10",
                        label="Model",
                        info="DQT CNN CIFAR-10 or CIFAR-100",
                    )
                    image_input = gr.Image(
                        label="Upload an image or use your webcam",
                        type="numpy",
                        sources=["upload", "webcam"],
                    )
                    classify_btn = gr.Button("🔍 Classify", variant="primary")
                    status_img = gr.Markdown("🖥️ CPU | ⏳ waiting")
                with gr.Column(scale=3):
                    label_out = gr.Markdown("Upload an image to classify.")
                    top3_plot = gr.Plot(label="Top-3 predictions")

            classify_btn.click(
                fn=classify_image,
                inputs=[gr.State(onnx_dir), image_input, model_choice],
                outputs=[label_out, top3_plot, status_img],
            )

        with gr.Tab("📊 Benchmarks"):
            gr.Markdown(
                "### PH-Neuro vs the world\n"
                "| Metric | PH-Neuro DQT | Conventional |\n"
                "|---|---|---|"
            )
            bench_table = gr.Dataframe(
                headers=["Model", "Params", "Packed 2-bit", "ONNX", "Accuracy / PPL", "Train time"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
            )
            refresh_btn = gr.Button("🔄 Refresh")
            refresh_btn.click(fn=build_benchmarks, inputs=[gr.State(onnx_dir)], outputs=[bench_table])

            gr.Markdown(
                "---\n"
                "**Key takeaway:** 102M parameters in **25 MB** (2-bit packed) — "
                "20× smaller than GPT-2's 500 MB, trained in ~2 h on one RTX 4060."
            )

        # Load the benchmark table once at startup.
        demo.load(fn=build_benchmarks, inputs=[gr.State(onnx_dir)], outputs=[bench_table])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PH-Neuro M2.5 — public Gradio demo (CPU, onnxruntime).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR,
                        help="Directory with the M2.5 .onnx / .ternary files.")
    parser.add_argument("--server-name", default=DEFAULT_SERVER_NAME,
                        help="Bind address (0.0.0.0 = share on LAN).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port to serve the demo on.")
    parser.add_argument("--share", action="store_true",
                        help="Create a temporary public shareable link.")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "gradio is required — install it with: .venv/bin/pip install gradio"
        ) from exc

    # Fail fast if no models are present at all.
    onnx_dir = os.path.abspath(args.onnx_dir)
    present = [f for f in (TEXT_ONNX, CIFAR10_ONNX, CIFAR100_ONNX) if os.path.exists(os.path.join(onnx_dir, f))]
    if not present:
        raise SystemExit(
            f"No M2.5 ONNX models found in {onnx_dir}.\n"
            "Run the export pipeline first (see docs/export_guide.md)."
        )
    print(f"Found models: {', '.join(present)}")

    demo = build_app(onnx_dir)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
