"""BrainWrapper — wrap a frozen CausalLM with local surprise-modulated plasticity.

Implements the locked Step 0.4 API surface for Phase 1.1 (E031):

    model = AutoModelForCausalLM.from_pretrained("<model id>")
    brain = BrainWrapper(model)
    brain.learn(texts, steps=1000)   # adapt plastic weights, no backprop
    brain.generate(prompt)           # use model + plastic weights
    brain.save("my_brain.pt")        # save plastic weights only

Mechanism recap (see ``docs/brain/03-architecture.md``):

* Tiny float32 vector biases are injected at each block's output projections
  (``o_proj`` + ``down_proj`` for SmolLM2 / ``attn.c_proj`` + ``mlp.c_proj``
  for GPT-2) via **output-modification forward hooks**.
* Update rule (3-factor Hebbian): ``Δb = η·M·mean_t(post)`` where ``M`` is a
  global float32 surprise scalar from :class:`SurpriseModulator`.
* **No backprop**: the training loop runs under ``torch.no_grad()`` and only
  does local tensor adds.
* **Identity guarantee (I1)**: with plasticity disabled (or all biases zero)
  the wrapped model is bit-identical to the frozen unwrapped model.
"""

from __future__ import annotations

import contextlib
import glob
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.brain.block_wrappers import (
    InjectionPoint,
    get_block_container,
    get_block_wrapper,
)
from ph_neuro.brain.modulator import SurpriseModulator

logger = logging.getLogger("ph_neuro.brain")

CHECKPOINT_FORMAT = "ph_neuro_brain_checkpoint"
SAVE_FORMAT = "ph_neuro_brain_save"
CHECKPOINT_VERSION = 1


# ── GPU helpers (module-level so the runner can gate before model load) ──


def gpu_free_mb() -> int | None:
    """Free GPU memory in MiB (from ``nvidia-smi``), or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - nvidia-smi can be missing/broken
        return None


def check_gpu_free(
    min_free_gb: float, policy: str = "exit", log: logging.Logger = logger
) -> None:
    """Gate on free GPU memory per the operational spec.

    Args:
        min_free_gb: minimum free GiB required.
        policy: ``"exit"`` (default) exits with code 1 for a supervisor to
            retry; ``"wait"`` polls every 60 s; ``"warn"`` logs and proceeds.
    """
    free_mb = gpu_free_mb()
    if free_mb is None:
        log.warning("nvidia-smi unavailable — skipping GPU memory check")
        return
    free_gb = free_mb / 1024.0
    if free_gb >= min_free_gb:
        log.info("GPU check OK: %.2f GiB free (need >= %.2f)", free_gb, min_free_gb)
        return
    msg = f"only {free_gb:.2f} GiB free, need >= {min_free_gb:.2f} GiB"
    if policy == "exit":
        log.warning("%s — exiting (supervisor will retry)", msg)
        sys.exit(1)
    if policy == "wait":
        while free_gb < min_free_gb:
            log.warning("%s — waiting 60 s", msg)
            time.sleep(60)
            free_mb = gpu_free_mb()
            if free_mb is None:
                return
            free_gb = free_mb / 1024.0
        log.info("GPU check OK (after wait): %.2f GiB free", free_gb)
        return
    if policy == "warn":
        log.warning("%s — proceeding anyway (risk accepted)", msg)
        return
    raise ValueError(f"gpu_policy={policy!r} not in ('exit', 'wait', 'warn')")


def _auto_min_free_gb(model: nn.Module) -> float:
    """Auto default GPU gate: clamp(params_bytes_bf16·2.2/1e9, 2.0, 8.0)."""
    n = sum(p.numel() for p in model.parameters())
    bytes_bf16 = n * 2
    return max(2.0, min(8.0, bytes_bf16 * 2.2 / 1e9))


# ── data helpers ───────────────────────────────────────────────────


def tokenize_list(texts, tokenizer, *, seq_truncate: int | None = None) -> torch.Tensor:
    """Tokenize a list of strings into a flat LongTensor (no special tokens)."""
    ids: list[int] = []
    for text in texts:
        if not text:
            continue
        toks = tokenizer(text, add_special_tokens=False).input_ids
        if seq_truncate is not None:
            toks = toks[:seq_truncate]
        ids.extend(toks)
    return torch.tensor(ids, dtype=torch.long)


def cyclic_batch_iter(
    tokens: torch.Tensor, batch_size: int, seq_len: int, seed: int | None
):
    """Yield ``{"input_ids": (B,S), "attention_mask": (B,S)}`` batches forever.

    Tokens are packed into non-overlapping ``seq_len`` blocks; block order is
    shuffled deterministically with ``seed`` and cycled. Batches shorter than
    ``batch_size`` are wrapped around (a partial final batch is dropped only
    if it is entirely empty).
    """
    if tokens.numel() == 0:
        raise ValueError("empty token stream for batch iterator")
    n_blocks = tokens.numel() // seq_len
    if n_blocks == 0:
        # Not even one full block: pad by cycling the stream.
        full = tokens.repeat((seq_len // tokens.numel()) + 1)
        tokens = full
        n_blocks = tokens.numel() // seq_len
    blocks = tokens[: n_blocks * seq_len].view(n_blocks, seq_len)
    order = list(range(n_blocks))
    if seed is not None:
        rng = torch.Generator().manual_seed(int(seed) & 0xFFFFFFFF)
        perm = torch.randperm(n_blocks, generator=rng).tolist()
        order = perm
    pos = 0
    while True:
        batch_blocks = []
        while len(batch_blocks) < batch_size:
            batch_blocks.append(blocks[order[pos % n_blocks]])
            pos += 1
        ids = torch.stack(batch_blocks, dim=0)  # (B,S)
        yield {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ── BrainWrapper ───────────────────────────────────────────────────


class BrainWrapper:
    """Wrap a frozen pre-trained CausalLM with local plasticity (Step 0.4 spec).

    The frozen backbone never changes; tiny plastic bias vectors are injected
    at each block's ``o_proj``+``down_proj`` (SmolLM2) /
    ``attn.c_proj``+``mlp.c_proj`` (GPT-2) and updated from locally captured
    activations and a global surprise scalar. No backprop.
    """

    def __init__(
        self,
        model: nn.Module,
        plasticity: str = "vector_bias",
        modulator_cfg: dict | None = None,
        *,
        lr: float = 1e-3,  # η
        decay_rate: float = 0.0,  # λ; 0.0 = off
        dtype: torch.dtype = torch.float32,
        tokenizer=None,
        device=None,
        checkpoint_dir: str | None = None,  # None → checkpointing disabled
        checkpoint_every: int = 100,  # N steps between checkpoints
        min_free_gb: float | None = None,  # GPU gate; auto from model size
        log: logging.Logger | None = None,
    ) -> None:
        if plasticity != "vector_bias":
            raise NotImplementedError(
                f"plasticity={plasticity!r} not implemented (Phase 1.1 = 'vector_bias')"
            )
        self.model = model
        self.plasticity = plasticity
        self.lr = float(lr)
        self.decay_rate = float(decay_rate)
        self.dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = int(checkpoint_every)
        self.log = log if log is not None else logger

        # device
        if device is None:
            p = next(model.parameters(), None)
            self.device = torch.device(p.device) if p is not None else torch.device("cpu")
        else:
            self.device = torch.device(device)

        # frozen + eval mode (I4: dropout is inert; no grad ever)
        self.model.eval()
        self.model.requires_grad_(False)

        # injection points
        self.block_wrapper = get_block_wrapper(model)
        self.container = get_block_container(model)
        self._injection_points: list[InjectionPoint] = []
        for i, block in enumerate(self.container):
            self._injection_points.extend(self.block_wrapper.get_injection_points(block, i))
        for ip in self._injection_points:
            ip.bias = ip.bias.to(self.device)
        self.log.info(
            "BrainWrapper: %d blocks, %d injection points, %d plastic params (%.1f KB fp32)",
            len(self.container),
            len(self._injection_points),
            self.plastic_parameter_count(),
            self.plastic_memory_bytes() / 1024,
        )

        # modulator
        self.modulator_cfg = dict(modulator_cfg or {})
        self.modulator = SurpriseModulator.from_config(self.modulator_cfg)

        # hook state
        self._active = True  # bias injection on/off
        self._capture = False  # capture pre/post activations (learn only)
        self._last_pre: dict[str, torch.Tensor] = {}
        self._last_post: dict[str, torch.Tensor] = {}
        self._register_hooks()

        # tokenizer (optional — only for learn(list[str]) / generate)
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = self._resolve_tokenizer()

        # GPU gate
        self.min_free_gb = (
            float(min_free_gb) if min_free_gb is not None else _auto_min_free_gb(model)
        )

    # ── tokenizer ───────────────────────────────────────────────────

    def _resolve_tokenizer(self):
        try:
            tok = self.model.get_tokenizer()
            if tok is not None:
                return tok
        except Exception:  # noqa: BLE001 - API may not exist on the model
            pass
        name = getattr(self.model.config, "_name_or_path", None)
        if name is None:
            raise ValueError(
                "cannot resolve a tokenizer; pass tokenizer=... to BrainWrapper"
            )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)

    # ── hooks ───────────────────────────────────────────────────────

    def _make_pre_hook(self, ip: InjectionPoint):
        def pre_hook(module, args):
            if self._capture:
                x = args[0]
                self._last_pre[ip.name] = x.detach().to(torch.float32)
            return None

        return pre_hook

    def _make_post_hook(self, ip: InjectionPoint):
        def post_hook(module, args, output):
            out = output
            if self._active and ip.bias.ne(0).any():
                out = output + ip.bias.to(output.dtype)
            if self._capture:
                self._last_post[ip.name] = out.detach().to(torch.float32)
            return out

        return post_hook

    def _register_hooks(self) -> None:
        for ip in self._injection_points:
            ip.pre_handle = ip.module.register_forward_pre_hook(self._make_pre_hook(ip))
            ip.post_handle = ip.module.register_forward_hook(self._make_post_hook(ip))

    # ── plasticity context ──────────────────────────────────────────

    @contextlib.contextmanager
    def without_plasticity(self):
        """Temporarily disable bias injection (hooks return output unchanged).

        Nested use is safe (the previous state is restored on exit). This is
        the mechanism for frozen-baseline eval: the wrapped model *is* the
        frozen model by construction.
        """
        prev = self._active
        self._active = False
        try:
            yield self
        finally:
            self._active = prev

    # ── learn ───────────────────────────────────────────────────────

    def learn(
        self,
        texts_or_dataloader,
        steps: int,
        *,
        batch_size: int = 4,
        seq_len: int = 256,
        gpu_policy: str = "exit",
        warmup_steps: int = 0,
        seed: int | None = None,
    ) -> list[dict]:
        """Adapt the plastic biases (no backprop). Returns per-step metrics.

        Args:
            texts_or_dataloader: ``list[str]`` (tokenized/packed/cycled
                internally) or an ``Iterable`` yielding
                ``{"input_ids": (B,S) LongTensor, "attention_mask": (B,S)}``.
            steps: number of update steps.
            batch_size, seq_len: packing for ``list[str]`` input.
            gpu_policy: ``"exit" | "wait" | "warn"`` GPU memory gate.
            warmup_steps: first N steps get ``M = 0`` (EMA still updates) —
                used to settle the surprise baseline on the source domain.
            seed: deterministic batch order (for ``list[str]`` input).

        Returns:
            One metric dict per step: ``{step, loss, ema_loss, surprise_s,
            modulator_M, mean_abs_delta_b, mean_abs_b, tokens_seen}``.
        """
        if steps <= 0:
            return []
        self.model.eval()
        self.model.requires_grad_(False)
        # The model is already resident in VRAM; gate only on the headroom a
        # learn pass needs (activations), not the full-model gate. The
        # full-model pre-load gate is the runner's job (before load).
        self._check_gpu(gpu_policy, self._residual_gate_gb())

        start_step = self._resume(steps)
        if start_step >= steps:
            self.log.info(
                "learn: already complete (step %d >= %d); skipping", start_step, steps
            )
            return []

        data_iter = self._make_batch_iter(texts_or_dataloader, batch_size, seq_len, seed)
        # Skip already-completed batches so resume reproduces the same stream.
        for _ in range(start_step):
            next(data_iter)

        prev_handlers = self._install_signal_handlers()
        metrics: list[dict] = []
        self._capture = False
        try:
            for step in range(start_step, steps):
                self._signal_step = step
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.log.warning("data iterator exhausted at step %d", step)
                    break
                ids = batch["input_ids"].to(self.device)
                mask = batch.get("attention_mask")
                if mask is None:
                    mask = torch.ones_like(ids)
                mask = mask.to(self.device)

                # 1. Forward (frozen); hooks capture post at every injection point
                self._capture = True
                try:
                    with torch.no_grad():
                        logits = self.model(input_ids=ids, attention_mask=mask).logits
                finally:
                    self._capture = False

                # 2. Loss L — float32, manual CE (not the model's internal loss)
                V = logits.size(-1)  # noqa: N806 - spec notation
                L = F.cross_entropy(  # noqa: N806 - spec notation
                    logits[..., :-1, :].to(torch.float32).reshape(-1, V),
                    ids[..., 1:].reshape(-1),
                )

                # 3–5. EMA → surprise s → modulator M (all float32)
                s, M = self.modulator.update(L)  # noqa: N806 - spec notation
                if warmup_steps and step < warmup_steps:
                    M = 0.0  # noqa: N806 - spec notation

                # 6. Plastic update per injection point (float32)
                mean_abs_delta = 0.0
                for ip in self._injection_points:
                    post = self._last_post[ip.name]  # (B,S,d) float32
                    delta_b = self.lr * M * post.mean(dim=(0, 1))  # (d,) float32
                    ip.bias.add_(delta_b)
                    if self.decay_rate > 0:
                        ip.bias.mul_(1.0 - self.decay_rate)
                    mean_abs_delta += delta_b.abs().mean().item()

                # 7. Metrics
                ema_now = (
                    float(self.modulator.ema_loss)
                    if self.modulator.ema_loss is not None
                    else float(L)  # constant mode has no EMA
                )
                metrics.append(
                    {
                        "step": step,
                        "loss": L.item(),
                        "ema_loss": ema_now,
                        "surprise_s": s,
                        "modulator_M": M,
                        "mean_abs_delta_b": mean_abs_delta / len(self._injection_points),
                        "mean_abs_b": self._mean_abs_bias(),
                        "tokens_seen": (step + 1) * ids.numel(),
                    }
                )
                if (step + 1) % max(self.checkpoint_every, 1) == 0:
                    self._save_checkpoint(step + 1)
                if step % 10 == 0 or step == steps - 1:
                    m = metrics[-1]
                    self.log.info(
                        "step %d/%d loss=%.4f ema=%.4f s=%+.4f M=%.3f |Δb|=%.3e |b|=%.3e",
                        step,
                        steps,
                        m["loss"],
                        m["ema_loss"],
                        m["surprise_s"],
                        m["modulator_M"],
                        m["mean_abs_delta_b"],
                        m["mean_abs_b"],
                    )
        finally:
            self._capture = False
            self._restore_signal_handlers(prev_handlers)

        self._save_checkpoint(steps)
        self.log.info("learn: done %d steps", len(metrics))
        return metrics

    # ── generate ────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        **generate_kwargs,
    ) -> str:
        """Generate with hooks active (plastic ON). ``with brain.without_plasticity():``
        returns the frozen output."""
        if self.tokenizer is None:
            raise ValueError("tokenizer required for generate()")
        tok = self.tokenizer
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
            self.log.warning("pad_token_id was None — set to eos_token_id")
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                **generate_kwargs,
            )
        return tok.decode(out[0], skip_special_tokens=True)

    # ── evaluate (protocol §5, §10) ─────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        texts=None,
        *,
        ids: torch.Tensor | None = None,
        window: int = 512,
        stride: int = 256,
        mode: str = "both",
    ) -> dict:
        """Sliding-window perplexity for the frozen and/or plastic model.

        Args:
            texts: list of strings (tokenized with the wrapper tokenizer).
            ids: alternative — a flat LongTensor of pre-tokenized ids
                (mutually exclusive with ``texts``).
            window: fixed context window (locked = 512 for both models).
            stride: window stride (locked = 256, 50% overlap).
            mode: ``"both"`` (default) computes frozen + plastic;
                ``"frozen"`` only; ``"plastic"`` only.

        Returns:
            ``{"frozen": {...}|None, "plastic": {...}}`` where each summary is
            ``{"ppl", "mean_nll", "n_tokens", "per_block": {"nll": [...],
            "tokens": [...]}}`` (per-block = per sliding window, same order
            for frozen & plastic → naturally paired).
        """
        if ids is None:
            if texts is None or self.tokenizer is None:
                raise ValueError("pass texts (list[str]) with a tokenizer, or ids")
            ids = tokenize_list(texts, self.tokenizer)
        if ids.dtype != torch.long:
            ids = ids.long()
        ids = ids.to(self.device)
        n = ids.numel()
        if n < 2:
            raise ValueError(f"too few tokens for eval: {n}")

        def run(active: bool) -> list[tuple[float, int]]:
            blocks: list[tuple[float, int]] = []
            for begin in range(0, n, stride):
                end = min(begin + window, n)
                chunk = ids[begin:end].unsqueeze(0)  # (1, L)
                if chunk.size(-1) < 2:
                    continue
                if active:
                    logits = self.model(input_ids=chunk).logits
                else:
                    with self.without_plasticity():
                        logits = self.model(input_ids=chunk).logits
                shift_l = logits[..., :-1, :].to(torch.float32).reshape(-1, logits.size(-1))
                shift_t = chunk[..., 1:].reshape(-1)
                nll = F.cross_entropy(shift_l, shift_t, reduction="sum").item()
                blocks.append((nll, int(shift_t.numel())))
            return blocks

        def summarize(blocks: list[tuple[float, int]]) -> dict:
            total_nll = sum(b[0] for b in blocks)
            total_tok = sum(b[1] for b in blocks)
            mean_nll = total_nll / total_tok
            return {
                "ppl": float(math.exp(mean_nll)),
                "mean_nll": float(mean_nll),
                "n_tokens": int(total_tok),
                "per_block": {
                    "nll": [b[0] for b in blocks],
                    "tokens": [b[1] for b in blocks],
                },
            }

        result = {"frozen": None, "plastic": None}
        if mode in ("both", "frozen"):
            result["frozen"] = summarize(run(active=False))
        if mode in ("both", "plastic"):
            result["plastic"] = summarize(run(active=True))
        return result

    # ── consolidation (Phase 2.3 placeholder) ───────────────────────

    def consolidate(self) -> dict:
        """Phase 2.3 placeholder — stable signature, no-op in Phase 1.1."""
        self.log.warning("consolidate() not implemented until Phase 2.3")
        return {"status": "not_implemented", "phase": "2.3"}

    # ── serialization ───────────────────────────────────────────────

    def state_dict(self) -> OrderedDict:
        """Flat plastic state: ``plastic.{ip.name} → bias`` (float32)."""
        sd: OrderedDict = OrderedDict()
        for ip in self._injection_points:
            sd[f"plastic.{ip.name}"] = ip.bias.detach().clone().to(torch.float32)
        return sd

    def load_state_dict(self, state, strict: bool = True) -> BrainWrapper:
        """Restore plastic biases from a state dict; validates keys + shapes."""
        expected = {f"plastic.{ip.name}" for ip in self._injection_points}
        provided = set(state.keys())
        missing = expected - provided
        extra = provided - expected
        if strict and missing:
            raise KeyError(f"missing plastic keys: {sorted(missing)}")
        if strict and extra:
            raise KeyError(f"unexpected keys: {sorted(extra)}")
        for ip in self._injection_points:
            key = f"plastic.{ip.name}"
            if key not in state:
                continue
            t = torch.as_tensor(state[key]).detach().to(torch.float32).to(self.device)
            if tuple(t.shape) != (ip.out_features,):
                raise ValueError(
                    f"shape mismatch for {key}: expected {(ip.out_features,)}, "
                    f"got {tuple(t.shape)}"
                )
            ip.bias.copy_(t)
        return self

    def _config_dict(self) -> dict:
        return {
            "plasticity": self.plasticity,
            "model_type": self.model.config.model_type,
            "n_layers": len(self.container),
            "hidden_size": self._injection_points[0].out_features
            if self._injection_points
            else None,
            "modulator_cfg": self.modulator_cfg,
            "alpha": self.modulator.alpha,
            "s0": self.modulator.s0,
            "k": self.modulator.k,
            "M_max": self.modulator.M_max,
            "eta": self.lr,
            "decay_rate": self.decay_rate,
            "dtype": str(self.dtype),
        }

    def save(self, path: str) -> BrainWrapper:
        """Save plastic weights only (atomic: temp file + ``os.replace``)."""
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "format": SAVE_FORMAT,
            "version": CHECKPOINT_VERSION,
            "plastic": self.state_dict(),
            "config": self._config_dict(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.log.info("saved plastic state -> %s", path)
        return self

    def load(self, path: str) -> BrainWrapper:
        """Restore plastic biases from ``save()`` output (shape-validated)."""
        payload = torch.load(path, weights_only=False)
        if isinstance(payload, dict) and "plastic" in payload:
            state = payload["plastic"]
        elif isinstance(payload, dict) and all(
            k.startswith("plastic.") for k in payload
        ):
            state = payload
        else:
            raise ValueError(f"unrecognized save format in {path}")
        try:
            self.load_state_dict(state)
        except KeyError as exc:
            raise ValueError(
                f"architecture mismatch loading {path}: {exc}"
            ) from exc
        self.log.info("loaded plastic state <- %s", path)
        return self

    # ── public helpers ──────────────────────────────────────────────

    def plastic_parameter_count(self) -> int:
        return sum(int(ip.bias.numel()) for ip in self._injection_points)

    def plastic_memory_bytes(self) -> int:
        return self.plastic_parameter_count() * 4  # float32

    def injection_point_names(self) -> list[str]:
        return [ip.name for ip in self._injection_points]

    def summary(self) -> dict:
        return {
            "model_type": self.model.config.model_type,
            "blocks": len(self.container),
            "injection_points": len(self._injection_points),
            "plastic_params": self.plastic_parameter_count(),
            "plastic_bytes": self.plastic_memory_bytes(),
            "lr": self.lr,
            "decay_rate": self.decay_rate,
            "modulator_mode": self.modulator.mode,
            "min_free_gb": self.min_free_gb,
        }

    def to(self, device) -> BrainWrapper:
        self.device = torch.device(device)
        for ip in self._injection_points:
            ip.bias = ip.bias.to(self.device)
        return self

    def set_lr(self, eta: float) -> None:
        self.lr = float(eta)

    def set_decay_rate(self, lmbda: float) -> None:
        self.decay_rate = float(lmbda)

    # ── internals ───────────────────────────────────────────────────

    def _mean_abs_bias(self) -> float:
        if not self._injection_points:
            return 0.0
        total = sum(float(ip.bias.abs().mean()) for ip in self._injection_points)
        return total / len(self._injection_points)

    def _check_gpu(self, policy: str, min_free_gb: float) -> None:
        if self.device.type != "cuda":
            self.log.info("device is CPU — skipping GPU memory check")
            return
        check_gpu_free(min_free_gb, policy, self.log)

    def _residual_gate_gb(self) -> float:
        """Post-load GPU gate: model is resident, gate only on headroom for
        activations (~0.3× the full-model gate, floor 1 GiB)."""
        return max(1.0, self.min_free_gb * 0.3)

    def _make_batch_iter(self, texts_or_dataloader, batch_size, seq_len, seed):
        if isinstance(texts_or_dataloader, list):
            if not texts_or_dataloader:
                raise ValueError("empty texts passed to learn()")
            if self.tokenizer is None:
                raise ValueError("tokenizer required for list[str] learn() input")
            tokens = tokenize_list(texts_or_dataloader, self.tokenizer)
            return cyclic_batch_iter(tokens, batch_size, seq_len, seed)
        if isinstance(texts_or_dataloader, torch.Tensor):
            return cyclic_batch_iter(texts_or_dataloader, batch_size, seq_len, seed)
        return iter(texts_or_dataloader)

    # ── checkpoints ─────────────────────────────────────────────────

    def _save_checkpoint(self, step: int) -> None:
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        state = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "step": int(step),
            "plastic": self.state_dict(),
            "ema_loss": (
                float(self.modulator.ema_loss)
                if self.modulator.ema_loss is not None
                else None
            ),
            "ema_initialized": self.modulator.initialized,
            "config": self._config_dict(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        paths = [
            os.path.join(self.checkpoint_dir, f"brain_ckpt_step{int(step)}.pt"),
            os.path.join(self.checkpoint_dir, "brain_latest.pt"),
        ]
        for p in paths:
            tmp = f"{p}.tmp.{os.getpid()}"
            torch.save(state, tmp)
            os.replace(tmp, p)  # atomic on POSIX
        self.log.info("checkpoint saved at step %d -> %s", step, self.checkpoint_dir)

    def _resume(self, steps: int) -> int:
        """Return the step to resume from; restores plastic + EMA state."""
        if not self.checkpoint_dir or not os.path.isdir(self.checkpoint_dir):
            return 0
        best_step = -1
        best_path = None
        for p in glob.glob(os.path.join(self.checkpoint_dir, "brain_ckpt_step*.pt")):
            m = re.match(r".*brain_ckpt_step(\d+)\.pt$", p)
            if not m:
                continue
            n = int(m.group(1))
            if n < steps and n > best_step:
                best_step, best_path = n, p
        # Skip-if-exists: a completed run (any checkpoint at/after steps) is
        # never restarted.
        for p in glob.glob(os.path.join(self.checkpoint_dir, "brain_ckpt_step*.pt")):
            m = re.match(r".*brain_ckpt_step(\d+)\.pt$", p)
            if m and int(m.group(1)) >= steps:
                return steps
        if best_path is None:
            return 0
        ckpt = torch.load(best_path, weights_only=False)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            self.log.warning("unrecognized checkpoint format in %s — ignoring", best_path)
            return 0
        self.load_state_dict(ckpt["plastic"])
        ema = ckpt.get("ema_loss")
        self.modulator.ema_loss = (
            torch.as_tensor(ema, dtype=torch.float32) if ema is not None else None
        )
        self.modulator.initialized = bool(ckpt.get("ema_initialized", ema is not None))
        self.log.info("resumed from step %d (%s)", best_step, best_path)
        return best_step

    # ── signals ─────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> dict:
        self._signal_step = 0
        self._second_signal = False

        def handler(signum, frame):  # noqa: ARG001 - stdlib signature
            if self._second_signal:
                self.log.warning("second signal received — exiting immediately")
                os._exit(130)
            self._second_signal = True
            self.log.warning(
                "signal %s received — saving checkpoint at step %d",
                signum,
                self._signal_step,
            )
            try:
                self._save_checkpoint(self._signal_step)
            except Exception as exc:  # noqa: BLE001
                self.log.error("checkpoint save on signal failed: %s", exc)
            os._exit(130)

        prev = {
            s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)
        }
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        return prev

    @staticmethod
    def _restore_signal_handlers(prev: dict) -> None:
        for s, h in prev.items():
            signal.signal(s, h)
