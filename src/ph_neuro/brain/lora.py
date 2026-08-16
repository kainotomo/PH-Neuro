"""Minimal manual LoRA adapters (E032 Part D / E034).

Backprop LoRA at the **same parameter budget** as the BrainWrapper low-rank
local modes: rank-1 ``A:(r, d_in)``/``B:(d_out, r)`` pairs injected at every
``o_proj``/``down_proj`` (LLaMA/SmolLM2) or ``attn.c_proj``/``mlp.c_proj``
(GPT-2) via forward hooks. ``peft`` is not a dependency, so this is the
project's own minimal implementation (~50 lines), identical in structure and
init to the local low-rank plastic representation — only the update rule
differs (AdamW backprop here vs local Hebbian/predictive-coding in the Brain
Wrapper).

``LoRAAdapter`` is testable in isolation (used by ``run_e032_lora.py`` and
``run_e034_lora.py``) and by ``tests/brain/test_e034_lora.py``.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from ph_neuro.brain.block_wrappers import (
    _get_in_features,
    _get_out_features,
    get_block_container,
    get_block_wrapper,
)
from ph_neuro.layers.ste_dqt import stochastic_round
from ph_neuro.layers.ste_linear import ste_sign
from ph_neuro.utils.packing import pack_ternary


class LoRAAdapter:
    """One trainable LoRA pair injected at one frozen projection module.

    ``output + (B @ (A @ x))`` via a forward hook; A/B are real
    ``nn.Parameters`` so gradients flow back through the hook. Init follows
    the E032 convention: ``A ~ N(0, 1/sqrt(d_in))``, ``B = 0`` (the injection
    is exactly zero at construction → the frozen model is unchanged; identical
    to the local low-rank mode's init, so the comparison isolates the update
    rule, not the init).
    """

    def __init__(self, module, rank: int, device, dtype=torch.float32):
        self.name = module.__class__.__name__
        self.out_features = _get_out_features(module)
        self.in_features = _get_in_features(module)
        self.rank = int(rank)
        self.device = torch.device(device)
        # Scaled random projection init (matches the local low-rank mode).
        self.A = torch.randn(
            self.rank, self.in_features, dtype=dtype, device=self.device
        ) * (1.0 / math.sqrt(self.in_features))
        self.B = torch.zeros(self.out_features, self.rank, dtype=dtype, device=self.device)
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)
        # Frozen-baseline eval disables the injection (hooks pass through).
        self.enabled = True
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if not self.enabled:
            return output
        x = args[0]
        t = torch.einsum("ri,bsi->bsr", self.A.to(output.dtype), x)
        return output + torch.einsum("or,bsr->bso", self.B.to(output.dtype), t)

    def set_enabled(self, enabled: bool) -> None:
        """Temporarily disable/enable the injection (frozen-baseline eval)."""
        self.enabled = bool(enabled)

    def parameters(self):
        yield self.A
        yield self.B

    def remove(self):
        self.handle.remove()

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            [("A", self.A.detach().clone().to(torch.float32)),
             ("B", self.B.detach().clone().to(torch.float32))]
        )

    def load_state_dict(self, state) -> None:
        self.A.data.copy_(state["A"].to(self.device))
        self.B.data.copy_(state["B"].to(self.device))

    def n_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def mean_abs(self) -> float:
        """Mean |A| + |B| (a cheap plastic-weight diagnostic, E032 convention)."""
        return float(self.A.detach().abs().mean()) + float(
            self.B.detach().abs().mean()
        )


def build_lora_adapters(model, rank: int, device) -> list[LoRAAdapter]:
    """Attach a LoRA adapter at every o_proj/down_proj (llama) / c_proj (gpt2)."""
    container = get_block_container(model)
    wrapper = get_block_wrapper(model)
    adapters: list[LoRAAdapter] = []
    for i, block in enumerate(container):
        for path in wrapper.block_paths:
            mod = block
            for part in path.split("."):
                mod = getattr(mod, part)
            adapters.append(LoRAAdapter(mod, rank, device))
    return adapters


def n_lora_params(adapters: list[LoRAAdapter]) -> int:
    return sum(ad.n_params() for ad in adapters)


def all_lora_weights(adapters: list[LoRAAdapter]) -> torch.Tensor:
    """Concatenated A+B flattened weights (for magnitude diagnostics)."""
    return torch.cat(
        [ad.A.detach().flatten() for ad in adapters]
        + [ad.B.detach().flatten() for ad in adapters]
    )


# ═══════════════════════════════════════════════════════════════════════
# E035 — ternary LoRA adapters (Step 2.2)
#
# Three ternarization paths on the same rank-1 344,064-param budget as the
# E034 float gated LoRA, all behind one ``TernaryLoRAAdapter`` interface:
#
#   ta — POST-TRAINING quantization (CAT-Q style): a float A/B pair is trained
#        first (E034 checkpoint reuse), then quantized to ternary with
#        per-matrix scale factors (``Q = sign(W)``, ``s = mean|W|`` — the
#        L1-optimal per-matrix ternary approximation). An optional short
#        calibration fine-tune (STE through the ternary weights, constant lr)
#        then refines the float latents and re-quantizes.
#   tb — DQT (``ste_dqt.py`` mechanics): int8 ternary buffers updated by
#        stochastic rounding of float accumulation buffers; forward uses the
#        int8 weights via a custom autograd Function (gradients route to the
#        float buffers — STE); ``apply_stochastic_rounding()`` after each
#        optimizer step.
#   tc — STE (``ste_linear.py`` mechanics): float latent scores; forward uses
#        ``sign(latent)`` with identity backward.
#
# All modes keep ``B = 0`` at construction (injection exactly zero → frozen
# model unchanged, invariant I1) and carry **per-matrix scale factors** so the
# ternary injection ``delta = (s_A·s_B)·(B_tern @ (A_tern @ x))`` matches the
# float adapter's magnitude (~0.01) instead of the raw O(√d_in) ternary delta,
# which would destroy the frozen residual stream (see 09-e035-ternary-lora.md
# §1 for the rationale). Scales are computed from the trained weights for T-A,
# trainable for T-B/T-C.
# ═══════════════════════════════════════════════════════════════════════


def ternary_quantize(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-matrix ternary quantization (CAT-Q style): ``w ≈ s · sign(w)``.

    Args:
        w: Float weight tensor of any shape.

    Returns:
        ``(Q, s)`` where ``Q = sign(w)`` as int8 in {-1, 0, +1} and ``s`` is
        the per-matrix fp32 scale ``mean(|w|)`` — the L1-optimal single scale
        for the fixed sign pattern. For a zero matrix ``s = 0`` (identity).
    """
    s = w.detach().abs().mean()
    Q = w.detach().sign().to(torch.int8)
    return Q, s


class _TernaryDQTInjection(torch.autograd.Function):
    """DQT forward/backward for the LoRA injection ``delta = (s_A·s_B)·(Bₜ @ (Aₜ @ x))``.

    Forward uses the **int8 ternary buffers** ``A_tern``/``B_tern`` (cast to
    the input dtype). Backward routes gradients to the **float accumulation
    buffers** ``A_float``/``B_float`` and the scale factors ``A_scale``/
    ``B_scale`` (STE — the quantization step is passed through as identity),
    mirroring ``ste_dqt.py``'s ``_DQTGradFn`` for a full layer but adapted to
    the LoRA rank-1 einsum.

    Shapes: ``x:(b,s,d_in)``, ``A:(r,d_in)``, ``B:(d_out,r)`` → ``t:(b,s,r)``,
    ``delta:(b,s,d_out)``.
    """

    @staticmethod
    def forward(ctx, x, A_float, A_tern, B_float, B_tern, A_scale, B_scale):
        ctx.save_for_backward(x, A_tern, B_tern, A_scale, B_scale)
        dt = x.dtype
        t = torch.einsum("ri,bsi->bsr", A_tern.to(dt), x)
        delta = torch.einsum("or,bsr->bso", B_tern.to(dt), t)
        return delta * (A_scale.to(dt) * B_scale.to(dt))

    @staticmethod
    def backward(ctx, grad_delta):
        x, A_tern, B_tern, A_scale, B_scale = ctx.saved_tensors
        dt = grad_delta.dtype
        A_t = A_tern.to(dt)
        B_t = B_tern.to(dt)
        s = A_scale.to(dt) * B_scale.to(dt)
        t = torch.einsum("ri,bsi->bsr", A_t, x)
        # True gradient w.r.t. the actual (scaled) injection → flows into the
        # model (the model's loss depends on the real delta magnitude).
        g = grad_delta * s
        grad_t = torch.einsum("or,bso->bsr", B_t, g)
        grad_x = torch.einsum("ri,bsr->bsi", A_t, grad_t)
        # Gain-decoupled STE gradients for the float accumulation buffers:
        # they see the *unscaled* ternary gradient so B can flip out of its
        # all-zero (identity) start instead of being frozen by the tiny scale
        # product (s_A·s_B ≈ 1e-3). The per-matrix scales handle the output
        # magnitude; AdamW's per-param normalization absorbs the constant
        # rescaling, so this is a learning-rate-like gain on the buffers.
        gu = grad_delta
        grad_tu = torch.einsum("or,bso->bsr", B_t, gu)
        grad_A_float = torch.einsum("bsr,bsi->ri", grad_tu, x)
        grad_B_float = torch.einsum("bso,bsr->or", gu, t)
        # True product-rule gradients for the scale factors.
        unscaled = torch.einsum("or,bsr->bso", B_t, t)
        grad_A_scale = B_scale * (unscaled * grad_delta).sum()
        grad_B_scale = A_scale * (unscaled * grad_delta).sum()
        return (
            grad_x,
            grad_A_float,
            None,
            grad_B_float,
            None,
            grad_A_scale,
            grad_B_scale,
        )


class TernaryLoRAAdapter:
    """Ternary LoRA adapter — one adapter injected at one frozen projection.

    Three ternarization modes (E035 / Step 2.2); see the module docstring.
    Injected via a forward hook as ``output + delta`` (same sites/pattern as
    :class:`LoRAAdapter`); ``A ~ N(0, 1/sqrt(d_in))``-analog init and ``B = 0``
    (identity at construction, invariant I1 — tested).

    Args:
        module: The frozen projection module to attach to.
        rank: LoRA rank (1 in E034/E035).
        device: Torch device.
        mode: ``"ta"`` (post-training quantize), ``"tb"`` (DQT), ``"tc"`` (STE).
        dtype: Float dtype for the accumulation/latent buffers (default fp32).
    """

    MODES = ("ta", "tb", "tc")

    def __init__(
        self,
        module,
        rank: int,
        device,
        mode: str = "tb",
        dtype=torch.float32,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.name = module.__class__.__name__
        self.out_features = _get_out_features(module)
        self.in_features = _get_in_features(module)
        self.rank = int(rank)
        self.device = torch.device(device)
        self.mode = mode
        self.dtype = dtype
        self.enabled = True
        self.handle = module.register_forward_hook(self._hook)

        r, din, dout = self.rank, self.in_features, self.out_features

        if mode == "ta":
            # Float A/B (E034 init: A ~ N(0, 1/sqrt(d_in)), B = 0).
            self.A = torch.randn(r, din, dtype=dtype, device=self.device) * (
                1.0 / math.sqrt(din)
            )
            self.B = torch.zeros(dout, r, dtype=dtype, device=self.device)
            self.A.requires_grad_(True)
            self.B.requires_grad_(True)
            # Filled by quantize().
            self.A_tern = torch.zeros(r, din, dtype=torch.int8, device=self.device)
            self.B_tern = torch.zeros(dout, r, dtype=torch.int8, device=self.device)
            self.A_scale = torch.tensor(0.0, dtype=dtype, device=self.device)
            self.B_scale = torch.tensor(0.0, dtype=dtype, device=self.device)
            self.phase = "float"  # float | quantized | calib

        elif mode == "tb":
            # DQT: float accumulation buffers + int8 ternary buffers, faithful
            # to ste_dqt.py's init convention — BOTH A and B float buffers
            # start at N(0, 0.1), stochastically rounded to ~10% nonzero int8
            # ternary at construction (the M2 DQT convention). This is a
            # pre-registered amendment (09-e035 §1): an all-zero B (identity)
            # cannot wake up a DQT adapter — stochastic rounding flips one
            # entry at a time, so with B_tern = 0 the adapter would stay
            # frozen at init (measured). The frozen-baseline guarantee still
            # holds exactly because eval disables the hooks (the adapter is a
            # removable module; disabled → bit-identical frozen output).
            self.A_float = torch.randn(r, din, dtype=dtype, device=self.device) * 0.1
            self.B_float = torch.randn(dout, r, dtype=dtype, device=self.device) * 0.1
            self.A_float.requires_grad_(True)
            self.B_float.requires_grad_(True)
            self.A_tern = stochastic_round(self.A_float.detach())
            self.B_tern = stochastic_round(self.B_float.detach())
            # Trainable per-matrix scales: A_scale normalizes the ternary A to
            # unit row-norm contribution (like float A's ~1), B_scale tracks
            # the effective B magnitude (~0.01, matching the float adapter's).
            n_nz = max(int((self.A_tern != 0).sum().item()), 1)
            self.A_scale = torch.tensor(
                1.0 / math.sqrt(n_nz), dtype=dtype, device=self.device
            )
            self.B_scale = torch.tensor(1e-2, dtype=dtype, device=self.device)
            self.A_scale.requires_grad_(True)
            self.B_scale.requires_grad_(True)
            self._prev_A = self.A_tern.clone()
            self._prev_B = self.B_tern.clone()

        else:  # tc
            # STE: float latent scores (ste_linear.py default N(0, 0.1)); B = 0
            # → sign(B) = 0 → identity at construction.
            self.A_latent = torch.randn(r, din, dtype=dtype, device=self.device) * 0.1
            self.B_latent = torch.zeros(dout, r, dtype=dtype, device=self.device)
            self.A_latent.requires_grad_(True)
            self.B_latent.requires_grad_(True)
            # Trainable per-matrix scales (A dense ±1 → 1/sqrt(d_in) row norm).
            self.A_scale = torch.tensor(
                1.0 / math.sqrt(din), dtype=dtype, device=self.device
            )
            self.B_scale = torch.tensor(1e-2, dtype=dtype, device=self.device)
            self.A_scale.requires_grad_(True)
            self.B_scale.requires_grad_(True)

    # ── forward ────────────────────────────────────────────────────

    def _hook(self, module, args, output):
        if not self.enabled:
            return output
        return output + self.forward_delta(args[0])

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        """The LoRA injection ``delta`` for the current mode/phase."""
        dt = x.dtype
        if self.mode == "ta":
            if self.phase == "float":
                t = torch.einsum("ri,bsi->bsr", self.A.to(dt), x)
                return torch.einsum("or,bsr->bso", self.B.to(dt), t)
            s = (self.A_scale.to(dt) * self.B_scale.to(dt))
            if self.phase == "calib":
                # STE through the float latents; scales fixed.
                A_t = ste_sign(self.A).to(dt)
                B_t = ste_sign(self.B).to(dt)
            else:  # quantized: use the int8 ternary snapshot + fixed scales
                A_t = self.A_tern.to(dt)
                B_t = self.B_tern.to(dt)
            t = torch.einsum("ri,bsi->bsr", A_t, x)
            return s * torch.einsum("or,bsr->bso", B_t, t)

        if self.mode == "tb":
            return _TernaryDQTInjection.apply(
                x,
                self.A_float,
                self.A_tern,
                self.B_float,
                self.B_tern,
                self.A_scale,
                self.B_scale,
            )

        # tc — STE through the latent scores, trainable scales.
        s = (self.A_scale.to(dt) * self.B_scale.to(dt))
        A_t = ste_sign(self.A_latent).to(dt)
        B_t = ste_sign(self.B_latent).to(dt)
        t = torch.einsum("ri,bsi->bsr", A_t, x)
        return s * torch.einsum("or,bsr->bso", B_t, t)

    # ── lifecycle ──────────────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> None:
        """Temporarily disable/enable the injection (frozen-baseline eval)."""
        self.enabled = bool(enabled)

    def parameters(self):
        """Yield the trainable parameters for the optimizer."""
        if self.mode == "ta":
            if self.phase in ("float", "calib"):
                yield self.A
                yield self.B
            return
        if self.mode == "tb":
            yield self.A_float
            yield self.B_float
            yield self.A_scale
            yield self.B_scale
        else:
            yield self.A_latent
            yield self.B_latent
            yield self.A_scale
            yield self.B_scale

    def apply_after_step(self) -> dict | None:
        """T-B: stochastic-round the float buffers into the int8 ternary buffers.

        Must be called AFTER ``optimizer.step()``. Returns flip statistics
        (``{A_flip_rate, A_n_flips, B_flip_rate, B_n_flips}``) or ``None`` for
        non-DQT modes.
        """
        if self.mode != "tb":
            return None
        stats: dict[str, float] = {}
        for name, fbuf in (("A", self.A_float), ("B", self.B_float)):
            tern = getattr(self, f"{name}_tern")
            new = stochastic_round(fbuf.detach())
            n_flips = int((tern != new).sum().item())
            stats[f"{name}_flip_rate"] = n_flips / max(tern.numel(), 1)
            stats[f"{name}_n_flips"] = n_flips
            tern.copy_(new)
            setattr(self, f"_prev_{name}", tern.clone())
        return stats

    def quantize(self) -> None:
        """T-A: convert the trained float A/B to a ternary snapshot + scales."""
        if self.mode != "ta":
            return
        self.A_tern, self.A_scale = ternary_quantize(self.A)
        self.B_tern, self.B_scale = ternary_quantize(self.B)
        self.phase = "quantized"

    def set_phase(self, phase: str) -> None:
        """T-A: switch between ``float`` / ``quantized`` / ``calib`` phases."""
        if self.mode != "ta":
            return
        if phase not in ("float", "quantized", "calib"):
            raise ValueError(f"unknown ta phase {phase!r}")
        self.phase = phase

    def ternary_snapshot(self):
        """Current ternary weights + scales ``(A_tern, B_tern, A_scale, B_scale)``."""
        if self.mode == "ta":
            if self.phase == "float":
                A_tern, A_scale = ternary_quantize(self.A)
                B_tern, B_scale = ternary_quantize(self.B)
                return A_tern, B_tern, A_scale, B_scale
            return self.A_tern, self.B_tern, self.A_scale, self.B_scale
        if self.mode == "tb":
            return self.A_tern, self.B_tern, self.A_scale, self.B_scale
        # tc
        A_tern = self.A_latent.detach().sign().to(torch.int8)
        B_tern = self.B_latent.detach().sign().to(torch.int8)
        return A_tern, B_tern, self.A_scale, self.B_scale

    # ── state / diagnostics ────────────────────────────────────────

    def state_dict(self) -> OrderedDict:
        d: OrderedDict = OrderedDict()
        if self.mode == "ta":
            d["A"] = self.A.detach().clone().to(torch.float32)
            d["B"] = self.B.detach().clone().to(torch.float32)
            d["A_tern"] = self.A_tern.clone()
            d["B_tern"] = self.B_tern.clone()
            d["A_scale"] = self.A_scale.detach().clone().to(torch.float32)
            d["B_scale"] = self.B_scale.detach().clone().to(torch.float32)
            d["phase"] = self.phase
        elif self.mode == "tb":
            d["A_float"] = self.A_float.detach().clone().to(torch.float32)
            d["B_float"] = self.B_float.detach().clone().to(torch.float32)
            d["A_tern"] = self.A_tern.clone()
            d["B_tern"] = self.B_tern.clone()
            d["A_scale"] = self.A_scale.detach().clone().to(torch.float32)
            d["B_scale"] = self.B_scale.detach().clone().to(torch.float32)
        else:  # tc
            d["A_latent"] = self.A_latent.detach().clone().to(torch.float32)
            d["B_latent"] = self.B_latent.detach().clone().to(torch.float32)
            d["A_scale"] = self.A_scale.detach().clone().to(torch.float32)
            d["B_scale"] = self.B_scale.detach().clone().to(torch.float32)
        return d

    def load_state_dict(self, state) -> None:
        if self.mode == "ta":
            self.A.data.copy_(state["A"].to(self.device))
            self.B.data.copy_(state["B"].to(self.device))
            self.A_tern.copy_(state["A_tern"].to(self.device))
            self.B_tern.copy_(state["B_tern"].to(self.device))
            self.A_scale.data.copy_(state["A_scale"].to(self.device))
            self.B_scale.data.copy_(state["B_scale"].to(self.device))
            self.phase = state.get("phase", "quantized")
        elif self.mode == "tb":
            self.A_float.data.copy_(state["A_float"].to(self.device))
            self.B_float.data.copy_(state["B_float"].to(self.device))
            self.A_tern.copy_(state["A_tern"].to(self.device))
            self.B_tern.copy_(state["B_tern"].to(self.device))
            self.A_scale.data.copy_(state["A_scale"].to(self.device))
            self.B_scale.data.copy_(state["B_scale"].to(self.device))
        else:  # tc
            self.A_latent.data.copy_(state["A_latent"].to(self.device))
            self.B_latent.data.copy_(state["B_latent"].to(self.device))
            self.A_scale.data.copy_(state["A_scale"].to(self.device))
            self.B_scale.data.copy_(state["B_scale"].to(self.device))

    def remove(self):
        self.handle.remove()

    def n_params(self) -> int:
        return self.rank * (self.in_features + self.out_features)

    def mean_abs(self) -> float:
        """Mean |effective weights| (float buffers / ternary |Q|·scale)."""
        if self.mode == "ta":
            return float(self.A.detach().abs().mean()) + float(
                self.B.detach().abs().mean()
            )
        if self.mode == "tb":
            return float(self.A_tern.float().abs().mean() * self.A_scale.item()) + float(
                self.B_tern.float().abs().mean() * self.B_scale.item()
            )
        return float(
            self.A_latent.detach().sign().abs().mean() * self.A_scale.item()
        ) + float(self.B_latent.detach().sign().abs().mean() * self.B_scale.item())


def build_ternary_lora_adapters(
    model, rank: int, device, mode: str = "tb"
) -> list[TernaryLoRAAdapter]:
    """Attach a ternary LoRA adapter at every o_proj/down_proj (llama) / c_proj (gpt2)."""
    container = get_block_container(model)
    wrapper = get_block_wrapper(model)
    adapters: list[TernaryLoRAAdapter] = []
    for i, block in enumerate(container):
        for path in wrapper.block_paths:
            mod = block
            for part in path.split("."):
                mod = getattr(mod, part)
            adapters.append(TernaryLoRAAdapter(mod, rank, device, mode=mode))
    return adapters


def pack_ternary_adapters(adapters: list[TernaryLoRAAdapter]) -> torch.Tensor:
    """Concatenate all ternary A/B weights and 2-bit pack them.

    Returns an int8 tensor of ``ceil(n_params / 4)`` bytes — the on-device
    representation. Scale factors are stored separately (2 fp32 per matrix).
    """
    parts: list[torch.Tensor] = []
    for ad in adapters:
        A_tern, B_tern, _, _ = ad.ternary_snapshot()
        parts.append(A_tern.flatten())
        parts.append(B_tern.flatten())
    allw = torch.cat(parts).to(torch.int8)
    return pack_ternary(allw)


def ternary_storage_report(
    adapters: list[TernaryLoRAAdapter], packed: torch.Tensor
) -> dict:
    """Float32 vs 2-bit-packed storage for the ternary adapter budget."""
    n_params = sum(ad.n_params() for ad in adapters)
    float32_bytes = n_params * 4
    packed_bytes = int(packed.numel())  # int8 bytes (4 ternary weights/byte)
    n_scales = 2 * len(adapters)
    scale_bytes = n_scales * 4
    total_packed = packed_bytes + scale_bytes
    return {
        "n_params": n_params,
        "float32_bytes": float32_bytes,
        "packed_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "total_packed_bytes": total_packed,
        "reduction_factor": float(float32_bytes / max(total_packed, 1)),
    }


# ═══════════════════════════════════════════════════════════════════
# E036 — consolidation machinery (Step 2.3)
#
# Sleep-like consolidation on top of the T-C (STE) ternary LoRA adapter: a
# persistent LONG-TERM store (LT) accumulates the top-K% (by |ΔW| magnitude)
# of each domain's SHORT-TERM (ST) latent-score changes; ST is reset and
# warm-started from LT before each new domain. See 10-e036-consolidation.md
# for the pre-registered design (K = 10%, add rule, no LT decay).
#
# LT is represented as a list of per-adapter **latent states** — detached
# float32 copies of ``{A_latent, B_latent, A_scale, B_scale}`` (the same
# tensors a T-C adapter carries). ST is the live T-C adapter attached to the
# model. The transfer only moves latent-score **signs** into LT (LT keeps the
# init scale factors, so the injection magnitude stays ~0.01); the per-domain
# sparse deltas are stored for the storage accounting.
# ═══════════════════════════════════════════════════════════════════


def tc_latent_state(adapter: TernaryLoRAAdapter) -> OrderedDict:
    """Detached **CPU** float32 copy of a T-C adapter's latent scores + scales.

    All E036 latent states (LT and ST snapshots) live on CPU float32 — the
    delta math (``latent_change_topk``, ``add_delta_to_lt``) is pure tensor
    bookkeeping and stays off the CUDA graph; ``tc_set_latent_state`` moves
    back to the adapter's device on warm-start.
    """
    return OrderedDict(
        [
            ("A_latent", adapter.A_latent.detach().cpu().to(torch.float32)),
            ("B_latent", adapter.B_latent.detach().cpu().to(torch.float32)),
            ("A_scale", adapter.A_scale.detach().cpu().to(torch.float32)),
            ("B_scale", adapter.B_scale.detach().cpu().to(torch.float32)),
        ]
    )


def tc_set_latent_state(adapter: TernaryLoRAAdapter, state) -> None:
    """Overwrite a T-C adapter's latent scores + scales from a saved state."""
    adapter.A_latent.data.copy_(state["A_latent"].to(adapter.device))
    adapter.B_latent.data.copy_(state["B_latent"].to(adapter.device))
    adapter.A_scale.data.copy_(state["A_scale"].to(adapter.device))
    adapter.B_scale.data.copy_(state["B_scale"].to(adapter.device))


def zero_lt_state(adapters: list[TernaryLoRAAdapter]) -> list[OrderedDict]:
    """A fresh (empty) long-term store: zero latents, canonical init scales.

    ``sign(0) = 0`` → a fresh LT injects nothing (identity). Scales are set to
    the canonical T-C init (``A_scale = 1/sqrt(d_in)``, ``B_scale = 1e-2``) so
    LT's injection magnitude matches the float adapter (~0.01) once signs
    accumulate.
    """
    out: list[OrderedDict] = []
    for ad in adapters:
        din = ad.in_features
        out.append(
            OrderedDict(
                [
                    ("A_latent", torch.zeros(ad.rank, din, dtype=torch.float32)),
                    ("B_latent", torch.zeros(ad.out_features, ad.rank, dtype=torch.float32)),
                    ("A_scale", torch.tensor(1.0 / math.sqrt(din), dtype=torch.float32)),
                    ("B_scale", torch.tensor(1e-2, dtype=torch.float32)),
                ]
            )
        )
    return out


def latent_change_topk(
    init_states: list[OrderedDict],
    final_states: list[OrderedDict],
    k: float = 0.10,
) -> dict:
    """Top-K% (by |ΔW| magnitude) latent-score change from init → final.

    Args:
        init_states / final_states: per-adapter latent states (E036 ST init
            vs ST final — the ST init is the warm-started LT copy, so ΔW is
            the domain's own change).
        k: fraction of the **global** budget (all A+B entries across all
            adapters) to transfer, by |ΔW| magnitude.

    Returns:
        ``{"delta": [per-adapter OrderedDict with sparse A_latent/B_latent —
        nonzero only at the kept positions, holding the true float ΔW],
        "n_kept": int, "kept_frac": float, "k": k, "threshold": float}``.
    """
    # Flatten all per-adapter A/B changes to pick the global top-K threshold.
    all_abs: list[torch.Tensor] = []
    for init, fin in zip(init_states, final_states):
        dA = fin["A_latent"].to(torch.float32) - init["A_latent"].to(torch.float32)
        dB = fin["B_latent"].to(torch.float32) - init["B_latent"].to(torch.float32)
        all_abs.append(dA.abs().flatten())
        all_abs.append(dB.abs().flatten())
    mags = torch.cat(all_abs)
    n_total = mags.numel()
    n_keep = int(round(n_total * k))
    n_keep = min(max(n_keep, 1), n_total)
    # The k-th largest magnitude = the keep threshold (k=0.10 → the 90th pct).
    threshold = float(torch.topk(mags, n_keep, largest=True).values.min())

    delta: list[OrderedDict] = []
    for init, fin in zip(init_states, final_states):
        dA = fin["A_latent"].to(torch.float32) - init["A_latent"].to(torch.float32)
        dB = fin["B_latent"].to(torch.float32) - init["B_latent"].to(torch.float32)
        mA = (dA.abs() >= threshold).to(dA.dtype)
        mB = (dB.abs() >= threshold).to(dB.dtype)
        delta.append(
            OrderedDict(
                [
                    ("A_latent", dA * mA),
                    ("B_latent", dB * mB),
                ]
            )
        )
    kept = int((mags >= threshold).sum().item())
    return {
        "delta": delta,
        "n_kept": kept,
        "kept_frac": float(kept) / n_total,
        "k": k,
        "threshold": threshold,
    }


def add_delta_to_lt(lt_states: list[OrderedDict], delta: dict) -> list[OrderedDict]:
    """In place: ``lt += delta`` (the E036 add rule; sparse top-K add)."""
    for i, st in enumerate(lt_states):
        d = delta["delta"][i]
        st["A_latent"] = st["A_latent"] + d["A_latent"].to(torch.float32)
        st["B_latent"] = st["B_latent"] + d["B_latent"].to(torch.float32)
    return lt_states


def warm_start_st_from_lt(adapters: list[TernaryLoRAAdapter], lt_states) -> None:
    """Warm-start the short-term adapters from the long-term store.

    Copies LT's latent scores + scales into ST, so the next domain begins by
    injecting the accumulated store (the forward-transfer mechanism). ST's
    ΔW for the next transfer is then measured relative to this warm-start.
    """
    for ad, st in zip(adapters, lt_states):
        tc_set_latent_state(ad, st)


def sparse_delta_storage(n_params: int, n_kept: int) -> dict:
    """On-disk bytes for one sparse ternary delta (bitmap + packed signs).

    The delta is stored as (a) a 1-bit presence mask over the full parameter
    budget, (b) the 2-bit packed ternary signs of the kept entries, (c) two
    fp32 per-A/B scales. This is the on-device sparse-ternary format
    (bitmap-CSR variant); the conservative int32-index variant is reported
    separately by the runner.
    """
    mask_bytes = math.ceil(n_params / 8)
    sign_bytes = math.ceil(n_kept / 4)  # 2-bit packed
    scale_bytes = 8
    return {
        "mask_bytes": mask_bytes,
        "sign_bytes": sign_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": mask_bytes + sign_bytes + scale_bytes,
        "int32_index_variant_bytes": n_kept * 4 + sign_bytes + scale_bytes,
    }
