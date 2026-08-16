"""Unit tests for the E036 consolidation machinery (Step 2.3).

Covers:
* ``tc_latent_state`` / ``tc_set_latent_state`` — T-C latent-state round-trip.
* ``zero_lt_state`` — a fresh LT injects nothing (identity) and carries the
  canonical init scales.
* ``latent_change_topk`` — keeps ~K% of the **global** budget by |ΔW|
  magnitude, preserves the true float Δ at the kept positions, and drops the
  rest; correct threshold and kept-fraction bookkeeping.
* ``add_delta_to_lt`` — sparse additive accumulation.
* ``warm_start_st_from_lt`` — ST's latents + scales are overwritten by LT, so
  the next domain begins injecting the accumulated store.
* ``sparse_delta_storage`` — the bitmap + packed-signs on-disk accounting.
* An end-to-end mini-consolidation: two "domains" on a tiny model — transfer
  a top-K delta after each, verify LT accumulates disjoint knowledge and that
  warm-starting changes the ST injection.

Also tests ``steps_to_plateau`` (from the aggregator) — the adaptation-speed
plateau definition.
"""

from __future__ import annotations

import math

import pytest
import torch

from ph_neuro.brain.lora import (
    add_delta_to_lt,
    latent_change_topk,
    sparse_delta_storage,
    tc_latent_state,
    tc_set_latent_state,
    warm_start_st_from_lt,
    zero_lt_state,
    build_ternary_lora_adapters,
)
from ph_neuro.examples.aggregate_e036 import steps_to_plateau
from tests.brain._models import tiny_llama


def _adapters(layers: int = 2, hidden: int = 32, mode: str = "tc"):
    model = tiny_llama(layers=layers, hidden=hidden)
    ads = build_ternary_lora_adapters(model, rank=1, device="cpu", mode=mode)
    return model, ads


# ── latent state round-trip ────────────────────────────────────────


class TestLatentState:
    def test_roundtrip(self):
        _, ads = _adapters()
        for ad in ads:
            st = tc_latent_state(ad)
            assert set(st.keys()) == {"A_latent", "B_latent", "A_scale", "B_scale"}
            assert st["A_latent"].dtype == torch.float32
            # Overwrite with a known value and round-trip back.
            ad.A_latent.data.fill_(0.7)
            st2 = tc_latent_state(ad)
            assert torch.allclose(st2["A_latent"], torch.full_like(st2["A_latent"], 0.7))
            tc_set_latent_state(ad, st)
            assert torch.equal(ad.A_latent.detach().cpu(), st["A_latent"])
            assert torch.equal(ad.B_latent.detach().cpu(), st["B_latent"])
            assert torch.equal(ad.A_scale.detach().cpu(), st["A_scale"])

    def test_zero_lt_identity(self):
        """A fresh LT injects nothing (sign(0)=0) → bit-identical to frozen."""
        torch.manual_seed(0)
        model, ads = _adapters()
        ids = torch.randint(0, 128, (1, 16))
        raw = model(ids).logits
        lt = zero_lt_state(ads)
        for st in lt:
            assert (st["A_latent"] == 0).all()
            assert (st["B_latent"] == 0).all()
            din = st["A_latent"].shape[-1]
            assert st["A_scale"].item() == pytest.approx(1.0 / math.sqrt(din))
            assert st["B_scale"].item() == pytest.approx(1e-2)
        # Load the zero LT into fresh adapters → identity.
        for ad, st in zip(ads, lt):
            tc_set_latent_state(ad, st)
        assert torch.equal(model(ids).logits, raw)


# ── latent_change_topk ─────────────────────────────────────────────


class TestLatentChangeTopk:
    def test_keeps_top_k_by_magnitude(self):
        model, ads = _adapters()
        init = zero_lt_state(ads)
        # Perturb the final state: a few large changes + many tiny ones.
        final = [tc_latent_state(ad) for ad in ads]
        rng = torch.Generator().manual_seed(1)
        big = torch.randn(final[0]["A_latent"].shape, generator=rng) * 5.0
        small = torch.randn(final[0]["A_latent"].shape, generator=rng) * 0.001
        final[0]["A_latent"] = big + small
        final[0]["B_latent"] = torch.randn(final[0]["B_latent"].shape, generator=rng) * 0.5

        out = latent_change_topk(init, final, k=0.10)
        n_total = sum(st["A_latent"].numel() + st["B_latent"].numel() for st in final)
        assert out["n_kept"] == pytest.approx(round(n_total * 0.10), abs=1)
        assert out["kept_frac"] == pytest.approx(0.10, abs=0.02)
        # The big-change entries are kept; the tiny-change entries are dropped.
        keptA = out["delta"][0]["A_latent"]
        assert (keptA.abs() > 1.0).sum().item() > 0  # big entries retained
        # Dropped entries are exactly zero.
        nz = (keptA != 0).sum().item() + (out["delta"][0]["B_latent"] != 0).sum().item()
        assert nz == out["n_kept"]
        # Kept positions carry the TRUE float delta (big + small), not a sign.
        mask = keptA != 0
        if mask.any():
            assert torch.allclose(keptA[mask], final[0]["A_latent"][mask], atol=1e-6)

    def test_global_threshold_across_adapters(self):
        model, ads = _adapters(layers=2)
        init = zero_lt_state(ads)
        final = [tc_latent_state(ad) for ad in ads]
        # Make the LAST adapter's B_latent changes huge (they dominate the
        # global top-K), so the first adapter's small changes get dropped.
        n = len(ads)
        final[n - 1]["B_latent"] = torch.full_like(
            final[n - 1]["B_latent"], 3.0)
        final[0]["A_latent"] = torch.full_like(final[0]["A_latent"], 0.001)
        out = latent_change_topk(init, final, k=0.05)
        # The last adapter's B entries are all kept; the first adapter's tiny
        # A entries are all dropped.
        assert (out["delta"][n - 1]["B_latent"] != 0).all()
        assert (out["delta"][0]["A_latent"] == 0).all()

    def test_k_0_and_k_1(self):
        model, ads = _adapters()
        init = zero_lt_state(ads)
        final = [tc_latent_state(ad) for ad in ads]
        final[0]["A_latent"] = torch.randn_like(final[0]["A_latent"])
        out = latent_change_topk(init, final, k=0.0)
        assert out["n_kept"] >= 1  # clamped to at least one entry
        out1 = latent_change_topk(init, final, k=1.0)
        n_total = sum(st["A_latent"].numel() + st["B_latent"].numel() for st in final)
        assert out1["n_kept"] == n_total


# ── add / warm-start ───────────────────────────────────────────────


class TestAddAndWarmStart:
    def test_add_accumulates(self):
        model, ads = _adapters()
        lt = zero_lt_state(ads)
        init = zero_lt_state(ads)
        final = [tc_latent_state(ad) for ad in ads]
        final[0]["A_latent"] = torch.ones_like(final[0]["A_latent"]) * 0.5
        d1 = latent_change_topk(init, final, k=0.10)
        add_delta_to_lt(lt, d1)
        # Sum of the deltas across adapters = the top-K masked Δ.
        tot1 = sum(st["A_latent"].sum().item() + st["B_latent"].sum().item()
                   for st in lt)
        assert tot1 == pytest.approx(
            sum(x.sum().item() for x in [d1["delta"][0]["A_latent"],
                                         d1["delta"][0]["B_latent"]]))
        # Second add (different domain) accumulates.
        final2 = [tc_latent_state(ad) for ad in ads]
        final2[1]["B_latent"] = torch.ones_like(final2[1]["B_latent"]) * -0.7
        d2 = latent_change_topk(lt, final2, k=0.10)
        add_delta_to_lt(lt, d2)
        tot2 = sum(st["A_latent"].sum().item() + st["B_latent"].sum().item()
                   for st in lt)
        assert tot2 != tot1  # LT grew

    def test_warm_start_copies_lt_into_st(self):
        model, ads = _adapters()
        lt = zero_lt_state(ads)
        ids = torch.randint(0, 128, (1, 16))
        raw = model(ids).logits  # fresh ST (zero B_latent) → identity
        # Give LT a nontrivial sign pattern on BOTH A and B (a zero B_latent
        # → sign=0 → no injection, so both must be nonzero for an active test).
        for i, st in enumerate(lt):
            st["A_latent"] = torch.ones_like(st["A_latent"]) * (0.25 if i % 2 else -0.25)
            st["B_latent"] = torch.ones_like(st["B_latent"]) * 0.1
        warm_start_st_from_lt(ads, lt)
        for ad, st in zip(ads, lt):
            assert torch.equal(ad.A_latent.detach().cpu(), st["A_latent"])
            assert torch.equal(ad.B_latent.detach().cpu(), st["B_latent"])
            assert torch.equal(ad.A_scale.detach().cpu(), st["A_scale"])
        # The warmed-up ST now injects the LT pattern (nonzero delta vs raw).
        assert not torch.equal(model(ids).logits, raw)  # injection active


# ── storage accounting ─────────────────────────────────────────────


class TestSparseDeltaStorage:
    def test_sizes(self):
        n_params = 344_064
        n_kept = round(n_params * 0.10)
        s = sparse_delta_storage(n_params, n_kept)
        assert s["mask_bytes"] == math.ceil(n_params / 8)
        assert s["sign_bytes"] == math.ceil(n_kept / 4)
        assert s["total_bytes"] == s["mask_bytes"] + s["sign_bytes"] + s["scale_bytes"]
        assert s["int32_index_variant_bytes"] == n_kept * 4 + s["sign_bytes"] + s["scale_bytes"]
        # Sanity: bitmap delta is far smaller than the int32-index variant.
        assert s["total_bytes"] < s["int32_index_variant_bytes"]

    def test_small_k_wins(self):
        n_params = 344_064
        k = 0.01
        s = sparse_delta_storage(n_params, round(n_params * k))
        assert s["total_bytes"] < n_params  # far below a full adapter


# ── end-to-end mini consolidation on a tiny model ──────────────────


class TestMiniConsolidation:
    def test_two_domain_consolidation(self):
        """Transfer after each of two 'domains'; LT accumulates and warm-start
        changes the ST injection (forward-transfer mechanism alive)."""
        torch.manual_seed(0)
        model, ads = _adapters(layers=2, hidden=32)
        lt = zero_lt_state(ads)

        # Domain 1: nudge all ST latents (a crude 'adaptation').
        for ad in ads:
            ad.A_latent.data.add_(0.3)
            ad.B_latent.data.fill_(0.1)
        st1 = [tc_latent_state(ad) for ad in ads]
        d1 = latent_change_topk(lt, st1, k=0.20)
        add_delta_to_lt(lt, d1)
        # After consolidation, LT is nonzero and ST is warm-started from it.
        assert any((st["A_latent"] != 0).any() for st in lt)
        warm_start_st_from_lt(ads, lt)
        st_init2 = [tc_latent_state(ad) for ad in ads]
        # Domain 2: ST (warm-started) adapts further.
        for ad in ads:
            ad.A_latent.data.add_(0.2)
            ad.B_latent.data.sub_(0.05)
        st2 = [tc_latent_state(ad) for ad in ads]
        d2 = latent_change_topk(st_init2, st2, k=0.20)
        assert d2["n_kept"] > 0
        # Δ2 is measured relative to the warm-start (not the raw ST2 state).
        assert not torch.equal(d2["delta"][0]["A_latent"], st2[0]["A_latent"])
        add_delta_to_lt(lt, d2)
        # The LT accumulated both domains.
        assert any(st["A_latent"].abs().sum().item() > 0 for st in lt)


# ── steps_to_plateau (aggregator helper) ───────────────────────────


class TestStepsToPlateau:
    def test_plateau_detection(self):
        probes = [
            {"adapt_step": 296, "ppl": 15.0},
            {"adapt_step": 306, "ppl": 14.0},
            {"adapt_step": 316, "ppl": 13.5},
            {"adapt_step": 326, "ppl": 13.4},  # plateau from here (within 0.5%)
            {"adapt_step": 336, "ppl": 13.42},
            {"adapt_step": 393, "ppl": 13.41},
        ]
        assert steps_to_plateau(probes) == 326

    def test_no_early_plateau(self):
        """Non-converging sequence → only the (trivial) last point qualifies."""
        probes = [
            {"adapt_step": 296, "ppl": 15.0},
            {"adapt_step": 306, "ppl": 14.5},
            {"adapt_step": 316, "ppl": 14.4},
        ]
        # Last point always qualifies (within 0.5% of itself) — so the result
        # is the last step, i.e. "plateaued only at the end" (not faster).
        assert steps_to_plateau(probes) == 316

    def test_empty(self):
        assert steps_to_plateau([]) is None
