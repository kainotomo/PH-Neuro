"""Unit tests for block wrappers — injection-point discovery + hook capture.

Verifies the real module paths on genuinely-structured tiny models:
* SmolLM2 (model_type 'llama'): ``model.model.layers[i].self_attn.o_proj`` +
  ``mlp.down_proj`` (nn.Linear).
* GPT-2 (model_type 'gpt2'): ``model.transformer.h[i].attn.c_proj`` +
  ``mlp.c_proj`` (Conv1D with ``.nf``).
"""

from __future__ import annotations

import pytest
import torch

from ph_neuro.brain.block_wrappers import (
    GPT2BlockWrapper,
    SmolLM2BlockWrapper,
    _get_out_features,
    get_block_container,
    get_block_wrapper,
)
from tests.brain._models import random_token_ids, tiny_gpt2, tiny_llama


class TestSmolLM2Wrapper:
    def test_returns_two_injection_points(self):
        model = tiny_llama(layers=2)
        wrapper = SmolLM2BlockWrapper()
        container = get_block_container(model)
        pts = []
        for i, block in enumerate(container):
            pts.extend(wrapper.get_injection_points(block, i))
        assert len(pts) == 4  # 2 layers × 2 points
        names = [p.name for p in pts]
        assert "L00.o_proj" in names and "L00.down_proj" in names
        assert "L01.o_proj" in names and "L01.down_proj" in names

    def test_hooks_fire_at_real_modules(self):
        model = tiny_llama(layers=2, hidden=32)
        wrapper = SmolLM2BlockWrapper()
        pts = wrapper.get_injection_points(next(iter(get_block_container(model))), 0)
        # each InjectionPoint points at the real projection module
        assert pts[0].module is model.model.layers[0].self_attn.o_proj
        assert pts[1].module is model.model.layers[0].mlp.down_proj
        assert isinstance(pts[0].module, torch.nn.Linear)

    def test_out_features_match_hidden(self):
        model = tiny_llama(layers=2, hidden=32)
        wrapper = SmolLM2BlockWrapper()
        container = get_block_container(model)
        pts = wrapper.get_injection_points(container[0], 0)
        for p in pts:
            assert p.bias.shape == (32,)
            assert p.out_features == 32


class TestGPT2Wrapper:
    def test_returns_two_injection_points_per_block(self):
        model = tiny_gpt2(layers=3)
        wrapper = GPT2BlockWrapper()
        pts = []
        for i, block in enumerate(get_block_container(model)):
            pts.extend(wrapper.get_injection_points(block, i))
        assert len(pts) == 6
        names = {p.name for p in pts}
        assert {"L00.attn_c_proj", "L00.mlp_c_proj", "L02.attn_c_proj"} <= names

    def test_conv1d_nf_handling(self):
        model = tiny_gpt2(hidden=32)
        block = get_block_container(model)[0]
        mod = block.attn.c_proj
        assert type(mod).__name__ == "Conv1D"  # HF Conv1D (plain nn.Module)
        # HF's Conv1D has .nf but no .out_features/.in_features (verified 2026-08-12)
        assert mod.nf == 32
        assert not hasattr(mod, "out_features")
        assert not hasattr(mod, "in_features")
        assert _get_out_features(mod) == 32
        wrapper = GPT2BlockWrapper()
        pts = wrapper.get_injection_points(block, 0)
        assert pts[0].out_features == 32
        assert pts[0].bias.shape == (32,)

    def test_hooks_fire_at_conv1d(self):
        model = tiny_gpt2()
        wrapper = GPT2BlockWrapper()
        pts = wrapper.get_injection_points(get_block_container(model)[0], 0)
        assert pts[0].module is model.transformer.h[0].attn.c_proj
        assert pts[1].module is model.transformer.h[0].mlp.c_proj


class TestFactories:
    def test_detection_by_model_type(self):
        assert isinstance(get_block_wrapper(tiny_llama()), SmolLM2BlockWrapper)
        assert isinstance(get_block_wrapper(tiny_gpt2()), GPT2BlockWrapper)

    def test_container_paths(self):
        llama = tiny_llama(layers=2)
        gpt2 = tiny_gpt2(layers=3)
        assert get_block_container(llama) is llama.model.layers
        assert get_block_container(gpt2) is gpt2.transformer.h

    def test_unsupported_model_type_raises(self):
        class FakeCfg:
            model_type = "bert"

        class FakeModel:
            config = FakeCfg()

        with pytest.raises(NotImplementedError):
            get_block_wrapper(FakeModel())
        with pytest.raises(NotImplementedError):
            get_block_container(FakeModel())


class TestHookCapture:
    def test_post_capture_shape(self):
        model = tiny_llama(layers=2, hidden=32, device="cpu")
        wrapper = SmolLM2BlockWrapper()
        pts = []
        for i, block in enumerate(get_block_container(model)):
            pts.extend(wrapper.get_injection_points(block, i))
        captured = {}

        def make_hook(name):
            def hook(module, args, output):
                captured[name] = output.detach().float()
                return output
            return hook

        handles = [p.module.register_forward_hook(make_hook(p.name)) for p in pts]
        ids = random_token_ids(batch=2, seq=16)
        with torch.no_grad():
            model(input_ids=ids)
        for h in handles:
            h.remove()
        assert set(captured) == {p.name for p in pts}
        for out in captured.values():
            assert tuple(out.shape) == (2, 16, 32)  # (B, S, d_out)
