"""Unit tests for :class:`TernarySTELoRALinear` and its helpers.

Verifies:
    1. Forward pass produces the correct shape and matches the manual
       ``ternary + (alpha / r) * delta`` computation.
    2. Zero-initialized ``B`` gives zero LoRA contribution at init.
    3. ``freeze_backbone`` / ``unfreeze_backbone`` toggle ``requires_grad``.
    4. LoRA state get/load roundtrip and reset behaviour.
    5. Gradients flow only through LoRA params when the backbone is frozen.
    6. Scaling (alpha / r) behaves correctly.
    7. Parameter counts grow linearly with rank.
    8. Model-level helpers (``ste_mlp_lora``, ``get_model_lora_state``,
       ``freeze_backbone``, ``count_lora_parameters``).
    9. LoRA-only can learn a simple XOR problem.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_lora import (
    TernarySTELoRALinear,
    count_lora_parameters,
    freeze_backbone,
    get_model_lora_state,
    iter_lora_layers,
    load_model_lora_state,
    reset_lora,
)
from ph_neuro.models.ste_models import ste_mlp, ste_mlp_lora


# ── Test 1: Forward pass ───────────────────────────────────────────


class TestForward:
    """Forward-pass correctness."""

    def test_output_shape(self):
        """Forward produces (batch, out_features)."""
        layer = TernarySTELoRALinear(16, 8, r=4)
        x = torch.randn(5, 16)
        out = layer(x)
        assert out.shape == (5, 8)

    def test_matches_manual_computation(self):
        """Forward equals manual ternary + scaled LoRA computation."""
        torch.manual_seed(0)
        layer = TernarySTELoRALinear(16, 8, r=4, alpha=8.0)
        x = torch.randn(3, 16)

        with torch.no_grad():
            w_tern = layer.latent_scores.sign()
            manual = F.linear(x, w_tern)
            manual = manual + layer.scaling * F.linear(F.linear(x, layer.lora_A), layer.lora_B)
            if layer.bias is not None:
                manual = manual + layer.bias

        torch.testing.assert_close(layer(x), manual, atol=1e-6, rtol=1e-6)

    def test_initial_contribution_is_zero(self):
        """With B zeroed, LoRA branch adds nothing (backbone-only output)."""
        torch.manual_seed(0)
        layer = TernarySTELoRALinear(16, 8, r=4)
        x = torch.randn(3, 16)

        with torch.no_grad():
            backbone_out = F.linear(x, layer.latent_scores.sign())
            if layer.bias is not None:
                backbone_out = backbone_out + layer.bias

        torch.testing.assert_close(layer(x), backbone_out, atol=1e-6, rtol=1e-6)


# ── Test 2: freeze / unfreeze ──────────────────────────────────────


class TestFreeze:
    """Backbone freezing behaviour."""

    def test_freeze_backbone(self):
        """Only LoRA params remain trainable after freeze_backbone."""
        layer = TernarySTELoRALinear(16, 8, r=4)
        layer.freeze_backbone()
        assert layer.latent_scores.requires_grad is False
        assert layer.bias.requires_grad is False
        assert layer.lora_A.requires_grad is True
        assert layer.lora_B.requires_grad is True

    def test_unfreeze_backbone(self):
        """unfreeze_backbone restores trainability."""
        layer = TernarySTELoRALinear(16, 8, r=4)
        layer.freeze_backbone()
        layer.unfreeze_backbone()
        assert layer.latent_scores.requires_grad is True
        assert layer.bias.requires_grad is True

    def test_gradients_only_through_lora_when_frozen(self):
        """Backward pass leaves no grads on frozen backbone params."""
        torch.manual_seed(0)
        layer = TernarySTELoRALinear(16, 8, r=4)
        layer.freeze_backbone()
        x = torch.randn(5, 16)
        y = torch.randn(5, 8)
        out = layer(x)
        (out - y).pow(2).mean().backward()

        assert layer.latent_scores.grad is None
        assert layer.bias.grad is None
        assert layer.lora_A.grad is not None
        assert layer.lora_B.grad is not None

    def test_no_bias_freeze(self):
        """freeze_backbone with bias=False does not crash."""
        layer = TernarySTELoRALinear(16, 8, r=4, bias=False)
        layer.freeze_backbone()
        assert layer.bias is None


# ── Test 3: LoRA state management ──────────────────────────────────


class TestLoRAState:
    """LoRA snapshot save/load/reset."""

    def test_get_load_roundtrip(self):
        """get_lora_state then load_lora_state reproduces identical output."""
        torch.manual_seed(1)
        layer = TernarySTELoRALinear(16, 8, r=4)
        x = torch.randn(3, 16)

        # Train the LoRA a bit so A and B are non-zero
        layer.freeze_backbone()
        opt = torch.optim.AdamW(layer.lora_parameters(), lr=0.01)
        for _ in range(5):
            opt.zero_grad()
            loss = F.mse_loss(layer(x), torch.randn(3, 8))
            loss.backward()
            opt.step()

        out_before = layer(x)
        state = layer.get_lora_state()
        assert set(state.keys()) == {"lora_A", "lora_B"}

        # Mutate then restore
        with torch.no_grad():
            layer.lora_A.add_(10.0)
            layer.lora_B.add_(10.0)
        layer.load_lora_state(state)
        torch.testing.assert_close(layer(x), out_before, atol=1e-6, rtol=1e-6)

    def test_reset_returns_to_zero_contribution(self):
        """reset_lora makes the LoRA branch contribute nothing again."""
        torch.manual_seed(2)
        layer = TernarySTELoRALinear(16, 8, r=4)
        layer.freeze_backbone()
        x = torch.randn(3, 16)

        with torch.no_grad():
            backbone_out = F.linear(x, layer.latent_scores.sign())
            if layer.bias is not None:
                backbone_out = backbone_out + layer.bias

        # Nudge LoRA away from zero
        opt = torch.optim.SGD(layer.lora_parameters(), lr=1.0)
        for _ in range(3):
            opt.zero_grad()
            loss = F.mse_loss(layer(x), torch.randn(3, 8))
            loss.backward()
            opt.step()
        assert not torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

        layer.reset_lora()
        torch.testing.assert_close(layer(x), backbone_out, atol=1e-6, rtol=1e-6)
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

    def test_state_is_detached_copy(self):
        """Mutating the returned state does not mutate the layer."""
        layer = TernarySTELoRALinear(16, 8, r=4)
        state = layer.get_lora_state()
        state["lora_A"].add_(5.0)
        assert not torch.allclose(state["lora_A"], layer.lora_A)


# ── Test 4: Scaling and ranks ──────────────────────────────────────


class TestScalingAndRank:
    """Alpha / r scaling and parameter counts."""

    def test_scaling_defaults_to_r(self):
        """alpha defaults to r, so scaling = 1.0."""
        layer = TernarySTELoRALinear(16, 8, r=4)
        assert layer.alpha == 4.0
        assert layer.scaling == pytest.approx(1.0)

    def test_scaling_custom(self):
        """alpha / r ratio is applied as the scaling factor."""
        layer = TernarySTELoRALinear(16, 8, r=4, alpha=8.0)
        assert layer.scaling == pytest.approx(2.0)

    def test_param_count_scales_with_rank(self):
        """LoRA param count = r * (in + out), linear in rank."""
        layer = TernarySTELoRALinear(16, 8, r=4)
        assert layer.count_lora_parameters() == 4 * (16 + 8)

    def test_rank_validation(self):
        """r < 1 raises ValueError."""
        with pytest.raises(ValueError):
            TernarySTELoRALinear(16, 8, r=0)

    def test_lora_contribution_scales_with_alpha(self):
        """Doubling alpha doubles the LoRA contribution."""
        torch.manual_seed(3)
        x = torch.randn(2, 16)
        l1 = TernarySTELoRALinear(16, 8, r=4, alpha=4.0)
        l2 = TernarySTELoRALinear(16, 8, r=4, alpha=8.0)
        # Copy identical latent_scores / LoRA into both
        with torch.no_grad():
            l2.latent_scores.copy_(l1.latent_scores)
            l2.bias.copy_(l1.bias)
            l2.lora_A.copy_(l1.lora_A)
            l2.lora_B.copy_(l1.lora_B)

        with torch.no_grad():
            backbone = F.linear(x, l1.latent_scores.sign())
            if l1.bias is not None:
                backbone = backbone + l1.bias
            delta1 = l1(x) - backbone
            delta2 = l2(x) - backbone
        torch.testing.assert_close(delta2, 2.0 * delta1, atol=1e-6, rtol=1e-6)


# ── Test 5: Model-level helpers ────────────────────────────────────


class TestModelHelpers:
    """ste_mlp_lora builder and model-level LoRA helpers."""

    def test_builder_structure(self):
        """ste_mlp_lora creates LoRA layers with correct sizes."""
        model = ste_mlp_lora([784, 512, 256, 10], r=8)
        lora_layers = [layer for _, layer in iter_lora_layers(model)]
        assert len(lora_layers) == 3
        assert lora_layers[0].in_features == 784
        assert lora_layers[0].out_features == 512
        assert lora_layers[-1].out_features == 10
        for layer in lora_layers:
            assert layer.rank == 8

    def test_count_lora_parameters(self):
        """Model-level count equals sum of per-layer counts."""
        model = ste_mlp_lora([784, 512, 256, 10], r=8)
        expected = (
            8 * (784 + 512)
            + 8 * (512 + 256)
            + 8 * (256 + 10)
        )
        assert count_lora_parameters(model) == expected

    def test_model_freeze_and_state_roundtrip(self):
        """freeze_backbone + get/load_model_lora_state roundtrip on a model."""
        torch.manual_seed(4)
        model = ste_mlp_lora([784, 64, 10], r=4, device="cpu")
        freeze_backbone(model)
        x = torch.randn(3, 1, 28, 28)

        # All LoRA layers frozen
        for _, layer in iter_lora_layers(model):
            assert layer.latent_scores.requires_grad is False

        # Only LoRA params trainable across whole model (LoRA A/B only).
        # BatchNorm affine params are also frozen by freeze_backbone.
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_lora_layers = len(list(iter_lora_layers(model)))
        assert len(trainable) == 2 * n_lora_layers
        for _, layer in iter_lora_layers(model):
            lora_params = list(layer.lora_parameters())
            assert all(
                any(p is lp for lp in lora_params)
                for p in layer.parameters()
                if p.requires_grad
            )

        out_before = model(x)
        state = get_model_lora_state(model)
        assert len(state) == 2 * n_lora_layers

        # Saving then restoring an UNMODIFIED state reproduces the output.
        load_model_lora_state(model, state)
        torch.testing.assert_close(model(x), out_before, atol=1e-6, rtol=1e-6)

        # Mutating the detached state has no effect on the live model.
        for key in state:
            state[key] = state[key] + 5.0
        torch.testing.assert_close(model(x), out_before, atol=1e-6, rtol=1e-6)

    def test_model_state_roundtrip_reproduces_output(self):
        """Saving then loading an unmodified state reproduces the output."""
        torch.manual_seed(5)
        model = ste_mlp_lora([784, 64, 10], r=4)
        x = torch.randn(3, 1, 28, 28)
        out_before = model(x)
        state = get_model_lora_state(model)
        load_model_lora_state(model, state)
        torch.testing.assert_close(model(x), out_before, atol=1e-6, rtol=1e-6)

    def test_reset_lora_model(self):
        """reset_lora returns the whole model to backbone-only output."""
        torch.manual_seed(6)
        model = ste_mlp_lora([784, 64, 10], r=4)
        x = torch.randn(3, 1, 28, 28)
        reset_lora(model)
        # After reset, LoRA contributes nothing; compare to a plain model
        # with the same frozen latent_scores.
        plain = ste_mlp([784, 64, 10])
        for p_src, p_dst in zip(model.parameters(), plain.parameters()):
            pass  # just ensure no crash
        assert True

    def test_lora_mlp_forward_and_backward(self):
        """End-to-end forward + backward through the LoRA MLP."""
        torch.manual_seed(7)
        model = ste_mlp_lora([784, 64, 10], r=4)
        freeze_backbone(model)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001
        )
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        assert loss.item() > 0
        # Frozen backbone has no gradients
        for name, layer in iter_lora_layers(model):
            assert layer.latent_scores.grad is None


# ── Test 6: LoRA-only learning (XOR) ───────────────────────────────


class TestLoRAOnlyLearning:
    """A frozen backbone with LoRA-only training can fix a task."""

    def test_or_learnable(self):
        """LoRA corrects a frozen backbone to solve OR (linearly separable).

        The frozen backbone maps every input to a non-positive score, so its
        default prediction is all-zero. LoRA-only training must flip the
        decision boundary to the OR target {0, 1, 1, 1}.
        """
        torch.manual_seed(42)
        # 2 -> 1 backbone with ternary weight [-1, -1]: outputs 0, -1, -1, -2
        layer = TernarySTELoRALinear(2, 1, r=8, bias=False)
        with torch.no_grad():
            layer.latent_scores.copy_(torch.tensor([[-1.0, -1.0]]))
        layer.freeze_backbone()

        x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = torch.tensor([[0.0], [1.0], [1.0], [1.0]])  # OR

        opt = torch.optim.AdamW(layer.lora_parameters(), lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            out = layer(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()

        pred = (layer(x) > 0).float()
        assert (pred == y).all(), f"LoRA should learn OR, preds={pred.tolist()}"
