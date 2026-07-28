"""Tests for TernaryHebbianConv2d layer.

Tests cover:
    - Layer creation and properties
    - Forward pass correctness (shape, no autograd graph)
    - Ternary weight invariant
    - Hebbian update mechanics
    - Hysteresis / refresh_weights
    - Freeze control
    - Device movement
"""

from __future__ import annotations

import pytest
import torch

from ph_neuro.layers.conv import TernaryHebbianConv2d


# ── Helpers ──────────────────────────────────────────────────────


def _make_input(
    n: int = 4,
    c: int = 3,
    h: int = 8,
    w: int = 8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a random float input tensor."""
    return torch.randn(n, c, h, w, device=device)


def _check_ternary(w: torch.Tensor) -> bool:
    return bool(torch.all((w == -1) | (w == 0) | (w == 1)).item())


def _expected_conv_out(in_size, kernel, stride, padding, dilation=1):
    """Compute expected output spatial size."""
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


# ── Tests ────────────────────────────────────────────────────────


class TestCreate:
    """Verify layer creation and basic properties."""

    def test_default_params(self):
        """Creating with default params should work."""
        layer = TernaryHebbianConv2d(3, 64, kernel_size=3)
        assert layer.in_channels == 3
        assert layer.out_channels == 64
        assert layer.kernel_size == (3, 3)

    def test_weights_initialized_zero(self):
        """Weights should start as all zeros."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        w = layer.weight.unpack()
        assert torch.all(w == 0)

    def test_scores_initialized_random(self):
        """Latent scores should be small random values, not all zero."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        scores = layer.latent_scores
        assert not torch.all(scores == 0)
        assert scores.shape == (4, 3, 3, 3)
        assert scores.dtype == torch.float16

    def test_tuple_kernel(self):
        """Kernel size as tuple should work."""
        layer = TernaryHebbianConv2d(3, 8, kernel_size=(3, 5), padding=(1, 2), stride=2)
        assert layer.kernel_size == (3, 5)

    def test_dilation(self):
        """Dilation should be stored correctly."""
        layer = TernaryHebbianConv2d(3, 8, kernel_size=3, dilation=2)
        assert layer._dilation == (2, 2)

    def test_extra_repr(self):
        """extra_repr should return a non-empty string."""
        layer = TernaryHebbianConv2d(3, 8, kernel_size=3)
        assert len(layer.extra_repr()) > 10


class TestForwardShape:
    """Verify forward pass output shape."""

    @pytest.mark.parametrize(
        "in_c, out_c, k, s, p, in_h, in_w, expected_h, expected_w",
        [
            (3, 8, 3, 1, 0, 8, 8, 6, 6),  # no padding
            (3, 8, 3, 1, 1, 8, 8, 8, 8),  # same padding
            (3, 16, 3, 2, 1, 8, 8, 4, 4),  # stride 2
            (1, 4, 5, 1, 2, 10, 10, 10, 10),  # 5×5 kernel with padding
        ],
    )
    def test_output_shape(self, in_c, out_c, k, s, p, in_h, in_w, expected_h, expected_w):
        """Output shape should match expected convolution output."""
        layer = TernaryHebbianConv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        # Bootstrap weights so output is non-zero
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)

        x = _make_input(n=4, c=in_c, h=in_h, w=in_w)
        out = layer(x)
        assert out.shape == (4, out_c, expected_h, expected_w), (
            f"Expected (4, {out_c}, {expected_h}, {expected_w}), got {out.shape}"
        )

    def test_cifar10_input_shape(self):
        """CIFAR-10 input (3×32×32) through conv(3→64, 3×3, pad=1) should be (N, 64, 32, 32)."""
        layer = TernaryHebbianConv2d(3, 64, kernel_size=3, padding=1)
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)

        x = _make_input(n=2, c=3, h=32, w=32)
        out = layer(x)
        assert out.shape == (2, 64, 32, 32)


class TestForwardTernary:
    """Verify forward pass uses ternary weights."""

    def test_weights_ternary_after_init(self):
        """Weights must be in {-1, 0, +1} after creation."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        w = layer.weight.unpack()
        assert _check_ternary(w)

    def test_weights_ternary_after_forward(self):
        """Forward pass should not change weight tensor properties."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)
        w_before = layer.weight.unpack().clone()
        assert _check_ternary(w_before)

        x = _make_input(n=2, c=3, h=8, w=8)
        _ = layer(x)
        w_after = layer.weight.unpack()
        assert _check_ternary(w_after), "Weights must remain ternary after forward pass"

    def test_no_autograd_graph(self):
        """Forward pass should not create autograd graph (requires_grad must be False)."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)
        x = _make_input(n=2, c=3, h=8, w=8)

        with torch.no_grad():
            out = layer(x)

        assert not out.requires_grad, "Output should not require grad"


class TestHebbianUpdate:
    """Verify Hebbian update mechanics."""

    def test_update_changes_scores(self):
        """Hebbian update should change latent scores."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)
        x = _make_input(n=2, c=3, h=8, w=8)

        # Forward to get output
        out = layer(x)

        scores_before = layer.latent_scores.clone()
        post = torch.sign(out).to(torch.int8)

        layer.hebbian_update(x, post, lr=0.01)
        scores_after = layer.latent_scores

        assert not torch.allclose(scores_before, scores_after), (
            "Scores should change after Hebbian update"
        )

    def test_update_no_op_when_frozen(self):
        """Hebbian update should be a no-op when layer is frozen."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)
        x = _make_input(n=2, c=3, h=8, w=8)
        out = layer(x)

        layer.requires_hebbian_(False)
        scores_before = layer.latent_scores.clone()
        post = torch.sign(out).to(torch.int8)

        layer.hebbian_update(x, post, lr=1.0)
        scores_after = layer.latent_scores

        assert torch.equal(scores_before, scores_after), (
            "Scores should not change when layer is frozen"
        )

    def test_update_with_padding(self):
        """Hebbian update should work with non-zero padding."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3, padding=1)
        from ph_neuro.training.greedy import _init_conv_connectivity
        _init_conv_connectivity(layer, density=0.5)
        x = _make_input(n=2, c=3, h=8, w=8)
        out = layer(x)

        post = torch.sign(out).to(torch.int8)
        layer.hebbian_update(x, post, lr=0.01)
        # Just verify it doesn't crash — scores changed
        assert not torch.all(layer.latent_scores == 0)


class TestRefreshWeights:
    """Verify hysteresis threshold mechanism."""

    def test_refresh_preserves_ternary(self):
        """refresh_weights should keep weights in {-1, 0, +1}."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        layer.refresh_weights()
        w = layer.weight.unpack()
        assert _check_ternary(w)

    def test_high_scores_activate_weights(self):
        """Scores above theta_upper should activate weights."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3, theta_upper=5.0, theta_lower=1.0)
        # Force some scores above theta_upper
        layer._latent_scores.scores[0, 0, 1, 1] = 10.0
        layer._latent_scores.scores[1, 0, 0, 0] = -8.0

        layer.refresh_weights()
        w = layer.weight.unpack()

        assert w[0, 0, 1, 1] == 1, "Positive high score should activate to +1"
        assert w[1, 0, 0, 0] == -1, "Negative high score should activate to -1"

    def test_low_scores_deactivate_weights(self):
        """Scores below theta_lower should deactivate active weights."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3, theta_upper=5.0, theta_lower=1.0)
        # Manually set a weight active with a low score — should deactivate
        # First, activate it
        layer._latent_scores.scores[0, 0, 1, 1] = 10.0
        layer.refresh_weights()
        assert layer.weight.unpack()[0, 0, 1, 1] == 1

        # Now drop the score below theta_lower
        layer._latent_scores.scores[0, 0, 1, 1] = 0.5
        layer.refresh_weights()
        assert layer.weight.unpack()[0, 0, 1, 1] == 0, (
            "Score below theta_lower should deactivate weight"
        )

    def test_hysteresis_gap(self):
        """Hysteresis gap should prevent flip-flopping."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3, theta_upper=5.0, theta_lower=1.0)
        # Set score in the gap (between lower and upper)
        layer._latent_scores.scores[0, 0, 1, 1] = 3.0

        # Start with weight = 0 — should NOT activate (3 < 5)
        layer.refresh_weights()
        assert layer.weight.unpack()[0, 0, 1, 1] == 0

        # Now start with weight = +1 — should NOT deactivate (3 > 1)
        layer._ternary_weight._data[0, 0, 1, 1] = 1
        layer._latent_scores.scores[0, 0, 1, 1] = 3.0
        layer.refresh_weights()
        assert layer.weight.unpack()[0, 0, 1, 1] == 1, (
            "Weight should stay active in hysteresis gap"
        )


class TestFreeze:
    """Verify freeze/unfreeze logic."""

    def test_requires_hebbian_default(self):
        """Layer should be Hebbian-enabled by default."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        assert layer._hebbian_enabled

    def test_requires_hebbian_false(self):
        """Freezing should disable Hebbian updates."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        result = layer.requires_hebbian_(False)
        assert not layer._hebbian_enabled
        assert result is layer  # should return self

    def test_requires_hebbian_true(self):
        """Unfreezing should re-enable Hebbian updates."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        layer.requires_hebbian_(False)
        layer.requires_hebbian_(True)
        assert layer._hebbian_enabled


class TestDeviceMovement:
    """Verify .to() moves all internal tensors."""

    def test_to_cpu(self):
        """Moving to CPU should work (always available)."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        device = torch.device("cpu")
        layer = layer.to(device)
        assert layer.weight._data.device == device
        assert layer.latent_scores.device == device

    @pytest.mark.cuda
    def test_to_cuda(self):
        """Moving to CUDA should work when GPU is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3).to(device)
        assert layer.weight._data.device.type == "cuda"
        assert layer.latent_scores.device.type == "cuda"


class TestDecay:
    """Verify homeostatic decay."""

    def test_decay_reduces_scores(self):
        """Apply decay should reduce score magnitudes."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        layer._latent_scores.scores.fill_(10.0)
        layer.apply_decay(decay_rate=0.1)

        expected = 10.0 - 0.1 * 10.0  # = 9.0
        assert torch.allclose(
            layer.latent_scores,
            torch.full_like(layer.latent_scores, expected),
        ), "Decay should reduce scores"

    def test_decay_no_op_when_frozen(self):
        """Apply decay should be a no-op when layer is frozen."""
        layer = TernaryHebbianConv2d(3, 4, kernel_size=3)
        layer._latent_scores.scores.fill_(10.0)
        layer.requires_hebbian_(False)
        layer.apply_decay(decay_rate=0.1)

        assert torch.allclose(
            layer.latent_scores,
            torch.full_like(layer.latent_scores, 10.0),
        ), "Decay should not apply when frozen"


class TestCompetitiveHebbian:
    """Verify the conv competitive Hebbian training path."""

    def test_competitive_epoch_updates_weights(self):
        """Competitive Hebbian epoch should change weights on synthetic data."""
        from ph_neuro.training.greedy import _init_conv_connectivity, train_conv_competitive_epoch
        from torch.utils.data import DataLoader, TensorDataset

        layer = TernaryHebbianConv2d(3, 4, kernel_size=3, theta_upper=1.0, theta_lower=0.3)
        _init_conv_connectivity(layer, density=0.5)

        # Use structured input (all +1) so Hebbian update is strong
        x = torch.ones(8, 3, 8, 8)
        y = torch.randint(0, 3, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        w_before = layer.weight.unpack().clone()
        metrics = train_conv_competitive_epoch(
            conv_layer=layer,
            loader=loader,
            frozen_encoder=None,
            device=torch.device("cpu"),
            lr=1.0,
            decay=0.0,
            epsilon=0.0,
        )

        w_after = layer.weight.unpack()
        assert not torch.equal(w_before, w_after), "Weights should change after competitive epoch"
        assert 0.0 <= metrics["flip_rate"] <= 1.0
        assert metrics["n_flips"] >= 0
