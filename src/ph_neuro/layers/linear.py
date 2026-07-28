"""Ternary Hebbian linear layer.

The fundamental building block: a fully-connected layer with ternary
weights learned via Hebbian plasticity instead of backpropagation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.core.latent_scores import LatentScoreTensor
from ph_neuro.core.ternary_tensor import TernaryTensor


def ternary_sign(x: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    """Map activations to {-1, 0, +1}.

    Args:
        x: Input tensor.
        epsilon: Values in ``(-epsilon, +epsilon)`` map to 0.
            With ``epsilon=0``, only exact zeros stay 0.

    Returns:
        int8 tensor with values in {-1, 0, +1}.
    """
    if epsilon > 0:
        x = torch.where(torch.abs(x) < epsilon, torch.zeros_like(x), x)
    return torch.sign(x).to(torch.int8)


class TernaryHebbianLinear(nn.Module):
    """Linear layer with ternary weights and Hebbian learning.

    Stores weights natively as {-1, 0, +1} with associated latent float
    scores. The forward pass uses ternary weights. Learning updates the
    latent scores via a Hebbian rule; ternary weights are periodically
    refreshed via a hysteresis threshold mechanism.

    No ``.backward()`` is called for learning — this layer learns purely
    through local Hebbian plasticity.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        theta_upper: Hysteresis upper threshold — latent score must exceed
            this to activate a synapse (0 → ±1).
        theta_lower: Hysteresis lower threshold — latent score must fall
            below this to deactivate a synapse (±1 → 0).
        bias: If ``True``, adds a learnable bias.
        hebbian_rule: One of ``'basic'``, ``'anti'``, ``'oja'``, ``'bcm'``.
            Default is ``'basic'``.

    Attributes:
        weight: Ternary weight tensor (read-only view via ``TernaryTensor``).
        latent_scores: fp16 latent scores accessible for inspection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta_upper: float = 5.0,
        theta_lower: float = 1.0,
        bias: bool = False,
        hebbian_rule: str = "basic",
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._theta_upper = theta_upper
        self._theta_lower = theta_lower
        self._hebbian_rule = hebbian_rule
        self._device = device

        # Ternary weight storage (starts all zeros)
        self._ternary_weight = TernaryTensor(
            (out_features, in_features), packed=False, device=device
        )

        # Latent scores (small random init)
        self._latent_scores = LatentScoreTensor(
            (out_features, in_features), init_std=0.1, device=device
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> TernaryTensor:
        """Ternary weight tensor (read-only)."""
        return self._ternary_weight

    @property
    def latent_scores(self) -> torch.Tensor:
        """fp16 latent scores (read-only)."""
        return self._latent_scores.scores

    def to(self, *args, **kwargs):
        """Move all tensors (including custom storage) to the given device."""
        super().to(*args, **kwargs)
        # Extract device from args/kwargs — nn.Module.to accepts device, dtype, or
        # (device, dtype) positional, or device= keyword. We handle all variants.
        device = None
        if args:
            device = args[0] if isinstance(args[0], (torch.device, str, int)) else args[0][0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            device = torch.device(device)
            self._ternary_weight._data = self._ternary_weight._data.to(device)
            self._latent_scores.scores = self._latent_scores.scores.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: ternary MatMul + optional sign activation.

        Args:
            x: Input tensor, shape ``(batch, in_features)``.

        Returns:
            Output tensor, shape ``(batch, out_features)``.
        """
        # Ensure weights are on the same device as input
        w = self._ternary_weight.to_dense()
        if w.device != x.device:
            w = w.to(x.device)
        out = F.linear(x.to(w.dtype), w)
        if self.bias is not None:
            out = out + self.bias
        return out

    def hebbian_update(
        self,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor,
        lr: float,
    ) -> None:
        """Apply Hebbian update to latent scores.

        Args:
            pre_activation: Pre-synaptic activations, shape ``(batch, in_features)``.
                Should be in {-1, 0, +1} (ternary).
            post_activation: Post-synaptic activations, shape ``(batch, out_features)``.
                Should be in {-1, 0, +1} (ternary).
            lr: Learning rate for the Hebbian update.
        """
        pre = pre_activation.to(torch.int8)
        post = post_activation.to(torch.int8)
        # Ensure latent scores are on the same device
        if self._latent_scores.scores.device != pre.device:
            self._latent_scores.scores = self._latent_scores.scores.to(pre.device)

        if self._hebbian_rule == "basic":
            self._latent_scores.apply_hebbian(pre, post, lr)
        elif self._hebbian_rule == "anti":
            delta = -lr * (pre.T.float() @ post.float()) / pre.shape[0]
            self._latent_scores.scores += delta
        elif self._hebbian_rule == "oja":
            import ph_neuro.core.hebbian_rules as rules

            self._latent_scores.scores = rules.oja_update(self._latent_scores.scores, pre, post, lr)
        elif self._hebbian_rule == "bcm":
            import ph_neuro.core.hebbian_rules as rules

            self._latent_scores.scores = rules.bcm_update(self._latent_scores.scores, pre, post, lr)
        else:
            raise ValueError(f"Unknown Hebbian rule: {self._hebbian_rule}")

    def apply_decay(self, decay_rate: float) -> None:
        """Apply homeostatic decay to latent scores.

        Args:
            decay_rate: Decay factor.
        """
        self._latent_scores.apply_decay(decay_rate)

    def refresh_weights(self) -> None:
        """Refresh ternary weights from latent scores using hysteresis.

        For each synapse:
        - If weight is 0 and ``|score| > theta_upper`` → flip to ``sign(score)``
        - If weight is ±1 and ``|score| < theta_lower`` → flip to 0

        The hysteresis gap ``(theta_upper - theta_lower)`` prevents oscillation.
        """
        scores = self._latent_scores.scores
        current_weights = self._ternary_weight.unpack()

        # Ensure both tensors are on the same device
        if scores.device != current_weights.device:
            current_weights = current_weights.to(scores.device)

        new_weights = current_weights.clone()

        # Activate: weight == 0 and |score| > theta_upper
        activate_mask = (current_weights == 0) & (scores.abs() > self._theta_upper)
        new_weights[activate_mask] = scores[activate_mask].sign().to(torch.int8)

        # Deactivate: weight != 0 and |score| < theta_lower
        deactivate_mask = (current_weights != 0) & (scores.abs() < self._theta_lower)
        new_weights[deactivate_mask] = 0

        self._ternary_weight = TernaryTensor(
            (self._out_features, self._in_features),
            packed=self._ternary_weight.packed,
            device=self._device,
        )
        # Rebuild the ternary tensor with new weights
        if self._ternary_weight.packed:
            self._ternary_weight = TernaryTensor.pack(new_weights)
        else:
            self._ternary_weight._data = new_weights

    def extra_repr(self) -> str:
        return (
            f"in_features={self._in_features}, "
            f"out_features={self._out_features}, "
            f"theta_upper={self._theta_upper}, "
            f"theta_lower={self._theta_lower}, "
            f"rule={self._hebbian_rule}"
        )
