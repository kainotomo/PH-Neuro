"""Latent score storage paired with ternary weights.

Each ternary weight has an associated float score (fp16) that tracks the
cumulative Hebbian evidence. The ternary weight is derived from the latent
score via a hysteresis threshold mechanism.
"""

from __future__ import annotations

import torch


class LatentScoreTensor:
    """fp16 scores paired with each ternary weight.

    The latent score accumulates Hebbian updates over time. Ternary weights
    are derived by comparing scores against upper/lower hysteresis thresholds.

    Args:
        shape: Tensor shape (typically ``(out_features, in_features)``).
        init_std: Standard deviation for random normal initialization.
        device: Torch device.

    Attributes:
        scores: fp16 tensor of the same shape.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        init_std: float = 0.1,
        device: torch.device | str | None = None,
    ):
        self._scores: torch.Tensor = (
            torch.randn(shape, dtype=torch.float16, device=device) * init_std
        )

    @property
    def scores(self) -> torch.Tensor:
        """The latent score tensor (fp16)."""
        return self._scores

    @scores.setter
    def scores(self, value: torch.Tensor) -> None:
        self._scores = value

    def get_ternary(
        self,
        theta_upper: float,
        theta_lower: float,
        current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert scores to ternary weights using hysteresis.

        When ``current`` is provided, full hysteresis logic applies:
        - If current weight is 0 and ``|score| > theta_upper`` → activate to ``sign(score)``
        - If current weight is ±1 and ``|score| < theta_lower`` → deactivate to 0
        - Otherwise → keep current weight

        When ``current`` is ``None``, simple thresholding is used:
        ``|score| > theta_upper → sign(score)``, everything else 0.

        Args:
            theta_upper: Threshold to activate a synapse (0 → ±1).
            theta_lower: Threshold to deactivate a synapse (±1 → 0).
            current: Optional current ternary weights to use for hysteresis.

        Returns:
            int8 tensor with values in {-1, 0, +1}.
        """
        if current is not None:
            # Full hysteresis: use current weights as starting point
            ternary = current.clone().to(dtype=torch.int8)
            # Activate: current == 0 and |score| > theta_upper
            activate = (current == 0) & (self._scores.abs() > theta_upper)
            ternary[activate] = self._scores[activate].sign().to(torch.int8)
            # Deactivate: current != 0 and |score| < theta_lower
            deactivate = (current != 0) & (self._scores.abs() < theta_lower)
            ternary[deactivate] = 0
            return ternary
        else:
            # Simple thresholding (backward-compatible)
            ternary = torch.zeros_like(self._scores, dtype=torch.int8)
            active_mask = self._scores.abs() > theta_upper
            ternary[active_mask] = self._scores[active_mask].sign().to(torch.int8)
            return ternary

    def apply_hebbian(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        lr: float,
    ) -> None:
        """Apply Hebbian update in-place.

        ``Δscore = lr × postᵀ @ pre / batch_size``

        Args:
            pre: Pre-activations, shape ``(batch, in_features)``, ternary {-1,0,+1}.
            post: Post-activations, shape ``(batch, out_features)``, ternary {-1,0,+1}.
            lr: Learning rate.
        """
        delta = lr * (post.T.to(self._scores.dtype) @ pre.to(self._scores.dtype))
        delta = delta / pre.shape[0]
        self._scores += delta

    def apply_decay(self, decay_rate: float) -> None:
        """Homeostatic decay: ``score -= decay_rate × score``.

        Slowly drifts unused scores toward zero, preventing unbounded growth.

        Args:
            decay_rate: Decay factor.
        """
        self._scores -= decay_rate * self._scores

    def clone(self) -> LatentScoreTensor:
        """Return a deep copy."""
        t = LatentScoreTensor.__new__(LatentScoreTensor)
        t._scores = self._scores.clone()
        return t

    def __repr__(self) -> str:
        return f"LatentScoreTensor(shape={tuple(self._scores.shape)}, dtype={self._scores.dtype})"
