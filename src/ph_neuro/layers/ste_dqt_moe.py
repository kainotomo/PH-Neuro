"""Mixture of Experts (MoE) layer with DQT ternary experts.

Implements :class:`TernaryDQTMoELayer` — a top-K sparse MoE layer where each
expert is a :class:`TernaryDQTLinear` (Direct Quantized Training, E017) and a
tiny float router selects the top-K experts per sample. Only the selected
experts run in the forward pass (grouped execution per expert), so the active
parameter count is ``top_k / n_experts`` of the expert layer.

Key features:
    - Top-K routing via a small float linear router (784 → N experts).
    - Weighted sum combination: softmax router weights over the selected K
      experts, re-normalized to sum to 1 (convex combination).
    - Load balancing tracking: per-expert selection share + per-sample
      coverage, accumulated over training.
    - Optional Switch-Transformer style auxiliary load balancing loss
      (``aux_load_balance_loss``) to discourage expert collapse.
    - DQT experts store ternary weights as int8 and update them via stochastic
      rounding of a float accumulation buffer (no latent float scores).

Usage::

    moe = TernaryDQTMoELayer(784, 128, n_experts=4, top_k=2)
    out = moe(x)                      # (B, 128) weighted sum of active experts
    fracs = moe.selection_fractions() # load balancing metric (sums to 1)
    aux = moe.aux_load_balance_loss() # optional aux loss for the optimizer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ph_neuro.layers.ste_dqt import TernaryDQTLinear, stochastic_round


# ── MoE Layer ───────────────────────────────────────────────────────


class TernaryDQTMoELayer(nn.Module):
    """Top-K Mixture of Experts layer with TernaryDQTLinear experts.

    Args:
        in_features: Size of each input sample (expert input dim).
        expert_width: Output width of each expert.
        n_experts: Total number of experts.
        top_k: Number of active experts per sample (must be <= n_experts).
        init_std: Init std for the float accumulation buffers of the experts.
        router_init_std: Init std for the (float) router weights.

    Attributes:
        router: Float ``nn.Linear(in_features, n_experts)`` — the router.
        experts: ``ModuleList`` of ``n_experts`` ``TernaryDQTLinear`` layers.
        selection_counts: Buffer counting total selections per expert.
        n_selections: Buffer counting total selections (sum over experts).
        coverage_counts: Buffer counting samples where each expert was active.
        n_samples: Buffer counting total samples routed.
    """

    def __init__(
        self,
        in_features: int,
        expert_width: int,
        n_experts: int,
        top_k: int,
        init_std: float = 0.1,
        router_init_std: float = 0.02,
    ):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k ({top_k}) must be in [1, n_experts={n_experts}]")

        self.in_features = in_features
        self.expert_width = expert_width
        self.n_experts = n_experts
        self.top_k = top_k

        # Tiny float router (not quantized — it is 0.8% of params for the
        # pilot config and needs full precision for stable top-K selection).
        self.router = nn.Linear(in_features, n_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)

        # DQT ternary experts
        self.experts = nn.ModuleList(
            TernaryDQTLinear(in_features, expert_width, bias=False)
            for _ in range(n_experts)
        )
        if abs(init_std - 0.1) > 1e-6:
            # Re-initialize experts with a custom init std
            for expert in self.experts:
                nn.init.normal_(expert.weight_float, mean=0.0, std=init_std)
                expert.weight_ternary = stochastic_round(expert.weight_float.data)

        # Load balancing tracking buffers
        self.register_buffer("selection_counts", torch.zeros(n_experts))
        self.register_buffer("n_selections", torch.zeros(1))
        self.register_buffer("coverage_counts", torch.zeros(n_experts))
        self.register_buffer("n_samples", torch.zeros(1))

        # Stored routing info from the most recent forward (used for the aux
        # loss and metrics). Overwritten every forward pass.
        self.last_logits: torch.Tensor | None = None
        self.last_indices: torch.Tensor | None = None
        self.last_weights: torch.Tensor | None = None
        self.last_probs: torch.Tensor | None = None

    # ── Forward ─────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with top-K routing and grouped expert execution.

        Only the selected experts run: samples are grouped per expert and each
        expert is invoked on exactly the samples that selected it, so the FLOPs
        and memory of the expert layer scale with ``top_k / n_experts``.

        Args:
            x: Input tensor, shape ``(batch, in_features)``.
            return_routing: If ``True``, also return ``(logits, indices,
                weights)``.

        Returns:
            Combined expert output, shape ``(batch, expert_width)`` — the
            weighted sum (re-normalized softmax) of the top-K expert outputs.
            If ``return_routing``, returns a tuple of the output and routing
            tensors.
        """
        batch = x.shape[0]

        # Router: top-K selection
        logits = self.router(x)  # (B, N)
        _, indices = logits.topk(self.top_k, dim=-1)  # (B, K) — Long, no grad

        # Weights: softmax over all experts, take selected, re-normalize to 1
        probs = torch.softmax(logits, dim=-1)  # (B, N)
        weights = probs.gather(1, indices)  # (B, K)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balancing bookkeeping (no grad)
        with torch.no_grad():
            self.selection_counts += torch.bincount(
                indices.flatten(), minlength=self.n_experts
            ).float()
            self.n_selections += indices.numel()
            sel_mask = (
                indices.unsqueeze(-1)
                == torch.arange(self.n_experts, device=indices.device)
            ).any(dim=1)  # (B, N) — which experts each sample picked
            self.coverage_counts += sel_mask.float().sum(dim=0)
            self.n_samples += batch

        # Grouped expert execution — only run experts that were selected.
        combined = torch.zeros(batch, self.expert_width, device=x.device, dtype=x.dtype)
        for e in range(self.n_experts):
            rows, k_pos = (indices == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            expert_out = self.experts[e](x[rows])  # (n, expert_width)
            w = weights[rows, k_pos].unsqueeze(-1)  # (n, 1)
            combined.index_add_(0, rows, expert_out * w)

        # Store routing info for the aux loss / metrics (current graph)
        self.last_logits = logits
        self.last_indices = indices
        self.last_weights = weights
        self.last_probs = probs

        if return_routing:
            return combined, logits, indices, weights
        return combined

    # ── Load balancing helpers ──────────────────────────────────────

    @torch.no_grad()
    def selection_fractions(self) -> torch.Tensor:
        """Share of all selections that went to each expert (sums to 1).

        Ideal uniform routing gives ``1 / n_experts`` per expert.
        """
        if self.n_selections.item() <= 0:
            return torch.full((self.n_experts,), 1.0 / self.n_experts)
        return self.selection_counts / self.n_selections

    @torch.no_grad()
    def coverage_fractions(self) -> torch.Tensor:
        """Fraction of samples where each expert was among the top-K.

        Ideal uniform routing gives ``top_k / n_experts`` per expert.
        """
        if self.n_samples.item() <= 0:
            return torch.full((self.n_experts,), float(self.top_k) / self.n_experts)
        return self.coverage_counts / self.n_samples

    def aux_load_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer style auxiliary load balancing loss.

        ``L = n_experts * sum_i(f_i * P_i)`` where ``f_i`` is the fraction of
        selections dispatched to expert ``i`` and ``P_i`` is the mean router
        probability of expert ``i``. Minimized (value 1.0) when routing is
        perfectly uniform. Must be called right after the forward pass while
        the computation graph is alive.

        Returns:
            A scalar tensor (requires grad) >= 1.0; lower is more balanced.
        """
        logits = self.last_logits
        indices = self.last_indices
        if logits is None or indices is None:
            return torch.tensor(
                float(self.n_experts), device=self.router.weight.device, requires_grad=True
            )
        probs = torch.softmax(logits, dim=-1)  # (B, N)
        f = torch.bincount(
            indices.flatten(), minlength=self.n_experts
        ).float() / indices.numel()
        p = probs.mean(dim=0)
        return self.n_experts * (f * p).sum()

    # ── Utilities ───────────────────────────────────────────────────

    @torch.no_grad()
    def reset_usage_stats(self) -> None:
        """Reset the load balancing counters (e.g. between training/eval)."""
        self.selection_counts.zero_()
        self.n_selections.zero_()
        self.coverage_counts.zero_()
        self.n_samples.zero_()

    @torch.no_grad()
    def get_weight_stats(self) -> dict[str, float]:
        """Aggregate ternary weight stats across all experts.

        Returns:
            Dict with ``pos_pct``, ``neg_pct``, ``zero_pct`` (percentages).
        """
        total = zeros = pos = neg = 0
        for expert in self.experts:
            w = expert.weight_ternary
            n = w.numel()
            total += n
            zeros += (w == 0).sum().item()
            pos += (w == 1).sum().item()
            neg += (w == -1).sum().item()
        if total == 0:
            return {"pos_pct": 0.0, "neg_pct": 0.0, "zero_pct": 0.0}
        return {
            "pos_pct": 100.0 * pos / total,
            "neg_pct": 100.0 * neg / total,
            "zero_pct": 100.0 * zeros / total,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count parameters: router, experts (float buffers), total."""
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = sum(p.numel() for p in self.experts.parameters())
        return {
            "router": router_params,
            "experts": expert_params,
            "total": router_params + expert_params,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, expert_width={self.expert_width}, "
            f"n_experts={self.n_experts}, top_k={self.top_k}"
        )
