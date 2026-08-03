#!/usr/bin/env python3
"""Pilot Mixture-of-Experts (MoE) experiment with DQT ternary experts (E019).

Trains two models on MNIST and compares them under an equal total-parameter
budget:

**MoE DQT** (sparse activation):
    Input (784)
      → Router (784 → N_experts) — selects top-K per sample
      → MoE layer: N_experts × TernaryDQTLinear(784, 128)
      → Weighted sum of the K active expert outputs
      → ReLU
      → TernaryDQTLinear(128, 10)   (weighted sum keeps width 128, not 128*K)
      → Output (10)

**Dense DQT** (baseline, same total params):
    Input (784) → TernaryDQTLinear(784, 512) → ReLU
    → TernaryDQTLinear(512, 10) → Output (10)

The dense hidden width is ``n_experts * expert_width`` so both models share the
same total parameter budget; the MoE activates only ``top_k / n_experts`` of
its expert parameters per input.

Usage:
    # Pilot (defaults): N=4 experts, top-K=2, lr=0.01, 30 epochs, seed 42
    python -m ph_neuro.examples.run_moe_dqt

    # Full params
    python -m ph_neuro.examples.run_moe_dqt --epochs 60 --lr 0.01 \
        --n-experts 4 --top-k 2 --expert-width 128 --seed 42

    # Retry with load-balancing aux loss if experts collapse
    python -m ph_neuro.examples.run_moe_dqt --lb-coef 0.01

Output:
    JSON file: ``{output_dir}/results_mnist_seed{seed}.json``
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from ph_neuro.examples._utils import print_header
from ph_neuro.layers.ste_dqt import TernaryDQTLinear
from ph_neuro.layers.ste_dqt_moe import TernaryDQTMoELayer

warnings.filterwarnings("ignore", category=UserWarning, module="torch.quantization")


# ── Data ────────────────────────────────────────────────────────────


def get_mnist_loaders(
    batch_size: int = 128, num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders."""
    from torchvision import transforms
    from torchvision.datasets import MNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


# ── Models ──────────────────────────────────────────────────────────


class DenseDQTMLP(nn.Module):
    """Dense single-hidden-layer MLP with TernaryDQTLinear layers.

    Matches the MoE depth (one hidden layer) and total parameter budget.
    """

    def __init__(
        self, in_features: int, hidden: int, out_features: int = 10, use_bn: bool = False,
    ):
        super().__init__()
        self.hidden = TernaryDQTLinear(in_features, hidden, bias=False)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden) if use_bn else None
        self.output = TernaryDQTLinear(hidden, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.hidden(x)
        x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.output(x)


class MoEDQTMLP(nn.Module):
    """Minimal MoE MLP: float router → top-K DQT experts → DQT output layer."""

    def __init__(
        self,
        in_features: int,
        expert_width: int,
        n_experts: int,
        top_k: int,
        out_features: int = 10,
        init_std: float = 0.1,
        use_bn: bool = False,
    ):
        super().__init__()
        self.moe = TernaryDQTMoELayer(
            in_features, expert_width, n_experts, top_k, init_std=init_std,
        )
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(expert_width) if use_bn else None
        self.output = TernaryDQTLinear(expert_width, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = self.moe(x)
        x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.output(x)


def build_dense_dqt(in_features: int = 784, out_features: int = 10,
                    n_experts: int = 4, expert_width: int = 128,
                    use_bn: bool = False) -> DenseDQTMLP:
    """Dense baseline with the same parameter budget as the MoE model."""
    hidden = n_experts * expert_width
    return DenseDQTMLP(in_features, hidden, out_features, use_bn=use_bn)


def build_moe_dqt(in_features: int = 784, out_features: int = 10,
                  n_experts: int = 4, top_k: int = 2,
                  expert_width: int = 128, use_bn: bool = False) -> MoEDQTMLP:
    """Minimal MoE MLP (pilot config)."""
    return MoEDQTMLP(
        in_features, expert_width, n_experts, top_k, out_features, use_bn=use_bn,
    )


# ── Helpers ─────────────────────────────────────────────────────────


@torch.no_grad()
def apply_stochastic_rounding(model: nn.Module) -> None:
    """Apply DQT stochastic rounding to every TernaryDQTLinear in the model."""
    for module in model.modules():
        if isinstance(module, TernaryDQTLinear):
            module.apply_stochastic_rounding()


@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    """Total number of learnable parameters (float buffers)."""
    return sum(p.numel() for p in model.parameters())


def compute_active_params(model: nn.Module) -> tuple[int, float]:
    """Active parameters per input.

    Dense: all parameters are active.
    MoE: router + ``top_k / n_experts`` of the expert params + output layer.

    Returns:
        Tuple of ``(active_params, active_fraction)``.
    """
    if isinstance(model, MoEDQTMLP):
        moe = model.moe
        router_params = moe.router.weight.numel()
        expert_params = sum(p.numel() for p in moe.experts.parameters())
        output_params = sum(p.numel() for p in model.output.parameters())
        active = router_params + (moe.top_k / moe.n_experts) * expert_params + output_params
        total = count_parameters(model)
        return int(round(active)), active / total
    total = count_parameters(model)
    return total, 1.0


@torch.no_grad()
def weight_sparsity(model: nn.Module) -> dict[str, float]:
    """Aggregate ternary weight stats across all DQT layers."""
    total = zeros = pos = neg = 0
    for module in model.modules():
        if isinstance(module, TernaryDQTLinear):
            w = module.weight_ternary
            n = w.numel()
            total += n
            zeros += (w == 0).sum().item()
            pos += (w == 1).sum().item()
            neg += (w == -1).sum().item()
    if total == 0:
        return {"sparsity_pct": 0.0, "pos_pct": 0.0, "neg_pct": 0.0}
    return {
        "sparsity_pct": 100.0 * zeros / total,
        "pos_pct": 100.0 * pos / total,
        "neg_pct": 100.0 * neg / total,
    }


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate test accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ── Training ────────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lb_coef: float,
    router_lr: float | None = None,
    max_patience: int = 10,
) -> dict:
    """Train a DQT model (dense or MoE) with AdamW + stochastic rounding.

    For MoE models with ``lb_coef > 0``, the Switch-Transformer aux load
    balancing loss is added to the cross-entropy loss.

    Returns:
        Dict of training metrics including per-epoch load balancing history
        for MoE models.
    """
    is_moe = isinstance(model, MoEDQTMLP)
    moe = model.moe if is_moe else None

    if is_moe and router_lr is not None:
        # Separate (smaller) LR for the router — standard MoE practice to slow
        # down routing collapse while experts learn at the full LR.
        router_ids = {id(p) for p in model.moe.router.parameters()}
        expert_group = [p for p in model.parameters() if id(p) not in router_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": expert_group, "lr": lr},
                {"params": model.moe.router.parameters(), "lr": router_lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_start = time.time()
    best_acc = 0.0
    best_epoch = 0
    final_acc = 0.0
    patience = 0
    epochs_trained = 0

    # Per-epoch histories
    history = {
        "train_acc": [],
        "test_acc": [],
        "loss": [],
        "sparsity_pct": [],
    }
    lb_history: list[dict] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_samples = 0
        correct = 0
        total = 0

        if moe is not None:
            moe.reset_usage_stats()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            if lb_coef > 0 and moe is not None:
                loss = loss + lb_coef * moe.aux_load_balance_loss()
            loss.backward()
            optimizer.step()
            apply_stochastic_rounding(model)

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            correct += out.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        if scheduler is not None:
            scheduler.step()

        train_acc = correct / max(total, 1)
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        epochs_trained = epoch

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        final_acc = test_acc

        stats = weight_sparsity(model)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(total_loss / max(n_samples, 1))
        history["sparsity_pct"].append(stats["sparsity_pct"])

        # Load balancing for this epoch (MoE only)
        if moe is not None:
            sel = moe.selection_fractions().cpu().tolist()
            cov = moe.coverage_fractions().cpu().tolist()
            lb_history.append({
                "selection_fractions": sel,
                "coverage_fractions": cov,
            })

        tag = "MoE" if is_moe else "Dense"
        print(
            f"  [{tag}] Epoch {epoch:3d}/{epochs}  "
            f"Train: {100 * train_acc:5.2f}%  "
            f"Test: {100 * test_acc:5.2f}%  "
            f"Best: {100 * best_acc:5.2f}%  "
            f"Sparsity: {stats['sparsity_pct']:4.1f}%  "
            f"Time: {epoch_time:4.1f}s"
        )
        if is_moe:
            sel_str = ", ".join(f"{s:.3f}" for s in lb_history[-1]["selection_fractions"])
            print(f"         expert selection share: [{sel_str}]")

        if patience >= max_patience:
            print(f"  [{tag}] Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    total_time = time.time() - total_start

    result: dict = {
        "best_accuracy": float(best_acc),
        "final_accuracy": float(final_acc),
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "training_time_seconds": float(total_time),
        "weight_stats": weight_sparsity(model),
        "train_acc_history": history["train_acc"],
        "test_acc_history": history["test_acc"],
        "loss_history": history["loss"],
        "sparsity_history": history["sparsity_pct"],
    }

    if moe is not None:
        # Final load balancing: average of the last 5 epochs
        n_avg = min(5, len(lb_history))
        recent = lb_history[-n_avg:] if lb_history else []
        if recent:
            avg_sel = [
                sum(e["selection_fractions"][i] for e in recent) / len(recent)
                for i in range(moe.n_experts)
            ]
            avg_cov = [
                sum(e["coverage_fractions"][i] for e in recent) / len(recent)
                for i in range(moe.n_experts)
            ]
        else:
            avg_sel = [1.0 / moe.n_experts] * moe.n_experts
            avg_cov = [float(moe.top_k) / moe.n_experts] * moe.n_experts
        result["load_balancing"] = {
            "selection_fractions": avg_sel,
            "coverage_fractions": avg_cov,
            "balance_ratio": (min(avg_sel) / max(avg_sel)) if max(avg_sel) > 0 else 0.0,
            "ideal_selection_fraction": 1.0 / moe.n_experts,
            "ideal_coverage_fraction": float(moe.top_k) / moe.n_experts,
            "history": lb_history,
        }

    return result


# ── Main ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot MoE experiment with DQT ternary experts (E019)"
    )
    parser.add_argument("--dataset", default="mnist", help="Dataset (mnist)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-width", type=int, default=128)
    parser.add_argument("--lb-coef", type=float, default=0.0,
                        help="Load-balancing aux loss coefficient (0 = off)")
    parser.add_argument("--router-lr", type=float, default=None,
                        help="Separate (usually smaller) LR for the router "
                             "(default: same as --lr)")
    parser.add_argument("--use-bn", action="store_true",
                        help="Add BatchNorm after the hidden layer in BOTH "
                             "models (matches DQT best-practice config)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--output-dir", default="moe_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print_header(
        f"MoE DQT Pilot (E019): {args.n_experts} experts, top-{args.top_k}, "
        f"lr={args.lr}, {args.epochs}ep, seed={args.seed}"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"MNIST: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")  # type: ignore[arg-type]
    print()

    # ── Models ──────────────────────────────────────────────────────
    dense_model = build_dense_dqt(
        n_experts=args.n_experts, expert_width=args.expert_width, use_bn=args.use_bn,
    ).to(device)
    moe_model = build_moe_dqt(
        n_experts=args.n_experts, top_k=args.top_k, expert_width=args.expert_width,
        use_bn=args.use_bn,
    ).to(device)

    dense_total = count_parameters(dense_model)
    moe_total = count_parameters(moe_model)
    moe_active, moe_active_frac = compute_active_params(moe_model)

    print("─" * 78)
    print(f"  Dense DQT   params: {dense_total:>8,}   active/input: {dense_total:>8,} (100%)")
    print(f"  MoE DQT     params: {moe_total:>8,}   active/input: {moe_active:>8,} "
          f"({100 * moe_active_frac:.1f}%)")
    print(f"    router: {moe_model.moe.router.weight.numel():,}  "
          f"experts: {args.n_experts}×{args.expert_width}  top-K: {args.top_k}")
    print("─" * 78)
    print()

    # ── Train dense ─────────────────────────────────────────────────
    print_header("Training Dense DQT baseline")
    dense_result = train_model(
        dense_model, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        lb_coef=0.0,
    )
    print()

    # ── Train MoE ───────────────────────────────────────────────────
    print_header("Training MoE DQT")
    moe_result = train_model(
        moe_model, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        lb_coef=args.lb_coef, router_lr=args.router_lr,
    )
    print()

    # ── Result dict ─────────────────────────────────────────────────
    result = {
        "experiment": "moe_dqt",
        "dataset": args.dataset,
        "seed": args.seed,
        "device": str(device),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "n_experts": args.n_experts,
            "top_k": args.top_k,
            "expert_width": args.expert_width,
            "lb_coef": args.lb_coef,
            "router_lr": args.router_lr,
            "use_bn": args.use_bn,
            "dense_arch": "Dense DQT MLP [784, N*W, 10]",
            "moe_arch": "Router(784->N) + N×TernaryDQTLinear(784,W) + TernaryDQTLinear(W,10)",
        },
        "params": {
            "dense_total": dense_total,
            "moe_total": moe_total,
            "moe_active_per_input": moe_active,
            "moe_active_fraction": moe_active_frac,
        },
        "dense": dense_result,
        "moe": moe_result,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"results_{args.dataset}_seed{args.seed}.json"
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────
    print_header("Comparison Summary")
    print(f"  Dense DQT best: {100 * dense_result['best_accuracy']:.2f}%  "
          f"(ep {dense_result['best_epoch']})  "
          f"sparsity {dense_result['weight_stats']['sparsity_pct']:.1f}%  "
          f"{dense_result['training_time_seconds']:.0f}s")
    print(f"  MoE DQT   best: {100 * moe_result['best_accuracy']:.2f}%  "
          f"(ep {moe_result['best_epoch']})  "
          f"sparsity {moe_result['weight_stats']['sparsity_pct']:.1f}%  "
          f"{moe_result['training_time_seconds']:.0f}s")
    if "load_balancing" in moe_result:
        lb = moe_result["load_balancing"]
        sel = lb["selection_fractions"]
        print(f"  MoE expert selection share: "
              f"[{', '.join(f'{s:.3f}' for s in sel)}]  "
              f"(ideal {lb['ideal_selection_fraction']:.3f}, "
              f"balance_ratio {lb['balance_ratio']:.3f})")
    print(f"\n  Δ accuracy (MoE − dense): "
          f"{100 * (moe_result['best_accuracy'] - dense_result['best_accuracy']):+.2f}pp")
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
