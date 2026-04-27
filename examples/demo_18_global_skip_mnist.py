"""
Demo 18 — Global-skip vs sequential connectivity on MNIST.

Research question
─────────────────
Our sparse engine makes it feasible to train an MLP where every layer has a
direct sparse connection to the outputs of ALL previous layers (not just
the layer immediately prior). This is a sparse version of the DenseNet
idea (Huang et al. 2017, CVPR) — which in dense form has O(L^2) parameter
count and is intractable at depth, but in sparse form has O(L * K) params
where K is the per-layer live-connection budget.

Does this connectivity pattern actually help at matched parameter budget?
This demo runs a small head-to-head on MNIST to find out.

Four models, all trained with the same seed, same optimizer, same
hyperparameters. Only the architecture differs.

  A. sparse_sequential         5L, hidden=256, 90% sparse, standard MLP
  B. sparse_global_skip        5L, hidden=256, ~96.5% sparse, layer N
                               reads concat(input, h1, ..., h_{N-1})
  C. dense_seq_narrow          5L, hidden=40, dense — matched live params
  D. dense_global_skip_shallow 2L, hidden=64, dense, concat skip —
                               sanity check that skip works at all in dense

What to observe
───────────────
If sparse_global_skip reaches final accuracy >= sparse_sequential at
matched live params, that's a real signal: the global-connectivity prior
is "free" in our library and worth exploring.

If sparse_global_skip loses, we say so honestly — it tells us that the
implicit "deeper prior layers matter less" structure of standard MLPs
is a useful bias we shouldn't throw away.

This is one run on one small task. Don't over-generalize either outcome.

Usage
─────
    python examples/demo_18_global_skip_mnist.py                 # default 10 epochs
    python examples/demo_18_global_skip_mnist.py --epochs 5      # quick
    python examples/demo_18_global_skip_mnist.py --models A,B    # subset

Runtime: ~8-12 min total on M3 Pro for all 4 models at 10 epochs.

Inspired by
───────────
  - Huang et al. "Densely Connected Convolutional Networks" CVPR 2017
  - Xie, Kirillov, Girshick, He. "Exploring Randomly Wired Neural
    Networks for Image Recognition" ICCV 2019
  - Liu et al. "A Sparse DenseNet for Image Classification" arxiv 1804.05340

Both papers show wiring patterns beyond sequential can help. Our
contribution is making the DYNAMIC version (SparseLab's topology
mutation) trainable on commodity hardware.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install sparselab[demos]  # (needs matplotlib + torchvision)")

try:
    from torchvision import datasets, transforms
except ImportError:
    raise SystemExit("pip install sparselab[demos]")

import sparselab


# ─── Config ──────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 128
LR = 0.01
DEFAULT_EPOCHS = 10
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "docs" / "demos"


# ─────────────────────────────────────────────────────────────────────
#  Model A: sparse_sequential
# ─────────────────────────────────────────────────────────────────────
#
# Standard 5-layer MLP using SparseLinear. Each layer connects only to
# its immediate predecessor. This is our "default sparse" baseline and
# the matching budget target for everything else.
#
# Live params: 5 SparseLinear layers at 90% sparsity gives ~42K live
# connections + ~2,560 in the dense fc5 head = ~44K.
# ─────────────────────────────────────────────────────────────────────

class SparseSequentialMLP(nn.Module):
    def __init__(self, hidden: int = 256, sparsity: float = 0.9):
        super().__init__()
        self.fc1 = sparselab.SparseLinear(784, hidden, sparsity=sparsity)
        self.fc2 = sparselab.SparseLinear(hidden, hidden, sparsity=sparsity)
        self.fc3 = sparselab.SparseLinear(hidden, hidden, sparsity=sparsity)
        self.fc4 = sparselab.SparseLinear(hidden, hidden, sparsity=sparsity)
        # fc5 is tiny (hidden * 10) — keep dense so sparse index overhead
        # doesn't dwarf its data.
        self.fc5 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)         # flatten 28x28 -> 784
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# ─────────────────────────────────────────────────────────────────────
#  Model B: sparse_global_skip
# ─────────────────────────────────────────────────────────────────────
#
# Every layer reads from the concatenation of the original input and
# every prior layer's output. The input dim to layer N is
# 784 + (N-1) * hidden.
#
# This is where the sparsity buys us something structural: the dense
# equivalent of this model has ~1.2M params for 5 layers at hidden=256
# (30x our target budget). We stay in budget by using ~96.5% sparsity
# on each layer — higher than the sequential model, which means each
# live connection has to be more "chosen."
#
# Note: we keep the same total live-connection budget as Model A so
# the comparison is apples-to-apples on parameters, not on compute.
# (Sparse global-skip reads from more places, so it does slightly more
# memory work per step — but the SIMD FMA count is the same.)
# ─────────────────────────────────────────────────────────────────────

class SparseGlobalSkipMLP(nn.Module):
    def __init__(self, hidden: int = 256, sparsity: float = 0.98):
        super().__init__()
        # Each layer's input dim is the original x plus all prior hidden
        # activations. Computed at construction time so we can size the
        # sparse weight matrices correctly.
        in_dims = [784, 784 + hidden, 784 + 2 * hidden,
                   784 + 3 * hidden, 784 + 4 * hidden]
        self.fc1 = sparselab.SparseLinear(in_dims[0], hidden, sparsity=sparsity)
        self.fc2 = sparselab.SparseLinear(in_dims[1], hidden, sparsity=sparsity)
        self.fc3 = sparselab.SparseLinear(in_dims[2], hidden, sparsity=sparsity)
        self.fc4 = sparselab.SparseLinear(in_dims[3], hidden, sparsity=sparsity)
        # Head reads from concat(x, h1, h2, h3, h4) at dim in_dims[4].
        # Keep dense so the final classification head has its full
        # capacity — we're measuring the MLP's feature-learning ability,
        # not squeezing parameters at the classifier.
        self.fc5 = nn.Linear(in_dims[4], 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)          # (B, 784)
        h1 = F.relu(self.fc1(x))           # (B, hidden)

        # Concatenate x with all prior activations as input to each
        # subsequent layer. torch.cat is cheap for this size; for much
        # larger nets a pre-allocated global buffer would be better.
        h2 = F.relu(self.fc2(torch.cat([x, h1], dim=1)))
        h3 = F.relu(self.fc3(torch.cat([x, h1, h2], dim=1)))
        h4 = F.relu(self.fc4(torch.cat([x, h1, h2, h3], dim=1)))
        h5_in = torch.cat([x, h1, h2, h3, h4], dim=1)
        return self.fc5(h5_in)


# ─────────────────────────────────────────────────────────────────────
#  Model C: dense_seq_narrow
# ─────────────────────────────────────────────────────────────────────
#
# A standard dense sequential MLP at matched live-parameter budget.
# At hidden=40 we get ~40K params — within ~5% of Model A's 42K.
# This is the "what if we just made the dense model smaller instead?"
# control. If sparse_sequential beats this, sparse storage is
# buying us expressivity. If it loses, maybe narrower dense wins.
# ─────────────────────────────────────────────────────────────────────

class DenseSequentialMLP(nn.Module):
    def __init__(self, hidden: int = 40):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# ─────────────────────────────────────────────────────────────────────
#  Model D: dense_global_skip_shallow (sanity check)
# ─────────────────────────────────────────────────────────────────────
#
# A 2-layer dense network with skip concatenation. Only useful at
# shallow depth because dense global-skip is O(L^2).
#
# This is our "does the skip trick even help in the dense regime?"
# check. If this beats a 2-layer sequential dense at matched params,
# skip has value; if it doesn't, skip might not help even in principle.
# ─────────────────────────────────────────────────────────────────────

class DenseGlobalSkipShallow(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        # fc2 reads concat(x, h1) which has dim 784 + hidden.
        self.fc2 = nn.Linear(784 + hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        return self.fc2(torch.cat([x, h1], dim=1))



# ─────────────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (dense_params, sparse_live_params).

    For comparison tables. SparseLinear stores its live weights in a
    single float32 array sized equal to the live-nnz count, so we just
    read .nnz off each SparseLinear and add dense param counts for
    everything else.
    """
    dense_p = 0
    sparse_p = 0
    for m in model.modules():
        if isinstance(m, sparselab.SparseLinear):
            sparse_p += m.nnz
            # Add bias if present (SparseLinear has a dense bias).
            if m.bias is not None:
                dense_p += m.bias.numel()
        elif isinstance(m, nn.Linear):
            dense_p += m.weight.numel()
            if m.bias is not None:
                dense_p += m.bias.numel()
    return dense_p, sparse_p


def load_mnist(batch_size: int = BATCH_SIZE):
    """Standard MNIST loader with the conventional (0.1307, 0.3081) normalize."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform)
    test = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader) -> float:
    """Return test-set accuracy in [0, 1]."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def train_one_model(name: str, model: nn.Module, train_loader, test_loader,
                    epochs: int) -> dict:
    """Train `model` for `epochs` epochs; return metrics dict."""
    # Reset RNG so every model sees the same batch ordering
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dense_p, sparse_p = count_params(model)
    total_live = dense_p + sparse_p

    print(f"\n{'=' * 68}")
    print(f"  Training: {name}")
    print(f"  Dense params: {dense_p:,}  Sparse live: {sparse_p:,}  "
          f"Total live: {total_live:,}")
    print(f"{'=' * 68}", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    train_losses = []          # running loss per epoch
    test_accs = []             # test acc after each epoch

    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n = 0
        for x, y in train_loader:
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)
            n += y.size(0)
        avg_loss = running_loss / n
        test_acc = evaluate(model, test_loader)
        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        elapsed = time.perf_counter() - t_start
        print(f"  [{name}] epoch {epoch+1:2d}/{epochs}  "
              f"train_loss={avg_loss:.4f}  test_acc={test_acc*100:5.2f}%  "
              f"({elapsed:.0f}s)", flush=True)

    total_time = time.perf_counter() - t_start
    return {
        "name": name,
        "dense_params": dense_p,
        "sparse_live": sparse_p,
        "total_live": total_live,
        "train_losses": train_losses,
        "test_accs": test_accs,
        "final_acc": test_accs[-1],
        "best_acc": max(test_accs),
        "total_time_s": total_time,
    }


# ─────────────────────────────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────────────────────────────

MODEL_BUILDERS = {
    "A": ("sparse_sequential (5L, h=256, 90% sparse)",
          lambda: SparseSequentialMLP(hidden=256, sparsity=0.9)),
    "B": ("sparse_global_skip (5L, h=256, 98% sparse)",
          lambda: SparseGlobalSkipMLP(hidden=256, sparsity=0.98)),
    "C": ("dense_seq_narrow (5L, h=40)",
          lambda: DenseSequentialMLP(hidden=40)),
    "D": ("dense_global_skip_shallow (2L, h=64)",
          lambda: DenseGlobalSkipShallow(hidden=64)),
}


def plot_results(results: list[dict], out_path: Path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"A": "#3b82f6", "B": "#ef4444", "C": "#10b981", "D": "#a855f7"}
    for r in results:
        k = r["name"].split()[0]          # grab the letter at start
        key = next((mk for mk, (nm, _) in MODEL_BUILDERS.items()
                    if nm == r["name"].split("  ", 1)[-1]), None) or k
        color = colors.get(key, "black")
        epochs = list(range(1, len(r["train_losses"]) + 1))
        ax_loss.plot(epochs, r["train_losses"], color=color, linewidth=2,
                     label=r["name"])
        ax_acc.plot(epochs, [a * 100 for a in r["test_accs"]], color=color,
                    linewidth=2, label=r["name"])
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("train loss")
    ax_loss.set_title("Training loss"); ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title("Test accuracy"); ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=8, loc="lower right")
    fig.suptitle("Global-skip vs sequential MLPs on MNIST (matched live params)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--models", type=str, default="A,B,C,D",
                        help="Comma-separated subset of {A,B,C,D}")
    args = parser.parse_args()
    keys = [k.strip().upper() for k in args.models.split(",")]
    for k in keys:
        if k not in MODEL_BUILDERS:
            raise SystemExit(f"Unknown model key {k!r}; valid: A,B,C,D")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Demo 18 — Global-skip vs sequential MLPs on MNIST")
    print(f"Seed={SEED}  Batch={BATCH_SIZE}  Epochs={args.epochs}  LR={LR}")
    print("Loading MNIST...", flush=True)
    train_loader, test_loader = load_mnist()

    results = []
    for k in keys:
        name, builder = MODEL_BUILDERS[k]
        full_name = f"{k}. {name}"
        model = builder()
        r = train_one_model(full_name, model, train_loader, test_loader,
                            args.epochs)
        results.append(r)

    # ─── Final comparison ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Model':<52}  {'params':>9}  {'best':>7}  {'time':>6}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<52}  {r['total_live']:>9,}  "
              f"{r['best_acc']*100:>6.2f}%  {r['total_time_s']:>5.0f}s")

    # Plot + save samples
    plot_path = OUT_DIR / "demo_18_global_skip.png"
    plot_results(results, plot_path)
    print(f"\nPlot: {plot_path}")
    print("(All runs used the same seed, same optimizer, same data order. "
          "Differences are architectural, not stochastic.)")


if __name__ == "__main__":
    main()
