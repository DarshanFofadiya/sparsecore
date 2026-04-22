"""
Demo 11 — Static vs SET vs RigL at 90% sparsity (milestone 4f).

The question demo_10 left open: SET tied with Static at 10 epochs
because its random regrow policy is a coin flip. RigL uses the
dense gradient to pick GROW positions — does that information
advantage turn into real accuracy?

This demo trains three identical MLPs (same seed, same
hyperparameters, same training budget) with three different
sparsity algorithms:
  - Static: baseline, frozen random mask
  - SET:    drop smallest-|value|, regrow random empties
  - RigL:   drop smallest-|value|, regrow at top-|dL/dW| empties

How to run
──────────
    python examples/demo_11_rigl_vs_set_vs_static.py

Needs: pip install sparsecore[demos]

Runtime: ~6-10 minutes on M3 Pro.

What to look at
───────────────
  1. Final best accuracy for each algorithm. Publisher literature says
     RigL > SET ≥ Static on short training budgets. We want to see
     that here too.
  2. Churn column: both SET and RigL mutate topology, but RigL's
     mutations should be *better informed*.
  3. The saved plot shows accuracy trajectories side-by-side.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install sparsecore[demos]")

try:
    from torchvision import datasets, transforms
except ImportError:
    raise SystemExit("pip install sparsecore[demos]")

import sparsecore


warnings.filterwarnings("ignore", category=UserWarning)


# ─── Config ──────────────────────────────────────────────────────────
HIDDEN = 512
BATCH_SIZE = 128
LR = 0.01
SPARSITY = 0.9
EPOCHS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DROP_FRACTION = 0.3
UPDATE_FREQ = 100


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


def build_model():
    torch.manual_seed(0)
    fc1 = sparsecore.SparseLinear(784, HIDDEN, sparsity=SPARSITY, bias=False)
    fc2 = nn.Linear(HIDDEN, 10, bias=False)
    return fc1, fc2


def evaluate(fc1, fc2, loader):
    with torch.no_grad():
        correct, total = 0, 0
        for imgs, labels in loader:
            x = imgs.view(imgs.size(0), -1)
            logits = fc2(F.relu(fc1(x)))
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_one_path(name, algo, train_loader, test_loader):
    print(f"\n─── Training '{name}' ───")
    fc1, fc2 = build_model()
    fc1.apply(algo)
    opt = torch.optim.SGD(list(fc1.parameters()) + list(fc2.parameters()), lr=LR)

    eval_epochs, eval_accs = [], []
    n_updates = 0
    prev_cols = np.array(fc1._csr.col_indices, copy=True)
    total_churn = 0

    t_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        for imgs, labels in train_loader:
            x = imgs.view(imgs.size(0), -1)
            opt.zero_grad()
            logits = fc2(F.relu(fc1(x)))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            algo.step()

        # Count churn and updates
        curr_cols = np.asarray(fc1._csr.col_indices)
        epoch_churn = int((prev_cols != curr_cols).sum())
        total_churn += epoch_churn
        prev_cols = curr_cols.copy()

        acc = evaluate(fc1, fc2, test_loader)
        eval_epochs.append(epoch)
        eval_accs.append(acc)
        print(f"    ep {epoch:>2d}  acc={acc*100:.2f}%  (total churn: {total_churn})")

    return {
        "name": name,
        "eval_epochs": eval_epochs,
        "eval_accs": eval_accs,
        "best_acc": max(eval_accs),
        "best_ep": eval_epochs[eval_accs.index(max(eval_accs))],
        "total_s": time.perf_counter() - t_start,
        "total_churn": total_churn,
    }


def plot_comparison(results, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    colors = {"Static": "#3b82f6", "SET": "#f59e0b", "RigL": "#ef4444"}

    for r in results:
        ax.plot(r["eval_epochs"], [a * 100 for a in r["eval_accs"]],
                color=colors.get(r["name"], "black"),
                linewidth=2,
                label=f'{r["name"]} (best {r["best_acc"]*100:.2f}% @ep{r["best_ep"]}, churn {r["total_churn"]})')

    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"DST algorithm comparison at {SPARSITY*100:.0f}% sparsity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    print(f"\nSparseCore demo 11 — Static vs SET vs RigL at {SPARSITY*100:.0f}% sparsity")
    print(f"HIDDEN={HIDDEN}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}")
    print(f"DST: drop_fraction={DROP_FRACTION}, update_freq={UPDATE_FREQ}")
    print("=" * 80)

    print("Loading MNIST...")
    train_loader, test_loader = load_mnist()

    algos = [
        ("Static", sparsecore.Static(sparsity=SPARSITY)),
        ("SET", sparsecore.SET(
            sparsity=SPARSITY,
            drop_fraction=DROP_FRACTION,
            update_freq=UPDATE_FREQ,
            seed=42,
        )),
        ("RigL", sparsecore.RigL(
            sparsity=SPARSITY,
            drop_fraction=DROP_FRACTION,
            update_freq=UPDATE_FREQ,
            seed=42,
        )),
    ]

    results = []
    for name, algo in algos:
        r = train_one_path(name, algo, train_loader, test_loader)
        results.append(r)

    # Summary
    print()
    print("=" * 80)
    print(f"{'algorithm':<10s}  {'best acc':>10s}  {'at epoch':>9s}  "
          f"{'total churn':>12s}  {'time':>8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10s}  {r['best_acc']*100:>9.2f}%  "
              f"{r['best_ep']:>9d}  {r['total_churn']:>12d}  {r['total_s']:>7.0f}s")

    # Gap from Static
    static_best = results[0]["best_acc"]
    print()
    for r in results[1:]:
        gap = (r["best_acc"] - static_best) * 100
        if gap > 0.2:
            verdict = "wins — DST information advantage shows up"
        elif gap > -0.2:
            verdict = "ties within noise"
        else:
            verdict = "loses — more epochs likely needed"
        print(f"  {r['name']} vs Static: {gap:+.2f} pp → {verdict}")
    print("=" * 80)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        "demo_11_rigl_vs_set_vs_static.png",
    )
    plot_comparison(results, out_path)
    print(f"Saved plot: {out_path}\n")


if __name__ == "__main__":
    main()
