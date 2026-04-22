"""
Demo 12 — Drop-fraction sweep for SET and RigL (milestone 4f follow-up).

Demo 11 showed SET and RigL both losing to Static at 10 epochs on MNIST
with drop_fraction=0.3. One hypothesis: 30% churn every 100 steps is
too disruptive at short training budgets — too many fresh zero-init
connections that the model hasn't had time to retrain before the next
mutation.

This demo varies drop_fraction across {0.1, 0.2, 0.3} for both SET and
RigL, keeping everything else (sparsity, update_freq, epochs, seeds)
identical. We're looking for the sweet spot where DST's mutation
advantage outweighs the mutation noise.

How to run
──────────
    python examples/demo_12_drop_fraction_sweep.py

Needs: pip install sparsecore[demos]

Runtime: ~10-12 minutes on M3 Pro (7 training runs).

What to look at
───────────────
  1. The best drop_fraction for each algorithm. Lower should help both
     at short training budgets; higher should help at long budgets.
  2. Does any DST configuration beat Static? If so, which one?
  3. The plot shows all 7 accuracy curves on one axis.
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
UPDATE_FREQ = 100
DROP_FRACTIONS = [0.1, 0.2, 0.3]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


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
    print(f"─── {name} ───", flush=True)
    fc1, fc2 = build_model()
    fc1.apply(algo)
    opt = torch.optim.SGD(list(fc1.parameters()) + list(fc2.parameters()), lr=LR)

    eval_epochs, eval_accs = [], []
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
        acc = evaluate(fc1, fc2, test_loader)
        eval_epochs.append(epoch)
        eval_accs.append(acc)
    final_acc = eval_accs[-1]
    best_acc = max(eval_accs)
    print(f"    final={final_acc*100:.2f}%  best={best_acc*100:.2f}%  "
          f"time={time.perf_counter()-t_start:.0f}s",
          flush=True)
    return {
        "name": name,
        "eval_epochs": eval_epochs,
        "eval_accs": eval_accs,
        "final_acc": final_acc,
        "best_acc": best_acc,
    }


def plot_all(results, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    # Color by algorithm family, line style by drop_fraction.
    color_by_family = {"Static": "#3b82f6", "SET": "#f59e0b", "RigL": "#ef4444"}
    style_by_df = {0.1: "-", 0.2: "--", 0.3: ":"}

    for r in results:
        # Parse name like "RigL(0.1)" or "Static"
        if r["name"] == "Static":
            color = color_by_family["Static"]
            style = "-"
            lw = 3
        else:
            family = r["name"].split("(")[0]
            df = float(r["name"].split("(")[1].rstrip(")"))
            color = color_by_family[family]
            style = style_by_df[df]
            lw = 2
        ax.plot(r["eval_epochs"],
                [a * 100 for a in r["eval_accs"]],
                color=color, linestyle=style, linewidth=lw,
                label=f'{r["name"]}  best={r["best_acc"]*100:.2f}%')

    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(
        f"DST drop-fraction sweep at {SPARSITY*100:.0f}% sparsity "
        f"(update_freq={UPDATE_FREQ})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    print(f"\nSparseCore demo 12 — drop-fraction sweep at {SPARSITY*100:.0f}% sparsity")
    print(f"HIDDEN={HIDDEN}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}  "
          f"UPDATE_FREQ={UPDATE_FREQ}")
    print(f"DROP_FRACTIONS: {DROP_FRACTIONS}")
    print("=" * 80, flush=True)
    print("Loading MNIST...", flush=True)
    train_loader, test_loader = load_mnist()

    results = []

    # Baseline: Static
    results.append(train_one_path(
        "Static", sparsecore.Static(sparsity=SPARSITY),
        train_loader, test_loader,
    ))

    # SET across drop_fractions
    for df in DROP_FRACTIONS:
        algo = sparsecore.SET(
            sparsity=SPARSITY, drop_fraction=df,
            update_freq=UPDATE_FREQ, seed=42,
        )
        results.append(train_one_path(f"SET({df})", algo, train_loader, test_loader))

    # RigL across drop_fractions
    for df in DROP_FRACTIONS:
        algo = sparsecore.RigL(
            sparsity=SPARSITY, drop_fraction=df,
            update_freq=UPDATE_FREQ, seed=42,
        )
        results.append(train_one_path(f"RigL({df})", algo, train_loader, test_loader))

    # Summary
    print()
    print("=" * 80)
    print(f"{'config':<12s}  {'final acc':>10s}  {'best acc':>10s}  "
          f"{'gap vs Static':>14s}")
    print("-" * 80)
    static_best = results[0]["best_acc"]
    for r in results:
        gap = (r["best_acc"] - static_best) * 100
        marker = "  ←" if gap > 0.1 else ""
        print(f"{r['name']:<12s}  {r['final_acc']*100:>9.2f}%  "
              f"{r['best_acc']*100:>9.2f}%  {gap:>+13.2f}pp{marker}")
    print("=" * 80)

    # Identify winner
    best_r = max(results, key=lambda r: r["best_acc"])
    print(f"\nBest configuration: {best_r['name']}  ({best_r['best_acc']*100:.2f}%)")

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        "demo_12_drop_fraction_sweep.png",
    )
    plot_all(results, out_path)
    print(f"Saved plot: {out_path}\n")


if __name__ == "__main__":
    main()
