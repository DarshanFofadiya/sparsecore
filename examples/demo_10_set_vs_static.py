"""
Demo 10 — SET vs Static at 90% sparsity (milestone 4e).

The question: does DST (dynamic sparse training) actually help on MNIST?

Demo 8 established: at 90% sparsity, a Static random mask converges to
97.45% test accuracy, vs dense 98.06% — a 0.61 pp gap. SET is supposed
to close part of that gap by moving the few available weights to where
the task actually needs them.

This demo trains two identical MLPs — same seed, same hyperparameters,
same total training budget — but one uses sparsecore.Static (frozen
random mask) and the other uses sparsecore.SET (drop smallest-
magnitude K% every 100 steps, regrow K% random empties).

How to run
──────────
    python examples/demo_10_set_vs_static.py

Needs: pip install sparsecore[demos]

Runtime: ~5-10 minutes on M3 Pro.

What to look at
───────────────
  1. Final best test accuracy for SET vs Static. SET should match or
     beat Static. Published SET results show ~1pp improvement in
     this regime.
  2. Churn column: how many connections SET actually moves. At 100
     updates over ~5k batches this is ~500k slot rewrites total —
     a small fraction of total training time.
  3. The plot: you should see SET's accuracy trajectory weave above
     Static's after the first few updates kick in.
"""

from __future__ import annotations

import os
import time

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


# ─── Config ──────────────────────────────────────────────────────────
HIDDEN = 512
BATCH_SIZE = 128
LR = 0.01
SPARSITY = 0.9
EPOCHS = 10
EVAL_EVERY = 1
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# SET parameters (paper defaults)
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
    """Two-layer MLP with sparse first layer and dense output. Same
    seed every time — the only difference between runs is which
    sparsity algorithm we attach."""
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


def count_churn(cols_history):
    """Compute total number of connections that changed between
    consecutive topology snapshots. Used to visualize SET activity."""
    churn = []
    for i in range(1, len(cols_history)):
        diff = (cols_history[i - 1] != cols_history[i]).sum()
        churn.append(int(diff))
    return churn


def train_one_path(name, algo, train_loader, test_loader):
    print(f"\n─── Training '{name}' ({algo!r}) ───")
    fc1, fc2 = build_model()
    fc1.apply(algo)
    opt = torch.optim.SGD(list(fc1.parameters()) + list(fc2.parameters()), lr=LR)

    eval_epochs, eval_accs = [], []
    cols_history = [np.array(fc1._csr.col_indices, copy=True)]
    t_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        for imgs, labels in train_loader:
            x = imgs.view(imgs.size(0), -1)
            opt.zero_grad()
            logits = fc2(F.relu(fc1(x)))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            # This is the DST step — for Static it's a no-op.
            algo.step()

        cols_history.append(np.array(fc1._csr.col_indices, copy=True))
        if epoch % EVAL_EVERY == 0:
            acc = evaluate(fc1, fc2, test_loader)
            eval_epochs.append(epoch)
            eval_accs.append(acc)
            print(f"    ep {epoch:>2d}  acc={acc*100:.2f}%")

    churn = count_churn(cols_history)
    total_churn = sum(churn)
    print(
        f"    final best acc: {max(eval_accs)*100:.2f}%  "
        f"(total churn: {total_churn} slots over {len(churn)} epochs)"
    )
    return {
        "name": name,
        "eval_epochs": eval_epochs,
        "eval_accs": eval_accs,
        "best_acc": max(eval_accs),
        "best_ep": eval_epochs[eval_accs.index(max(eval_accs))],
        "total_s": time.perf_counter() - t_start,
        "churn_per_epoch": churn,
    }


def plot_comparison(static_result, set_result, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy curves
    ax1.plot(static_result["eval_epochs"],
             [a * 100 for a in static_result["eval_accs"]],
             color="#3b82f6", linewidth=2,
             label=f'Static (best {static_result["best_acc"]*100:.2f}%)')
    ax1.plot(set_result["eval_epochs"],
             [a * 100 for a in set_result["eval_accs"]],
             color="#ef4444", linewidth=2,
             label=f'SET (best {set_result["best_acc"]*100:.2f}%)')
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test accuracy (%)")
    ax1.set_title(f"Static vs SET at {SPARSITY*100:.0f}% sparsity")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # Churn per epoch
    epochs = list(range(1, EPOCHS + 1))
    ax2.bar(epochs, set_result["churn_per_epoch"],
            color="#ef4444", alpha=0.7, label="SET churn")
    ax2.bar(epochs, static_result["churn_per_epoch"],
            color="#3b82f6", alpha=0.7, label="Static churn")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("# topology changes per epoch")
    ax2.set_title("Topology mutation activity")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    print(f"\nSparseCore demo 10 — SET vs Static at {SPARSITY*100:.0f}% sparsity")
    print(f"HIDDEN={HIDDEN}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}")
    print(f"SET: drop_fraction={DROP_FRACTION}, update_freq={UPDATE_FREQ}")
    print("=" * 80)

    print("Loading MNIST...")
    train_loader, test_loader = load_mnist()

    static_algo = sparsecore.Static(sparsity=SPARSITY)
    set_algo = sparsecore.SET(
        sparsity=SPARSITY,
        drop_fraction=DROP_FRACTION,
        update_freq=UPDATE_FREQ,
        seed=42,
    )

    static_result = train_one_path("Static", static_algo, train_loader, test_loader)
    set_result = train_one_path("SET", set_algo, train_loader, test_loader)

    gap = (set_result["best_acc"] - static_result["best_acc"]) * 100
    print()
    print("=" * 80)
    print(f"  Static best: {static_result['best_acc']*100:.2f}% at ep "
          f"{static_result['best_ep']}  ({static_result['total_s']:.0f}s)")
    print(f"  SET    best: {set_result['best_acc']*100:.2f}% at ep "
          f"{set_result['best_ep']}  ({set_result['total_s']:.0f}s)")
    print(f"  Gap:   {gap:+.2f} pp (SET - Static; positive = SET wins)")
    print()
    if gap > 0.2:
        print("  → SET improves accuracy at matched sparsity. DST helps.")
    elif gap > -0.2:
        print("  → SET matches Static within noise.")
    else:
        print("  → Static wins — possibly the update schedule or drop "
              "fraction is tuned badly, or SET just doesn't help here.")
    print("=" * 80)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos", "demo_10_set_vs_static.png"
    )
    plot_comparison(static_result, set_result, out_path)
    print(f"Saved plot: {out_path}\n")


if __name__ == "__main__":
    main()
