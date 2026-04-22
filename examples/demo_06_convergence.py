"""
Demo 6 — Convergence to exhaustion: does sparse catch up to dense given
enough training budget, or is there a hard capacity ceiling?

What this demo asks
───────────────────
demo_05 ran 3 epochs and showed sparse training loses 1-13 pp accuracy
vs dense, depending on sparsity. The open question: is that gap a
capacity ceiling (the sparse model literally cannot represent what
dense can) or a convergence issue (give it enough steps and it
catches up)?

This matters for positioning. If the gap is purely convergence, then
"sparse is slower but matches dense eventually" is the truthful
narrative. If it's a capacity ceiling, the story becomes "trade X pp
accuracy for Y× less memory at each sparsity level" — no free lunch.

How we answer
─────────────
For each sparsity in {50%, 70%, 80%, 90%, 95%, 99%}:
  - Train both dense (same mask at init, leaked afterwards) and sparse
    (true PaddedCSR) up to MAX_EPOCHS
  - Early-stop each if test accuracy hasn't improved in PATIENCE
    consecutive evaluations
  - Record the FULL accuracy-vs-step curve for both paths
  - Report final best accuracy and the step where that happened

How to run
──────────
    python examples/demo_06_convergence.py

Needs: pip install sparsecore[demos]

Runtime: 30-45 minutes on M3 Pro.

What to look at
───────────────
  1. Best accuracy column — at each sparsity, how close does sparse get?
  2. Plateau step — step at which sparse stops improving. If it's still
     rising at MAX_EPOCHS, the gap might close further.
  3. The saved plot shows full convergence trajectories side-by-side.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

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
from sparsecore import PaddedCSR
from sparsecore.ops import _SpMMFunction


warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────

HIDDEN = 512
BATCH_SIZE = 128
LR = 0.01
MAX_EPOCHS = 15
PATIENCE = 4            # consecutive no-improvement eval points → stop
EVAL_EVERY_EPOCHS = 1   # run test-set eval this often
SPARSITIES = [0.7, 0.9, 0.99]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@dataclass
class ConvergenceTrace:
    sparsity: float
    path: str                      # "dense" or "sparse"
    eval_epochs: list[int]         # epoch at which each eval was recorded
    eval_accuracies: list[float]   # test accuracy at each eval
    best_accuracy: float
    best_epoch: int
    stopped_early: bool
    total_epochs_trained: int
    total_seconds: float


# ─────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────

def get_loaders():
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


# ─────────────────────────────────────────────────────────────────────
#  Eval helpers
# ─────────────────────────────────────────────────────────────────────

def evaluate_dense(fc1, fc2, test_loader):
    with torch.no_grad():
        correct, total = 0, 0
        for imgs, labels in test_loader:
            x = imgs.view(imgs.size(0), -1).t()
            h = F.relu(fc1.weight @ x)
            logits = fc2.weight @ h
            correct += (logits.t().argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_sparse(W_csr, fc2, test_loader):
    with torch.no_grad():
        correct, total = 0, 0
        for imgs, labels in test_loader:
            x = imgs.view(imgs.size(0), -1).t()
            h = F.relu(sparsecore.spmm(W_csr, x))
            logits = fc2.weight @ h
            correct += (logits.t().argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────
#  Train-to-convergence with early stopping
# ─────────────────────────────────────────────────────────────────────

def train_dense_to_convergence(sparsity, train_loader, test_loader) -> ConvergenceTrace:
    """Dense path trained up to MAX_EPOCHS with early stopping."""
    torch.manual_seed(0)
    fc1 = nn.Linear(784, HIDDEN, bias=False)
    with torch.no_grad():
        mask = torch.rand_like(fc1.weight) >= sparsity
        fc1.weight.data *= mask.float()
    fc2 = nn.Linear(HIDDEN, 10, bias=False)
    opt = torch.optim.SGD([fc1.weight, fc2.weight], lr=LR)

    eval_epochs, eval_accs = [], []
    best_acc, best_epoch = 0.0, 0
    stale = 0
    stopped_early = False

    t_start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        for imgs, labels in train_loader:
            x = imgs.view(imgs.size(0), -1).t()
            opt.zero_grad()
            h = F.relu(fc1.weight @ x)
            logits = fc2.weight @ h
            loss = F.cross_entropy(logits.t(), labels)
            loss.backward()
            opt.step()

        if epoch % EVAL_EVERY_EPOCHS == 0:
            acc = evaluate_dense(fc1, fc2, test_loader)
            eval_epochs.append(epoch)
            eval_accs.append(acc)
            if acc > best_acc + 1e-4:
                best_acc, best_epoch = acc, epoch
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                stopped_early = True
                break

    return ConvergenceTrace(
        sparsity=1.0 - mask.float().mean().item(),
        path="dense",
        eval_epochs=eval_epochs,
        eval_accuracies=eval_accs,
        best_accuracy=best_acc,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        total_epochs_trained=eval_epochs[-1],
        total_seconds=time.perf_counter() - t_start,
    )


def train_sparse_to_convergence(sparsity, train_loader, test_loader) -> ConvergenceTrace:
    """Sparse path trained up to MAX_EPOCHS with early stopping."""
    torch.manual_seed(0)
    bound = 1.0 / (784 ** 0.5)
    W1_init = (torch.rand(HIDDEN, 784) * 2 - 1) * bound
    mask1 = (torch.rand(HIDDEN, 784) >= sparsity).float()
    W1_init = W1_init * mask1
    W1_csr = PaddedCSR.from_dense(W1_init)
    W1_values_t = torch.from_numpy(np.asarray(W1_csr.values)).requires_grad_(True)
    fc2 = nn.Linear(HIDDEN, 10, bias=False)

    eval_epochs, eval_accs = [], []
    best_acc, best_epoch = 0.0, 0
    stale = 0
    stopped_early = False

    t_start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        for imgs, labels in train_loader:
            x = imgs.view(imgs.size(0), -1).t()
            if W1_values_t.grad is not None: W1_values_t.grad.zero_()
            if fc2.weight.grad is not None: fc2.weight.grad.zero_()
            h = F.relu(_SpMMFunction.apply(W1_values_t, W1_csr, x, "simd"))
            logits = fc2.weight @ h
            loss = F.cross_entropy(logits.t(), labels)
            loss.backward()
            with torch.no_grad():
                W1_values_t.data -= LR * W1_values_t.grad
                W1_csr.values[:] = W1_values_t.data.numpy()
                fc2.weight.data -= LR * fc2.weight.grad

        if epoch % EVAL_EVERY_EPOCHS == 0:
            acc = evaluate_sparse(W1_csr, fc2, test_loader)
            eval_epochs.append(epoch)
            eval_accs.append(acc)
            if acc > best_acc + 1e-4:
                best_acc, best_epoch = acc, epoch
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                stopped_early = True
                break

    return ConvergenceTrace(
        sparsity=1.0 - W1_csr.nnz / (HIDDEN * 784),
        path="sparse",
        eval_epochs=eval_epochs,
        eval_accuracies=eval_accs,
        best_accuracy=best_acc,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        total_epochs_trained=eval_epochs[-1],
        total_seconds=time.perf_counter() - t_start,
    )


# ─────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print("═" * 90)
    print("SparseCore demo 6 — Convergence to exhaustion")
    print(
        f"Up to {MAX_EPOCHS} epochs with early stopping "
        f"(patience={PATIENCE} evals)  |  "
        f"LR={LR}  batch={BATCH_SIZE}"
    )
    print("═" * 90)


def print_row(dense: ConvergenceTrace, sparse: ConvergenceTrace):
    gap = dense.best_accuracy - sparse.best_accuracy
    dense_tag = "early" if dense.stopped_early else "maxed"
    sparse_tag = "early" if sparse.stopped_early else "maxed"
    print(
        f"{sparse.sparsity * 100:>6.1f}%  "
        f"dense  best={dense.best_accuracy * 100:>5.2f}%  "
        f"@ep{dense.best_epoch:>2d}  "
        f"(stopped {dense_tag} @ep{dense.total_epochs_trained})  "
        f"{dense.total_seconds:>5.0f}s"
    )
    print(
        f"         sparse best={sparse.best_accuracy * 100:>5.2f}%  "
        f"@ep{sparse.best_epoch:>2d}  "
        f"(stopped {sparse_tag} @ep{sparse.total_epochs_trained})  "
        f"{sparse.total_seconds:>5.0f}s  "
        f"gap={gap * 100:>5.2f}pp"
    )
    print()


def plot_curves(dense_results, sparse_results, out_path):
    fig, axes = plt.subplots(1, len(dense_results), figsize=(3.5 * len(dense_results), 4.5), sharey=True)
    for ax, dr, sr in zip(axes, dense_results, sparse_results):
        ax.plot(dr.eval_epochs, [a * 100 for a in dr.eval_accuracies],
                color="#3b82f6", label=f"dense", linewidth=2)
        ax.plot(sr.eval_epochs, [a * 100 for a in sr.eval_accuracies],
                color="#ef4444", label=f"sparse", linewidth=2)
        ax.set_title(f"{sr.sparsity * 100:.0f}% sparse")
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        # Mark the best-ever point on each curve
        ax.scatter([dr.best_epoch], [dr.best_accuracy * 100], color="#1d4ed8", zorder=5, s=40)
        ax.scatter([sr.best_epoch], [sr.best_accuracy * 100], color="#991b1b", zorder=5, s=40)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("Convergence to exhaustion — dense vs sparse at each sparsity", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"\nSaved convergence plot to: {out_path}")


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print_header()
    print("Loading MNIST...")
    train_loader, test_loader = get_loaders()

    dense_results, sparse_results = [], []
    for i, s in enumerate(SPARSITIES):
        print(f"\n─── sparsity {s * 100:.0f}% ({i+1}/{len(SPARSITIES)}) ───")
        dr = train_dense_to_convergence(s, train_loader, test_loader)
        sr = train_sparse_to_convergence(s, train_loader, test_loader)
        print_row(dr, sr)
        dense_results.append(dr)
        sparse_results.append(sr)

    # Summary table
    print("═" * 90)
    print(f"{'sparsity':>9s}  {'dense best':>11s}  {'sparse best':>12s}  {'gap':>8s}  "
          f"{'dense ep':>9s}  {'sparse ep':>10s}")
    print("─" * 90)
    for dr, sr in zip(dense_results, sparse_results):
        gap = (dr.best_accuracy - sr.best_accuracy) * 100
        print(
            f"{sr.sparsity * 100:>8.1f}%  "
            f"{dr.best_accuracy * 100:>10.2f}%  "
            f"{sr.best_accuracy * 100:>11.2f}%  "
            f"{gap:>6.2f}pp  "
            f"{dr.best_epoch:>8d}  "
            f"{sr.best_epoch:>9d}"
        )
    print("═" * 90)

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "demo_06_convergence.png")
    plot_curves(dense_results, sparse_results, out_path)
    print()


if __name__ == "__main__":
    main()
