"""
Demo 7 — Train to real convergence at 90% sparsity.

The question demo_06 left open: at 15 epochs, both dense and sparse at
90% sparsity were STILL improving. Did the gap close with more time,
or is it a real capacity ceiling?

This demo answers that specific question by removing the 15-epoch cap
and letting both models run until early stopping kicks in. We use
patience=10 (very conservative) so we genuinely wait for each model to
plateau before calling it converged.

How to run
──────────
    python examples/demo_07_90pct_convergence.py

Needs: pip install sparsecore[demos]

Runtime: ~8-15 minutes on M3 Pro.

What to look at
───────────────
  1. Final best accuracy for each path — this is the real plateau.
  2. Epoch where each plateaued — are the plateaus at similar epochs,
     or does sparse need significantly more training?
  3. The final gap — if < 1pp, it's essentially a convergence story.
     If > 2pp, there's a structural capacity ceiling even at 90%.
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
from sparsecore import PaddedCSR
from sparsecore.ops import _SpMMFunction


warnings.filterwarnings("ignore", category=UserWarning)


HIDDEN = 512
BATCH_SIZE = 128
LR = 0.01
SPARSITY = 0.9
MAX_EPOCHS = 100       # hard safety cap
PATIENCE = 10          # very patient — wait 10 epochs for any improvement
EVAL_EVERY = 1
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


def train_to_plateau(name: str, train_step, evaluate, train_loader, test_loader):
    """Generic training loop that stops when patience runs out."""
    eval_epochs, eval_accs = [], []
    best_acc, best_ep = 0.0, 0
    stale = 0

    t_start = time.perf_counter()
    for ep in range(1, MAX_EPOCHS + 1):
        for imgs, labels in train_loader:
            train_step(imgs, labels)
        if ep % EVAL_EVERY == 0:
            acc = evaluate(test_loader)
            eval_epochs.append(ep)
            eval_accs.append(acc)
            if acc > best_acc + 1e-4:
                best_acc, best_ep = acc, ep
                stale = 0
                print(f"    [{name}] ep {ep:>3d}  acc={acc*100:.2f}%  (new best)")
            else:
                stale += 1
                if ep % 5 == 0:
                    print(f"    [{name}] ep {ep:>3d}  acc={acc*100:.2f}%  (stale for {stale})")
            if stale >= PATIENCE:
                print(f"    [{name}] plateau reached at ep {ep}, best was ep {best_ep}")
                break
    else:
        print(f"    [{name}] hit MAX_EPOCHS={MAX_EPOCHS}, not converged?")

    return {
        "eval_epochs": eval_epochs,
        "eval_accs": eval_accs,
        "best_acc": best_acc,
        "best_ep": best_ep,
        "total_epochs": eval_epochs[-1] if eval_epochs else 0,
        "total_s": time.perf_counter() - t_start,
    }


def make_dense_path(sparsity: float):
    torch.manual_seed(0)
    fc1 = nn.Linear(784, HIDDEN, bias=False)
    with torch.no_grad():
        mask = torch.rand_like(fc1.weight) >= sparsity
        fc1.weight.data *= mask.float()
    fc2 = nn.Linear(HIDDEN, 10, bias=False)
    opt = torch.optim.SGD([fc1.weight, fc2.weight], lr=LR)

    def step(imgs, labels):
        x = imgs.view(imgs.size(0), -1).t()
        opt.zero_grad()
        h = F.relu(fc1.weight @ x)
        logits = fc2.weight @ h
        loss = F.cross_entropy(logits.t(), labels)
        loss.backward()
        opt.step()

    def evaluate(loader):
        with torch.no_grad():
            correct, total = 0, 0
            for imgs, labels in loader:
                x = imgs.view(imgs.size(0), -1).t()
                h = F.relu(fc1.weight @ x)
                logits = fc2.weight @ h
                correct += (logits.t().argmax(1) == labels).sum().item()
                total += labels.size(0)
        return correct / total

    return step, evaluate


def make_sparse_path(sparsity: float):
    torch.manual_seed(0)
    bound = 1.0 / (784 ** 0.5)
    W1_init = (torch.rand(HIDDEN, 784) * 2 - 1) * bound
    mask = (torch.rand(HIDDEN, 784) >= sparsity).float()
    W1_init = W1_init * mask
    W_csr = PaddedCSR.from_dense(W1_init)
    W_vals = torch.from_numpy(np.asarray(W_csr.values)).requires_grad_(True)
    fc2 = nn.Linear(HIDDEN, 10, bias=False)

    def step(imgs, labels):
        x = imgs.view(imgs.size(0), -1).t()
        if W_vals.grad is not None: W_vals.grad.zero_()
        if fc2.weight.grad is not None: fc2.weight.grad.zero_()
        h = F.relu(_SpMMFunction.apply(W_vals, W_csr, x, "simd"))
        logits = fc2.weight @ h
        loss = F.cross_entropy(logits.t(), labels)
        loss.backward()
        with torch.no_grad():
            W_vals.data -= LR * W_vals.grad
            W_csr.values[:] = W_vals.data.numpy()
            fc2.weight.data -= LR * fc2.weight.grad

    def evaluate(loader):
        with torch.no_grad():
            correct, total = 0, 0
            for imgs, labels in loader:
                x = imgs.view(imgs.size(0), -1).t()
                h = F.relu(sparsecore.spmm(W_csr, x))
                logits = fc2.weight @ h
                correct += (logits.t().argmax(1) == labels).sum().item()
                total += labels.size(0)
        return correct / total

    return step, evaluate


def plot_curves(d, s, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(d["eval_epochs"], [a * 100 for a in d["eval_accs"]],
            color="#3b82f6", label=f'dense (best {d["best_acc"]*100:.2f}% @ep{d["best_ep"]})',
            linewidth=2)
    ax.plot(s["eval_epochs"], [a * 100 for a in s["eval_accs"]],
            color="#ef4444", label=f'sparse (best {s["best_acc"]*100:.2f}% @ep{s["best_ep"]})',
            linewidth=2)
    ax.scatter([d["best_ep"]], [d["best_acc"] * 100], color="#1d4ed8", s=60, zorder=5)
    ax.scatter([s["best_ep"]], [s["best_acc"] * 100], color="#991b1b", s=60, zorder=5)
    ax.set_title(f"Convergence at {SPARSITY*100:.0f}% sparsity — plateau of each path")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    print(f"\nSparseCore demo 7 — {SPARSITY*100:.0f}% sparsity convergence test")
    print(f"MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}  LR={LR}  batch={BATCH_SIZE}")
    print("=" * 80)

    print("Loading MNIST...")
    train_loader, test_loader = load_mnist()

    print(f"\n─── DENSE path at {SPARSITY*100:.0f}% sparsity (mask simulated) ───")
    d_step, d_eval = make_dense_path(SPARSITY)
    d = train_to_plateau("dense", d_step, d_eval, train_loader, test_loader)
    print(f"    Total: {d['total_s']:.0f}s over {d['total_epochs']} epochs")

    print(f"\n─── SPARSE path at {SPARSITY*100:.0f}% sparsity (PaddedCSR) ───")
    s_step, s_eval = make_sparse_path(SPARSITY)
    s = train_to_plateau("sparse", s_step, s_eval, train_loader, test_loader)
    print(f"    Total: {s['total_s']:.0f}s over {s['total_epochs']} epochs")

    gap = (d["best_acc"] - s["best_acc"]) * 100
    print()
    print("=" * 80)
    print(f"  Dense  plateau: {d['best_acc']*100:.2f}%  at epoch {d['best_ep']}")
    print(f"  Sparse plateau: {s['best_acc']*100:.2f}%  at epoch {s['best_ep']}")
    print(f"  Gap: {gap:.2f}pp")
    print()
    if gap < 1.0:
        print("  → Gap < 1pp. Sparse effectively matches dense at plateau.")
        print("    Story is: sparse is slower per epoch but reaches same quality.")
    elif gap < 2.5:
        print("  → Gap in 1-2.5pp range. Modest structural difference.")
        print("    Story is: small quality tax for the 82% memory savings.")
    else:
        print("  → Gap > 2.5pp. Real capacity ceiling at this sparsity + random init.")
        print("    Motivates RigL-style regrow (milestone 4e).")
    print("=" * 80)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos", "demo_07_90pct_curves.png"
    )
    plot_curves(d, s, out_path)
    print(f"Saved plot: {out_path}\n")


if __name__ == "__main__":
    main()
