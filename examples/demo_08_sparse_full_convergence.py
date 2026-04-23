"""
Demo 8 — Train the 90% sparse model to real convergence, no epoch cap.

The question demo_07 left open: sparse was still improving when it hit
MAX_EPOCHS=100 at 97.09% (vs dense's plateau of 98.06%, gap 0.97 pp).
Does sparse actually *converge* to dense's accuracy given the time it
wants, or does it plateau short?

This demo removes the tight cap. MAX_EPOCHS=500 is a safety valve, not
a target — the real stop criterion is PATIENCE=10 epochs with no test
accuracy improvement. We run dense the same way in the same script so
the numbers are directly comparable from one clean run.

How to run
──────────
    python examples/demo_08_sparse_full_convergence.py

Needs: pip install sparselab[demos]

Runtime: depends. Dense took ~4 min last time (plateau at ep 82). Sparse
was doing ~7s/epoch at 90% sparsity; if it needs 150–200 epochs to
plateau, that's ~20–25 min. Budget 30 min total.

What to look at
───────────────
  1. Did sparse trigger early stopping (good: it actually plateaued)
     or hit MAX_EPOCHS (we still don't know its true ceiling).
  2. Final gap. <0.5pp = "effectively matches". >1pp = real structural
     difference at this sparsity level.
  3. Ratio of epochs needed. If sparse needs 2–3× dense's epoch count
     to reach the same accuracy, that's a clean "you pay in epochs,
     not accuracy" story for the launch blog.
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
    raise SystemExit("pip install sparselab[demos]")

try:
    from torchvision import datasets, transforms
except ImportError:
    raise SystemExit("pip install sparselab[demos]")

import sparselab
from sparselab import PaddedCSR
from sparselab.ops import _SpMMFunction


warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
#
# MAX_EPOCHS is a safety valve; PATIENCE is the real stop rule.
# 500 epochs × ~7s (sparse) = ~60 min absolute worst case.

HIDDEN = 512
BATCH_SIZE = 128
LR = 0.01
SPARSITY = 0.9
MAX_EPOCHS = 500
PATIENCE = 10
EVAL_EVERY = 1
HEARTBEAT_EVERY = 20   # once a run goes long, a quiet progress note every N epochs
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
    """Patience-driven training loop. MAX_EPOCHS is purely a safety cap."""
    eval_epochs, eval_accs = [], []
    best_acc, best_ep = 0.0, 0
    stale = 0

    t_start = time.perf_counter()
    last_heartbeat = 0
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
                # Every new best is printed; very noisy early, tapers off.
                print(f"    [{name}] ep {ep:>4d}  acc={acc*100:.2f}%  (new best)")
                last_heartbeat = ep
            else:
                stale += 1
                # A quiet heartbeat every HEARTBEAT_EVERY stale epochs so
                # the user sees the run is alive without spam.
                if ep - last_heartbeat >= HEARTBEAT_EVERY:
                    elapsed = time.perf_counter() - t_start
                    print(f"    [{name}] ep {ep:>4d}  acc={acc*100:.2f}%  "
                          f"(stale {stale}/{PATIENCE}, elapsed {elapsed:.0f}s)")
                    last_heartbeat = ep

            if stale >= PATIENCE:
                print(f"    [{name}] converged at ep {ep}, best was ep {best_ep}")
                break
    else:
        # for-else: only runs if we exhausted MAX_EPOCHS without break.
        print(f"    [{name}] hit MAX_EPOCHS={MAX_EPOCHS} without plateauing "
              f"(still-climbing behaviour)")

    return {
        "eval_epochs": eval_epochs,
        "eval_accs": eval_accs,
        "best_acc": best_acc,
        "best_ep": best_ep,
        "total_epochs": eval_epochs[-1] if eval_epochs else 0,
        "total_s": time.perf_counter() - t_start,
        "converged": stale >= PATIENCE,
    }


def make_dense_path(sparsity: float):
    """Dense path trained with a random mask applied only at init.
    After init, the zeros leak (dense gradient fills them). Same seed as
    sparse path so the comparison is fair at identical starting weights."""
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
    """True sparse path: weight lives in PaddedCSR; training only touches
    the `values` buffer. Same seed and init distribution as dense path."""
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
        # Manual optimizer: walk the torch grad back into the CSR buffer
        # so the next forward sees updated weights.
        with torch.no_grad():
            W_vals.data -= LR * W_vals.grad
            W_csr.values[:] = W_vals.data.numpy()
            fc2.weight.data -= LR * fc2.weight.grad

    def evaluate(loader):
        with torch.no_grad():
            correct, total = 0, 0
            for imgs, labels in loader:
                x = imgs.view(imgs.size(0), -1).t()
                h = F.relu(sparselab.spmm(W_csr, x))
                logits = fc2.weight @ h
                correct += (logits.t().argmax(1) == labels).sum().item()
                total += labels.size(0)
        return correct / total

    return step, evaluate


def plot_curves(d, s, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(d["eval_epochs"], [a * 100 for a in d["eval_accs"]],
            color="#3b82f6",
            label=f'dense (best {d["best_acc"]*100:.2f}% @ep{d["best_ep"]}, '
                  f'{"converged" if d["converged"] else "still climbing"})',
            linewidth=2)
    ax.plot(s["eval_epochs"], [a * 100 for a in s["eval_accs"]],
            color="#ef4444",
            label=f'sparse (best {s["best_acc"]*100:.2f}% @ep{s["best_ep"]}, '
                  f'{"converged" if s["converged"] else "still climbing"})',
            linewidth=2)
    ax.scatter([d["best_ep"]], [d["best_acc"] * 100], color="#1d4ed8", s=60, zorder=5)
    ax.scatter([s["best_ep"]], [s["best_acc"] * 100], color="#991b1b", s=60, zorder=5)
    ax.set_title(f"Full convergence at {SPARSITY*100:.0f}% sparsity "
                 f"(patience={PATIENCE}, MAX_EPOCHS={MAX_EPOCHS})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    print(f"\nSparseLab demo 8 — full convergence at {SPARSITY*100:.0f}% sparsity")
    print(f"MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}  LR={LR}  batch={BATCH_SIZE}")
    print("=" * 80)

    print("Loading MNIST...")
    train_loader, test_loader = load_mnist()

    print(f"\n─── DENSE path at {SPARSITY*100:.0f}% sparsity (mask simulated) ───")
    d_step, d_eval = make_dense_path(SPARSITY)
    d = train_to_plateau("dense", d_step, d_eval, train_loader, test_loader)
    print(f"    Total: {d['total_s']:.0f}s over {d['total_epochs']} epochs "
          f"({'converged' if d['converged'] else 'did not converge'})")

    print(f"\n─── SPARSE path at {SPARSITY*100:.0f}% sparsity (PaddedCSR) ───")
    s_step, s_eval = make_sparse_path(SPARSITY)
    s = train_to_plateau("sparse", s_step, s_eval, train_loader, test_loader)
    print(f"    Total: {s['total_s']:.0f}s over {s['total_epochs']} epochs "
          f"({'converged' if s['converged'] else 'did not converge'})")

    gap = (d["best_acc"] - s["best_acc"]) * 100
    print()
    print("=" * 80)
    print(f"  Dense  plateau: {d['best_acc']*100:.2f}%  at epoch {d['best_ep']}  "
          f"({'converged' if d['converged'] else 'hit max epochs'})")
    print(f"  Sparse plateau: {s['best_acc']*100:.2f}%  at epoch {s['best_ep']}  "
          f"({'converged' if s['converged'] else 'hit max epochs'})")
    print(f"  Gap: {gap:.2f}pp")
    print(f"  Epoch ratio: sparse needed {s['best_ep'] / max(d['best_ep'], 1):.1f}× "
          f"dense's epochs to reach its best")
    print()
    if d["converged"] and s["converged"]:
        if gap < 0.5:
            print("  → CLEAN RESULT: both converged, gap < 0.5pp.")
            print("    Story: sparse at 90% matches dense on MNIST, full stop.")
        elif gap < 1.5:
            print("  → Both converged, gap < 1.5pp. Small structural difference.")
        else:
            print(f"  → Both converged with a {gap:.2f}pp gap. Real capacity difference.")
    elif not s["converged"]:
        print("  → Sparse hit MAX_EPOCHS without plateauing. Raise MAX_EPOCHS and rerun.")
    else:
        print("  → Dense hit MAX_EPOCHS without plateauing. Unusual; investigate.")
    print("=" * 80)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos", "demo_08_full_convergence.png"
    )
    plot_curves(d, s, out_path)
    print(f"Saved plot: {out_path}\n")


if __name__ == "__main__":
    main()
