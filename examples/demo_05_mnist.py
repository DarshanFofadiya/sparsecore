"""
Demo 5 — MNIST training to convergence at multiple sparsities.

What this demo proves
─────────────────────
demo_04 showed that sparse training runs. demo_04b showed the speed
story honestly (we lose to Apple AMX on Apple Silicon). This demo
answers the other critical question: **at what sparsity does our
model still learn the task?**

We train a single-hidden-layer MLP on MNIST at 7 sparsity levels:
    784 → 512 hidden → 10
The hidden layer is the sparse one. Input and output layers stay
dense (too small to matter, and representing "the pixel" or "the
class" benefits from full capacity).

Saves a PNG with loss curves so you can watch convergence shape.

How to run
──────────
    python examples/demo_05_mnist.py

Needs torchvision + matplotlib:
    pip install sparselab[demos]

What to look at
───────────────
  1. Loss curves: all sparsities should decrease, but higher sparsity
     should converge slower / to a higher floor. Classic DST result.
  2. Test accuracy: watch how much we lose per sparsity step. At 50-70%
     you should lose <2% accuracy; at 95% maybe 5%; at 99% maybe 10%.
  3. Wall-clock per step at each sparsity: should shrink as sparsity
     rises (fewer FMAs), though still slower than dense on Apple AMX.
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
    matplotlib.use("Agg")  # headless: no interactive window
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "demo_05 requires matplotlib. Install with: pip install sparselab[demos]"
    )

try:
    from torchvision import datasets, transforms
except ImportError:
    raise SystemExit(
        "demo_05 requires torchvision. Install with: pip install sparselab[demos]"
    )

import sparselab
from sparselab import PaddedCSR


warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────

# Net shape: 784 (28*28) → HIDDEN → 10 classes
HIDDEN = 512

# Sparsity levels to sweep on the hidden layer's W
SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

# Training config
BATCH_SIZE = 128
NUM_EPOCHS = 3       # 3 epochs × 469 batches = ~1400 steps
LR = 0.01
LOG_EVERY = 50       # record loss every this many steps
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ─────────────────────────────────────────────────────────────────────
#  Memory accounting — at-rest model size
# ─────────────────────────────────────────────────────────────────────
#
#  "How much memory does this model take?" is genuinely hard to measure
#  under torch because its allocator caches freed memory and tracemalloc
#  only sees Python-heap allocations. Attempting to measure the peak
#  dynamic allocation of a single training step gave us nonsensical
#  numbers (tracemalloc) or allocator-noise-dominated readings (psutil).
#
#  The most useful, verifiable number is the AT-REST model size:
#  how many bytes does a checkpointed version of this model occupy?
#  This is what a user cares about when asking "does my model fit in
#  memory?" — the dynamic memory during a step is smaller than the
#  at-rest parameter storage for any model big enough to matter.
#
#  We compute it by summing the actual bytes of each tensor that makes
#  up the layer's state. No allocator, no noise, no lies.
# ─────────────────────────────────────────────────────────────────────

def dense_model_bytes(shape: tuple[int, int]) -> int:
    """
    At-rest bytes for dense layer state: just the weight + gradient tensors.

    - weight:      M * K * 4 bytes (float32)
    - weight.grad: same size (allocated during backward, stays around
                   until the next zero_grad or explicit release)

    Total: 8 * M * K bytes.

    IMPORTANT: the dense weight tensor stores the full M*K grid regardless
    of sparsity, because a mask-simulated approach never frees zero cells.
    This is what Cerebras, rigl-torch, and most DST libraries do.
    """
    M, K = shape
    return 2 * M * K * 4


def sparse_model_bytes(W_csr) -> int:
    """
    At-rest bytes for a PaddedCSR layer's training state.

    Storage (all tracked by PaddedCSR):
      - values:       total_capacity * 4 bytes (float32)
      - col_indices:  total_capacity * 4 bytes (int32)
      - row_start:    nrows * 4 bytes (int32)
      - row_nnz:      nrows * 4 bytes (int32)
      - row_capacity: nrows * 4 bytes (int32)

    Gradient (our design: aligned to values for in-place SGD):
      - dW_values: total_capacity * 4 bytes (float32)

    We do NOT count the transient Wᵀ here because it's freed after
    each backward pass. At-rest memory is what persists, not peak
    transient memory. The transient Wᵀ would only matter if multiple
    Wᵀ's existed simultaneously, which they don't.

    This function gives us an honest apples-to-apples comparison with
    dense_model_bytes. The difference is exactly what the sparse
    storage format trades for: index overhead vs. row-by-row zero
    elimination.
    """
    cap = W_csr.total_capacity
    nrows = W_csr.nrows
    values_b = cap * 4      # float32 weight values
    cols_b = cap * 4        # int32 column indices
    rowdata_b = 3 * nrows * 4  # row_start + row_nnz + row_capacity
    grad_b = cap * 4        # dW_values gradient buffer
    return values_b + cols_b + rowdata_b + grad_b


@dataclass
class RunResult:
    sparsity: float                  # actual, not target
    losses: list[float]              # running loss recorded every LOG_EVERY steps
    test_accuracy: float             # final accuracy on MNIST test set
    ms_per_step: float               # mean wall-clock per train step
    total_seconds: float             # total training time
    num_live_params: int             # count of learnable params in hidden layer
    model_bytes: int                 # at-rest bytes of the hidden layer's
                                     # state (weight + gradient + any index
                                     # structure). Computed exactly from
                                     # tensor sizes, not measured.


# ─────────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────────

def get_mnist_loaders():
    """Load MNIST. Downloads to ./data/ on first run, ~12 MB."""
    transform = transforms.Compose([
        transforms.ToTensor(),              # [0, 255] uint8 → [0.0, 1.0] float32
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST stats
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────
#  Two training functions — dense and sparse — share everything else
# ─────────────────────────────────────────────────────────────────────

def train_dense(train_loader, test_loader, sparsity: float) -> RunResult:
    """
    Train an all-dense MLP. `sparsity` controls how many hidden-layer
    weights start at 0 (but the tensor is stored dense and the mask
    isn't re-applied after each step — the sparsity is leaked).

    This is the dense-simulated baseline: what Cerebras and rigl-torch
    do. We include it for comparison but it's not the product we ship.
    """
    torch.manual_seed(0)
    # Input layer: 784 → HIDDEN (dense)
    fc1 = nn.Linear(784, HIDDEN, bias=False)
    # Apply a sparsity mask at init but don't enforce it afterwards
    with torch.no_grad():
        mask = torch.rand_like(fc1.weight) >= sparsity
        fc1.weight.data *= mask.float()
    # Output layer: HIDDEN → 10
    fc2 = nn.Linear(HIDDEN, 10, bias=False)

    opt = torch.optim.SGD([fc1.weight, fc2.weight], lr=LR)

    model_bytes = dense_model_bytes((HIDDEN, 784))

    losses: list[float] = []
    num_live_params = int(mask.sum().item())
    t_start = time.perf_counter()
    step_times: list[float] = []

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Flatten 28x28 → 784; transpose so shape is (784, batch) for matmul
            x = images.view(images.size(0), -1).t()  # (784, B)

            t_step = time.perf_counter()

            opt.zero_grad()
            h = fc1.weight @ x                 # (HIDDEN, B)
            h = F.relu(h)
            logits = fc2.weight @ h            # (10, B)
            loss = F.cross_entropy(logits.t(), labels)
            loss.backward()
            opt.step()

            step_times.append(time.perf_counter() - t_step)
            if (len(losses) * LOG_EVERY) < (epoch * len(train_loader) + batch_idx + 1):
                losses.append(loss.item())

    total_s = time.perf_counter() - t_start

    # Evaluate
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            x = images.view(images.size(0), -1).t()
            h = F.relu(fc1.weight @ x)
            logits = fc2.weight @ h
            pred = logits.t().argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        acc = correct / total

    return RunResult(
        sparsity=1.0 - (mask.float().mean().item()),
        losses=losses,
        test_accuracy=acc,
        ms_per_step=np.mean(step_times) * 1000,
        total_seconds=total_s,
        num_live_params=num_live_params,
        model_bytes=model_bytes,
    )


def train_sparse(train_loader, test_loader, sparsity: float) -> RunResult:
    """
    Train an MLP where the hidden layer is a SparseLab ``SparseLinear``
    backed by PaddedCSR storage. Input and output layers stay dense.

    This is the flagship: our actual product's training loop, as of
    milestone 4b. Notice how little of this function actually mentions
    SparseLab — the whole training loop looks identical to a dense
    PyTorch training loop, except for one line:

        fc1 = sparselab.SparseLinear(784, HIDDEN, sparsity=sparsity, bias=False)

    That is the "two-line adoption" promise stated in PROJECT_OVERVIEW.md.
    """
    torch.manual_seed(0)

    # The ONE sparse-specific line in this training loop.
    fc1 = sparselab.SparseLinear(784, HIDDEN, sparsity=sparsity, bias=False)
    fc2 = nn.Linear(HIDDEN, 10, bias=False)

    # Standard torch.optim usage — no special sparse-aware optimizer.
    opt = torch.optim.SGD(list(fc1.parameters()) + list(fc2.parameters()), lr=LR)

    # For the memory-at-rest column of the results table: read the live
    # count off the SparseLinear layer. The bytes helper still operates
    # on the underlying PaddedCSR.
    num_live_params = fc1.nnz
    model_bytes = sparse_model_bytes(fc1._csr)

    losses: list[float] = []
    t_start = time.perf_counter()
    step_times: list[float] = []

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (images, labels) in enumerate(train_loader):
            # (B, 784) — SparseLinear handles the (*, H_in) -> (*, H_out) shape contract.
            x = images.view(images.size(0), -1)

            t_step = time.perf_counter()

            opt.zero_grad()
            h = F.relu(fc1(x))            # sparse forward
            logits = fc2(h)               # dense output layer
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

            step_times.append(time.perf_counter() - t_step)
            if (len(losses) * LOG_EVERY) < (epoch * len(train_loader) + batch_idx + 1):
                losses.append(loss.item())

    total_s = time.perf_counter() - t_start

    # Evaluate. We keep using the plain sparselab.spmm() for eval
    # (no autograd overhead). Same weights via fc1._csr.
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            x = images.view(images.size(0), -1)
            h = F.relu(fc1(x))
            logits = fc2(h)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        acc = correct / total

    return RunResult(
        sparsity=1.0 - fc1.density,
        losses=losses,
        test_accuracy=acc,
        ms_per_step=np.mean(step_times) * 1000,
        total_seconds=total_s,
        num_live_params=num_live_params,
        model_bytes=model_bytes,
    )


# ─────────────────────────────────────────────────────────────────────
#  Reporting — text summary + matplotlib plot
# ─────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print("═" * 78)
    print("SparseLab demo 5 — MNIST training to convergence at multiple sparsities")
    print(f"Hidden layer: {HIDDEN}   epochs: {NUM_EPOCHS}   batch: {BATCH_SIZE}   lr: {LR}")
    print("═" * 78)


def print_summary(dense_results, sparse_results):
    print()
    print(
        f"{'SPARSITY':>9s}  {'LIVE PARAMS':>12s}  "
        f"{'dense acc':>10s}  {'sparse acc':>11s}  "
        f"{'dense ms':>10s}  {'sparse ms':>11s}  "
        f"{'dense KB':>10s}  {'sparse KB':>11s}  {'mem ratio':>10s}"
    )
    print("─" * 120)
    for dr, sr in zip(dense_results, sparse_results):
        dense_kb = dr.model_bytes / 1024
        sparse_kb = sr.model_bytes / 1024
        ratio = sparse_kb / dense_kb if dense_kb > 0 else float("nan")
        print(
            f"{dr.sparsity * 100:>8.1f}%  "
            f"{sr.num_live_params:>12,}  "
            f"{dr.test_accuracy * 100:>9.2f}%  "
            f"{sr.test_accuracy * 100:>10.2f}%  "
            f"{dr.ms_per_step:>9.2f}   "
            f"{sr.ms_per_step:>10.2f}   "
            f"{dense_kb:>9.1f}   "
            f"{sparse_kb:>10.1f}   "
            f"{ratio * 100:>8.1f}%"
        )
    print()
    print("  'KB' is at-rest bytes of the hidden layer's state (weight + gradient + indices),")
    print("  computed exactly from tensor sizes. Does not include transient backward buffers")
    print("  or activations, which are similar between dense and sparse paths at this batch size.")
    print()


def plot_curves(dense_results, sparse_results, output_path: str):
    """Save a side-by-side loss-curve plot — dense left, sparse right."""
    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Build a colormap across sparsities: light at 0%, dark at 99%
    cmap = plt.get_cmap("viridis")
    n = len(dense_results)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for i, (dr, sr, color) in enumerate(zip(dense_results, sparse_results, colors)):
        steps_d = np.arange(len(dr.losses)) * LOG_EVERY
        steps_s = np.arange(len(sr.losses)) * LOG_EVERY
        ax_d.plot(steps_d, dr.losses, color=color, label=f"{dr.sparsity * 100:.0f}%")
        ax_s.plot(steps_s, sr.losses, color=color, label=f"{sr.sparsity * 100:.0f}%")

    for ax, title in [(ax_d, "Dense (mask-simulated)"), (ax_s, "SparseLab (true sparse)")]:
        ax.set_xlabel("training step")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="sparsity", loc="upper right", fontsize=9)
    ax_d.set_ylabel("cross-entropy loss (log scale)")

    fig.suptitle(
        f"MNIST MLP ({HIDDEN} hidden) — loss curves at multiple sparsities",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Saved loss-curve plot to: {output_path}")


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print_header()
    print("Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders()
    print(f"  train: {len(train_loader.dataset):,} images, "
          f"test: {len(test_loader.dataset):,} images")

    dense_results: list[RunResult] = []
    sparse_results: list[RunResult] = []

    for sparsity in SPARSITIES:
        print(f"\n─── Training at sparsity {sparsity * 100:.0f}% ───")
        print(f"  dense  ", end="", flush=True)
        dr = train_dense(train_loader, test_loader, sparsity)
        print(f"acc={dr.test_accuracy*100:.2f}%, {dr.total_seconds:.1f}s")
        print(f"  sparse ", end="", flush=True)
        sr = train_sparse(train_loader, test_loader, sparsity)
        print(f"acc={sr.test_accuracy*100:.2f}%, {sr.total_seconds:.1f}s")
        dense_results.append(dr)
        sparse_results.append(sr)

    print_summary(dense_results, sparse_results)

    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "demos", "demo_05_mnist_curves.png")
    plot_curves(dense_results, sparse_results, out_path)

    print()
    print("═" * 78)
    print("What to try next:")
    print("  - Raise NUM_EPOCHS to 10 — see if sparse eventually closes the accuracy gap")
    print("  - Change HIDDEN to 1024 or 2048 — more capacity = more room for sparsity")
    print("  - Add a second hidden layer (784 → 512 → 512 → 10) with sparse on both")
    print()


if __name__ == "__main__":
    main()
