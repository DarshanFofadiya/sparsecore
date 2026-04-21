"""
Demo 4 — End-to-end sparse training with autograd (Milestone 4a).

What this demo proves
─────────────────────
Everything from milestones 1-3 was forward-pass only. This is the first
demo where loss.backward() actually works on a sparse layer — the moment
SparseCore crosses from "compute engine" to "training framework."

We set up a trivial regression task:
    y = W_true @ x + ε

where W_true is a fixed random matrix. Then we train TWO models to fit
this relationship:
  1. A dense nn.Linear (PyTorch's built-in, all 8,192 params learnable)
  2. Our SparseCore forward/backward (only live-slot params learnable)

Both run 200 steps of SGD. The loss curve is what you watch.

How to run
──────────
    python examples/demo_04_autograd.py

What to look at
───────────────
  1. Both losses must drop. If the sparse loss stays flat, autograd is
     broken and we have a bug.
  2. The sparse loss should bottom out higher than dense (at some
     noise floor determined by the sparsity). That's expected — a
     90%-sparse model has 10x fewer parameters, so it can't perfectly
     represent a random dense W_true. The gap shrinks with lower
     sparsity.
  3. The "padding slots still zero" line at the end. Confirms that the
     optimizer updates only live weights, not padding — which is the
     whole DST thesis in one assertion.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import torch.nn as nn

import sparsecore
from sparsecore import PaddedCSR
from sparsecore.ops import _SpMMFunction


warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
#  Problem setup
# ─────────────────────────────────────────────────────────────────────

# Dimensions: W is (M, K), x is (K, batch), y is (M, batch).
# Tiny enough to train fast, big enough for SIMD to matter.
M, K, BATCH = 32, 64, 16

# Sparsity level of our trained model. W_true is still dense; our
# sparse model only has (1 - SPARSITY) * M * K learnable params.
SPARSITY = 0.9      # 90% sparse → 10% of cells learnable

# Training config
NUM_STEPS = 200
LR = 0.05


# ─────────────────────────────────────────────────────────────────────
#  Data generation
# ─────────────────────────────────────────────────────────────────────

def make_data(seed: int = 42):
    """Generate W_true, training inputs X, and targets Y."""
    torch.manual_seed(seed)
    W_true = torch.randn(M, K, dtype=torch.float32)
    X = torch.randn(K, BATCH, dtype=torch.float32)
    # Slight noise so the loss floor is above zero for both models
    noise = torch.randn(M, BATCH, dtype=torch.float32) * 0.01
    Y_target = W_true @ X + noise
    return W_true, X, Y_target


# ─────────────────────────────────────────────────────────────────────
#  Dense baseline — vanilla nn.Linear training
# ─────────────────────────────────────────────────────────────────────

def train_dense(X: torch.Tensor, Y_target: torch.Tensor) -> list[float]:
    """
    Train a dense nn.Linear to fit the targets. Returns per-step losses.

    nn.Linear stores weight as (out_features, in_features) = (M, K),
    matching our sparse W shape. No bias — we want pure W @ X learning.
    """
    torch.manual_seed(0)
    layer = nn.Linear(K, M, bias=False)   # W shape (M, K)
    opt = torch.optim.SGD(layer.parameters(), lr=LR)
    losses = []

    # nn.Linear computes Y = X @ Wᵀ. We want Y = W @ X. To match shapes,
    # transpose our input/target so nn.Linear and our sparse path see
    # equivalent problems. We care about the loss curve, not the exact
    # computation path.
    x_in = X.T              # (BATCH, K)
    y_target = Y_target.T   # (BATCH, M)

    for step in range(NUM_STEPS):
        opt.zero_grad()
        y_pred = layer(x_in)
        loss = ((y_pred - y_target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses


# ─────────────────────────────────────────────────────────────────────
#  Sparse training — our SparseCore path
# ─────────────────────────────────────────────────────────────────────

def train_sparse(X: torch.Tensor, Y_target: torch.Tensor) -> tuple[list[float], PaddedCSR, torch.Tensor]:
    """
    Train a sparse W (90% zeros) to fit the targets. Uses our custom
    autograd Function — loss.backward() flows through spmm_grad_w and
    spmm_simd(Wᵀ) entirely through SparseCore.

    Returns (losses, final_W_csr, final_W_values_tensor).
    """
    torch.manual_seed(0)
    # Initialize a sparse W: same init scale as nn.Linear for fair compare.
    # nn.Linear default: uniform [-1/√K, 1/√K]. We use that.
    bound = 1.0 / (K ** 0.5)
    W_dense = (torch.rand(M, K) * 2 - 1) * bound
    # Impose the sparsity pattern (which 10% of cells are live). This
    # pattern stays FIXED throughout training — milestone 4c will add
    # dynamic topology mutation; for now, the mask is chosen once.
    mask = torch.rand(M, K) >= SPARSITY
    W_dense = W_dense * mask.float()

    # Build PaddedCSR from the initialized dense weights.
    W_csr = PaddedCSR.from_dense(W_dense)

    # Expose W.values as a torch tensor for autograd tracking.
    # W.values is writable (4a design change) so we can do in-place SGD.
    W_values_t = torch.from_numpy(np.asarray(W_csr.values)).requires_grad_(True)

    losses = []
    for step in range(NUM_STEPS):
        # Clear grad
        if W_values_t.grad is not None:
            W_values_t.grad.zero_()

        # Forward. _SpMMFunction registers W_values_t as the tracked
        # tensor, so its .grad will receive dW_values after backward().
        Y_pred = _SpMMFunction.apply(W_values_t, W_csr, X, "simd")
        loss = ((Y_pred - Y_target) ** 2).mean()

        # Backward — this is what we shipped in 4a-v
        loss.backward()

        # SGD step on the tracked tensor. Update the underlying W.values
        # through the writable view so next forward sees new weights.
        with torch.no_grad():
            W_values_t.data -= LR * W_values_t.grad
            # W.values is a writable zero-copy view of C++ memory; writing
            # into it updates what our C++ kernel will read next forward.
            W_csr.values[:] = W_values_t.data.numpy()

        losses.append(loss.item())

    return losses, W_csr, W_values_t


# ─────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print("═" * 70)
    print("SparseCore demo 4 — End-to-end sparse training with autograd")
    print(
        f"W shape: ({M}, {K})  |  "
        f"sparsity: {SPARSITY * 100:.0f}%  |  "
        f"batch: {BATCH}  |  steps: {NUM_STEPS}"
    )
    print("═" * 70)


def print_loss_curve(dense_losses: list[float], sparse_losses: list[float]):
    # Sample a few checkpoints to print
    checkpoints = [0, 10, 50, 100, 150, NUM_STEPS - 1]
    print(f"{'STEP':>6s}  {'dense loss':>12s}  {'sparse loss':>12s}")
    print("─" * 40)
    for step in checkpoints:
        print(
            f"{step:>6d}  {dense_losses[step]:>12.5f}  {sparse_losses[step]:>12.5f}"
        )
    print()


def print_summary(
    dense_losses: list[float],
    sparse_losses: list[float],
    W_csr: PaddedCSR,
    W_values_t: torch.Tensor,
):
    dense_final = dense_losses[-1]
    sparse_final = sparse_losses[-1]
    dense_start = dense_losses[0]
    sparse_start = sparse_losses[0]

    print(f"Final loss:")
    print(f"  dense:  {dense_final:.5f}  (reduction: {dense_start:.2f} → {dense_final:.5f})")
    print(f"  sparse: {sparse_final:.5f}  (reduction: {sparse_start:.2f} → {sparse_final:.5f})")
    print()

    # Padding-slots-still-zero verification: the whole DST thesis in one check
    values = np.asarray(W_csr.values)
    col_indices = np.asarray(W_csr.col_indices)
    padding_mask = col_indices == -1
    padding_values = values[padding_mask]
    max_padding_abs = float(np.abs(padding_values).max()) if len(padding_values) else 0.0
    print(
        f"Padding slots still zero after training: "
        f"max |padding_value| = {max_padding_abs:.2e}  "
        f"{'✓' if max_padding_abs == 0.0 else '× LEAKED!'}"
    )
    # This line is the thesis in an assert: the optimizer never touched
    # padding slots because their gradient was always 0.
    print()

    # What fraction of possible weights are we actually using?
    total_cells = M * K
    live_cells = W_csr.nnz
    print(
        f"Parameter count: {live_cells:,} live / {total_cells:,} possible "
        f"({100 * live_cells / total_cells:.1f}% active)"
    )

    print()
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print_header()
    W_true, X, Y_target = make_data()

    t0 = time.perf_counter()
    dense_losses = train_dense(X, Y_target)
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    sparse_losses, W_csr, W_values_t = train_sparse(X, Y_target)
    t_sparse = time.perf_counter() - t0

    print_loss_curve(dense_losses, sparse_losses)
    print_summary(dense_losses, sparse_losses, W_csr, W_values_t)

    print(f"Training wall clock:")
    print(f"  dense  ({NUM_STEPS} steps): {t_dense * 1000:.1f} ms")
    print(f"  sparse ({NUM_STEPS} steps): {t_sparse * 1000:.1f} ms")
    print()
    print("What to try next:")
    print("  - Change SPARSITY at the top of this file to 0.99 — with 10x")
    print("    fewer params, expect slower convergence and a higher loss floor")
    print("  - Raise NUM_STEPS to 1000 — sparse catches up further")
    print("  - Swap SGD for Adam (nn.Linear supports it natively; our path")
    print("    needs manual state tracking — milestone 4b adds SparseLinear)")
    print()


if __name__ == "__main__":
    main()
