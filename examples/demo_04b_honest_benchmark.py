"""
Demo 4b — Honest training-step benchmark at multiple sparsities.

What this demo proves (and un-proves)
─────────────────────────────────────
The 40× "sparse vs dense" number in demo_04_autograd.py compared our
raw kernel path against nn.Linear's full framework call. That was not a
fair matmul benchmark — most of the gap was Python overhead around
nn.Linear / Optimizer, not compute.

This demo strips away the framework differences by timing ONLY the
core train-step math for both paths:
  - Sparse path: spmm forward + spmm_grad_w backward + in-place SGD
  - Dense path:  torch.matmul forward + transpose backward + in-place SGD

Both use raw tensors, no nn.Module wrapping, no optimizer object.

How to run
──────────
    python examples/demo_04b_honest_benchmark.py

What to look at
───────────────
  1. Time per step at each sparsity level. Sparse gets faster as
     sparsity rises; dense stays roughly constant.
  2. Crossover point: the sparsity above which our sparse path beats
     the dense matmul. On Apple Silicon with AMX, expect somewhere
     in the 90-99% range (matching demo_03).
  3. Loss convergence after N steps at each sparsity. The sparse
     model has fewer parameters so it converges to a higher loss
     floor, proportional to sparsity.

Shape chosen to be transformer-adjacent (not as tiny as demo_04).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch

import sparselab
from sparselab import PaddedCSR
from sparselab.ops import _SpMMFunction


warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────

# Size matters for this comparison. Too small → Python overhead dominates.
# We picked a shape where torch.matmul is clearly working: (M, K) @ (K, N)
# = 512 × 256 × 64, doing ~8M FLOPs per forward pass.
M, K, N = 512, 256, 64

SPARSITIES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

NUM_STEPS = 100
LR = 0.01

# Number of inner-loop trials for each step timing — taking the min
# across trials filters out one-off scheduler jitter on a busy machine.
TIMING_TRIALS = 5


# ─────────────────────────────────────────────────────────────────────
#  Data generation
# ─────────────────────────────────────────────────────────────────────

def make_data(seed: int = 7):
    """Generate a fixed target pattern."""
    torch.manual_seed(seed)
    W_true = torch.randn(M, K, dtype=torch.float32)
    X = torch.randn(K, N, dtype=torch.float32)
    Y_target = W_true @ X  # noiseless target
    return W_true, X, Y_target


# ─────────────────────────────────────────────────────────────────────
#  Dense training step (stripped of nn.Module / optimizer overhead)
# ─────────────────────────────────────────────────────────────────────

def build_dense_model(sparsity: float):
    """Initialize a dense weight with the given fraction of zero cells.

    Even at high sparsity, the dense path stores and computes on the
    full (M, K) tensor — zeros included. This is the "dense-simulated
    sparse training" baseline we are trying to avoid.
    """
    torch.manual_seed(0)
    W_init = torch.randn(M, K, dtype=torch.float32) * (1.0 / (K ** 0.5))
    mask = (torch.rand(M, K) >= sparsity).float()
    W_dense = (W_init * mask).clone().requires_grad_(True)
    return W_dense


def dense_train_step(W_dense: torch.Tensor, X: torch.Tensor, Y_target: torch.Tensor, lr: float):
    """One training step with a dense W, no framework ceremony."""
    Y_pred = W_dense @ X
    loss = ((Y_pred - Y_target) ** 2).mean()
    if W_dense.grad is not None:
        W_dense.grad.zero_()
    loss.backward()
    with torch.no_grad():
        W_dense.data -= lr * W_dense.grad
    return loss.item()


# ─────────────────────────────────────────────────────────────────────
#  Sparse training step (our path)
# ─────────────────────────────────────────────────────────────────────

def build_sparse_model(sparsity: float):
    """Initialize a sparse weight via PaddedCSR at the given sparsity."""
    torch.manual_seed(0)
    W_init = torch.randn(M, K, dtype=torch.float32) * (1.0 / (K ** 0.5))
    mask = (torch.rand(M, K) >= sparsity).float()
    W_sparse_dense = W_init * mask
    W_csr = PaddedCSR.from_dense(W_sparse_dense)
    W_values_t = torch.from_numpy(np.asarray(W_csr.values)).requires_grad_(True)
    return W_csr, W_values_t


def sparse_train_step(W_csr: PaddedCSR, W_values_t: torch.Tensor, X: torch.Tensor, Y_target: torch.Tensor, lr: float):
    """One training step with PaddedCSR + our autograd kernels."""
    Y_pred = _SpMMFunction.apply(W_values_t, W_csr, X, "simd")
    loss = ((Y_pred - Y_target) ** 2).mean()
    if W_values_t.grad is not None:
        W_values_t.grad.zero_()
    loss.backward()
    with torch.no_grad():
        W_values_t.data -= lr * W_values_t.grad
        W_csr.values[:] = W_values_t.data.numpy()
    return loss.item()


# ─────────────────────────────────────────────────────────────────────
#  Benchmark one sparsity level
# ─────────────────────────────────────────────────────────────────────

def benchmark_sparsity(sparsity: float, X, Y_target):
    """
    Build both models at this sparsity, time one step of each (taking
    min of TIMING_TRIALS), then run NUM_STEPS to measure loss convergence.
    """
    # ─── Build models ─────────────────────────────────────────────────
    W_dense = build_dense_model(sparsity)
    W_csr, W_values_t = build_sparse_model(sparsity)

    # Actual sparsity in our sparse model (stochastic — close to target)
    actual_sparsity = 1.0 - W_csr.nnz / (M * K)

    # ─── Warmup ──────────────────────────────────────────────────────
    dense_train_step(W_dense, X, Y_target, LR)
    sparse_train_step(W_csr, W_values_t, X, Y_target, LR)

    # ─── Time one step, min of TIMING_TRIALS ─────────────────────────
    dense_times = []
    for _ in range(TIMING_TRIALS):
        t0 = time.perf_counter()
        dense_train_step(W_dense, X, Y_target, LR)
        dense_times.append((time.perf_counter() - t0) * 1000)
    dense_ms = min(dense_times)

    sparse_times = []
    for _ in range(TIMING_TRIALS):
        t0 = time.perf_counter()
        sparse_train_step(W_csr, W_values_t, X, Y_target, LR)
        sparse_times.append((time.perf_counter() - t0) * 1000)
    sparse_ms = min(sparse_times)

    # ─── Full convergence run ────────────────────────────────────────
    # Reset models to initial state so trial timings didn't affect loss
    W_dense = build_dense_model(sparsity)
    W_csr, W_values_t = build_sparse_model(sparsity)

    dense_initial_loss = dense_train_step(W_dense, X, Y_target, LR)
    sparse_initial_loss = sparse_train_step(W_csr, W_values_t, X, Y_target, LR)

    for _ in range(NUM_STEPS - 1):
        dense_final = dense_train_step(W_dense, X, Y_target, LR)
        sparse_final = sparse_train_step(W_csr, W_values_t, X, Y_target, LR)

    return {
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "nnz": W_csr.nnz,
        "dense_ms_per_step": dense_ms,
        "sparse_ms_per_step": sparse_ms,
        "speedup": dense_ms / sparse_ms,
        "dense_initial": dense_initial_loss,
        "dense_final": dense_final,
        "sparse_initial": sparse_initial_loss,
        "sparse_final": sparse_final,
    }


# ─────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print("═" * 96)
    print(
        f"SparseLab demo 4b — Honest training-step benchmark "
        f"(matmul vs matmul, no nn.Module overhead)"
    )
    print(
        f"W: ({M}, {K})   X: ({K}, {N})   "
        f"{NUM_STEPS} steps per sparsity   "
        f"{TIMING_TRIALS} timing trials"
    )
    print("═" * 96)


def print_row_header():
    print(
        f"{'SPARS':>7s}  {'NNZ':>9s}  "
        f"{'dense ms':>10s}  {'sparse ms':>11s}  {'SPEEDUP':>9s}  "
        f"{'dense loss':>21s}  {'sparse loss':>21s}"
    )
    print(f"{'':>7s}  {'':>9s}  "
          f"{'(1 step)':>10s}  {'(1 step)':>11s}  "
          f"{'':>9s}  {'start → end':>21s}  {'start → end':>21s}")
    print("─" * 96)


def print_row(r):
    icon = "✓" if r["speedup"] >= 1.0 else "×"
    print(
        f"{r['actual_sparsity'] * 100:>6.1f}%  "
        f"{r['nnz']:>9,}  "
        f"{r['dense_ms_per_step']:>9.2f}   "
        f"{r['sparse_ms_per_step']:>10.2f}   "
        f"{r['speedup']:>7.2f}x {icon}  "
        f"{r['dense_initial']:>8.2f} → {r['dense_final']:<9.4f}  "
        f"{r['sparse_initial']:>8.2f} → {r['sparse_final']:<9.4f}"
    )


def print_summary(results):
    print("─" * 96)

    crossover = None
    for r in results:
        if r["speedup"] >= 1.0:
            crossover = r["actual_sparsity"]
            break

    if crossover is not None:
        print(
            f"  Crossover: sparse training beats dense starting at "
            f"{crossover * 100:.0f}% sparsity (incl. backward + weight update)"
        )
    else:
        print(
            f"  No crossover in this sweep. Dense wins at every sparsity "
            f"level for this shape on Apple Silicon AMX."
        )

    # Best speedup
    best = max(results, key=lambda r: r["speedup"])
    print(
        f"  Peak speedup: {best['speedup']:.2f}x at "
        f"{best['actual_sparsity'] * 100:.0f}% sparsity"
    )

    # Loss floor story
    # How much does sparse lose vs dense at each sparsity?
    print()
    print("  Convergence — sparse loss floor vs dense (lower is more expressive):")
    for r in results:
        gap = r["sparse_final"] / max(r["dense_final"], 1e-6)
        print(
            f"    {r['actual_sparsity'] * 100:>5.1f}%: "
            f"dense={r['dense_final']:.4f}, "
            f"sparse={r['sparse_final']:.4f}  "
            f"(sparse is {gap:.1f}x dense's loss)"
        )
    print("═" * 96)
    print()

    print("What to try next:")
    print("  - Raise NUM_STEPS to 500 — the higher-sparsity runs haven't")
    print("    fully converged yet at 100 steps")
    print("  - Change M, K, N to a real transformer layer size")
    print("    (e.g. M=2048, K=512, N=512) — speedup curve shifts upward")
    print("  - Force kernel='scalar' on the sparse path to isolate what")
    print("    our NEON kernel actually adds vs auto-vectorized scalar")
    print()


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print_header()
    _, X, Y_target = make_data()

    print_row_header()
    results = []
    for sparsity in SPARSITIES:
        r = benchmark_sparsity(sparsity, X, Y_target)
        results.append(r)
        print_row(r)

    print_summary(results)


if __name__ == "__main__":
    main()
