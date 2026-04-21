"""
Demo 3 — Sparse × Dense matmul benchmark (Milestone 3d).

What this demo proves
─────────────────────
We have two moving parts working together:
  (a) PaddedCSR — our custom sparse storage format (Milestone 3b)
  (b) NEON SpMM — a SIMD inner loop that skips zero entries entirely (3c/3d)

The question we care about as a project:
  "At what sparsity does our actually-sparse SpMM overtake Apple's
   hyper-optimized dense matmul (torch.matmul, backed by AMX)?"

The answer (from this demo) is what makes "sparse training on a MacBook"
a viable story instead of a wishful one.

How to run
──────────
    python examples/demo_03_spmm.py

What to look at
───────────────
  1. The ORACLE DIFF column — must always be < ~1e-4. That's our
     correctness evidence: our kernel agrees with torch.matmul.
  2. The SPEEDUP column — the key number. Anywhere > 1.0x means
     we're faster than Apple's dense AMX at that sparsity.
  3. The CROSSOVER message at the end — the lowest sparsity at which
     we beat torch.matmul. Intel benchmarks put this around 75% for
     MKL; Apple Silicon (AMX) is unknown — this demo discovers it.

What "good" looks like
──────────────────────
  - Zero correctness failures (every row passes the oracle check)
  - Crossover somewhere in the 80-95% range means our story holds:
    sparse-from-scratch DST research lives at 80-95% sparsity.
  - At very low sparsity (0-50%) we will LOSE, possibly badly (10x+).
    That is expected and fine — sparse is not meant for low sparsity.
  - At 99% we should win decisively (10x+ ideally).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch

import sparsecore
from sparsecore import PaddedCSR


# Quiet PyTorch's beta-state sparse CSR warning — we know.
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────
#  Config — tune these to change what we're benchmarking.
# ─────────────────────────────────────────────────────────────────────

# Shape chosen to be transformer-adjacent:
#   M = 2048  — "out_features" (FFN inner dim, ~GPT-2-small scale)
#   K = 512   — "in_features"  (embedding dim)
#   N = 512   — "batch * seq_len" per forward pass
# The matmul Y = W @ X computes 2048×512 output for 2048×512 weights,
# with 512×512 dense activations. This is what one FFN layer of a small
# transformer actually does.
M, K, N = 2048, 512, 512

# Sparsity levels to sweep. Higher numbers = fewer nonzeros.
SPARSITIES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]

# Wall-clock timing: average over this many runs after a warmup pass.
NUM_RUNS = 10


# ─────────────────────────────────────────────────────────────────────
#  Timing helper.
# ─────────────────────────────────────────────────────────────────────

def time_it(fn, num_runs: int = NUM_RUNS) -> float:
    """
    Run fn() once to warm caches, then time num_runs invocations.
    Returns the mean wall-clock time in milliseconds.

    We use time.perf_counter, which is the highest-resolution monotonic
    clock available in Python. For ms-scale work it's far more reliable
    than time.time().
    """
    fn()  # warmup — first call often includes allocator / cache effects
    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000.0  # → milliseconds


# ─────────────────────────────────────────────────────────────────────
#  Benchmark one (M, K, N, sparsity) configuration.
# ─────────────────────────────────────────────────────────────────────

def benchmark(sparsity: float) -> dict:
    """
    At a given sparsity, build W and X, time our SpMM vs torch.matmul,
    and verify they agree with each other.
    """
    # Use a fixed seed per sparsity so runs are reproducible.
    torch.manual_seed(int(sparsity * 10_000))

    # Build the dense weight, then zero out (sparsity) fraction of it.
    W_dense = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) >= sparsity
    W_dense = W_dense * mask.float()

    # Actual measured sparsity (stochastic, close to target).
    actual_sparsity = 1.0 - (W_dense != 0).float().mean().item()

    # Convert to our sparse format and a dense input activation.
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(K, N, dtype=torch.float32)

    # ─── Correctness: our result vs torch.matmul ─────────────────────
    Y_ours = sparsecore.spmm(W_csr, X)          # NEON path (default)
    Y_oracle = W_dense @ X
    max_diff = (Y_ours - Y_oracle).abs().max().item()

    # ─── Timings ─────────────────────────────────────────────────────
    ms_ours = time_it(lambda: sparsecore.spmm(W_csr, X))
    ms_torch = time_it(lambda: W_dense @ X)

    return {
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "nnz": int((W_dense != 0).sum().item()),
        "ms_ours": ms_ours,
        "ms_torch": ms_torch,
        "speedup": ms_torch / ms_ours,  # >1.0 = we're faster
        "max_diff": max_diff,
    }


# ─────────────────────────────────────────────────────────────────────
#  Pretty-print a results table.
# ─────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print("═" * 78)
    print(f"SparseCore demo 3 — SpMM benchmark on Apple Silicon (NEON)")
    print(f"Shape: ({M}, {K}) @ ({K}, {N}) → ({M}, {N})   "
          f"Runs per cell: {NUM_RUNS}")
    print("═" * 78)
    print(f"{'SPARSITY':>10s}  {'NNZ':>10s}  "
          f"{'torch (ms)':>12s}  {'ours (ms)':>11s}  "
          f"{'SPEEDUP':>9s}  {'ORACLE DIFF':>12s}")
    print("─" * 78)


def print_row(r):
    # Emoji guide: ✓ = we win, × = we lose, ! = correctness problem.
    speed_icon = "✓" if r["speedup"] >= 1.0 else "×"
    corr_icon = "!" if r["max_diff"] > 1e-3 else " "
    print(
        f"{r['actual_sparsity'] * 100:9.1f}%  "
        f"{r['nnz']:>10,}  "
        f"{r['ms_torch']:>11.2f}   "
        f"{r['ms_ours']:>10.2f}   "
        f"{r['speedup']:>7.2f}x {speed_icon}  "
        f"{r['max_diff']:>10.2e} {corr_icon}"
    )


def print_summary(results):
    print("─" * 78)

    # Find the crossover — lowest sparsity where we beat torch.
    crossover = None
    for r in results:
        if r["speedup"] >= 1.0:
            crossover = r["actual_sparsity"]
            break

    if crossover is not None:
        print(f"  ✓ Crossover: we beat torch.matmul starting at "
              f"{crossover * 100:.0f}% sparsity.")
    else:
        print("  × No crossover in this sweep. Try higher sparsity or "
              "a larger M/K.")

    # Max speedup in the sweep.
    max_r = max(results, key=lambda r: r["speedup"])
    print(f"  Peak speedup: {max_r['speedup']:.2f}x at "
          f"{max_r['actual_sparsity'] * 100:.0f}% sparsity "
          f"(nnz={max_r['nnz']:,}).")

    # Max correctness diff across the sweep.
    worst_diff = max(r["max_diff"] for r in results)
    print(f"  Max oracle diff across all sparsities: {worst_diff:.2e} "
          f"(must be << 1e-3 for correctness).")

    print("═" * 78)
    print()
    print("What to try next:")
    print("  - Bump N up to 2048 (longer seq_len) — NEON utilization rises")
    print("  - Try M=4096 (wider FFN) — more rows of work per call")
    print("  - Force the scalar kernel: sparsecore.spmm(W, X, kernel='scalar')")
    print("    and re-run, to see how much NEON actually helps us.")
    print()


# ─────────────────────────────────────────────────────────────────────
#  Main.
# ─────────────────────────────────────────────────────────────────────

def main():
    print_header()
    results = []
    for s in SPARSITIES:
        r = benchmark(s)
        results.append(r)
        print_row(r)
    print_summary(results)


if __name__ == "__main__":
    main()
