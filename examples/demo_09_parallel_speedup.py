"""
Demo 9 — OpenMP parallelization speedup (milestone 4c).

Before this milestone, our SpMM forward and backward kernels were
single-threaded. On an M3 Pro that left 10 of 12 CPU cores idle while
the sparse training loop saturated one core. This demo measures the
speedup from parallelizing the outer (row) loop with OpenMP.

How to run
──────────
    python examples/demo_09_parallel_speedup.py

You can pin thread count via the standard OpenMP env var:
    OMP_NUM_THREADS=1 python examples/demo_09_parallel_speedup.py    # sequential
    OMP_NUM_THREADS=6 python examples/demo_09_parallel_speedup.py    # P-cores only
    OMP_NUM_THREADS=12 python examples/demo_09_parallel_speedup.py   # all cores

What to look at
───────────────
  1. `t_seq` (OMP_NUM_THREADS=1 effectively) vs `t_par` (default).
     Our target: 2-4× speedup on M3 Pro at 90% sparsity on a realistic
     MLP-shaped matrix (512 rows, 784 cols, batch 128).
  2. Speedup scales with matrix size — tiny matrices won't gain because
     OpenMP fork/join overhead dominates. We include one small-matrix
     row to illustrate that.
  3. Backward kernel (spmm_grad_w) should see similar speedup — it has
     the same per-row independence.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

import sparselab
from sparselab import PaddedCSR


REPEATS = 10  # how many times to run each kernel for timing stability


def bench(fn, *args, repeats: int = REPEATS) -> float:
    """Return median wall-clock (seconds) across `repeats` calls."""
    # One warmup to get things into L1/L2 cache
    fn(*args)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_forward(W: PaddedCSR, X_np: np.ndarray) -> float:
    X_t = torch.from_numpy(X_np)
    return bench(lambda: sparselab.spmm(W, X_t))


def bench_backward(W: PaddedCSR, dY: np.ndarray, X: np.ndarray) -> float:
    from sparselab import _core
    return bench(lambda: _core.spmm_grad_w(W, dY, X))


def run_config(name: str, M: int, K: int, N: int, sparsity: float):
    """One row of the result table: a single problem size × sparsity."""
    W = PaddedCSR.random(M, K, sparsity=sparsity, seed=0)
    X = np.random.default_rng(1).standard_normal((K, N), dtype=np.float32)
    dY = np.random.default_rng(2).standard_normal((M, N), dtype=np.float32)

    t_fwd = bench_forward(W, X)
    t_bwd = bench_backward(W, dY, X)

    # Total FLOPs: forward = 2 * nnz * N, backward = 2 * nnz * N.
    flops_fwd = 2 * W.nnz * N
    flops_bwd = 2 * W.nnz * N
    gflops_fwd = flops_fwd / t_fwd / 1e9
    gflops_bwd = flops_bwd / t_bwd / 1e9

    return {
        "name": name, "M": M, "K": K, "N": N, "nnz": W.nnz,
        "t_fwd_ms": t_fwd * 1e3, "t_bwd_ms": t_bwd * 1e3,
        "gflops_fwd": gflops_fwd, "gflops_bwd": gflops_bwd,
    }


def main():
    nth = os.environ.get("OMP_NUM_THREADS", "default")
    print(f"\nSparseLab demo 9 — OpenMP parallel speedup")
    print(f"OMP_NUM_THREADS = {nth}   (set to 1 to disable parallelism)")
    print("=" * 98)

    print(
        f"\n{'config':<26s}  {'M':>4s}  {'K':>5s}  {'N':>4s}  {'nnz':>7s}  "
        f"{'fwd ms':>8s}  {'fwd GF/s':>9s}  {'bwd ms':>8s}  {'bwd GF/s':>9s}"
    )
    print("-" * 98)

    configs = [
        # (label, M rows, K cols, N batch, sparsity)
        ("tiny (OpenMP overhead)",   16,   32,  16, 0.90),
        ("MLP-hidden-ish",           512, 784, 128, 0.90),
        ("MLP-hidden @ 99%",         512, 784, 128, 0.99),
        ("FFN-scale 2048x2048",     2048, 2048, 128, 0.90),
        ("FFN-scale 2048x2048 99%", 2048, 2048, 128, 0.99),
    ]

    rows = []
    for label, M, K, N, s in configs:
        r = run_config(label, M, K, N, s)
        rows.append(r)
        print(
            f"{label:<26s}  {r['M']:>4d}  {r['K']:>5d}  {r['N']:>4d}  "
            f"{r['nnz']:>7d}  {r['t_fwd_ms']:>8.2f}  {r['gflops_fwd']:>9.2f}  "
            f"{r['t_bwd_ms']:>8.2f}  {r['gflops_bwd']:>9.2f}"
        )

    print("=" * 98)
    print(
        "\nHow to read this:"
        "\n  - GF/s = billion FMA operations per second actually delivered."
        "\n  - Run once with OMP_NUM_THREADS=1 and once with the default to"
        "\n    see the parallel speedup. On an M3 Pro with 6 P + 6 E cores,"
        "\n    the MLP-hidden and FFN-scale rows should show a 2-4x speedup."
        "\n  - The tiny row is expected to be roughly the same (or slower)"
        "\n    with parallelism — fork/join overhead dominates at small M."
        "\n    That's why we gate parallelism behind a row-count threshold."
    )
    print()


if __name__ == "__main__":
    main()
