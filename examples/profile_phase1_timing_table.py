"""
Phase 1 profiler — where does time go in a SparseLinear fwd+bwd?

Output: a table with one row per problem size, columns for each
component (kernel forward, kernel backward dW, W.transpose(), kernel
backward dX) plus the composed fwd+bwd time and the derived "python
overhead" (composed minus sum of kernels).

The point: identify which single component, at each problem size, is
the largest share of a training step. That tells us which optimization
to attempt first.

How to run
──────────
    python examples/profile_phase1_timing_table.py

No external deps beyond torch and the package itself.

Output shape:
    config  | fwd  | dW   | WT   | dX   | kernel_sum | composed | overhead | %kernel
    mnist   |  X   |  X   |  X   |  X   |     X      |    X     |    X     |   X%
    spike   | ...
    ffn-xl  | ...
"""

from __future__ import annotations

import time

import numpy as np
import torch

import sparsecore
from sparsecore import _core


CONFIGS = [
    # (name, K=in_features, M=out_features, N=batch_or_batch_x_seq, sparsity)
    ("tiny       (64x64 b=32)",      64,    64,    32, 0.9),
    ("mnist-ffn  (128x512 b=128)",  128,   512,   128, 0.9),
    ("spike      (128x512 b=1024)", 128,   512,  1024, 0.9),
    ("ffn-mid    (512x2048 b=1024)",512,  2048,  1024, 0.9),
]

N_TRIALS = 30          # trials per component, median taken
N_WARMUP = 5


def median_ms(fn) -> float:
    """Run fn() N_WARMUP times to warm caches, then N_TRIALS times;
    return median wall-clock in milliseconds."""
    for _ in range(N_WARMUP):
        fn()
    samples = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def measure_config(name: str, K: int, M: int, N: int, sparsity: float) -> dict:
    """One row of the table. Returns dict of component timings."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Build SparseLinear-shaped pieces:
    #   W: PaddedCSR (M rows, K cols) — matches nn.Linear's weight shape
    #       except we use our CSR
    W = sparsecore.PaddedCSR.random(M, K, sparsity=sparsity, seed=0)

    # Forward input is shaped (K, N) because our kernel takes (K, N).
    # At the SparseLinear level, users see (*, K) and we transpose inside.
    X_col = np.random.randn(K, N).astype(np.float32)
    dY = np.random.randn(M, N).astype(np.float32)

    # ─── Isolated kernel timings ──────────────────────────────────
    t_fwd = median_ms(lambda: _core.spmm_simd(W, X_col))
    t_dW  = median_ms(lambda: _core.spmm_grad_w(W, dY, X_col))
    t_WT  = median_ms(lambda: W.transpose())
    WT = W.transpose()
    t_dX  = median_ms(lambda: _core.spmm_simd(WT, dY))

    # ─── Composed fwd+bwd through SparseLinear (the full pipeline) ─
    # This is what a user actually experiences per-layer per-step.
    layer = sparsecore.SparseLinear(K, M, sparsity=sparsity, bias=False)
    # Force the layer's internal CSR to match our test W exactly so
    # comparisons are apples-to-apples.
    # (Actually easier to just rebuild with fresh seed; not crucial.)
    x_user = torch.randn(N, K, requires_grad=True)

    def one_fwd_bwd():
        if layer._values.grad is not None:
            layer._values.grad = None
        if x_user.grad is not None:
            x_user.grad = None
        y = layer(x_user)
        y.sum().backward()

    t_composed = median_ms(one_fwd_bwd)

    kernel_sum = t_fwd + t_dW + t_WT + t_dX
    overhead = t_composed - kernel_sum
    pct_kernel = kernel_sum / t_composed * 100 if t_composed > 0 else 0.0

    return {
        "name":       name,
        "fwd":        t_fwd,
        "dW":         t_dW,
        "WT":         t_WT,
        "dX":         t_dX,
        "kernel_sum": kernel_sum,
        "composed":   t_composed,
        "overhead":   overhead,
        "pct_kernel": pct_kernel,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print("=" * 104)
    print("Phase 1: where does time go per SparseLinear fwd+bwd (ms/call, median of 30)")
    print("=" * 104)
    header = (f"{'config':<30s}  "
              f"{'fwd':>6s}  {'dW':>6s}  {'WT':>6s}  {'dX':>6s}  "
              f"{'kernels':>8s}  {'composed':>9s}  {'python':>8s}  "
              f"{'%krn':>5s}")
    print(header)
    print("-" * 104)
    for r in rows:
        print(
            f"{r['name']:<30s}  "
            f"{r['fwd']:>6.3f}  {r['dW']:>6.3f}  {r['WT']:>6.3f}  {r['dX']:>6.3f}  "
            f"{r['kernel_sum']:>8.3f}  {r['composed']:>9.3f}  {r['overhead']:>8.3f}  "
            f"{r['pct_kernel']:>4.0f}%"
        )
    print("=" * 104)
    print()
    print("Legend:")
    print("  fwd      = spmm_simd(W, X) kernel")
    print("  dW       = spmm_grad_w(W, dY, X) kernel (backward w.r.t. weights)")
    print("  WT       = W.transpose() CSR materialization")
    print("  dX       = spmm_simd(WT, dY) kernel (backward w.r.t. input)")
    print("  kernels  = sum of the four components above (all C++ work)")
    print("  composed = full fwd+bwd through SparseLinear, with autograd")
    print("  python   = composed - kernels = Python + pybind11 overhead")
    print("  %krn     = fraction of composed time that was real kernel work")
    print()


def print_findings(rows: list[dict]) -> None:
    """Highlight the biggest component at each scale — tells us what to optimize."""
    print("─" * 104)
    print("Biggest single component at each scale:")
    for r in rows:
        components = [
            ("fwd",    r["fwd"]),
            ("dW",     r["dW"]),
            ("WT",     r["WT"]),
            ("dX",     r["dX"]),
            ("python", r["overhead"]),
        ]
        components.sort(key=lambda t: -t[1])
        top = components[0]
        pct = top[1] / r["composed"] * 100
        print(f"  {r['name']:<30s}  biggest: {top[0]:<8s} "
              f"({top[1]:.3f} ms, {pct:.0f}% of step)")
    print()


def main():
    print(f"Warmup={N_WARMUP}, trials={N_TRIALS} per measurement. "
          f"Running...")
    rows = []
    for name, K, M, N, sparsity in CONFIGS:
        r = measure_config(name, K, M, N, sparsity)
        rows.append(r)
    print_table(rows)
    print_findings(rows)


if __name__ == "__main__":
    main()
