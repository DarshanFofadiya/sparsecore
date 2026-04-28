"""
profile_dw_baseline.py — measure dW kernel throughput across platforms.

What this script does
─────────────────────
For the FFN shapes used in demos 15 and 16 (the 10M and 40M-scale
mini-GPTs), measure how fast our scalar + SIMD ``spmm_grad_w`` kernels
run vs a dense-BLAS oracle of the same math.

Originally written as the Gate-1 measurement before we wrote the NEON
dW kernel (issue #1, now shipped in v0.2.1). Now it serves two jobs:

  1. As a reproducible perf benchmark. Run it before and after any
     kernel change to confirm regressions don't sneak in.
  2. As the Gate-1 measurement for FUTURE SIMD work on other
     architectures. If the scalar baseline is at 10-20 GF/s and the
     ceiling is 100+ GF/s, a hand-written kernel is worth writing.
     If the scalar baseline is already near the ceiling, the compiler
     already did the work and we should invest elsewhere.

How to run
──────────
    python examples/profile_dw_baseline.py

Takes ~30 seconds. Uses 30-run median timing with 3 warmup calls per
shape. No commits, no build, no test side-effects — read-only
benchmarking.

What the output tells you
─────────────────────────
  scalar ms: the scalar kernel's wallclock on this shape.
  simd ms:   the _simd kernel's wallclock. On ARM64 (Apple Silicon,
             Linux aarch64) this is the hand-written NEON kernel.
             On x86_64 it falls back to scalar — AVX kernel is issue
             #2 and not yet built.
  dense ms:  what torch.matmul does for the "everything were dense"
             math — a lower bound for any dense path.
  si/sc:     simd / scalar. ~0.15 on ARM64 means NEON gave us a 6x
             speedup. ~1.0 on x86_64 confirms the scalar fallback.
  s.GF/s:    scalar throughput on the live slots.

Measured baseline (Apple M3 Pro, torch threads=6, 2026-04-27)
─────────────────────────────────────────────────────────────
Scalar (pre-NEON-dW):
  demo15 FFN up   (384 × 1536, N=2048, s=0.90):  16.6 ms,  14.4 GF/s
  demo15 FFN down (1536 × 384, N=2048, s=0.90):  16.2 ms,  14.8 GF/s
  demo16 FFN up   (640 × 2560, N=1024, s=0.90):  23.9 ms,  14.0 GF/s
  demo16 FFN down (2560 × 640, N=1024, s=0.90):  22.6 ms,  14.8 GF/s
NEON (post issue #1):
  demo15 FFN up   (384 × 1536, N=2048, s=0.90):   2.4 ms  (6.53×)
  demo15 FFN down (1536 × 384, N=2048, s=0.90):   2.3 ms  (6.67×)
  demo16 FFN up   (640 × 2560, N=1024, s=0.90):   3.3 ms  (6.37×)
  demo16 FFN down (2560 × 640, N=1024, s=0.90):   3.4 ms  (6.30×)

14 GF/s on scalar was consistent with Clang emitting sequential
scalar FMAs (one per cycle latency ≈ 14 GF/s on M-series). The 6.5×
speedup we achieved from the 8-wide dual-accumulator NEON kernel
closed most of the gap to the ~90-120 GF/s target (see milestone_12).

Reproducibility
───────────────
The torch.manual_seed() calls fix the random W, dY, X for each shape,
so wallclock numbers are directly comparable across runs on the same
machine. The only source of drift is thermal state — close other
heavy processes before running for best results.
"""

from __future__ import annotations

import statistics as stats
import time

import numpy as np
import torch

import sparselab  # noqa: F401  — needed to register _core
from sparselab import _core, PaddedCSR


# ─── Shapes to measure ─────────────────────────────────────────────
# (M, K, N, sparsity, label)
# First 4 rows: FFN up/down projections from demos 15 (10M) and 16 (40M).
# Last row: tiny shape to verify no regression on small layers.
# ─────────────────────────────────────────────────────────────────────

SHAPES = [
    ( 384, 1536, 2048, 0.90, "demo15_ffn_up     (384  x 1536 x N=2048, s=0.90)"),
    (1536,  384, 2048, 0.90, "demo15_ffn_down   (1536 x 384  x N=2048, s=0.90)"),
    ( 640, 2560, 1024, 0.90, "demo16_ffn_up     (640  x 2560 x N=1024, s=0.90)"),
    (2560,  640, 1024, 0.90, "demo16_ffn_down   (2560 x 640  x N=1024, s=0.90)"),
    (  64,   64,  128, 0.80, "tiny              (64   x 64   x N=128,  s=0.80)"),
]

N_WARMUP = 3
N_RUNS = 30


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def make_inputs(M: int, K: int, N: int, sparsity: float, seed: int = 42):
    """Build a reproducible PaddedCSR + dense dY, X for one shape."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    W_dense = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) >= sparsity
    W_dense = W_dense * mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)

    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()
    return W_csr, W_dense, dY, X


def time_scalar_dw(W_csr: PaddedCSR, dY: np.ndarray, X: np.ndarray) -> float:
    """Median wallclock for one spmm_grad_w call, in milliseconds."""
    # Warmup: first calls hit cold caches and OpenMP thread startup.
    for _ in range(N_WARMUP):
        _core.spmm_grad_w(W_csr, dY, X)

    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _core.spmm_grad_w(W_csr, dY, X)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(samples)


def time_simd_dw(W_csr: PaddedCSR, dY: np.ndarray, X: np.ndarray) -> float:
    """Median wallclock for one spmm_grad_w_simd call, in milliseconds.

    On ARM64 (Apple Silicon, Linux aarch64) this runs our hand-written
    NEON kernel and lands at ~6x the throughput of the scalar column
    on FFN shapes (see milestone_12). On x86_64 the _simd binding
    falls back to the scalar kernel — there is no hand-written AVX
    kernel yet (issue #2). Running this script on an x86 runner
    therefore produces si/sc ~= 1.0 across all shapes.
    """
    for _ in range(N_WARMUP):
        _core.spmm_grad_w_simd(W_csr, dY, X)

    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _core.spmm_grad_w_simd(W_csr, dY, X)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(samples)


def time_dense_oracle(W_dense: torch.Tensor, dY_np: np.ndarray, X_np: np.ndarray) -> float:
    """Median wallclock for the dense-BLAS oracle of the same math.

    Computes:
        G = dY @ X.T       # (M, N) @ (N, K) = (M, K)
        G_masked = G * mask

    This is what a dense autograd path would spend — a lower bound for
    any dense implementation. We include the elementwise mask step to
    be honest: our sparse kernel already skips zero slots, so a fair
    comparison has to include whatever the dense path would need to do
    to get the same final tensor layout.
    """
    dY = torch.from_numpy(dY_np)
    X  = torch.from_numpy(X_np)
    mask = (W_dense != 0.0).to(torch.float32)

    for _ in range(N_WARMUP):
        G = dY @ X.T
        _ = G * mask

    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        G = dY @ X.T
        G_masked = G * mask  # noqa: F841 — we time both ops
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(samples)


def compute_flops(M: int, K: int, N: int, nnz: int) -> int:
    """FLOPs for dW: one N-length dot product per live slot.

    Each dot is N multiplies + (N-1) adds ≈ 2N flops. With nnz slots
    total work = 2 * N * nnz.
    """
    return 2 * N * nnz


def format_gflops(flops: int, ms: float) -> str:
    if ms <= 0:
        return "   n/a"
    gflops = flops / (ms * 1e-3) / 1e9
    return f"{gflops:6.2f}"


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("SparseLab — dW kernel baseline measurement")
    print("(Gate 1 for GitHub issue #1: NEON dW kernel)")
    print("=" * 96)
    print(f"Torch threads: {torch.get_num_threads()}   "
          f"Runs: {N_RUNS} (median)   Warmup: {N_WARMUP}")
    print()
    print(f"{'Shape':<55} {'scalar':>8} {'simd':>8} {'dense':>8} "
          f"{'si/sc':>7} {'s.GF/s':>8}")
    print(f"{'':<55} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} "
          f"{'':>7} {'':>8}")
    print("-" * 96)

    for M, K, N, s, label in SHAPES:
        W_csr, W_dense, dY, X = make_inputs(M, K, N, s)

        t_scalar = time_scalar_dw(W_csr, dY, X)
        t_simd   = time_simd_dw(W_csr, dY, X)
        t_dense  = time_dense_oracle(W_dense, dY, X)
        si_over_sc = t_simd / t_scalar if t_scalar > 0 else float("inf")

        nnz = W_csr.nnz
        flops = compute_flops(M, K, N, nnz)

        print(f"{label:<55} {t_scalar:>8.2f} {t_simd:>8.2f} {t_dense:>8.2f} "
              f"{si_over_sc:>6.2f}x "
              f"{format_gflops(flops, t_scalar):>8}")

    print("=" * 96)
    print()
    print("How to interpret these numbers:")
    print()
    print("  simd/scalar ratio (si/sc):")
    print("    On ARM64 (Apple Silicon, Linux aarch64) we expect")
    print("    ~0.15-0.20 — the hand-written NEON kernel is ~5-6x")
    print("    faster than scalar on FFN shapes.")
    print("    On x86_64 we expect ~1.0 — the _simd binding falls")
    print("    back to scalar (AVX kernel is issue #2, not yet built).")
    print()
    print("  scalar / dense ratio:")
    print("    How much slower our sparse-aware scalar kernel is than a")
    print("    full dense torch.matmul on the same shape. Because we do")
    print("    (1 - sparsity) x less arithmetic, ratios > 1 indicate")
    print("    per-FLOP inefficiency we can recover with SIMD.")
    print()
    print("  s.GF/s (scalar throughput):")
    print("    Our kernel's actual arithmetic rate. Rough ceilings:")
    print("    - Apple M-series f32: ~150-200 GF/s (NEON)")
    print("    - Intel Sapphire Rapids f32: ~100-150 GF/s (AVX-512)")
    print("    - AMD Zen 4 f32:            ~80-120 GF/s (AVX2)")
    print("    Decision rule for new SIMD work:")
    print("      < 30 GF/s  -> scalar / no auto-vec, hand kernel worth writing")
    print("      30-80 GF/s -> partial auto-vec, some headroom")
    print("      > 80 GF/s  -> compiler already vectorized well, invest elsewhere")


if __name__ == "__main__":
    main()
