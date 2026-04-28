"""
Demo 17 — NEON-accelerated dW kernel, end-to-end.

What this demo proves
─────────────────────
dW (the gradient of the loss w.r.t. W at live slots) is the dominant
cost of sparse-from-scratch training. At 40M-param scale it was 62% of
a training step (milestone 10). This demo shows the NEON kernel port
cuts that cost by ~6.5x on Apple Silicon and translates into a
~2x faster end-to-end training step, via the same SparseLinear a user
would put in their model.

Two sections below:

  §1. Per-layer dW throughput table — scalar vs NEON vs dense-BLAS on
      the exact FFN shapes from demos 15 (10M) and 16 (40M).

  §2. End-to-end training step timing — a 3-layer sparse MLP, one full
      forward + loss + backward + optimizer step, scalar kernel vs
      NEON kernel. Same seeds, same inputs, same architecture — only
      the backward dW kernel differs.

What to watch for
─────────────────
  §1: all four FFN shapes should show >= 3x local speedup (NEON vs
      scalar). The demo prints "✓ Target met" per row and aborts with
      a clear error if any FFN shape misses.

  §2: end-to-end median should drop by roughly the fraction of step
      time dW was consuming. At FFN-heavy workloads we expect 30-50%
      reduction in per-step wallclock.

Usage
─────
    python examples/demo_17_dw_neon.py

Runtime: ~30 seconds. CPU-only, single-process.

Related
───────
  - Baseline-only profiling:   examples/profile_dw_baseline.py
  - Spec:                      issue #1 on GitHub
  - Prior art this mirrors:    examples/demo_04b_honest_benchmark.py
  - Milestone writeup:         docs/demos/milestone_12.md
"""

from __future__ import annotations

import statistics as stats
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sparselab
from sparselab import _core, PaddedCSR


# Required speedup on FFN shapes to call this demo a success. Matches
# the spec §8.2 ship threshold. Tiny shapes are checked as a no-regress
# sanity (>=0.8x) not as a speedup gate.
REQUIRED_FFN_SPEEDUP = 3.0
REQUIRED_TINY_RATIO = 0.8


# ─────────────────────────────────────────────────────────────────────
#  §1 — Per-layer dW throughput
# ─────────────────────────────────────────────────────────────────────

FFN_SHAPES = [
    ( 384, 1536, 2048, 0.90, "demo15 FFN up  (10M)   ", True),
    (1536,  384, 2048, 0.90, "demo15 FFN down(10M)   ", True),
    ( 640, 2560, 1024, 0.90, "demo16 FFN up  (40M)   ", True),
    (2560,  640, 1024, 0.90, "demo16 FFN down(40M)   ", True),
    (  64,   64,  128, 0.80, "tiny (no-regress)      ", False),
]

N_WARMUP = 3
N_RUNS = 30


def time_kernel(fn, W_csr, dY, X):
    """Median ms across N_RUNS calls; N_WARMUP throwaway warmups first."""
    for _ in range(N_WARMUP):
        fn(W_csr, dY, X)
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(W_csr, dY, X)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(samples)


def make_inputs(M, K, N, sparsity, seed=42):
    """Build a PaddedCSR + numpy dY, X at the requested shape."""
    torch.manual_seed(seed)
    W_dense = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) >= sparsity
    W_dense = W_dense * mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)
    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()
    return W_csr, dY, X


def run_layer_comparison():
    print("=" * 78)
    print("  §1. Per-layer dW throughput (scalar vs NEON)")
    print("=" * 78)
    print(f"    Torch threads: {torch.get_num_threads()}   "
          f"Runs: {N_RUNS} (median)   Warmup: {N_WARMUP}")
    print()
    print(f"    {'Shape':<25} {'scalar':>8} {'NEON':>8} {'speedup':>9} status")
    print(f"    {'':<25} {'(ms)':>8} {'(ms)':>8} {'':>9}")
    print("    " + "-" * 66)

    all_ffn_passed = True

    for M, K, N, s, label, is_ffn in FFN_SHAPES:
        W_csr, dY, X = make_inputs(M, K, N, s)

        t_scalar = time_kernel(_core.spmm_grad_w, W_csr, dY, X)
        t_neon   = time_kernel(_core.spmm_grad_w_simd, W_csr, dY, X)
        speedup = t_scalar / t_neon if t_neon > 0 else float("inf")

        if is_ffn:
            ok = speedup >= REQUIRED_FFN_SPEEDUP
            status = f"✓ >= {REQUIRED_FFN_SPEEDUP}x" if ok else f"✗ < {REQUIRED_FFN_SPEEDUP}x"
            if not ok:
                all_ffn_passed = False
        else:
            ratio = t_neon / t_scalar
            ok = ratio <= (1.0 / REQUIRED_TINY_RATIO)
            status = "✓ no regress" if ok else "✗ regress"

        print(f"    {label:<25} {t_scalar:>8.2f} {t_neon:>8.2f} "
              f"{speedup:>7.2f}x  {status}")

    print("    " + "-" * 66)
    if all_ffn_passed:
        print(f"    All FFN shapes met the >= {REQUIRED_FFN_SPEEDUP}x speedup target.")
    else:
        print(f"    At least one FFN shape missed the {REQUIRED_FFN_SPEEDUP}x target.")
        print("    See spec §8.2 for the diagnostic playbook (instruments, asm check).")
    print()
    return all_ffn_passed


# ─────────────────────────────────────────────────────────────────────
#  §2 — End-to-end training step timing
# ─────────────────────────────────────────────────────────────────────
#
#  Build a small 3-layer sparse MLP, time one full training step
#  (forward + loss + backward + optimizer.step()), once with
#  kernel="scalar" and once with kernel="simd". Same seed, same inputs,
#  same learning-rate-zero optimizer (so the test doesn't diverge
#  between runs for any other reason).
#
#  We use kernel-routing via sparselab.spmm(kernel=…) rather than the
#  default "auto" so the scalar/NEON branches are explicitly exercised
#  and the difference is attributable to the dW kernel alone.
# ─────────────────────────────────────────────────────────────────────


class SparseMLP(nn.Module):
    """3-layer sparse MLP (1536 → 640 → 640 → 10) at 90% sparsity."""
    def __init__(self):
        super().__init__()
        self.fc1 = sparselab.SparseLinear(1536, 640, sparsity=0.9, bias=False)
        self.fc2 = sparselab.SparseLinear( 640, 640, sparsity=0.9, bias=False)
        self.fc3 = sparselab.SparseLinear( 640,  10, sparsity=0.9, bias=False)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        return self.fc3(h)


def time_training_step(model, x, y, optimizer, n_runs=10):
    """Median wallclock for one full fwd+bwd+step iteration."""
    samples = []
    for _ in range(n_runs):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return stats.median(samples)


def run_end_to_end_comparison():
    print("=" * 78)
    print("  §2. End-to-end training step (3-layer sparse MLP)")
    print("=" * 78)

    # Build identical models twice — one with scalar backward, one with NEON.
    # Seed before each build so both start from the exact same weights.
    B, D_IN = 32, 1536

    def build_model_and_inputs(seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SparseMLP()
        x = torch.randn(B, D_IN)
        y = torch.randint(0, 10, (B,))
        # lr=0 so the optimizer step is a no-op — we measure per-step
        # cost without letting weight drift change subsequent timing.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        return model, x, y, optimizer

    # ─ Warmup + time with SCALAR backward path ─
    model_s, x_s, y_s, opt_s = build_model_and_inputs()
    _monkey_patch_kernel(model_s, "scalar")
    # Warmup
    for _ in range(3):
        opt_s.zero_grad(set_to_none=True)
        F.cross_entropy(model_s(x_s), y_s).backward()
        opt_s.step()
    t_scalar = time_training_step(model_s, x_s, y_s, opt_s)

    # ─ Warmup + time with NEON backward path ─
    model_n, x_n, y_n, opt_n = build_model_and_inputs()
    _monkey_patch_kernel(model_n, "simd")
    for _ in range(3):
        opt_n.zero_grad(set_to_none=True)
        F.cross_entropy(model_n(x_n), y_n).backward()
        opt_n.step()
    t_neon = time_training_step(model_n, x_n, y_n, opt_n)

    speedup = t_scalar / t_neon if t_neon > 0 else float("inf")

    print(f"    Model: 3-layer SparseMLP (1536 -> 640 -> 640 -> 10) @ 90% sparsity")
    print(f"    Batch: {B} samples, forward + loss + backward + step (lr=0)")
    print(f"    Runs:  10 (median), 3 warmup")
    print()
    print(f"    Backward kernel     Per-step wallclock")
    print("    " + "-" * 40)
    print(f"    scalar              {t_scalar:>7.2f} ms")
    print(f"    NEON                {t_neon:>7.2f} ms")
    print(f"    Speedup (end-to-end): {speedup:.2f}x")
    print()

    if speedup >= 1.3:
        print(f"    ✓ Meaningful end-to-end speedup delivered.")
    else:
        print(f"    (Smaller than layer-level speedup because forward,")
        print(f"     dX, and Python overhead dilute the dW-only win.")
        print(f"     Larger models amortize these better — see demo 16 scaling.)")
    print()
    return t_scalar, t_neon, speedup


def _monkey_patch_kernel(model: nn.Module, kernel: str):
    """Replace SparseLinear.forward with a kernel-specific version.

    The public SparseLinear.forward uses sparselab.spmm with
    kernel='auto'. For this demo we want explicit scalar vs NEON runs
    to attribute the difference to the dW kernel, so we patch each
    layer in place. The autograd SpMMFunction stashes ctx.kernel at
    forward time, so the backward pass picks up the right kernel
    automatically via the Phase-D dispatch in sparselab.ops.
    """
    from sparselab.ops import _SpMMFunction

    for module in model.modules():
        if isinstance(module, sparselab.SparseLinear):
            # Bind a closure that forces the chosen kernel.
            def make_forward(m, k):
                def forward(x):
                    orig_shape = x.shape
                    x_flat = x.reshape(-1, m.in_features).contiguous()
                    X_col = x_flat.t().contiguous()
                    Y_col = _SpMMFunction.apply(m._values, m._csr, X_col, k)
                    y_flat = Y_col.t()
                    if m.bias is not None:
                        y_flat = y_flat + m.bias
                    return y_flat.reshape(orig_shape[:-1] + (m.out_features,))
                return forward
            module.forward = make_forward(module, kernel)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔" + "═" * 76 + "╗")
    print("║  SparseLab demo 17 — NEON-accelerated dW kernel                            ║")
    print("║  (the v0.2.1 speedup, end-to-end)                                          ║")
    print("╚" + "═" * 76 + "╝")
    print()

    all_ffn_ok = run_layer_comparison()
    t_scalar_e2e, t_neon_e2e, speedup_e2e = run_end_to_end_comparison()

    print("=" * 78)
    print("  Summary")
    print("=" * 78)
    if all_ffn_ok and speedup_e2e >= 1.3:
        print(f"    dW NEON kernel delivers the v0.2.1 speedup target:")
        print(f"      - Per-layer:     >= 3x on all FFN shapes")
        print(f"      - End-to-end:    {speedup_e2e:.2f}x on a 3-layer sparse MLP training step")
        print(f"    Users on Apple Silicon get this automatically via the 'auto'")
        print(f"    kernel default in SparseLinear. No API change.")
        print()
    else:
        print(f"    Some targets missed. See per-row detail above.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
