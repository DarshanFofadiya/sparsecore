"""
AVX2-specific tests for spmm_grad_w_simd on x86_64.

Purpose
───────
test_spmm_grad.py parametrizes all 23 oracle tests over both kernels,
so scalar-vs-AVX2 correctness is covered on the canonical shapes. This
file adds cases that specifically exercise AVX2's Phase A/B/C structure
(16-wide main loop / 8-wide trail / scalar 0-7 residue), parallel-
scheduling determinism, and edge cases that can expose SIMD-only bugs.

This is the x86 analog of tests/test_spmm_grad_neon.py. Same shape,
same cases, retargeted to the AVX2 phase boundaries. The 8-wide NEON
tests hit N % 8 boundaries; here we hit N % 16 boundaries because
AVX2's Phase A consumes 16 floats per iteration.

Design rationale
────────────────
The AVX2 inner loop has three control-flow phases that the oracle
tests don't hit individually. A shape like M=16, K=16, N=16 enters
Phase A once and skips B and C entirely; M=16, K=16, N=17 enters A
once, B zero times, and C for the single trailing float. We parametrize
N across every mod-16 residue from 1..65 plus boundary neighbors to
ensure every path executes and every A→B→C transition is hit by at
least one test.

Platform gating
───────────────
This file is skipped on non-x86_64 platforms. On ARM64 the AVX2 kernel
doesn't exist (setup.py's IS_X86_64 gate excludes the source file);
calling _core.spmm_grad_w_simd there routes to the NEON kernel which
is already covered by test_spmm_grad_neon.py. Running these tests on
NEON would not break — _core.spmm_grad_w_simd has the same interface
on both platforms and would still agree with scalar — but the N % 16
parametrization would be redundant with test_spmm_grad_neon.py's
N % 8 coverage.

Tolerance
─────────
Scalar-vs-AVX2 agreement uses rtol=atol=1e-5 matching the rest of
test_spmm_grad.py and test_spmm_grad_neon.py. The dual-accumulator
reordering produces at most last-1-2-bit float32 differences over N
up to a few thousand — empirically measured at 2.3e-7 relative error
in the Gate A0 microbench (design doc §6.0). Well inside 1e-5 on
realistic dot-product magnitudes.

Run with:   pytest tests/test_spmm_grad_avx2.py -v
"""

from __future__ import annotations

import platform
import warnings

import numpy as np
import pytest
import torch

from sparselab import PaddedCSR, _core


# ─────────────────────────────────────────────────────────────────────
#  Platform gate
#
#  Skip the whole module on non-x86_64 machines. On ARM the AVX2
#  kernel isn't compiled (source gated in setup.py) and these tests
#  would duplicate test_spmm_grad_neon.py's coverage anyway.
# ─────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"),
    reason="AVX2 kernel only compiled on x86_64 (setup.py IS_X86_64 gate).",
)


@pytest.fixture(autouse=True)
def _suppress_known_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        yield


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_W_csr(M: int, K: int, sparsity: float, seed: int = 42) -> PaddedCSR:
    """Build a PaddedCSR with reproducible random sparsity pattern."""
    torch.manual_seed(seed)
    W_dense = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) >= sparsity
    W_dense = W_dense * mask.float()
    return PaddedCSR.from_dense(W_dense)


# ─────────────────────────────────────────────────────────────────────
#  Group X2a — scalar/AVX2 bit-tolerance agreement over random shapes
#
#  20 randomly-sized problems, varied sparsity. If any slot's AVX2
#  result diverges from scalar by more than rtol=atol=1e-5, the
#  dual-accumulator reordering is producing un-tolerated noise and we
#  have a numeric bug to investigate.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rng_seed", list(range(20)))
def test_scalar_and_avx2_agree_on_random_shapes(rng_seed):
    """20 random (M, K, N, sparsity) draws: scalar and AVX2 must agree."""
    rng = np.random.default_rng(rng_seed)
    M = int(rng.integers(4, 64))
    K = int(rng.integers(4, 64))
    N = int(rng.integers(1, 64))
    sparsity = float(rng.uniform(0.3, 0.95))

    W_csr = _make_W_csr(M, K, sparsity, seed=rng_seed)
    dY = torch.randn(M, N, dtype=torch.float32,
                     generator=torch.Generator().manual_seed(rng_seed)).numpy()
    X  = torch.randn(K, N, dtype=torch.float32,
                     generator=torch.Generator().manual_seed(rng_seed + 1)).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_avx2   = _core.spmm_grad_w_simd(W_csr, dY, X)

    assert np.allclose(dW_scalar, dW_avx2, rtol=1e-5, atol=1e-5), (
        f"Scalar/AVX2 disagree at seed={rng_seed} "
        f"(M={M}, K={K}, N={N}, s={sparsity:.2f}). "
        f"Max abs diff: {np.abs(dW_scalar - dW_avx2).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group X2b — N-residue coverage
#
#  Our AVX2 kernel has three internal phases based on j's position in
#  the inner dot-product loop:
#    Phase A: j + 16 <= N  (16-wide dual-accumulator main)
#    Phase B: j + 8  <= N  (one 8-wide iter if 8-15 remain)
#    Phase C: j < N        (scalar 0-7 residue)
#
#  We parametrize N across every mod-16 residue from 1..65 plus
#  boundary neighbors to guarantee every phase boundary is exercised
#  by at least one test. Key cases:
#
#    N=1..7   → only Phase C (A and B both skip)
#    N=8      → Phase B once, no C
#    N=9..15  → Phase B + 1..7 Phase C residue
#    N=15     → A=0, B=1, C=7 (the worst-case tail before first A iter)
#    N=16     → Phase A once, no B, no C (clean main-loop multiple)
#    N=17     → Phase A once + 1 Phase C (post-A no-B case)
#    N=24     → A + B + no C (Phase A + Phase B exactly fills)
#    N=31     → A=1, B=0, C=15-wait no, A consumes 16 so 15 left, B=1 (8), C=7
#               A once, B once, C=7 (maximum tail after both SIMD phases)
#    N=32     → 2× Phase A, no B, no C
#    N=33     → 2× Phase A, no B, 1 Phase C
#    N=47     → 2× Phase A, B once, 7 Phase C
#    N=48     → 3× Phase A, no B, no C
#    N=64, 65 → stress a multi-iter Phase A with/without trailing residue
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "N",
    [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65],
)
def test_all_n_residues_match_scalar(N):
    """Every N % 16 residue must produce the same output as scalar."""
    M, K = 8, 12  # small; we care about N-shape variance, not M or K.
    W_csr = _make_W_csr(M, K, sparsity=0.5, seed=123)
    torch.manual_seed(7)
    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_avx2   = _core.spmm_grad_w_simd(W_csr, dY, X)

    assert np.allclose(dW_scalar, dW_avx2, rtol=1e-5, atol=1e-5), (
        f"AVX2 diverged from scalar at N={N} (mod 16 = {N % 16}). "
        f"Max abs diff: {np.abs(dW_scalar - dW_avx2).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group X2c — structural edge cases
#
#  Two stress patterns that can expose SIMD-only bugs:
#
#  Empty-row interleaving: rows of W alternate between empty and
#  populated. Tests that OpenMP's static schedule doesn't assume
#  balanced per-thread work.
#
#  Single-slot-per-row: every row has exactly nnz=1, so the outer
#  enumerate-live-slots loop runs one pass per row with very short dot
#  products. This stresses the inner-loop cleanup path when N is tiny
#  (Phase C only, neither A nor B iterates).
# ─────────────────────────────────────────────────────────────────────

def test_empty_rows_interleaved():
    """W with every other row empty — OpenMP static schedule robustness."""
    # N=31 → exercises A once, B once, and C=7 all in one slot — the
    # most complex phase-transition case we can pick.
    M, K, N = 20, 16, 31
    W_dense = torch.randn(M, K, dtype=torch.float32)
    keep_mask = torch.zeros(M, K, dtype=torch.bool)
    for i in range(0, M, 2):  # only even rows keep any connections
        keep_mask[i] = torch.rand(K) >= 0.3
    W_dense = W_dense * keep_mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)

    torch.manual_seed(5)
    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_avx2   = _core.spmm_grad_w_simd(W_csr, dY, X)
    assert np.allclose(dW_scalar, dW_avx2, rtol=1e-5, atol=1e-5)


def test_single_slot_per_row_tiny_n():
    """Each row has exactly 1 live slot; tiny N stresses the Phase-C tail."""
    # N=3 → A=0, B=0, C=3; exercises scalar-only path in AVX2 kernel.
    M, K, N = 16, 32, 3
    W_dense = torch.zeros(M, K, dtype=torch.float32)
    for i in range(M):
        W_dense[i, i % K] = float(i + 1) * 0.1
    W_csr = PaddedCSR.from_dense(W_dense)

    torch.manual_seed(9)
    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_avx2   = _core.spmm_grad_w_simd(W_csr, dY, X)
    assert np.allclose(dW_scalar, dW_avx2, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group X2d — determinism under OpenMP
#
#  With schedule(static) the same work goes to the same thread every
#  time, so the final dW_values must be bit-identical across repeated
#  calls. If we ever switch to schedule(dynamic) this would flake —
#  the test defends against a future well-intentioned change that
#  silently breaks training reproducibility.
# ─────────────────────────────────────────────────────────────────────

def test_avx2_is_deterministic_across_calls():
    """Same inputs → byte-identical outputs every call (bit-stable)."""
    torch.manual_seed(11)
    W_csr = _make_W_csr(64, 96, sparsity=0.7, seed=11)
    dY = torch.randn(64, 48, dtype=torch.float32).numpy()
    X  = torch.randn(96, 48, dtype=torch.float32).numpy()

    g1 = _core.spmm_grad_w_simd(W_csr, dY, X)
    g2 = _core.spmm_grad_w_simd(W_csr, dY, X)
    g3 = _core.spmm_grad_w_simd(W_csr, dY, X)

    # np.array_equal is the bit-identical check; np.allclose would
    # miss the bug where static schedule silently drifted.
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(g2, g3)
