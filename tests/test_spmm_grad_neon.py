"""
NEON-specific tests for spmm_grad_w_simd.

Purpose
───────
test_spmm_grad.py parametrizes all 15 oracle tests over both kernels, so
scalar-vs-NEON correctness is covered on the 8 canonical shapes. This
file adds cases that specifically exercise NEON's Phase A/B/C structure
(8-wide main loop / 4-wide trail / scalar 1-3 residue), parallel-
scheduling determinism, and edge cases that can expose SIMD-only bugs.

Design rationale
────────────────
The NEON inner loop has three control-flow phases that the oracle tests
don't hit individually. A shape like M=16, K=16, N=8 enters Phase A once
and skips B and C entirely; M=16, K=16, N=9 enters A once, B zero
times, and C for the single trailing float. We parametrize N over every
mod-8 residue from 1-65 to ensure every path executes and every
A→B→C transition is hit by at least one test.

Tolerance
─────────
Scalar-vs-NEON agreement uses rtol=atol=1e-5 matching the rest of
test_spmm_grad.py. The dual-accumulator reordering produces at most
last-1-2-bit float32 differences over N up to a few thousand; we've
verified this is well inside 1e-5 relative on realistic dot-product
magnitudes (see design doc §7.3 and the bit-tolerance discussion).

Run with:   pytest tests/test_spmm_grad_neon.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sparselab import PaddedCSR, _core


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
#  Group C2a — scalar/NEON bit-tolerance agreement over random shapes
#
#  20 randomly-sized problems, varied sparsity. If any slot's NEON
#  result diverges from scalar by more than rtol=atol=1e-5, the
#  dual-accumulator reordering is producing un-tolerated noise and we
#  have a numeric bug to investigate.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rng_seed", list(range(20)))
def test_scalar_and_neon_agree_on_random_shapes(rng_seed):
    """20 random (M, K, N, sparsity) draws: scalar and NEON must agree."""
    rng = np.random.default_rng(rng_seed)
    M = int(rng.integers(4, 64))
    K = int(rng.integers(4, 64))
    N = int(rng.integers(1, 64))
    sparsity = float(rng.uniform(0.3, 0.95))

    W_csr = _make_W_csr(M, K, sparsity, seed=rng_seed)
    dY = torch.randn(M, N, dtype=torch.float32, generator=torch.Generator().manual_seed(rng_seed)).numpy()
    X  = torch.randn(K, N, dtype=torch.float32, generator=torch.Generator().manual_seed(rng_seed + 1)).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_neon   = _core.spmm_grad_w_simd(W_csr, dY, X)

    assert np.allclose(dW_scalar, dW_neon, rtol=1e-5, atol=1e-5), (
        f"Scalar/NEON disagree at seed={rng_seed} "
        f"(M={M}, K={K}, N={N}, s={sparsity:.2f}). "
        f"Max abs diff: {np.abs(dW_scalar - dW_neon).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group C2b — N-residue coverage
#
#  Our NEON kernel has three internal phases based on j's position in
#  the inner dot-product loop:
#    Phase A: j + 8 <= N  (8-wide main)
#    Phase B: j + 4 <= N  (one 4-wide iter if 4-7 remain)
#    Phase C: j < N       (scalar 1-3 residue)
#  We parametrize N across every mod-8 residue from 1..65 plus a few
#  boundary neighbors (15, 17, 31, 33, 63, 65) to guarantee every
#  phase boundary is exercised by at least one test.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "N",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65],
)
def test_all_n_residues_match_scalar(N):
    """Every N % 8 residue must produce the same output as scalar."""
    M, K = 8, 12  # small; we care about N-shape variance, not M or K.
    W_csr = _make_W_csr(M, K, sparsity=0.5, seed=123)
    torch.manual_seed(7)
    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_neon   = _core.spmm_grad_w_simd(W_csr, dY, X)

    assert np.allclose(dW_scalar, dW_neon, rtol=1e-5, atol=1e-5), (
        f"NEON diverged from scalar at N={N} (mod 8 = {N % 8}). "
        f"Max abs diff: {np.abs(dW_scalar - dW_neon).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group C2c — structural edge cases
#
#  Two stress patterns that can expose SIMD-only bugs:
#
#  Empty-row interleaving: rows of W alternate between empty and
#  populated. Tests that OpenMP's static schedule doesn't assume
#  balanced per-thread work.
#
#  Single-slot-per-row: every row has exactly nnz=1, so the outer
#  enumerate-live-slots loop runs one pass per row with very short dot
#  products. This stresses the inner-loop cleanup path when N is tiny.
# ─────────────────────────────────────────────────────────────────────

def test_empty_rows_interleaved():
    """W with every other row empty — OpenMP static schedule robustness."""
    M, K, N = 20, 16, 17   # N=17 exercises phases A, B, C all in one slot
    # Build a mask where rows 1, 3, 5, ... are completely empty.
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
    dW_neon   = _core.spmm_grad_w_simd(W_csr, dY, X)
    assert np.allclose(dW_scalar, dW_neon, rtol=1e-5, atol=1e-5)


def test_single_slot_per_row_tiny_n():
    """Each row has exactly 1 live slot; tiny N stresses the Phase-C tail."""
    M, K, N = 16, 32, 3   # N=3 — scalar tail only, skips A and B entirely
    # Build one live slot per row at a column picked by row index.
    W_dense = torch.zeros(M, K, dtype=torch.float32)
    for i in range(M):
        W_dense[i, i % K] = float(i + 1) * 0.1
    W_csr = PaddedCSR.from_dense(W_dense)

    torch.manual_seed(9)
    dY = torch.randn(M, N, dtype=torch.float32).numpy()
    X  = torch.randn(K, N, dtype=torch.float32).numpy()

    dW_scalar = _core.spmm_grad_w(W_csr, dY, X)
    dW_neon   = _core.spmm_grad_w_simd(W_csr, dY, X)
    assert np.allclose(dW_scalar, dW_neon, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group C2d — determinism under OpenMP
#
#  With schedule(static) the same work goes to the same thread every
#  time, so the final dW_values must be bit-identical across repeated
#  calls. If we ever switch to schedule(dynamic) this would flake —
#  the test defends against a future well-intentioned change that
#  silently breaks training reproducibility.
# ─────────────────────────────────────────────────────────────────────

def test_neon_is_deterministic_across_calls():
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
