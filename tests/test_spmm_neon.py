"""
Milestone 3d Oracle tests — NEON SpMM correctness.

The NEON kernel (_core.spmm_simd) must agree with:
  (a) the scalar kernel (_core.spmm_scalar) — they compute the same math
      in different orders; float non-associativity may cause ~1 ULP diffs
  (b) the PyTorch oracle (W_dense @ X) — the ultimate ground truth

Both comparisons use rtol=atol=1e-5.

Extra focus vs. test_spmm.py:
  - Tail-path stress: many N values that are NOT multiples of 4 (the
    SIMD width), because tail off-by-one bugs hide there
  - Extreme N values: N=1, N=2, N=3 (pure tail, no SIMD body runs)
  - Large N: confirms the main SIMD loop actually runs many iterations
  - Cross-check: every case tests scalar vs NEON directly, isolating
    any NEON-specific bug from the shared binding layer

Run with:  pytest tests/test_spmm_neon.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sparsecore import PaddedCSR, _core


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


def _make_sparse_W(
    M: int, K: int, sparsity: float, seed: int = 0
) -> tuple[PaddedCSR, torch.Tensor]:
    """Build (PaddedCSR, dense ground-truth) with given sparsity."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(M, K, generator=gen, dtype=torch.float32)
    keep_mask = torch.rand(M, K, generator=gen) >= sparsity
    W_dense = W * keep_mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)
    return W_csr, W_dense


# ─────────────────────────────────────────────────────────────────────
#  Group 1 — NEON vs scalar vs oracle (triangular correctness)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M,K,N",
    [
        # N covers all four tail residues (N % 4 = 0, 1, 2, 3).
        (32, 32, 32),   # N % 4 == 0 — pure SIMD, no tail
        (32, 32, 33),   # N % 4 == 1 — 1-element tail
        (32, 32, 34),   # N % 4 == 2 — 2-element tail
        (32, 32, 35),   # N % 4 == 3 — 3-element tail
        (16, 24, 1),    # N=1: entire inner loop is tail, SIMD body never runs
        (16, 24, 2),    # N=2: pure tail
        (16, 24, 3),    # N=3: pure tail
        (64, 64, 128),  # longer N, many SIMD body iterations
        (7, 13, 5),     # all prime
    ],
    ids=[
        "N_mod4_0", "N_mod4_1", "N_mod4_2", "N_mod4_3",
        "N_1", "N_2", "N_3", "N_128", "all_prime",
    ],
)
@pytest.mark.parametrize(
    "sparsity",
    [0.0, 0.5, 0.9, 0.99],
    ids=["dense", "50pct", "90pct", "99pct"],
)
def test_neon_matches_scalar_and_oracle(M, K, N, sparsity):
    """
    NEON must produce the same answer as the scalar kernel (within rtol=1e-5)
    and both must match torch.matmul. This triangular check isolates NEON
    bugs from shared-binding bugs.
    """
    if M * K < 4 and sparsity >= 0.9:
        pytest.skip("Not enough cells for >=90% sparsity to be meaningful.")

    W_csr, W_dense = _make_sparse_W(M, K, sparsity=sparsity, seed=42)
    X = torch.randn(K, N, dtype=torch.float32)
    X_np = X.numpy()

    Y_scalar = _core.spmm_scalar(W_csr, X_np)
    Y_neon = _core.spmm_simd(W_csr, X_np)
    Y_oracle = (W_dense @ X).numpy()

    assert Y_neon.shape == (M, N)
    assert Y_neon.dtype == np.float32

    assert np.allclose(Y_neon, Y_scalar, rtol=1e-5, atol=1e-5), (
        f"NEON vs scalar mismatch: max diff = "
        f"{np.abs(Y_neon - Y_scalar).max():.3e}"
    )
    assert np.allclose(Y_neon, Y_oracle, rtol=1e-5, atol=1e-5), (
        f"NEON vs oracle mismatch: max diff = "
        f"{np.abs(Y_neon - Y_oracle).max():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Edge cases specific to NEON
# ─────────────────────────────────────────────────────────────────────

def test_neon_fully_sparse_returns_zeros():
    """NEON on an all-zero W must still produce exactly zero output."""
    W_dense = torch.zeros(8, 16, dtype=torch.float32)
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(16, 7, dtype=torch.float32)
    Y = _core.spmm_simd(W_csr, X.numpy())
    np.testing.assert_array_equal(Y, np.zeros((8, 7), dtype=np.float32))


def test_neon_empty_rows_skipped():
    """
    Mixed empty / non-empty rows. The NEON kernel must skip empty rows
    without reading garbage and still correctly fill non-empty rows.
    """
    W_dense = torch.zeros(6, 16, dtype=torch.float32)
    # Rows 0, 3, 5 are non-empty; rows 1, 2, 4 are empty.
    W_dense[0, 2] = 1.5
    W_dense[0, 7] = -2.0
    W_dense[3, 10] = 3.25
    W_dense[5, 0] = -0.5
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(16, 11, dtype=torch.float32)

    Y_neon = _core.spmm_simd(W_csr, X.numpy())
    Y_oracle = (W_dense @ X).numpy()

    # Empty rows are exactly zero (they never ran the inner loop at all).
    np.testing.assert_array_equal(Y_neon[1], np.zeros(11))
    np.testing.assert_array_equal(Y_neon[2], np.zeros(11))
    np.testing.assert_array_equal(Y_neon[4], np.zeros(11))
    # Non-empty rows match the oracle.
    assert np.allclose(Y_neon, Y_oracle, rtol=1e-5, atol=1e-5)


def test_neon_large_N_main_loop_stress():
    """
    Run with N=1024 so the main SIMD loop does 256 iterations per live
    slot. Catches any error that only accumulates over many iterations
    (register spills, mis-managed accumulators).
    """
    W_csr, W_dense = _make_sparse_W(32, 48, sparsity=0.85, seed=11)
    X = torch.randn(48, 1024, dtype=torch.float32)

    Y_neon = _core.spmm_simd(W_csr, X.numpy())
    Y_oracle = (W_dense @ X).numpy()
    assert np.allclose(Y_neon, Y_oracle, rtol=1e-5, atol=1e-5)


def test_neon_padding_slots_not_touched():
    """
    With padding_ratio=1.0 (100% padding), half of values[] is padding.
    If the NEON kernel walks capacity instead of nnz, it would read
    value=0.0 × col_idx=-1, producing either a segfault (c = -1 is a
    huge unsigned index into X) or silent garbage. Our oracle compare
    catches both.
    """
    W_dense = torch.randn(24, 32) * (torch.rand(24, 32) > 0.75).float()
    W_csr = PaddedCSR.from_dense(W_dense, padding_ratio=1.0)
    X = torch.randn(32, 13, dtype=torch.float32)  # N=13: mod4==1 tail too

    Y_neon = _core.spmm_simd(W_csr, X.numpy())
    Y_oracle = (W_dense @ X).numpy()
    assert np.allclose(Y_neon, Y_oracle, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Numeric stability (harder-to-hit cases)
# ─────────────────────────────────────────────────────────────────────

def test_neon_very_small_values():
    """
    Small-magnitude inputs. Catches any place where we accidentally use
    a less-precise operation (like scalar FP add in the wrong order).
    """
    W_csr, W_dense = _make_sparse_W(8, 8, sparsity=0.5, seed=2)
    X = torch.randn(8, 8, dtype=torch.float32) * 1e-6  # tiny values

    Y_neon = _core.spmm_simd(W_csr, X.numpy())
    Y_oracle = (W_dense @ X).numpy()
    # Tolerance relaxes proportional to magnitude; 1e-5 relative is still
    # tight (last 5 of 7 decimal digits agree).
    assert np.allclose(Y_neon, Y_oracle, rtol=1e-5, atol=1e-10)


def test_neon_very_large_values():
    """Large-magnitude inputs — check we don't accidentally overflow."""
    W_csr, W_dense = _make_sparse_W(8, 8, sparsity=0.5, seed=3)
    X = torch.randn(8, 8, dtype=torch.float32) * 1e3

    Y_neon = _core.spmm_simd(W_csr, X.numpy())
    Y_oracle = (W_dense @ X).numpy()
    assert np.allclose(Y_neon, Y_oracle, rtol=1e-5, atol=1e-2)


def test_neon_deterministic():
    """
    Repeated calls with the same input produce bit-exact identical output.
    NEON kernels must not leak state between calls.
    """
    W_csr, _ = _make_sparse_W(32, 32, sparsity=0.8, seed=5)
    X = torch.randn(32, 17, dtype=torch.float32).numpy()

    Y1 = _core.spmm_simd(W_csr, X)
    Y2 = _core.spmm_simd(W_csr, X)
    Y3 = _core.spmm_simd(W_csr, X)
    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(Y2, Y3)
