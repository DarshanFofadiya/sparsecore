"""
Milestone 4a-ii Oracle tests — PaddedCSR.transpose().

What we verify:
  - Correctness: W.transpose().to_dense() == W.to_dense().T exactly
  - Shape: transpose of (M, K) has shape (K, M)
  - nnz preservation: transpose preserves nnz count
  - Edge cases: empty matrix, all-zero, square, tall, wide, single entry
  - Invariants: the transposed PaddedCSR passes its own assert_invariants()
  - Involution: transpose(transpose(W)) == W (up to padding layout)
  - Used correctly as input to SpMM: X @ W == (Wᵀ @ Xᵀ)ᵀ algebra holds

Oracle: torch.Tensor.T — bit-exact comparison (transpose is pure data
movement, no float arithmetic).

Design doc: docs/design/spmm_backward.md §3.2
Run with:  pytest tests/test_padded_csr_transpose.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import sparselab
from sparselab import PaddedCSR


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
#  Group 1 — Correctness vs torch.Tensor.T
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M,K,sparsity",
    [
        (3, 4, 0.5),       # small, uneven
        (8, 8, 0.0),       # fully dense square
        (8, 8, 0.5),       # 50% sparse square
        (8, 8, 0.9),       # 90% sparse square
        (16, 32, 0.8),     # wider than tall
        (32, 16, 0.8),     # taller than wide
        (1, 1, 0.0),       # smallest
        (1, 16, 0.5),      # row vector
        (16, 1, 0.5),      # column vector
        (64, 64, 0.99),    # very sparse
    ],
    ids=[
        "3x4_50", "8x8_dense", "8x8_50", "8x8_90",
        "16x32_80", "32x16_80", "1x1", "1x16", "16x1", "64x64_99",
    ],
)
def test_transpose_matches_torch(M, K, sparsity):
    """W.transpose().to_dense() must bit-exactly equal W.to_dense().T."""
    torch.manual_seed(42)
    W_dense = torch.randn(M, K, dtype=torch.float32)
    keep = torch.rand(M, K) >= sparsity
    W_dense = W_dense * keep.float()

    W_csr = PaddedCSR.from_dense(W_dense)
    WT = W_csr.transpose()

    # Shape is swapped
    assert WT.shape == (K, M), f"Expected shape ({K}, {M}), got {WT.shape}"
    # nnz is preserved
    assert WT.nnz == W_csr.nnz, (
        f"Transpose changed nnz: {W_csr.nnz} → {WT.nnz}"
    )
    # Bit-exact correctness — transpose is pure data movement.
    assert torch.equal(WT.to_dense(), W_dense.T), (
        f"Transpose mismatch:\nGot:\n{WT.to_dense()}\nExpected:\n{W_dense.T}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Invariants and edge cases
# ─────────────────────────────────────────────────────────────────────

def test_transpose_result_passes_invariants():
    """The transposed PaddedCSR must pass its own invariant check."""
    torch.manual_seed(0)
    W_dense = torch.randn(20, 30) * (torch.rand(20, 30) >= 0.7).float()
    W_csr = PaddedCSR.from_dense(W_dense)
    WT = W_csr.transpose()
    # If any slot-layout invariant was broken, this raises.
    WT.assert_invariants()


def test_transpose_empty_matrix():
    """Transposing an empty (0 nnz) matrix yields another empty matrix."""
    W_csr = PaddedCSR(nrows=5, ncols=8)  # empty-constructor → 0 nnz
    WT = W_csr.transpose()
    assert WT.shape == (8, 5)
    assert WT.nnz == 0
    WT.assert_invariants()


def test_transpose_all_zeros():
    """An all-zero dense input transposes to an all-zero output."""
    W_dense = torch.zeros(7, 11, dtype=torch.float32)
    W_csr = PaddedCSR.from_dense(W_dense)
    WT = W_csr.transpose()
    assert WT.shape == (11, 7)
    assert WT.nnz == 0
    assert torch.equal(WT.to_dense(), torch.zeros(11, 7))


def test_transpose_single_entry():
    """A matrix with exactly one live entry transposes deterministically."""
    W_dense = torch.zeros(5, 6, dtype=torch.float32)
    W_dense[2, 4] = 3.14
    W_csr = PaddedCSR.from_dense(W_dense)
    WT = W_csr.transpose()
    assert WT.shape == (6, 5)
    assert WT.nnz == 1
    # The entry should appear at Wᵀ[4, 2].
    assert WT.to_dense()[4, 2].item() == pytest.approx(3.14, abs=1e-6)
    # All other positions are zero.
    expected = torch.zeros(6, 5, dtype=torch.float32)
    expected[4, 2] = 3.14
    assert torch.equal(WT.to_dense(), expected)


def test_transpose_preserves_values():
    """
    Every live value in W must appear somewhere in Wᵀ with no duplication,
    no loss, no mutation.
    """
    torch.manual_seed(1)
    W_dense = torch.randn(12, 18) * (torch.rand(12, 18) >= 0.6).float()
    W_csr = PaddedCSR.from_dense(W_dense)
    WT = W_csr.transpose()

    # Collect live values from W_csr and WT, sorted for comparison.
    vals_in = np.array(W_csr.values)[
        np.where(np.array(W_csr.col_indices) >= 0)
    ]
    vals_out = np.array(WT.values)[
        np.where(np.array(WT.col_indices) >= 0)
    ]
    np.testing.assert_array_equal(np.sort(vals_in), np.sort(vals_out))


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Involution: transpose twice == identity
# ─────────────────────────────────────────────────────────────────────

def test_transpose_involution():
    """
    Transposing twice returns a matrix with the same logical content
    (dense view) as the original. Padding layout may differ.
    """
    torch.manual_seed(2)
    W_dense = torch.randn(15, 20) * (torch.rand(15, 20) >= 0.75).float()
    W_csr = PaddedCSR.from_dense(W_dense)
    WTT = W_csr.transpose().transpose()

    assert WTT.shape == W_csr.shape
    assert WTT.nnz == W_csr.nnz
    assert torch.equal(WTT.to_dense(), W_csr.to_dense())


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — Works correctly as input to SpMM
# ─────────────────────────────────────────────────────────────────────

def test_transpose_enables_wt_matmul_y():
    """
    The whole point of transpose is to compute dL/dX = Wᵀ @ dL/dY via
    our existing SpMM kernel. This test verifies that workflow end-to-end.

    For forward: Y = W @ X
    For backward on X: dL/dX = Wᵀ @ dL/dY

    Here we stand in for dL/dY with a random tensor and verify that
    spmm(Wᵀ, dY) produces the same result as W.T @ dY (dense oracle).
    """
    torch.manual_seed(3)
    M, K, N = 16, 24, 8
    W_dense = torch.randn(M, K) * (torch.rand(M, K) >= 0.7).float()
    dY = torch.randn(M, N, dtype=torch.float32)  # stands in for dL/dY

    W_csr = PaddedCSR.from_dense(W_dense)
    WT = W_csr.transpose()

    # Our path: spmm(Wᵀ, dY)
    dX_ours = sparselab.spmm(WT, dY)
    # Oracle path: dense W.T @ dY
    dX_oracle = W_dense.T @ dY

    assert dX_ours.shape == (K, N)
    assert torch.allclose(dX_ours, dX_oracle, rtol=1e-5, atol=1e-5), (
        f"spmm(Wᵀ, dY) didn't match W.T @ dY. "
        f"Max diff: {(dX_ours - dX_oracle).abs().max().item():.3e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 5 — padding_ratio parameter
# ─────────────────────────────────────────────────────────────────────

def test_transpose_respects_padding_ratio():
    """Custom padding_ratio is applied to the output."""
    W_dense = torch.randn(10, 10) * (torch.rand(10, 10) >= 0.7).float()
    W_csr = PaddedCSR.from_dense(W_dense)

    WT_no_pad = W_csr.transpose(padding_ratio=0.0)
    WT_full_pad = W_csr.transpose(padding_ratio=1.0)

    # With 0.0 padding, total_capacity should equal nnz (no extra slots).
    # Edge case: each row gets a floor of 1 slot even if nnz=0, so
    # total_capacity >= K. Equal when every row is non-empty.
    assert WT_no_pad.total_capacity >= WT_no_pad.nnz
    # With 1.0 padding, total_capacity should be significantly larger.
    assert WT_full_pad.total_capacity > WT_no_pad.total_capacity
    # Both must produce identical dense content.
    assert torch.equal(WT_no_pad.to_dense(), WT_full_pad.to_dense())


def test_transpose_rejects_negative_padding():
    """Negative padding_ratio is invalid (delegated to _compute_row_capacity)."""
    W_csr = PaddedCSR.from_dense(torch.randn(4, 4))
    with pytest.raises(ValueError, match="padding_ratio"):
        W_csr.transpose(padding_ratio=-0.1)
