"""
Tests for PaddedCSR.rewrite_row (milestone 4e, C++ chunk).

rewrite_row is the single mutation primitive used by DST algorithms.
These tests verify that every invariant from docs/design/padded_csr.md
§2.2 is preserved across mutations, and that malformed inputs fail
loudly with informative errors.
"""

import numpy as np
import pytest
import torch

import sparselab
from sparselab import PaddedCSR


# ─────────────────────────────────────────────────────────────────────
#  Happy path — common mutations work
# ─────────────────────────────────────────────────────────────────────

def test_rewrite_row_replaces_content():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    dense_before = W.to_dense().clone()

    # Replace row 2 with a known content.
    cols = np.array([0, 4, 7], dtype=np.int32)
    vals = np.array([1.5, -2.5, 3.5], dtype=np.float32)
    W.rewrite_row(2, cols, vals)

    dense_after = W.to_dense()
    # Rows 0, 1, 3 unchanged
    for i in (0, 1, 3):
        torch.testing.assert_close(dense_after[i], dense_before[i])
    # Row 2 matches exactly
    expected_row = torch.zeros(8)
    expected_row[0] = 1.5
    expected_row[4] = -2.5
    expected_row[7] = 3.5
    torch.testing.assert_close(dense_after[2], expected_row)


def test_rewrite_row_updates_nnz_correctly():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    original_row_nnz = list(np.asarray(W.row_nnz))

    # Shrink row 1
    W.rewrite_row(1, np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))
    assert np.asarray(W.row_nnz)[1] == 1

    # Expand row 3 up to its capacity
    cap3 = int(np.asarray(W.row_capacity)[3])
    new_cols = np.arange(cap3, dtype=np.int32)
    new_vals = np.arange(cap3, dtype=np.float32) * 0.1
    W.rewrite_row(3, new_cols, new_vals)
    assert np.asarray(W.row_nnz)[3] == cap3

    # Untouched rows stay the same
    assert np.asarray(W.row_nnz)[0] == original_row_nnz[0]
    assert np.asarray(W.row_nnz)[2] == original_row_nnz[2]


def test_rewrite_row_fills_padding_with_sentinel():
    """After a shrinking rewrite, the trailing slots must become
    (col=-1, val=0) so the SpMM kernel correctly skips them."""
    W = PaddedCSR.random(3, 6, sparsity=0.2, seed=0)
    # Shrink row 0 to a single entry
    W.rewrite_row(0, np.array([2], dtype=np.int32),
                      np.array([9.0], dtype=np.float32))

    cols = np.asarray(W.col_indices)
    vals = np.asarray(W.values)
    row_start = int(np.asarray(W.row_start)[0])
    row_nnz = int(np.asarray(W.row_nnz)[0])
    row_cap = int(np.asarray(W.row_capacity)[0])

    # Live slot
    assert cols[row_start] == 2
    assert vals[row_start] == pytest.approx(9.0)

    # All trailing slots are padding
    for k in range(row_nnz, row_cap):
        assert cols[row_start + k] == -1, f"slot {k}: col should be -1"
        assert vals[row_start + k] == 0.0, f"slot {k}: val should be 0"


def test_rewrite_row_preserves_all_invariants():
    W = PaddedCSR.random(6, 10, sparsity=0.4, seed=0)
    for row in range(6):
        # Pick 2 arbitrary valid columns (sorted)
        cols = np.array([row % 10, (row + 3) % 10], dtype=np.int32)
        cols = np.sort(cols)
        # If we happened to pick duplicates, de-dup
        cols = np.unique(cols)
        vals = np.arange(len(cols), dtype=np.float32) * 0.5
        W.rewrite_row(row, cols, vals)
    W.assert_invariants()


def test_rewrite_row_works_in_sequence():
    """Multiple rewrites to the same row should each be atomic —
    later writes overwrite earlier ones cleanly."""
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)

    for i in range(5):
        cols = np.array([i], dtype=np.int32)
        vals = np.array([float(i)], dtype=np.float32)
        W.rewrite_row(0, cols, vals)

    row0 = W.to_dense()[0]
    expected = torch.zeros(8)
    expected[4] = 4.0
    torch.testing.assert_close(row0, expected)
    W.assert_invariants()


# ─────────────────────────────────────────────────────────────────────
#  Error paths — bad input must fail loudly
# ─────────────────────────────────────────────────────────────────────

def test_rewrite_row_rejects_out_of_range_index():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    with pytest.raises(ValueError, match="out of range"):
        W.rewrite_row(10, np.array([0], dtype=np.int32),
                          np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="out of range"):
        W.rewrite_row(-1, np.array([0], dtype=np.int32),
                          np.array([1.0], dtype=np.float32))


def test_rewrite_row_rejects_unsorted_cols():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    with pytest.raises(ValueError, match="ascending"):
        W.rewrite_row(0, np.array([3, 1], dtype=np.int32),
                         np.array([1.0, 2.0], dtype=np.float32))


def test_rewrite_row_rejects_duplicate_cols():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    # Duplicates are explicitly forbidden (PyTorch CSR invariant).
    with pytest.raises(ValueError, match="ascending"):
        W.rewrite_row(0, np.array([2, 2, 5], dtype=np.int32),
                         np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_rewrite_row_rejects_out_of_range_cols():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    with pytest.raises(ValueError, match="out of range"):
        W.rewrite_row(0, np.array([0, 8], dtype=np.int32),  # 8 == ncols
                         np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="out of range"):
        W.rewrite_row(0, np.array([-1, 3], dtype=np.int32),
                         np.array([1.0, 2.0], dtype=np.float32))


def test_rewrite_row_rejects_length_mismatch():
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    with pytest.raises(ValueError, match="must equal"):
        W.rewrite_row(0, np.array([0, 3], dtype=np.int32),
                         np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_rewrite_row_rejects_oversize():
    """Can't grow a row beyond its row_capacity."""
    W = PaddedCSR.random(4, 8, sparsity=0.5, seed=0)
    cap0 = int(np.asarray(W.row_capacity)[0])
    cols = np.arange(cap0 + 1, dtype=np.int32)
    vals = np.ones(cap0 + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="capacity"):
        W.rewrite_row(0, cols, vals)


# ─────────────────────────────────────────────────────────────────────
#  Integration with SpMM
# ─────────────────────────────────────────────────────────────────────

def test_spmm_sees_rewritten_rows():
    """A forward pass after rewrite_row must use the new content.
    This is the integration test: prove the mutation is 'visible' to
    the downstream kernel."""
    W = PaddedCSR.random(3, 5, sparsity=0.0, seed=0)  # dense to start

    # Replace row 1 with a single entry at column 2.
    W.rewrite_row(1, np.array([2], dtype=np.int32),
                     np.array([7.0], dtype=np.float32))

    # Run forward. Y[1, j] should equal 7 * X[2, j].
    X = torch.ones(5, 4) * 2.0    # every entry = 2.0
    Y = sparselab.spmm(W, X)
    assert Y[1].allclose(torch.full((4,), 14.0))  # 7 * 2 = 14
