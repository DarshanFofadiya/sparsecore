"""
Milestone 3b Oracle tests — PaddedCSR construction, round-trip, invariants.

Covers:
  - Construction from dense tensors, torch.sparse_csr tensors, random factory
  - Round-trip fidelity (dense → PaddedCSR → dense preserves values exactly)
  - All 8 invariants hold after construction (checked via assert_invariants)
  - Padding ratio handling (0.0, 0.2, 0.5, 1.0)
  - Edge cases (empty matrix, all-zero, size 1, large)
  - Error paths (invalid padding_ratio, wrong dtype, non-2D, invalid sparsity)

These tests do NOT cover SpMM math — that's Milestone 3c's test suite.

Design doc: docs/design/padded_csr.md
Run with:  pytest tests/test_padded_csr.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sparsecore import PaddedCSR


# PyTorch's sparse_csr API emits a beta-state UserWarning. We know, we're using
# it on purpose, we don't want it cluttering test output.
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
#  Group 1 — Construction paths
# ─────────────────────────────────────────────────────────────────────

def test_empty_constructor_shape():
    """Empty PaddedCSR(nrows, ncols) has correct shape and zero nnz."""
    p = PaddedCSR(nrows=5, ncols=10)
    assert p.shape == (5, 10)
    assert p.nrows == 5
    assert p.ncols == 10
    assert p.nnz == 0
    assert p.total_capacity == 0


def test_from_dense_basic():
    """from_dense on the design-doc worked example produces expected layout."""
    W = torch.tensor([
        [3.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0, 5.0],
        [4.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    p = PaddedCSR.from_dense(W, padding_ratio=0.5)

    assert p.shape == (3, 4)
    assert p.nnz == 5
    # Each row gets ceil(nnz * 1.5) capacity, min 1:
    #   row 0: nnz=2 → ceil(3.0) = 3
    #   row 1: nnz=2 → ceil(3.0) = 3
    #   row 2: nnz=1 → ceil(1.5) = 2
    assert list(p.row_capacity) == [3, 3, 2]
    assert list(p.row_nnz) == [2, 2, 1]
    assert list(p.row_start) == [0, 3, 6]


def test_from_torch_sparse_csr_basic():
    """Construction via torch.sparse_csr_tensor produces the same result as from_dense."""
    W = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
    ], dtype=torch.float32)

    p1 = PaddedCSR.from_dense(W, padding_ratio=0.2)
    p2 = PaddedCSR.from_torch_sparse_csr(W.to_sparse_csr(), padding_ratio=0.2)

    # Both should reconstruct to the same dense matrix.
    assert torch.equal(p1.to_dense(), W)
    assert torch.equal(p2.to_dense(), W)


def test_random_factory_achieves_target_sparsity():
    """random() produces a matrix whose sparsity is near the requested target."""
    p = PaddedCSR.random(500, 500, sparsity=0.9, seed=42)
    assert p.shape == (500, 500)
    # Bernoulli with p=0.1 over 250k cells: stddev ≈ sqrt(250_000 * 0.1 * 0.9) ≈ 150.
    # 5σ bound is about ±750 cells ≈ ±0.003 in sparsity fraction. 0.01 is safe.
    assert abs(p.sparsity - 0.9) < 0.01


def test_random_is_reproducible_with_seed():
    """Same seed → same nnz (and by extension, same topology)."""
    p1 = PaddedCSR.random(100, 100, sparsity=0.9, seed=7)
    p2 = PaddedCSR.random(100, 100, sparsity=0.9, seed=7)
    assert p1.nnz == p2.nnz
    # Value arrays should be bitwise identical.
    assert np.array_equal(p1.values, p2.values)
    assert np.array_equal(p1.col_indices, p2.col_indices)


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Round-trip fidelity
# ─────────────────────────────────────────────────────────────────────

def _round_trip(W: torch.Tensor, padding_ratio: float = 0.2) -> torch.Tensor:
    return PaddedCSR.from_dense(W, padding_ratio=padding_ratio).to_dense()


@pytest.mark.parametrize("shape", [(1, 1), (1, 10), (10, 1), (7, 7), (50, 100)])
def test_round_trip_preserves_values(shape):
    """Dense → PaddedCSR → dense is the identity."""
    torch.manual_seed(sum(shape))
    W = torch.randn(*shape, dtype=torch.float32)
    # Mask so not every cell is live; gives meaningful sparsity.
    W = W * (torch.rand(*shape) > 0.5).float()
    assert torch.equal(W, _round_trip(W))


def test_round_trip_large_matrix():
    """4096 × 4096 at 95% sparsity — scale check."""
    torch.manual_seed(0)
    W = torch.randn(4096, 4096, dtype=torch.float32)
    W = W * (torch.rand(4096, 4096) > 0.95).float()
    assert torch.equal(W, _round_trip(W))


def test_round_trip_all_dense_row():
    """A row with every cell live still round-trips."""
    W = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],     # fully dense row
        [0.0, 0.0, 0.0, 0.0, 0.0],     # fully zero row
    ], dtype=torch.float32)
    assert torch.equal(W, _round_trip(W))


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Invariants
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("padding_ratio", [0.0, 0.2, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_invariants_hold_after_random_construction(padding_ratio, seed):
    """random() output satisfies all 8 invariants across many seeds."""
    p = PaddedCSR.random(50, 80, sparsity=0.85, padding_ratio=padding_ratio, seed=seed)
    p.assert_invariants()  # raises ValueError on any violation


def test_padding_ratio_zero_gives_tight_storage():
    """padding_ratio=0 gives capacity == nnz per row (no padding slots, modulo min 1)."""
    torch.manual_seed(0)
    W = torch.randn(10, 10) * (torch.rand(10, 10) > 0.5).float()
    p = PaddedCSR.from_dense(W, padding_ratio=0.0)
    # Every row: capacity == max(1, nnz).
    for n, cap in zip(list(p.row_nnz), list(p.row_capacity)):
        assert cap == max(1, int(n))


def test_padding_slots_count_matches_expectation():
    """total_capacity - nnz equals the sum of padding slots."""
    p = PaddedCSR.random(100, 100, sparsity=0.9, padding_ratio=0.3, seed=1)
    assert p.padding_slots == p.total_capacity - p.nnz


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — Edge cases
# ─────────────────────────────────────────────────────────────────────

def test_all_zero_matrix():
    """Matrix with zero live entries still constructs and round-trips."""
    W = torch.zeros(5, 5, dtype=torch.float32)
    p = PaddedCSR.from_dense(W)
    assert p.nnz == 0
    # Each row still gets capacity=1 (the min floor from _compute_row_capacity).
    assert all(c == 1 for c in list(p.row_capacity))
    assert torch.equal(W, p.to_dense())


def test_single_element_matrix():
    """1 × 1 matrix with a single live entry."""
    W = torch.tensor([[7.5]], dtype=torch.float32)
    p = PaddedCSR.from_dense(W)
    assert p.shape == (1, 1)
    assert p.nnz == 1
    assert torch.equal(W, p.to_dense())


def test_zero_row_matrix():
    """0 × 5 matrix (no rows at all) constructs and behaves."""
    p = PaddedCSR(nrows=0, ncols=5)
    assert p.shape == (0, 5)
    assert p.nnz == 0
    assert p.total_capacity == 0
    p.assert_invariants()


def test_threshold_filters_small_values():
    """from_dense(threshold=0.5) drops values with |W| ≤ 0.5."""
    W = torch.tensor([
        [0.1, 0.6, 0.0, -0.4],
        [1.0, -2.0, 0.3, 0.0],
    ], dtype=torch.float32)
    p = PaddedCSR.from_dense(W, threshold=0.5)
    # Live cells: (0,1)=0.6, (1,0)=1.0, (1,1)=-2.0. That's 3 live entries.
    assert p.nnz == 3
    # Reconstructed: dropped values appear as 0 in output.
    W_expected = torch.tensor([
        [0.0, 0.6, 0.0, 0.0],
        [1.0, -2.0, 0.0, 0.0],
    ], dtype=torch.float32)
    assert torch.equal(W_expected, p.to_dense())


# ─────────────────────────────────────────────────────────────────────
#  Group 5 — Error paths
# ─────────────────────────────────────────────────────────────────────

def test_from_dense_rejects_non_2d():
    """1-D and 3-D inputs are rejected."""
    with pytest.raises(ValueError, match="2-D"):
        PaddedCSR.from_dense(torch.zeros(5, dtype=torch.float32))
    with pytest.raises(ValueError, match="2-D"):
        PaddedCSR.from_dense(torch.zeros(2, 3, 4, dtype=torch.float32))


def test_from_dense_rejects_negative_padding_ratio():
    """Negative padding_ratio is invalid."""
    W = torch.zeros(3, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="padding_ratio"):
        PaddedCSR.from_dense(W, padding_ratio=-0.1)


def test_random_rejects_out_of_range_sparsity():
    """Sparsity must be in [0, 1)."""
    with pytest.raises(ValueError, match="sparsity"):
        PaddedCSR.random(10, 10, sparsity=-0.1)
    with pytest.raises(ValueError, match="sparsity"):
        PaddedCSR.random(10, 10, sparsity=1.0)
    with pytest.raises(ValueError, match="sparsity"):
        PaddedCSR.random(10, 10, sparsity=1.5)


def test_cpp_constructor_rejects_invariant_violation():
    """The C++ constructor validates and raises on invariant violations."""
    # Deliberately broken: values length 3 but col_indices length 2.
    from sparsecore import _core
    with pytest.raises(ValueError, match="Invariant 2 violated"):
        _core.PaddedCSR(
            nrows=2, ncols=3,
            values=[1.0, 2.0, 3.0],
            col_indices=[0, 1],
            row_start=[0, 2],
            row_nnz=[1, 1],
            row_capacity=[1, 1],
        )


def test_cpp_constructor_rejects_unsorted_columns():
    """Live columns within a row must be sorted ascending."""
    from sparsecore import _core
    with pytest.raises(ValueError, match="sorted"):
        _core.PaddedCSR(
            nrows=1, ncols=5,
            # Row 0: cols [3, 1] — unsorted — should fail invariant 6.
            values=[1.0, 2.0],
            col_indices=[3, 1],
            row_start=[0],
            row_nnz=[2],
            row_capacity=[2],
        )


def test_cpp_constructor_rejects_out_of_range_column():
    """col_indices[i] must be in [0, ncols)."""
    from sparsecore import _core
    with pytest.raises(ValueError, match="Invariant 6"):
        _core.PaddedCSR(
            nrows=1, ncols=3,
            values=[1.0],
            col_indices=[5],  # out of range for ncols=3
            row_start=[0],
            row_nnz=[1],
            row_capacity=[1],
        )


# ─────────────────────────────────────────────────────────────────────
#  Group 6 — Array view mutability
# ─────────────────────────────────────────────────────────────────────

def test_values_is_writable():
    """
    values[] is writable so optimizers can do W.values -= lr * dW_values
    in place. This is the canonical DST training step. All structural
    arrays stay read-only (see test_structural_arrays_are_read_only).
    """
    p = PaddedCSR.random(10, 10, sparsity=0.5, seed=0)
    # Capture a live value so we can restore it after the write test
    original = float(p.values[0])
    p.values[0] = 999.0  # should NOT raise
    assert p.values[0] == 999.0
    # Restore (tests must not leave state leaking across runs)
    p.values[0] = original


def test_structural_arrays_are_read_only():
    """
    col_indices, row_start, row_nnz, row_capacity stay read-only because
    mutating them would break PaddedCSR invariants. Topology mutation
    (growing/pruning connections) happens through explicit methods in
    milestone 4c, never via direct array writes.
    """
    p = PaddedCSR.random(10, 10, sparsity=0.5, seed=0)

    with pytest.raises(ValueError):
        p.col_indices[0] = 999
    with pytest.raises(ValueError):
        p.row_start[0] = 999
    with pytest.raises(ValueError):
        p.row_nnz[0] = 999
    with pytest.raises(ValueError):
        p.row_capacity[0] = 999


def test_views_reflect_cpp_memory():
    """Views have the expected dtypes matching the C++ field types."""
    p = PaddedCSR.random(5, 5, sparsity=0.5, seed=0)
    assert p.values.dtype == np.float32
    assert p.col_indices.dtype == np.int32
    assert p.row_start.dtype == np.int32
    assert p.row_nnz.dtype == np.int32
    assert p.row_capacity.dtype == np.int32
