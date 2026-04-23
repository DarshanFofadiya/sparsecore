"""
Milestone 3c Oracle tests — scalar SpMM correctness.

Covers:
  - Oracle correctness across shapes and sparsities (W @ X matches torch.matmul)
  - Edge cases (empty rows, fully-sparse W, size-1 dims, padding slots ignored)
  - Error paths (wrong dtype, shape mismatch, non-CPU device, wrong types)
  - Dtype + contiguity coercion (float64 input, transposed-view input)

Oracle: torch.matmul(W_dense, X) with rtol=atol=1e-5.
        float32 matmul accumulates in float32, so 1e-5 is tight-but-achievable.

Design doc: docs/design/spmm.md
Run with:  pytest tests/test_spmm.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import sparselab
from sparselab import PaddedCSR


# ─────────────────────────────────────────────────────────────────────
#  Fixtures + helpers
# ─────────────────────────────────────────────────────────────────────

# Suppress the beta-state warning from torch.sparse_csr — we know, we use it
# on purpose. Also suppress the "NumPy array is not writable" warning that
# fires when torch.from_numpy wraps our zero-copy view of C++ memory.
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
    """
    Build a random (M, K) sparse weight as both PaddedCSR and dense Tensor.

    Returns (W_csr, W_dense) where W_csr is the PaddedCSR we'll feed to
    spmm() and W_dense is the ground-truth tensor we'll use as the oracle.
    The two are numerically identical (PaddedCSR.from_dense round-trips
    losslessly).
    """
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(M, K, generator=gen, dtype=torch.float32)
    # Zero out (sparsity) fraction of entries uniformly at random.
    keep_mask = torch.rand(M, K, generator=gen) >= sparsity
    W_dense = W * keep_mask.float()
    W_csr = PaddedCSR.from_dense(W_dense)
    return W_csr, W_dense


# ─────────────────────────────────────────────────────────────────────
#  Group 1 — Oracle correctness across shapes and sparsities
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M,K,N",
    [
        (32, 32, 32),      # square, small
        (128, 64, 16),     # wider-than-tall W, narrow X
        (64, 128, 256),    # tall-narrow W, wide X (transformer-ish)
        (7, 13, 5),        # all prime, none divisible by SIMD lanes
        (1, 1, 1),         # smallest valid case
    ],
    ids=["square_32", "wide_W", "narrow_W", "prime_dims", "smallest"],
)
@pytest.mark.parametrize(
    "sparsity",
    [0.0, 0.5, 0.9, 0.99],
    ids=["dense", "50pct", "90pct", "99pct"],
)
def test_spmm_matches_dense_matmul(M, K, N, sparsity):
    """
    Our scalar SpMM must agree with torch.matmul(W_dense, X) to within
    float32 tolerance across shapes and sparsities.
    """
    # Skip 99% sparsity on the 1x1x1 case — there's no meaningful way to
    # have 99% of 1 entry be zero. (floor(1 * 0.01) = 0, torch may keep all.)
    if M * K < 4 and sparsity >= 0.9:
        pytest.skip("Not enough cells for >=90% sparsity to be meaningful.")

    W_csr, W_dense = _make_sparse_W(M, K, sparsity=sparsity, seed=42)
    X = torch.randn(K, N, dtype=torch.float32)

    Y_ours = sparselab.spmm(W_csr, X)
    Y_oracle = W_dense @ X

    assert Y_ours.shape == (M, N)
    assert Y_ours.dtype == torch.float32
    assert torch.allclose(Y_ours, Y_oracle, rtol=1e-5, atol=1e-5), (
        f"SpMM mismatch: max diff = "
        f"{(Y_ours - Y_oracle).abs().max().item():.3e}"
    )


def test_spmm_returns_cpu_tensor():
    """Return type is a CPU torch.Tensor regardless of input device hints."""
    W_csr, _ = _make_sparse_W(16, 16, sparsity=0.5)
    X = torch.randn(16, 8, dtype=torch.float32)
    Y = sparselab.spmm(W_csr, X)
    assert isinstance(Y, torch.Tensor)
    assert Y.device.type == "cpu"


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Edge cases
# ─────────────────────────────────────────────────────────────────────

def test_spmm_fully_sparse_W_returns_zeros():
    """If W is all zeros, Y must be exactly zero (no FMAs happened)."""
    W_dense = torch.zeros(10, 20, dtype=torch.float32)
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(20, 5, dtype=torch.float32)
    Y = sparselab.spmm(W_csr, X)
    assert torch.equal(Y, torch.zeros(10, 5, dtype=torch.float32))


def test_spmm_all_rows_empty():
    """Every row of W has 0 live entries → Y is all zeros."""
    # Build an all-zero PaddedCSR via the constructor (avoids from_dense
    # potentially optimizing the zero case).
    W_csr = PaddedCSR(nrows=8, ncols=16)  # empty-constructor path
    X = torch.randn(16, 4, dtype=torch.float32)
    Y = sparselab.spmm(W_csr, X)
    assert Y.shape == (8, 4)
    assert torch.equal(Y, torch.zeros(8, 4, dtype=torch.float32))


def test_spmm_single_live_entry():
    """
    W has exactly 1 live entry → Y has exactly 1 nonzero row, and its
    value is W[i,c] * X[c, :] for that (i, c).
    """
    W_dense = torch.zeros(4, 6, dtype=torch.float32)
    W_dense[2, 3] = 2.5  # the only live entry
    W_csr = PaddedCSR.from_dense(W_dense)

    X = torch.randn(6, 3, dtype=torch.float32)
    Y = sparselab.spmm(W_csr, X)

    # Rows 0, 1, 3 must be exactly zero (no live entry contributed).
    assert torch.equal(Y[0], torch.zeros(3))
    assert torch.equal(Y[1], torch.zeros(3))
    assert torch.equal(Y[3], torch.zeros(3))
    # Row 2 = 2.5 * X[3, :]
    assert torch.allclose(Y[2], 2.5 * X[3, :], rtol=1e-6, atol=1e-6)


def test_spmm_N_equals_1():
    """
    Common case during inference: X is a column vector (N=1). Must still
    match the oracle.
    """
    W_csr, W_dense = _make_sparse_W(32, 24, sparsity=0.8, seed=1)
    x = torch.randn(24, 1, dtype=torch.float32)
    y_ours = sparselab.spmm(W_csr, x)
    y_oracle = W_dense @ x
    assert y_ours.shape == (32, 1)
    assert torch.allclose(y_ours, y_oracle, rtol=1e-5, atol=1e-5)


def test_spmm_ignores_padding_slots():
    """
    Padding slots (col_idx=-1, value=0.0) must never be read. We use a
    padding_ratio > 0 to ensure padding slots exist; if the kernel
    accidentally walked them it would either (a) read col=-1 as a huge
    negative out-of-bounds index into X and segfault, or (b) read
    value=0.0 and produce correct numbers by accident. Our oracle
    comparison catches case (b).
    """
    W_dense = torch.randn(16, 32) * (torch.rand(16, 32) > 0.7).float()
    # padding_ratio=1.0 means every row has 100% extra slots — so nearly
    # half the values array is padding. If we were accidentally walking
    # capacity instead of nnz this would blow up.
    W_csr = PaddedCSR.from_dense(W_dense, padding_ratio=1.0)
    X = torch.randn(32, 8, dtype=torch.float32)

    Y_ours = sparselab.spmm(W_csr, X)
    Y_oracle = W_dense @ X
    assert torch.allclose(Y_ours, Y_oracle, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Error paths
# ─────────────────────────────────────────────────────────────────────

def test_spmm_rejects_1d_X():
    """X must be 2-D. A 1-D 'column vector' is ambiguous; reject it."""
    W_csr, _ = _make_sparse_W(8, 8, sparsity=0.5)
    x_1d = torch.randn(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="2-D"):
        sparselab.spmm(W_csr, x_1d)


def test_spmm_rejects_shape_mismatch():
    """W.ncols must equal X.shape[0]."""
    W_csr, _ = _make_sparse_W(M=8, K=16, sparsity=0.5)
    X_wrong = torch.randn(17, 4, dtype=torch.float32)  # K' = 17, not 16
    with pytest.raises(ValueError, match="shape|match|ncols"):
        sparselab.spmm(W_csr, X_wrong)


def test_spmm_rejects_wrong_W_type():
    """W must be a PaddedCSR, not a plain Tensor or anything else."""
    W_tensor = torch.randn(8, 16)
    X = torch.randn(16, 4)
    with pytest.raises(TypeError, match="PaddedCSR"):
        sparselab.spmm(W_tensor, X)


def test_spmm_rejects_wrong_X_type():
    """X must be a torch.Tensor, not a numpy array or list."""
    W_csr, _ = _make_sparse_W(8, 16, sparsity=0.5)
    with pytest.raises(TypeError, match="torch.Tensor"):
        sparselab.spmm(W_csr, np.random.randn(16, 4).astype(np.float32))


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="No non-CPU device available to test the rejection path",
)
def test_spmm_rejects_non_cpu_X():
    """CPU-only for v0.1. A GPU/MPS tensor must be rejected explicitly."""
    W_csr, _ = _make_sparse_W(8, 16, sparsity=0.5)
    device = "cuda" if torch.cuda.is_available() else "mps"
    X = torch.randn(16, 4, device=device)
    with pytest.raises(RuntimeError, match="CPU"):
        sparselab.spmm(W_csr, X)


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — Dtype + contiguity coercion
# ─────────────────────────────────────────────────────────────────────

def test_spmm_accepts_float64_X():
    """
    float64 input should be silently coerced to float32 and still produce
    a correct answer (within float32 tolerance).
    """
    W_csr, W_dense = _make_sparse_W(16, 24, sparsity=0.7, seed=2)
    X64 = torch.randn(24, 8, dtype=torch.float64)

    Y_ours = sparselab.spmm(W_csr, X64)
    Y_oracle = W_dense @ X64.float()  # oracle in float32 for a fair compare

    assert Y_ours.dtype == torch.float32
    assert torch.allclose(Y_ours, Y_oracle, rtol=1e-5, atol=1e-5)


def test_spmm_accepts_non_contiguous_X():
    """
    A transposed view of a tensor is non-contiguous. Our binding calls
    .contiguous() under the hood so this must work.
    """
    W_csr, W_dense = _make_sparse_W(10, 20, sparsity=0.5, seed=3)
    # Build X as a transposed view: the storage is (8, 20), the logical
    # tensor is (20, 8) and non-contiguous.
    X_base = torch.randn(8, 20, dtype=torch.float32)
    X_view = X_base.T  # shape (20, 8), stride (1, 20) — non-contiguous

    assert not X_view.is_contiguous(), "sanity: view should be non-contiguous"

    Y_ours = sparselab.spmm(W_csr, X_view)
    Y_oracle = W_dense @ X_view
    assert torch.allclose(Y_ours, Y_oracle, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group 5 — Stress / property checks
# ─────────────────────────────────────────────────────────────────────

def test_spmm_idempotent_under_repeat():
    """
    Same inputs produce the same output every call. Catches any hidden
    state the kernel might accidentally leak between calls.
    """
    W_csr, _ = _make_sparse_W(32, 32, sparsity=0.8, seed=7)
    X = torch.randn(32, 16, dtype=torch.float32)

    Y1 = sparselab.spmm(W_csr, X)
    Y2 = sparselab.spmm(W_csr, X)
    Y3 = sparselab.spmm(W_csr, X)
    assert torch.equal(Y1, Y2)
    assert torch.equal(Y2, Y3)


def test_spmm_different_seeds_differ():
    """
    Sanity check: two genuinely different random weight matrices produce
    different outputs (catches the "kernel returns zeros" failure mode).
    """
    W1, _ = _make_sparse_W(16, 16, sparsity=0.5, seed=1)
    W2, _ = _make_sparse_W(16, 16, sparsity=0.5, seed=2)
    X = torch.randn(16, 4, dtype=torch.float32)
    Y1 = sparselab.spmm(W1, X)
    Y2 = sparselab.spmm(W2, X)
    assert not torch.allclose(Y1, Y2)
