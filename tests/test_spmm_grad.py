"""
Milestone 4a-iv Oracle tests — spmm_grad_w correctness.

What we verify:
  - dW_values at live slots matches the dense PyTorch autograd oracle
  - Padding slots stay at 0.0 (critical for W.values -= lr * dW safety)
  - Output length equals W.total_capacity (aligns with W.values)
  - Shape invariants: dY (M, N), X (K, N), W (M, K)
  - Edge cases: empty W, empty rows, single entry, size-1 dims
  - Error paths: shape mismatches, wrong ndim
  - dtype coercion: float64 dY/X silently work

Oracle: dense PyTorch autograd — `W_dense_grad.backward(dY)` then read
        W_dense_grad.grad[i, c] for each live slot (i, c).
Tolerance: rtol=atol=1e-5 (float32 matmul precision).

Design doc: docs/design/spmm_backward.md §1.4
Run with:  pytest tests/test_spmm_grad.py -v
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
#  Kernel parametrization fixture
#
#  Every correctness test runs against BOTH the scalar and NEON SIMD
#  kernels to guarantee they agree within 1e-5. On x86, _simd routes
#  to scalar so the parametrization still runs but exercises the same
#  code path twice — that's cheap and keeps the test file portable.
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["scalar", "simd"])
def kernel_fn(request):
    """Parametrized dW kernel — resolves to scalar or simd at test time."""
    if request.param == "scalar":
        return _core.spmm_grad_w
    return _core.spmm_grad_w_simd


# ─────────────────────────────────────────────────────────────────────
#  Helper: dense PyTorch backward as oracle
# ─────────────────────────────────────────────────────────────────────

def _dense_oracle(W_dense: torch.Tensor, X: torch.Tensor, dY: torch.Tensor):
    """
    Compute dL/dW as PyTorch's dense autograd would, returning the full
    (M, K) gradient tensor.
    """
    W = W_dense.clone().requires_grad_(True)
    Y = W @ X
    Y.backward(dY)
    return W.grad  # shape (M, K)


def _extract_live_slots(W_csr: PaddedCSR):
    """Return (slots, rows, cols) for each live entry of W_csr."""
    col_indices = np.array(W_csr.col_indices)
    row_start = np.array(W_csr.row_start)
    row_nnz = np.array(W_csr.row_nnz)
    slots, rows, cols = [], [], []
    for i in range(W_csr.nrows):
        start = int(row_start[i])
        n_live = int(row_nnz[i])
        for s in range(n_live):
            slot = start + s
            slots.append(slot)
            rows.append(i)
            cols.append(int(col_indices[slot]))
    return np.array(slots), np.array(rows), np.array(cols)


def _padding_slots(W_csr: PaddedCSR):
    """Return the indices of all padding slots in W_csr.values."""
    row_start = np.array(W_csr.row_start)
    row_nnz = np.array(W_csr.row_nnz)
    row_capacity = np.array(W_csr.row_capacity)
    padding = []
    for i in range(W_csr.nrows):
        start = int(row_start[i])
        n_live = int(row_nnz[i])
        cap = int(row_capacity[i])
        for p in range(n_live, cap):
            padding.append(start + p)
    return np.array(padding)


# ─────────────────────────────────────────────────────────────────────
#  Group 1 — correctness vs dense autograd
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M,K,N,sparsity",
    [
        (4, 5, 3, 0.5),         # tiny, for eyeball debugging
        (16, 16, 8, 0.5),       # square, medium sparsity
        (32, 16, 4, 0.8),       # non-square, high sparsity
        (8, 32, 16, 0.9),       # wide K
        (64, 64, 32, 0.95),     # large, very sparse
        (1, 16, 8, 0.0),        # single output row, fully dense
        (16, 1, 8, 0.0),        # single input col, fully dense
        (16, 16, 1, 0.5),       # N=1 (column-vector upstream gradient)
    ],
    ids=[
        "tiny", "square_50", "nonsquare_80", "wide_K_90",
        "large_95", "M_1", "K_1", "N_1",
    ],
)
def test_spmm_grad_w_matches_dense_autograd(M, K, N, sparsity, kernel_fn):
    """For each live slot, our dW_values must match dense autograd's grad."""
    torch.manual_seed(42)
    W_dense = torch.randn(M, K, dtype=torch.float32)
    mask = torch.rand(M, K) >= sparsity
    W_dense = W_dense * mask.float()

    X = torch.randn(K, N, dtype=torch.float32)
    dY = torch.randn(M, N, dtype=torch.float32)

    W_csr = PaddedCSR.from_dense(W_dense)

    # Oracle: full dense gradient
    dL_dW_dense = _dense_oracle(W_dense, X, dY)

    # Ours: compact gradient at live slots only
    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())

    # Shape: must align with W.values
    assert dW_values.shape == (W_csr.total_capacity,), (
        f"dW_values shape {dW_values.shape} != total_capacity "
        f"{(W_csr.total_capacity,)}"
    )

    # Each live slot's gradient matches the dense oracle at that (i, c)
    slots, rows, cols = _extract_live_slots(W_csr)
    if len(slots) > 0:
        ours = dW_values[slots]
        oracle = dL_dW_dense[rows, cols].numpy()
        assert np.allclose(ours, oracle, rtol=1e-5, atol=1e-5), (
            f"dW mismatch at live slots. Max diff: "
            f"{np.abs(ours - oracle).max():.3e}"
        )


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — padding slots stay zero
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("padding_ratio", [0.0, 0.2, 0.5, 1.0])
def test_padding_slots_stay_zero(padding_ratio, kernel_fn):
    """
    Padding slots must stay at exactly 0.0 regardless of padding_ratio.
    If any padding slot contained random junk, W.values -= lr * dW would
    corrupt weights at positions that aren't even supposed to be active.
    """
    torch.manual_seed(1)
    W_dense = torch.randn(10, 12) * (torch.rand(10, 12) >= 0.6).float()
    X = torch.randn(12, 5, dtype=torch.float32)
    dY = torch.randn(10, 5, dtype=torch.float32)

    W_csr = PaddedCSR.from_dense(W_dense, padding_ratio=padding_ratio)
    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())

    padding = _padding_slots(W_csr)
    if len(padding) > 0:
        padding_vals = dW_values[padding]
        # Must be exactly zero, not "close to zero" — the memset guarantees it.
        assert np.all(padding_vals == 0.0), (
            f"Padding slots are not zero. Max abs value: "
            f"{np.abs(padding_vals).max():.3e}"
        )


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — edge cases
# ─────────────────────────────────────────────────────────────────────

def test_empty_W_returns_empty_grad(kernel_fn):
    """nnz = 0 → dW_values is length total_capacity but all zero."""
    W_csr = PaddedCSR(nrows=8, ncols=16)  # empty ctor → nnz = 0, cap = 0
    X = torch.randn(16, 5, dtype=torch.float32)
    dY = torch.randn(8, 5, dtype=torch.float32)
    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())
    assert dW_values.shape == (0,) or np.all(dW_values == 0.0)


def test_all_rows_empty_W(kernel_fn):
    """A W with all zero entries still yields an all-zero gradient."""
    W_dense = torch.zeros(6, 10, dtype=torch.float32)
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(10, 4, dtype=torch.float32)
    dY = torch.randn(6, 4, dtype=torch.float32)
    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())
    assert np.all(dW_values == 0.0)


def test_single_live_entry(kernel_fn):
    """One live entry at (i=2, c=3): dW_values[slot] = dY[2,:]·X[3,:]."""
    W_dense = torch.zeros(5, 7, dtype=torch.float32)
    W_dense[2, 3] = 1.0  # W[i,c] value itself doesn't matter for grad
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(7, 4, dtype=torch.float32)
    dY = torch.randn(5, 4, dtype=torch.float32)

    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())
    slots, rows, cols = _extract_live_slots(W_csr)
    assert len(slots) == 1
    expected = (dY[2] * X[3]).sum().item()
    assert dW_values[slots[0]] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_fully_dense_W(kernel_fn):
    """
    A fully dense W (0% sparsity) still works; every slot has a gradient.
    This is the stress case — every (i, k) pair requires a full dot product.
    """
    torch.manual_seed(3)
    W_dense = torch.randn(8, 8, dtype=torch.float32)  # no masking
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(8, 4, dtype=torch.float32)
    dY = torch.randn(8, 4, dtype=torch.float32)

    dL_dW_dense = _dense_oracle(W_dense, X, dY)
    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())

    slots, rows, cols = _extract_live_slots(W_csr)
    ours = dW_values[slots]
    oracle = dL_dW_dense[rows, cols].numpy()
    assert np.allclose(ours, oracle, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — error paths
# ─────────────────────────────────────────────────────────────────────

def test_rejects_1d_dY(kernel_fn):
    W_csr = PaddedCSR.from_dense(torch.randn(4, 4))
    X = torch.randn(4, 2, dtype=torch.float32)
    dY_bad = torch.randn(4, dtype=torch.float32)  # 1-D
    with pytest.raises(ValueError, match="2-D"):
        kernel_fn(W_csr, dY_bad.numpy(), X.numpy())


def test_rejects_1d_X(kernel_fn):
    W_csr = PaddedCSR.from_dense(torch.randn(4, 4))
    dY = torch.randn(4, 2, dtype=torch.float32)
    X_bad = torch.randn(4, dtype=torch.float32)  # 1-D
    with pytest.raises(ValueError, match="2-D"):
        kernel_fn(W_csr, dY.numpy(), X_bad.numpy())


def test_rejects_dY_row_mismatch(kernel_fn):
    """dY.shape[0] != W.nrows → error."""
    W_csr = PaddedCSR.from_dense(torch.randn(4, 6))
    dY_bad = torch.randn(5, 3, dtype=torch.float32)  # 5 != 4
    X = torch.randn(6, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="nrows|dY.shape"):
        kernel_fn(W_csr, dY_bad.numpy(), X.numpy())


def test_rejects_X_row_mismatch(kernel_fn):
    """X.shape[0] != W.ncols → error."""
    W_csr = PaddedCSR.from_dense(torch.randn(4, 6))
    dY = torch.randn(4, 3, dtype=torch.float32)
    X_bad = torch.randn(7, 3, dtype=torch.float32)  # 7 != 6
    with pytest.raises(ValueError, match="ncols|X.shape"):
        kernel_fn(W_csr, dY.numpy(), X_bad.numpy())


def test_rejects_N_mismatch(kernel_fn):
    """dY.shape[1] != X.shape[1] → error."""
    W_csr = PaddedCSR.from_dense(torch.randn(4, 6))
    dY = torch.randn(4, 3, dtype=torch.float32)
    X_bad = torch.randn(6, 5, dtype=torch.float32)  # 5 != 3
    with pytest.raises(ValueError, match="N|inner|shape"):
        kernel_fn(W_csr, dY.numpy(), X_bad.numpy())


# ─────────────────────────────────────────────────────────────────────
#  Group 5 — dtype handling
# ─────────────────────────────────────────────────────────────────────

def test_accepts_float64_inputs(kernel_fn):
    """float64 dY/X should be silently coerced to float32 (forcecast)."""
    torch.manual_seed(5)
    W_dense = torch.randn(8, 8) * (torch.rand(8, 8) >= 0.5).float()
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(8, 4, dtype=torch.float64)
    dY = torch.randn(8, 4, dtype=torch.float64)

    dW_values = kernel_fn(W_csr, dY.numpy(), X.numpy())
    # Should complete without error; correctness tested elsewhere
    assert dW_values.dtype == np.float32
    assert dW_values.shape == (W_csr.total_capacity,)


# ─────────────────────────────────────────────────────────────────────
#  Group 6 — determinism + idempotence
# ─────────────────────────────────────────────────────────────────────

def test_deterministic(kernel_fn):
    """Same inputs → bit-identical outputs every call."""
    torch.manual_seed(7)
    W_dense = torch.randn(10, 12) * (torch.rand(10, 12) >= 0.5).float()
    W_csr = PaddedCSR.from_dense(W_dense)
    X = torch.randn(12, 6, dtype=torch.float32).numpy()
    dY = torch.randn(10, 6, dtype=torch.float32).numpy()

    g1 = kernel_fn(W_csr, dY, X)
    g2 = kernel_fn(W_csr, dY, X)
    g3 = kernel_fn(W_csr, dY, X)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(g2, g3)
