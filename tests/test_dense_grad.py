"""
Tests for dense_grad kernel (milestone 4f, C++ chunk).

dense_grad computes the full (M, K) dense gradient G = dY @ X.T.
This is a plain dense matmul — no sparse structure involved. The
tests are mostly oracle tests against numpy's dense matmul.
"""

import numpy as np
import pytest

import sparselab
from sparselab import _core


# ─────────────────────────────────────────────────────────────────────
#  Oracle correctness
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M,K,N", [
    (5, 7, 4),        # tiny
    (32, 64, 16),     # medium (runs parallel)
    (128, 256, 32),   # larger
    (1, 1, 1),        # edge: scalar
    (1, 8, 4),        # edge: single row
    (8, 1, 4),        # edge: single col
])
def test_dense_grad_matches_numpy_reference(M, K, N):
    np.random.seed(0)
    dY = np.random.randn(M, N).astype(np.float32)
    X = np.random.randn(K, N).astype(np.float32)

    G_ours = _core.dense_grad(dY, X)
    G_ref = dY @ X.T

    assert G_ours.shape == (M, K)
    np.testing.assert_allclose(G_ours, G_ref, rtol=1e-5, atol=1e-5)


def test_dense_grad_empty_shape():
    """Empty matrices shouldn't crash."""
    G = _core.dense_grad(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((5, 4), dtype=np.float32),
    )
    assert G.shape == (0, 5)


def test_dense_grad_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        _core.dense_grad(
            np.zeros(5, dtype=np.float32),
            np.zeros((3, 5), dtype=np.float32),
        )


def test_dense_grad_rejects_inner_dim_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        _core.dense_grad(
            np.zeros((4, 6), dtype=np.float32),
            np.zeros((3, 5), dtype=np.float32),  # N=5 but dY has N=6
        )


# ─────────────────────────────────────────────────────────────────────
#  Parallel correctness at larger scale
# ─────────────────────────────────────────────────────────────────────

def test_dense_grad_parallel_deterministic():
    """Two runs with the same inputs must produce bit-identical outputs.
    Same static-schedule invariant as the SpMM kernels."""
    np.random.seed(42)
    dY = np.random.randn(128, 32).astype(np.float32)
    X = np.random.randn(64, 32).astype(np.float32)

    G1 = _core.dense_grad(dY, X)
    G2 = _core.dense_grad(dY, X)
    np.testing.assert_array_equal(G1, G2)


def test_dense_grad_at_transformer_ffn_scale():
    """Bigger test: 512 x 2048 matmul. Mostly a 'doesn't crash' smoke."""
    np.random.seed(0)
    dY = np.random.randn(512, 128).astype(np.float32) * 0.1
    X = np.random.randn(2048, 128).astype(np.float32) * 0.1

    G_ours = _core.dense_grad(dY, X)
    G_ref = dY @ X.T
    assert G_ours.shape == (512, 2048)
    # Larger scale → more float accumulation noise, relax tolerance.
    np.testing.assert_allclose(G_ours, G_ref, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────
#  Comparing against spmm_grad_w at live positions
# ─────────────────────────────────────────────────────────────────────

def test_dense_grad_matches_spmm_grad_w_at_live_positions():
    """spmm_grad_w gives us gradients at live slots (aligned with
    W.values). dense_grad gives us the FULL (M, K) gradient. For
    live positions these MUST agree — same math."""
    np.random.seed(0)
    W = sparselab.PaddedCSR.random(8, 12, sparsity=0.6, seed=0)
    dY = np.random.randn(8, 6).astype(np.float32)
    X = np.random.randn(12, 6).astype(np.float32)

    # Sparse path: gradient per live slot (length = total_capacity)
    dW_live = _core.spmm_grad_w(W, dY, X)

    # Dense path: gradient at every (i, k)
    G = _core.dense_grad(dY, X)

    # For each live slot s at (i, c), dW_live[s] should equal G[i, c].
    col_indices = np.asarray(W.col_indices)
    row_start = np.asarray(W.row_start)
    row_nnz = np.asarray(W.row_nnz)

    for i in range(W.nrows):
        start = int(row_start[i])
        n_live = int(row_nnz[i])
        for k in range(n_live):
            slot = start + k
            c = int(col_indices[slot])
            # Live-slot gradient must match dense gradient at (i, c).
            np.testing.assert_allclose(
                dW_live[slot], G[i, c],
                rtol=1e-5, atol=1e-5,
                err_msg=f"Mismatch at row {i}, col {c}, slot {slot}"
            )
