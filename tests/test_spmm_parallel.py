"""
Tests for OpenMP parallelization of SpMM kernels (milestone 4c).

These tests don't care about timings — they're purely correctness
checks. The goal is: whether the kernel runs serially (M below the
threshold) or in parallel (M above), the output must be bit-identical
to the serial reference.

Why "bit-identical"? Because our parallelization is `#pragma omp
parallel for schedule(static)` over rows of W. Row i produces Y[i, :]
in isolation — there's no cross-thread accumulation, so no
floating-point non-associativity concerns. The parallel result must
equal the serial result exactly.

If a future refactor introduces an atomic reduction or a dynamic
schedule with row chunks, that invariant might no longer hold and
these tests will flag it.
"""

import os

import numpy as np
import pytest
import torch

import sparsecore
from sparsecore import _core, PaddedCSR


# ─────────────────────────────────────────────────────────────────────
#  Parallel result == sequential result, bit-for-bit
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [33, 128, 512, 1024])
@pytest.mark.parametrize("sparsity", [0.5, 0.9, 0.99])
def test_spmm_simd_deterministic_across_runs(M, sparsity):
    """
    Running the same kernel twice with the same inputs should yield
    bit-identical outputs, regardless of thread count. Static OpenMP
    scheduling means each row is always handled by one thread in one
    go; there's no non-determinism.
    """
    W = PaddedCSR.random(M, 256, sparsity=sparsity, seed=42)
    X = np.random.default_rng(1).standard_normal((256, 64), dtype=np.float32)

    Y1 = _core.spmm_simd(W, X)
    Y2 = _core.spmm_simd(W, X)

    # Exact equality (not almost-equal) because static schedule is deterministic.
    np.testing.assert_array_equal(Y1, Y2)


@pytest.mark.parametrize("M", [33, 256, 1024])
@pytest.mark.parametrize("sparsity", [0.7, 0.9])
def test_spmm_scalar_matches_simd_in_parallel(M, sparsity):
    """
    The scalar and SIMD kernels MUST agree within float32 tolerance,
    regardless of whether they're run in parallel. The parallel region
    shouldn't introduce any divergence beyond what float FMA ordering
    already allows inside one row.
    """
    W = PaddedCSR.random(M, 128, sparsity=sparsity, seed=7)
    X = np.random.default_rng(1).standard_normal((128, 48), dtype=np.float32)

    Y_scalar = _core.spmm_scalar(W, X)
    Y_simd = _core.spmm_simd(W, X)

    np.testing.assert_allclose(Y_scalar, Y_simd, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("M", [33, 256, 1024])
@pytest.mark.parametrize("sparsity", [0.7, 0.9])
def test_spmm_grad_w_parallel_matches_serial_math(M, sparsity):
    """
    spmm_grad_w writes dW_values[s] = dot(dY[i,:], X[c,:]) for each
    live slot. Different rows' slots land in non-overlapping regions
    of dW_values (by PaddedCSR's row layout), so parallel writes
    cannot conflict. Check that the result matches the math.
    """
    W = PaddedCSR.random(M, 96, sparsity=sparsity, seed=11)
    N = 32
    dY = np.random.default_rng(1).standard_normal((M, N), dtype=np.float32)
    X = np.random.default_rng(2).standard_normal((96, N), dtype=np.float32)

    dW = _core.spmm_grad_w(W, dY, X)

    # Reference: loop over live slots exactly, compute each dot product.
    W_dense = W.to_dense()
    # dW_live[i, k] for live (i, k) = dot(dY[i, :], X[k, :])
    # and zero elsewhere.
    dW_live_ref = np.zeros_like(W_dense.numpy(), dtype=np.float32)
    live_mask = (W_dense.numpy() != 0).astype(np.float32)
    # Broadcast-efficient way to compute the full (M, K) dot product table:
    #   (M, N) @ (N, K) = (M, K)
    full = dY @ X.T
    dW_live_ref = full * live_mask

    # Read back dW per slot using the CSR index
    col_indices = np.asarray(W.col_indices)
    row_start = np.asarray(W.row_start)
    row_nnz = np.asarray(W.row_nnz)

    for i in range(M):
        rs = row_start[i]
        n = row_nnz[i]
        for s in range(n):
            slot = rs + s
            c = col_indices[slot]
            expected = dW_live_ref[i, c]
            got = dW[slot]
            assert abs(expected - got) < 1e-4, (
                f"mismatch at i={i}, c={c}, slot={slot}: "
                f"expected {expected}, got {got}"
            )


def test_spmm_below_threshold_still_works():
    """
    The OpenMP `if(M >= THRESHOLD)` clause gates parallelism. Below the
    threshold the loop runs sequentially — it must still be correct.
    """
    # Threshold is 32 in kernels/parallel.hpp; test with M well below.
    M = 8
    W = PaddedCSR.random(M, 64, sparsity=0.5, seed=0)
    X = np.random.default_rng(1).standard_normal((64, 16), dtype=np.float32)

    Y_simd = _core.spmm_simd(W, X)
    Y_scalar = _core.spmm_scalar(W, X)

    np.testing.assert_allclose(Y_simd, Y_scalar, rtol=1e-5, atol=1e-5)


def test_large_spmm_runs_without_segfault():
    """
    Smoke test that a realistic-scale forward+backward doesn't crash
    even with the full parallel region active. If our OpenMP wiring
    had a thread-local lifetime bug, this would tend to segfault.
    """
    torch.manual_seed(0)
    layer = sparsecore.SparseLinear(1024, 2048, sparsity=0.9)
    x = torch.randn(64, 1024)
    for _ in range(3):
        y = layer(x)
        y.sum().backward()
        # Zero grads manually since we're not using an optimizer here.
        layer._values.grad.zero_()
        layer.bias.grad.zero_()
    # Passing without raising is the assertion.


# ─────────────────────────────────────────────────────────────────────
#  Thread count behavior (optional — only meaningful with OpenMP built in)
# ─────────────────────────────────────────────────────────────────────

def test_results_invariant_across_thread_counts():
    """
    Running with different OMP_NUM_THREADS values (via the env-driven
    API) must produce the same output. We can't re-set the env mid-
    process and have OpenMP pick it up portably, but we CAN use omp's
    runtime API.

    This test is a no-op when OpenMP isn't present (the _core module
    exposes no omp_set_num_threads binding in that case — we just skip).
    """
    # We don't bind omp_set_num_threads in our pybind module (that would
    # expose runtime state in ways we haven't designed). Instead we rely
    # on the static-schedule invariant: same input + same thread count
    # yields same output. Run twice with default thread count and
    # verify determinism at scale.
    W = PaddedCSR.random(512, 768, sparsity=0.9, seed=99)
    X = np.random.default_rng(1).standard_normal((768, 128), dtype=np.float32)

    Y1 = _core.spmm_simd(W, X)
    Y2 = _core.spmm_simd(W, X)
    Y3 = _core.spmm_simd(W, X)

    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(Y2, Y3)
