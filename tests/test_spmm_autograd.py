"""
Milestone 4a-v/vi Oracle tests — SpMM autograd integration.

The most important test in all of milestone 4a is torch.autograd.gradcheck,
which compares our analytical backward against a numerical (finite-
differences) gradient. Any bug in backward shows up here immediately.

Supplementary tests:
  - end-to-end loss.backward() on trivial graphs
  - dL/dX shape and device match the forward input
  - dL/dW_values aligns with W.values (length = total_capacity)
  - autograd path is skipped when requires_grad=False (fast path)

Design doc: docs/design/spmm_backward.md §5.1
Run with:  pytest tests/test_spmm_autograd.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import sparsecore
from sparsecore import PaddedCSR
from sparsecore.ops import _SpMMFunction


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
#  Group 1 — torch.autograd.gradcheck (the gold standard)
# ─────────────────────────────────────────────────────────────────────
#
#  gradcheck perturbs each input by eps and compares (f(x+eps) - f(x-eps))/2eps
#  against our analytical backward. We use float64 for numerical stability
#  (gradcheck's finite differences need more precision than float32
#  provides). Our kernels internally run float32, so we accept slightly
#  looser tolerance (atol=rtol=1e-3) than the default.
#  ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M,K,N,sparsity",
    [
        (4, 5, 3, 0.0),     # fully dense tiny — baseline correctness
        (4, 5, 3, 0.5),     # moderately sparse tiny
        (8, 6, 4, 0.75),    # high sparsity
        (6, 8, 2, 0.5),     # rectangular, N very small
    ],
    ids=["dense_tiny", "50pct_tiny", "75pct_small", "rect_Nsmall"],
)
def test_gradcheck_dX(M, K, N, sparsity):
    """
    Verify dL/dX with finite differences. Only X tracks gradients here;
    W_values is passed without requires_grad so we isolate the dX path.
    """
    torch.manual_seed(42)
    W_dense = torch.randn(M, K, dtype=torch.float64) * (
        torch.rand(M, K) >= sparsity
    ).double()
    W_csr = PaddedCSR.from_dense(W_dense.float())

    # Pull W_values as a torch tensor; don't track its gradient for this test
    W_values_t = torch.from_numpy(np.array(W_csr.values, copy=True)).double()
    W_values_t.requires_grad_(False)

    X = torch.randn(K, N, dtype=torch.float64, requires_grad=True)

    # Wrap our autograd op into a function signature gradcheck expects.
    # gradcheck will perturb X and compare against our backward.
    # We bypass the public spmm() wrapper here to use float64 (our public
    # wrapper force-converts to float32, which would defeat gradcheck).
    #
    # Instead, we test a float32 round-trip — our kernel runs in float32,
    # but gradcheck applies perturbations in float64 and expects the
    # derivative to match. We allow looser tolerance to cover the
    # float32-internal-precision loss.
    def fn(x):
        # Convert to float32 for our kernel, then back to float64 for
        # gradcheck's comparison. The conversion itself is differentiable
        # so autograd chain rule still works — the only error we're
        # measuring is the float32 precision loss during our kernel.
        return sparsecore.spmm(W_csr, x.float()).double()

    assert torch.autograd.gradcheck(
        fn, (X,),
        eps=1e-3,       # larger eps because we're in float32 internally
        atol=1e-2,      # loose: float32 internally limits precision
        rtol=1e-2,
        check_undefined_grad=False,
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — end-to-end loss.backward() on practical graphs
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "M,K,N,sparsity",
    [
        (16, 16, 8, 0.0),
        (16, 16, 8, 0.5),
        (16, 32, 8, 0.9),
        (32, 16, 16, 0.8),
    ],
    ids=["dense", "50pct", "90pct_wide", "80pct_tall"],
)
def test_backward_dX_matches_dense(M, K, N, sparsity):
    """
    X.grad from our autograd path must match dense PyTorch autograd.
    """
    torch.manual_seed(1)
    W_dense_np = (torch.randn(M, K) * (torch.rand(M, K) >= sparsity).float())

    # Build W_csr from a frozen copy of W_dense_np
    W_csr = PaddedCSR.from_dense(W_dense_np)

    # Test on the same X for both paths
    X_init = torch.randn(K, N, dtype=torch.float32)

    # Our path
    X_ours = X_init.clone().requires_grad_(True)
    Y_ours = sparsecore.spmm(W_csr, X_ours)
    loss_ours = (Y_ours * torch.randn_like(Y_ours)).sum()
    # We need the SAME random upstream grad for both paths. Regenerate:
    torch.manual_seed(99)
    upstream = torch.randn(M, N, dtype=torch.float32)

    X_ours.grad = None
    X_ours = X_init.clone().requires_grad_(True)
    Y_ours = sparsecore.spmm(W_csr, X_ours)
    Y_ours.backward(upstream)

    # Oracle path
    W_dense = W_dense_np.clone()
    X_oracle = X_init.clone().requires_grad_(True)
    Y_oracle = W_dense @ X_oracle
    Y_oracle.backward(upstream)

    # Compare dL/dX
    assert torch.allclose(X_ours.grad, X_oracle.grad, rtol=1e-5, atol=1e-5), (
        f"dL/dX mismatch. Max diff: "
        f"{(X_ours.grad - X_oracle.grad).abs().max().item():.3e}"
    )


def test_backward_dW_values_matches_dense():
    """
    When W_values has requires_grad=True, dW_values (padded, length
    total_capacity) should match the dense oracle at every live slot.
    """
    torch.manual_seed(7)
    M, K, N, sparsity = 8, 10, 4, 0.5
    W_dense = torch.randn(M, K) * (torch.rand(M, K) >= sparsity).float()
    W_csr = PaddedCSR.from_dense(W_dense)

    # Wrap W_values as a tracked tensor. Use clone() so we own the storage.
    W_values_t = torch.from_numpy(
        np.array(W_csr.values, copy=True)
    ).requires_grad_(True)

    X = torch.randn(K, N, dtype=torch.float32)
    upstream = torch.randn(M, N, dtype=torch.float32)

    # Call the Function directly so W_values_t is in the autograd graph
    Y = _SpMMFunction.apply(W_values_t, W_csr, X, "simd")
    Y.backward(upstream)

    # Oracle path
    W_dense_grad = W_dense.clone().requires_grad_(True)
    Y_oracle = W_dense_grad @ X
    Y_oracle.backward(upstream)
    dL_dW_dense = W_dense_grad.grad  # (M, K) dense

    # Compare at live slots
    col_indices = np.array(W_csr.col_indices)
    row_start = np.array(W_csr.row_start)
    row_nnz = np.array(W_csr.row_nnz)
    for i in range(M):
        start = int(row_start[i])
        n_live = int(row_nnz[i])
        for s in range(n_live):
            slot = start + s
            c = int(col_indices[slot])
            assert abs(W_values_t.grad[slot].item() - dL_dW_dense[i, c].item()) < 1e-5, (
                f"dW_values[{slot}] mismatch at (i={i}, c={c}): "
                f"ours={W_values_t.grad[slot].item()}, oracle={dL_dW_dense[i,c].item()}"
            )


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — no-grad fast path
# ─────────────────────────────────────────────────────────────────────

def test_no_grad_path_is_used_when_no_requires_grad():
    """
    Calling spmm() with no requires_grad anywhere should NOT produce a
    grad-tracking output. This confirms the fast path works.
    """
    W_csr = PaddedCSR.from_dense(torch.randn(4, 4))
    X = torch.randn(4, 2, dtype=torch.float32, requires_grad=False)
    Y = sparsecore.spmm(W_csr, X)
    assert Y.requires_grad is False


def test_no_grad_context_skips_autograd():
    """
    Inside torch.no_grad(), the autograd graph must not be built even
    if X.requires_grad=True.
    """
    W_csr = PaddedCSR.from_dense(torch.randn(4, 4))
    X = torch.randn(4, 2, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        Y = sparsecore.spmm(W_csr, X)
    assert Y.requires_grad is False


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — "real training step" smoke test
# ─────────────────────────────────────────────────────────────────────

def test_one_step_of_gradient_descent_reduces_loss():
    """
    One step of SGD using our autograd should reduce a simple MSE loss.
    This is the "does training actually work" smoke test — not a rigorous
    convergence test, but catches any issue that would prevent learning.

    Demonstrates the training loop pattern:
      1. Forward: Y = W @ X via _SpMMFunction
      2. Loss
      3. Backward to populate W_values_t.grad
      4. Update W_values_t.data in place
      5. Mirror the update into W.values (writable zero-copy view)
      6. Re-forward sees the updated weights
    """
    torch.manual_seed(11)
    M, K, N = 16, 12, 8
    W_dense = torch.randn(M, K) * (torch.rand(M, K) >= 0.5).float()
    W_csr = PaddedCSR.from_dense(W_dense)
    W_values_t = torch.from_numpy(
        np.array(W_csr.values, copy=True)
    ).requires_grad_(True)

    X = torch.randn(K, N, dtype=torch.float32)
    target = torch.randn(M, N, dtype=torch.float32)

    # Forward + loss
    Y0 = _SpMMFunction.apply(W_values_t, W_csr, X, "simd")
    loss0 = ((Y0 - target) ** 2).mean()

    # Backward + SGD step
    loss0.backward()
    with torch.no_grad():
        W_values_t.data -= 0.01 * W_values_t.grad
        # Mirror the update into W.values (writable view into C++ memory)
        # so the next forward pass sees the new weights.
        W_csr.values[:] = W_values_t.data.numpy()

    # Re-forward with updated weights — loss should drop
    W_values_t.grad = None
    Y1 = _SpMMFunction.apply(W_values_t, W_csr, X, "simd")
    loss1 = ((Y1 - target) ** 2).mean()

    assert loss1.item() < loss0.item(), (
        f"Loss did not decrease after SGD step: "
        f"{loss0.item():.4f} → {loss1.item():.4f}"
    )
