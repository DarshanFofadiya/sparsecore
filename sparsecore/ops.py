"""
sparsecore.ops — sparse tensor operations.

This module holds the public-facing operation functions (spmm, and in
later milestones: spmm backward, elementwise ops, etc.). It is the
PyTorch-facing shim layer — accepts torch.Tensor inputs, dispatches to
the C++ kernels, returns torch.Tensor outputs.

Separation of concerns:
    sparsecore.layout  — how sparse data is stored and constructed
    sparsecore.ops     — what you can do with sparse data (forward pass ops)

Public API:
    spmm(W, X) -> torch.Tensor       # sparse-dense matmul Y = W @ X
                                     # autograd-aware: participates in
                                     # loss.backward() when W.values or
                                     # X has requires_grad=True
"""

from __future__ import annotations

import numpy as np
import torch

from sparsecore import _core
from sparsecore._core import PaddedCSR as _PaddedCSR


# ─────────────────────────────────────────────────────────────────────
#  Autograd Function — the bridge to PyTorch's training machinery
# ─────────────────────────────────────────────────────────────────────
#
#  Pattern reference:
#  https://pytorch.org/docs/stable/notes/extending.html
#
#  A torch.autograd.Function subclass has two static methods, forward
#  and backward. PyTorch's autograd engine calls forward during the
#  forward pass, stashes whatever ctx.save_for_backward captures, and
#  calls backward during loss.backward() with the upstream gradient.
#
#  Our implementation pattern: the first argument to forward() must be
#  the Tensor we want autograd to track gradients for. We pass
#  W_values (a torch.Tensor aliased to W.values's underlying storage)
#  so autograd hands us back dW_values in the right shape. The
#  PaddedCSR object itself travels through ctx as a non-differentiable
#  metadata blob.
# ─────────────────────────────────────────────────────────────────────

class _SpMMFunction(torch.autograd.Function):
    """
    Autograd-aware wrapper for Y = W @ X where W is a PaddedCSR.

    Forward:  Y = W @ X             (our spmm_simd kernel)
    Backward (w.r.t. W_values):     dW_values[s] = dot(dY[i,:], X[c,:])
                                    (our spmm_grad_w kernel)
    Backward (w.r.t. X):            dX = Wᵀ @ dY
                                    (our spmm_simd kernel on W.transpose())

    The first positional arg, W_values, is the torch.Tensor autograd
    tracks. It's a zero-copy alias of W.values; the C++ forward kernel
    reads through W (which points at the same memory), so updating
    W_values inplace (as an optimizer does) correctly updates what
    subsequent forward passes compute on.
    """

    @staticmethod
    def forward(ctx, W_values, W, X, kernel):
        # Save tensors needed for backward. X is required verbatim; we
        # save W_values (not used in backward math but autograd requires
        # saving at least one Tensor per tracked input for the gradient-
        # return contract).
        ctx.save_for_backward(W_values, X)
        # Non-tensor metadata on ctx is allowed and not tracked.
        ctx.W = W
        ctx.kernel = kernel

        # Delegate to the existing kernel dispatcher. Note we intentionally
        # bypass the public spmm() wrapper's requires_grad routing to
        # avoid an infinite loop — we're already inside the autograd path.
        kernel_fns = {
            "auto": _core.spmm_simd,
            "simd": _core.spmm_simd,
            "scalar": _core.spmm_scalar,
        }
        X_contig = X.contiguous()
        Y_np = kernel_fns[kernel](W, X_contig.numpy())
        return torch.from_numpy(Y_np)

    @staticmethod
    def backward(ctx, dY):
        W_values, X = ctx.saved_tensors
        W = ctx.W

        # dY arrives from the upstream layer. Ensure it's contiguous
        # float32 — same coercion the forward pass does for X.
        dY_f32 = dY.to(dtype=torch.float32).contiguous()
        X_f32 = X.to(dtype=torch.float32).contiguous()

        # ── Gradient w.r.t. W_values ──
        # Per design doc §1.4: one dot product per live slot, output
        # aligned with W.values (length = total_capacity, padding = 0).
        dW_np = _core.spmm_grad_w(W, dY_f32.numpy(), X_f32.numpy())
        dW_values = torch.from_numpy(dW_np)

        # ── Gradient w.r.t. X ──
        # Per design doc §1.3: dL/dX = Wᵀ @ dL/dY — just another SpMM.
        # We materialize Wᵀ once; future optimization could fuse this
        # into a single transpose-and-multiply kernel.
        WT = W.transpose()
        dX_np = _core.spmm_simd(WT, dY_f32.numpy())
        dX = torch.from_numpy(dX_np)

        # Returns must line up with forward's inputs:
        #   W_values (Tensor, tracked)  → dW_values
        #   W        (PaddedCSR, non-diff) → None
        #   X        (Tensor, tracked)  → dX
        #   kernel   (str, non-diff)    → None
        return dW_values, None, dX, None


# ─────────────────────────────────────────────────────────────────────
#  spmm — sparse-dense matrix multiply (public API)
# ─────────────────────────────────────────────────────────────────────

def spmm(
    W: _PaddedCSR,
    X: torch.Tensor,
    *,
    kernel: str = "auto",
) -> torch.Tensor:
    """
    Compute Y = W @ X, where W is a PaddedCSR and X is a dense 2-D tensor.

    This is the forward pass of a sparse Linear layer: in a typical
    transformer, W is the weight matrix (shape (out_features, in_features),
    stored sparse) and X is the input activation (shape (in_features,
    batch_size * seq_len), stored dense).

    Autograd:
        If X has requires_grad=True (the common case inside a training
        loop where X is an intermediate activation), this routes through
        an autograd-aware wrapper so loss.backward() will produce:
          - dL/dX: the dense gradient for X (via Wᵀ @ dY)
          - dL/dW_values: sparse gradient aligned with W.values
                          (only computed if W.values has requires_grad=True)

        If neither input tracks gradients (inference / no_grad context),
        this skips the autograd machinery and calls the kernel directly.

    Dispatch:
        - kernel="auto" (default): use NEON SIMD on ARM64 builds, falls
          back to scalar elsewhere. For this v0.1, "auto" always means
          NEON because the build targets Apple Silicon.
        - kernel="simd": force the NEON SIMD kernel. Errors on non-ARM.
        - kernel="scalar": force the scalar reference kernel (slower but
          bit-stable). Useful for debugging and benchmark baselines.

    Args:
        W: a PaddedCSR of shape (M, K). Inner dimension K must equal X's
           first dimension.
        X: a 2-D dense tensor of shape (K, N). Any float dtype is accepted;
           it will be converted to float32 for the computation. Must be on
           CPU — GPU tensors raise an error.
        kernel: which kernel to dispatch to. See above.

    Returns:
        A 2-D dense torch.Tensor Y of shape (M, N), dtype float32, on CPU.

    Raises:
        TypeError: if W is not a PaddedCSR or X is not a torch.Tensor.
        ValueError: if X is not 2-D, or shape mismatch, or unknown kernel.
        RuntimeError: if X is on a non-CPU device.

    Notes:
        - O(nnz(W) * N) multiply-adds, regardless of kernel. NEON cuts the
          wall-clock constant by ~3-4x on the inner loop.
        - For sparse-to-sparse DST, the optimizer updates W.values in-place;
          our autograd returns dW_values aligned to W.values (length =
          W.total_capacity, padding slots always 0).
    """
    # ─── Type checks ──────────────────────────────────────────────────
    if not isinstance(W, _PaddedCSR):
        raise TypeError(
            f"spmm: W must be a sparsecore.PaddedCSR, got {type(W).__name__}"
        )
    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"spmm: X must be a torch.Tensor, got {type(X).__name__}"
        )

    # ─── Device check ─────────────────────────────────────────────────
    if X.device.type != "cpu":
        raise RuntimeError(
            f"spmm: X must be on CPU (got device={X.device}). "
            f"SparseCore v0.1 is CPU-only; GPU support arrives post-v0.1."
        )

    # ─── Shape check ──────────────────────────────────────────────────
    if X.dim() != 2:
        raise ValueError(
            f"spmm: X must be 2-D, got shape={tuple(X.shape)}"
        )

    # ─── Kernel name validation ───────────────────────────────────────
    valid_kernels = {"auto", "simd", "scalar"}
    if kernel not in valid_kernels:
        raise ValueError(
            f"spmm: unknown kernel={kernel!r}. "
            f"Expected one of: {sorted(valid_kernels)}."
        )

    # ─── Dtype + contiguity coercion ──────────────────────────────────
    X_f32 = X.to(dtype=torch.float32).contiguous()

    # ─── Autograd routing ─────────────────────────────────────────────
    # If X needs grad, route through the autograd Function so backward
    # works automatically. We pull W_values out as a torch Tensor so
    # autograd can track it; at forward time it's effectively a no-op
    # view, at backward time it receives dW_values.
    #
    # In practice, during training:
    #   - X.requires_grad = True (it's an intermediate activation)
    #   - W_values.requires_grad will be True once SparseLinear (4b) wraps
    #     it as a nn.Parameter. For now, users have to make that choice
    #     themselves; if they want gradients on W they need
    #     W_values.requires_grad_(True) before the call.
    W_values_t = torch.from_numpy(np.array(W.values, copy=False))

    if X_f32.requires_grad or W_values_t.requires_grad:
        return _SpMMFunction.apply(W_values_t, W, X_f32, kernel)

    # ─── Fast path (no autograd) ──────────────────────────────────────
    kernel_fns = {
        "auto": _core.spmm_simd,
        "simd": _core.spmm_simd,
        "scalar": _core.spmm_scalar,
    }
    Y_np = kernel_fns[kernel](W, X_f32.numpy())
    return torch.from_numpy(Y_np)
