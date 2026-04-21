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
"""

from __future__ import annotations

import numpy as np
import torch

from sparsecore import _core
from sparsecore._core import PaddedCSR as _PaddedCSR


# ─────────────────────────────────────────────────────────────────────
#  spmm — sparse-dense matrix multiply
# ─────────────────────────────────────────────────────────────────────

def spmm(W: _PaddedCSR, X: torch.Tensor) -> torch.Tensor:
    """
    Compute Y = W @ X, where W is a PaddedCSR and X is a dense 2-D tensor.

    This is the forward pass of a sparse Linear layer: in a typical
    transformer, W is the weight matrix (shape (out_features, in_features),
    stored sparse) and X is the input activation (shape (in_features,
    batch_size * seq_len), stored dense).

    For v0.1 this dispatches to the scalar reference kernel. Milestone 3d
    will add a NEON SIMD fast path and dispatch based on shape/alignment;
    the public API does not change.

    Args:
        W: a PaddedCSR of shape (M, K). Inner dimension K must equal X's
           first dimension.
        X: a 2-D dense tensor of shape (K, N). Any float dtype is accepted;
           it will be converted to float32 for the computation. Must be on
           CPU — GPU tensors raise an error. Strided layout only (no
           sparse inputs).

    Returns:
        A 2-D dense torch.Tensor Y of shape (M, N), dtype float32, on CPU.
        Contains the product W @ X computed in scalar float32 arithmetic.

    Raises:
        TypeError: if W is not a PaddedCSR or X is not a torch.Tensor.
        ValueError: if X is not 2-D, or shape mismatch between W and X.
        RuntimeError: if X is on a non-CPU device.

    Notes:
        - O(nnz(W) * N) scalar multiply-adds. For an (M, K) W at sparsity s
          with typical Transformer shapes (N = batch*seq_len), this is
          (1-s) * M * K * N FMAs — up to 10x fewer than dense at 90% sparsity.
        - This function has no autograd support yet; call only from
          inference / no_grad contexts. Autograd arrives in Milestone 4a.
    """
    # ─── Type checks ──────────────────────────────────────────────────
    # The C++ binding would catch wrong types too, but a Python-level
    # check gives a clearer error message than a pybind11 type-cast failure.
    if not isinstance(W, _PaddedCSR):
        raise TypeError(
            f"spmm: W must be a sparsecore.PaddedCSR, got {type(W).__name__}"
        )
    if not isinstance(X, torch.Tensor):
        raise TypeError(
            f"spmm: X must be a torch.Tensor, got {type(X).__name__}"
        )

    # ─── Device check ─────────────────────────────────────────────────
    # v0.1 is CPU-only. A crisp error beats a mysterious segfault if the
    # user absentmindedly passes a .cuda() or .mps() tensor.
    if X.device.type != "cpu":
        raise RuntimeError(
            f"spmm: X must be on CPU (got device={X.device}). "
            f"SparseCore v0.1 is CPU-only; GPU support arrives post-v0.1."
        )

    # ─── Shape check ──────────────────────────────────────────────────
    # The C++ binding re-checks this, but failing here lets us report
    # the PyTorch-native shape syntax (X.shape) in the error.
    if X.dim() != 2:
        raise ValueError(
            f"spmm: X must be 2-D, got shape={tuple(X.shape)}"
        )

    # ─── Dtype + contiguity coercion ──────────────────────────────────
    # Convert to float32 and ensure C-contiguous layout. For float32
    # contiguous inputs these are both no-ops (PyTorch returns self).
    X_f32 = X.to(dtype=torch.float32).contiguous()

    # ─── Dispatch to the scalar kernel ────────────────────────────────
    # .numpy() gives a zero-copy view into the torch tensor's storage
    # (safe because we forced contiguous float32 above). The C++ side
    # reads through this view and writes into a freshly-allocated
    # output buffer, so there is no aliasing risk.
    Y_np = _core.spmm_scalar(W, X_f32.numpy())

    # ─── Wrap result back as a torch.Tensor ───────────────────────────
    # torch.from_numpy shares memory with the NumPy array — this is a
    # zero-copy hand-off. The lifetime of Y_np is tied to the returned
    # tensor via NumPy's refcount, so we don't need to copy.
    return torch.from_numpy(Y_np)
