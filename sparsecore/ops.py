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
        - Autograd support arrives in Milestone 4a; for now, call from
          inference / no_grad contexts only.
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

    # ─── Dtype + contiguity coercion ──────────────────────────────────
    X_f32 = X.to(dtype=torch.float32).contiguous()

    # ─── Kernel dispatch ──────────────────────────────────────────────
    # The map is explicit so adding future backends (e.g., avx2, rvv) is
    # a one-line change. "auto" is currently a synonym for "simd" on our
    # Apple Silicon build; a future runtime-dispatch layer will pick
    # based on CPU features detected at .so-load time.
    kernel_fns = {
        "auto": _core.spmm_simd,
        "simd": _core.spmm_simd,
        "scalar": _core.spmm_scalar,
    }
    if kernel not in kernel_fns:
        raise ValueError(
            f"spmm: unknown kernel={kernel!r}. "
            f"Expected one of: {sorted(kernel_fns.keys())}."
        )

    Y_np = kernel_fns[kernel](W, X_f32.numpy())
    return torch.from_numpy(Y_np)
