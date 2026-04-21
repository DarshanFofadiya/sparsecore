"""
sparsecore.layout — Python factories for PaddedCSR.

The C++ side (csrc/kernels/padded_csr.{hpp,cpp}) defines the data layout
and invariant checking. This module provides user-friendly constructors
that build the 6 arrays (values, col_indices, row_start, row_nnz,
row_capacity) from PyTorch tensors and pass them to the C++ constructor.

Public API:
    PaddedCSR.from_dense(W, *, threshold=0.0, padding_ratio=0.2)
    PaddedCSR.from_torch_sparse_csr(csr, *, padding_ratio=0.2)
    PaddedCSR.random(nrows, ncols, *, sparsity, padding_ratio=0.2, seed=None)

See docs/design/padded_csr.md for the full specification.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from sparsecore import _core

if TYPE_CHECKING:
    from sparsecore._core import PaddedCSR as _PaddedCSR


# ─────────────────────────────────────────────────────────────────────
#  Row-capacity computation
# ─────────────────────────────────────────────────────────────────────

def _compute_row_capacity(row_nnz: np.ndarray, padding_ratio: float) -> np.ndarray:
    """
    Given live-entry counts per row and a padding ratio, compute the
    allocated capacity per row.

    Formula: capacity[i] = max(1, ceil(nnz[i] * (1 + padding_ratio)))

    The floor of 1 ensures even empty rows can accept at least one new
    connection during Phase 4's grow operation without immediately
    needing a resize_row() call.

    Args:
        row_nnz: (nrows,) int32 array, live-entry count per row.
        padding_ratio: float >= 0, extra capacity as a fraction of nnz.

    Returns:
        (nrows,) int32 array of row capacities.
    """
    if padding_ratio < 0:
        raise ValueError(f"padding_ratio must be >= 0, got {padding_ratio}")
    scaled = np.ceil(row_nnz.astype(np.float64) * (1.0 + padding_ratio))
    capacity = np.maximum(1, scaled).astype(np.int32)
    return capacity


# ─────────────────────────────────────────────────────────────────────
#  Factory: from a torch sparse CSR tensor
# ─────────────────────────────────────────────────────────────────────

def from_torch_sparse_csr(
    csr: torch.Tensor,
    *,
    padding_ratio: float = 0.2,
) -> "_PaddedCSR":
    """
    Build a PaddedCSR from a torch.sparse_csr_tensor.

    This is the canonical entry point — other factories normalize to this
    path. PyTorch's CSR has tight rows (no padding); we convert by
    computing per-row capacities and placing live entries at the start
    of each padded row.

    Args:
        csr: a 2-D torch.sparse_csr tensor (float32 values, int32 or int64
             indices). Will be converted to float32/int32 internally.
        padding_ratio: extra capacity per row as a fraction of live nnz.
                       Default 0.2 = 20% padding, which empirically balances
                       memory overhead against grow frequency for typical
                       DST schedules (~10% connection churn per step).

    Returns:
        A new PaddedCSR with all 8 invariants satisfied.

    Raises:
        ValueError: if the input isn't a 2-D sparse CSR tensor, or if
                    construction fails invariant checking.
    """
    if csr.layout != torch.sparse_csr:
        raise ValueError(
            f"from_torch_sparse_csr expected layout=torch.sparse_csr, "
            f"got layout={csr.layout}"
        )
    if csr.dim() != 2:
        raise ValueError(
            f"from_torch_sparse_csr expected 2-D tensor, got {csr.dim()}-D"
        )

    nrows, ncols = csr.size()
    # Extract PyTorch CSR components. .contiguous().cpu() handles the rare
    # case of non-contiguous or GPU-side tensors; cheap when already
    # contiguous and on CPU.
    crow = csr.crow_indices().contiguous().cpu().numpy().astype(np.int32)
    col = csr.col_indices().contiguous().cpu().numpy().astype(np.int32)
    vals = csr.values().contiguous().cpu().numpy().astype(np.float32)

    # Compute per-row nnz from the cumulative crow_indices.
    # crow[i+1] - crow[i] is the count of live entries in row i.
    row_nnz = np.diff(crow).astype(np.int32)  # length nrows

    # Allocate per-row capacity with padding.
    row_capacity = _compute_row_capacity(row_nnz, padding_ratio)  # length nrows

    # row_start is the cumulative sum of row_capacity (with a 0 prepended).
    # This is the padded analog of PyTorch's crow_indices.
    row_start = np.concatenate(
        ([0], np.cumsum(row_capacity[:-1]))
    ).astype(np.int32)  # length nrows

    total_capacity = int(row_capacity.sum())

    # Allocate the padded slot arrays.
    padded_values = np.zeros(total_capacity, dtype=np.float32)
    padded_col = np.full(total_capacity, -1, dtype=np.int32)

    # Copy each row's tight live entries into the first row_nnz[i] slots
    # of the padded row. The remaining slots stay as (col=-1, value=0.0)
    # which is exactly the padding-slot invariant.
    for i in range(nrows):
        n_live = int(row_nnz[i])
        if n_live == 0:
            continue
        src_start = int(crow[i])
        src_end = int(crow[i + 1])
        dst_start = int(row_start[i])
        padded_values[dst_start : dst_start + n_live] = vals[src_start:src_end]
        padded_col[dst_start : dst_start + n_live] = col[src_start:src_end]

    # Hand off to the C++ constructor. It auto-validates invariants and
    # raises ValueError on any failure — if our conversion above is
    # correct, this never fires.
    return _core.PaddedCSR(
        nrows=int(nrows),
        ncols=int(ncols),
        values=padded_values.tolist(),
        col_indices=padded_col.tolist(),
        row_start=row_start.tolist(),
        row_nnz=row_nnz.tolist(),
        row_capacity=row_capacity.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────
#  Factory: from a dense tensor
# ─────────────────────────────────────────────────────────────────────

def from_dense(
    W: torch.Tensor,
    *,
    threshold: float = 0.0,
    padding_ratio: float = 0.2,
) -> "_PaddedCSR":
    """
    Build a PaddedCSR from a dense 2-D tensor.

    Entries where |W[i,j]| > threshold are considered live; everything
    else is treated as zero.

    Args:
        W: a 2-D dense float tensor (any float dtype; converted to float32).
        threshold: magnitude below which entries are considered zero.
                   Default 0.0 means "any nonzero value is live."
        padding_ratio: extra capacity per row as a fraction of live nnz.

    Returns:
        A new PaddedCSR with all 8 invariants satisfied.
    """
    if W.dim() != 2:
        raise ValueError(f"from_dense expected 2-D tensor, got {W.dim()}-D")

    W_f32 = W.to(dtype=torch.float32).contiguous()

    if threshold > 0:
        # Apply the threshold: anything with magnitude at or below threshold
        # becomes 0.0. Preserves the original values above threshold.
        mask = W_f32.abs() > threshold
        W_f32 = torch.where(mask, W_f32, torch.zeros_like(W_f32))

    # Delegate to torch.Tensor.to_sparse_csr for the dense→CSR conversion.
    # This is the path our Borrow-Don't-Reinvent steering recommends: reuse
    # PyTorch's dense-to-CSR logic rather than writing our own.
    return from_torch_sparse_csr(W_f32.to_sparse_csr(), padding_ratio=padding_ratio)


# ─────────────────────────────────────────────────────────────────────
#  Factory: random sparse matrix
# ─────────────────────────────────────────────────────────────────────

def random(
    nrows: int,
    ncols: int,
    *,
    sparsity: float,
    padding_ratio: float = 0.2,
    seed: int | None = None,
) -> "_PaddedCSR":
    """
    Build a PaddedCSR with a random sparsity pattern.

    Useful for benchmarks, tests, and quick prototypes. The live entries
    are placed uniformly at random; values are drawn from a standard
    normal distribution (mean 0, stddev 1).

    Args:
        nrows, ncols: matrix shape.
        sparsity: fraction of logical cells that should be zero.
                  Must be in [0, 1). sparsity=0.9 → 10% dense entries.
        padding_ratio: extra capacity per row as a fraction of live nnz.
        seed: random seed for reproducibility. None → non-deterministic.

    Returns:
        A new PaddedCSR with all 8 invariants satisfied.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # Build a dense tensor with values, then zero out (1-density) fraction
    # uniformly at random.
    density = 1.0 - sparsity
    W = torch.randn(nrows, ncols, generator=gen, dtype=torch.float32)
    keep_mask = torch.rand(nrows, ncols, generator=gen) < density
    W = W * keep_mask.float()

    return from_dense(W, threshold=0.0, padding_ratio=padding_ratio)


# ─────────────────────────────────────────────────────────────────────
#  Interop: PaddedCSR -> dense tensor
# ─────────────────────────────────────────────────────────────────────

def to_dense(p: "_PaddedCSR") -> torch.Tensor:
    """
    Convert a PaddedCSR back into a dense torch.Tensor.

    Scatter each live slot's value into position [row, col] of a
    zero-initialized dense tensor. Padding slots are skipped (their
    col_idx is -1, which isn't a valid position).

    This is the round-trip Oracle for from_dense: for any dense W,
    to_dense(from_dense(W)) should equal W exactly (when threshold=0).
    """
    nrows, ncols = p.shape
    dense = torch.zeros(nrows, ncols, dtype=torch.float32)

    # NumPy views of the underlying C++ vectors. Zero-copy, read-only.
    values = p.values
    col_indices = p.col_indices
    row_start = p.row_start
    row_nnz = p.row_nnz

    for i in range(nrows):
        n_live = int(row_nnz[i])
        if n_live == 0:
            continue
        start = int(row_start[i])
        # Live slots are contiguous at the start of each row's region.
        row_cols = col_indices[start : start + n_live]
        row_vals = values[start : start + n_live]
        dense[i, row_cols] = torch.from_numpy(np.ascontiguousarray(row_vals))

    return dense
