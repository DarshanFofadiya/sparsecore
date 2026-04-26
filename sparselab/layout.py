"""
sparselab.layout — Python factories for PaddedCSR.

The C++ side (csrc/kernels/padded_csr.{hpp,cpp}) defines the data layout
and invariant checking. This module provides user-friendly constructors
that build the 6 arrays (values, col_indices, row_start, row_nnz,
row_capacity) from PyTorch tensors and pass them to the C++ constructor.

Public API:
    PaddedCSR.from_dense(W, *, threshold=0.0, padding_ratio=0.2)
    PaddedCSR.from_torch_sparse_csr(csr, *, padding_ratio=0.2)
    PaddedCSR.random(nrows, ncols, *, sparsity, padding_ratio=0.2, seed=None)

See docs/design/padded_csr.md for the full specification.
See docs/design/padding_ratio.md for padding_ratio tradeoffs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from sparselab import _core

if TYPE_CHECKING:
    from sparselab._core import PaddedCSR as _PaddedCSR


__all__ = [
    "from_dense",
    "from_torch_sparse_csr",
    "random",
    "to_dense",
    "transpose",
    "transpose_with_perm",
]


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
                       (See docs/design/padding_ratio.md for tradeoffs).

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
                       (See docs/design/padding_ratio.md for tradeoffs).

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
                       (See docs/design/padding_ratio.md for tradeoffs).
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


# ─────────────────────────────────────────────────────────────────────
#  Transpose: build Wᵀ from W
#
#  This is the classical CSR-transpose in two passes (count + scatter).
#  See docs/design/spmm_backward.md §3.2 for the full algorithm.
#
#  Cost: O(nnz + M + K) time, one new PaddedCSR allocation. We use this
#  once per backward pass to turn W into Wᵀ so dL/dX = Wᵀ @ dL/dY can
#  reuse our existing SpMM kernel. Transpose is not in the SpMM hot
#  path — we can afford the O(nnz) walk.
# ─────────────────────────────────────────────────────────────────────

def transpose(p: "_PaddedCSR", *, padding_ratio: float = 0.2) -> "_PaddedCSR":
    """
    Return a new PaddedCSR that is the transpose of `p`.

    Transposition swaps rows and columns: if W has a live entry at (i, c)
    with value v, then Wᵀ has a live entry at (c, i) with the same value.

    Args:
        p: the PaddedCSR to transpose. Not mutated.
        padding_ratio: padding ratio for the output. Defaults to 0.2;
            for backward pass usage we could pick 0.0 (no padding, since
            we never insert into Wᵀ), but 0.2 matches our other factories
            and keeps options open.                       

    Returns:
        A new PaddedCSR of shape (p.ncols, p.nrows) with the same nnz.
        Its live entries within each row are contiguous (not sorted by
        original row index — see note below).

    Notes:
        - Live entries in each output row are NOT sorted by column. They
          appear in the order they were discovered during pass 2, which
          is the row-major order of p. This doesn't matter for SpMM
          correctness (the kernel walks live slots; order is irrelevant)
          but a future optimization could sort for cache locality.
    """
    M = p.nrows          # rows of input
    K = p.ncols          # cols of input → rows of output
    nnz = p.nnz

    # ─── Extract the live entries from p as flat arrays ───────────────
    # We only read the live portion of each row (indices
    # [row_start[i] : row_start[i] + row_nnz[i]]), skipping padding.
    # Result: three same-length arrays describing every live entry.
    values_view = p.values          # zero-copy NumPy view over C++ values
    col_view = p.col_indices        # zero-copy NumPy view over col_indices
    row_start = p.row_start         # length M
    row_nnz = p.row_nnz             # length M

    # Build an "entry stream": for each live entry, record (row, col, val).
    # This flattens the per-row slicing into one big triple of arrays.
    src_rows = np.empty(nnz, dtype=np.int32)
    src_cols = np.empty(nnz, dtype=np.int32)
    src_vals = np.empty(nnz, dtype=np.float32)
    offset = 0
    for i in range(M):
        n_live = int(row_nnz[i])
        if n_live == 0:
            continue
        start = int(row_start[i])
        src_rows[offset : offset + n_live] = i
        src_cols[offset : offset + n_live] = col_view[start : start + n_live]
        src_vals[offset : offset + n_live] = values_view[start : start + n_live]
        offset += n_live
    # Sanity: we should have exactly consumed nnz entries.
    assert offset == nnz, (
        f"transpose: entry-stream build consumed {offset} entries, "
        f"expected {nnz}. PaddedCSR invariants may be broken."
    )

    # ─── Pass 1: count per-output-row nnz ─────────────────────────────
    # In Wᵀ, row index is the original column index. So we count how
    # many live entries had each column value in the input.
    # np.bincount is the idiomatic "count-by-index" primitive.
    out_row_nnz = np.bincount(src_cols, minlength=K).astype(np.int32)
    # Length check: bincount returns length max(src_cols)+1 when no
    # minlength specified; with minlength=K we're guaranteed length K.

    # ─── Allocate output capacities and compute row_start ─────────────
    out_row_capacity = _compute_row_capacity(out_row_nnz, padding_ratio)
    # row_start[k] = sum of capacities for rows 0..k-1.
    # Shape: length K. Use cumulative sum with a 0 prepended.
    out_row_start = np.concatenate(
        ([0], np.cumsum(out_row_capacity[:-1]))
    ).astype(np.int32)

    # Allocate the slot arrays, initialized to the padding sentinel:
    # values=0.0, col_indices=-1. We'll overwrite live slots below.
    total_capacity = int(out_row_capacity.sum())
    out_values = np.zeros(total_capacity, dtype=np.float32)
    out_cols = np.full(total_capacity, -1, dtype=np.int32)

    # ─── Pass 2: scatter entries into output ──────────────────────────
    # write_cursor[k] = next free slot within output row k.
    # Starts at row_start[k] (the beginning of row k's region) and
    # increments by 1 for each entry placed in that row.
    write_cursor = out_row_start.copy()

    # Walk the entry stream. For each (src_row, src_col, val),
    # place it at (out_row=src_col, out_col=src_row) in Wᵀ.
    #
    # Vectorized version using fancy indexing — much faster than a
    # Python-level for loop for large nnz. We compute the destination
    # slot for each entry in one numpy operation.
    if nnz > 0:
        # For each entry e, dest_row_e = src_cols[e]. The slot inside
        # that row is the current cursor value, which we then advance.
        # To vectorize with incrementing cursor, we compute per-entry
        # "position within its output row" via a grouped-rank trick:
        #   - sort entries by their output row
        #   - within each group, position = 0, 1, 2, ...
        sort_order = np.argsort(src_cols, kind="stable")
        sorted_rows = src_cols[sort_order]         # ascending
        # within-group ranks: for each group of equal values, the rank
        # within that group is its index minus the first-occurrence index
        first_occurrence = np.searchsorted(sorted_rows, sorted_rows)
        within_group_rank = np.arange(nnz, dtype=np.int32) - first_occurrence
        # dest slot = row_start[dest_row] + within_group_rank
        dest_slots = out_row_start[sorted_rows] + within_group_rank
        # Scatter
        out_values[dest_slots] = src_vals[sort_order]
        out_cols[dest_slots] = src_rows[sort_order]
        # Advance cursors (not strictly needed after vectorized scatter,
        # but kept for clarity / future extension):
        write_cursor = out_row_start + out_row_nnz

    # ─── Hand off to the C++ constructor ──────────────────────────────
    # This will run the invariant checker; any bug in our transpose
    # algorithm shows up immediately as a ValueError.
    return _core.PaddedCSR(
        nrows=int(K),   # swapped
        ncols=int(M),   # swapped
        values=out_values.tolist(),
        col_indices=out_cols.tolist(),
        row_start=out_row_start.tolist(),
        row_nnz=out_row_nnz.tolist(),
        row_capacity=out_row_capacity.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────
#  transpose_with_perm — transpose that also returns a permutation map
#
#  Same output as `transpose(W)`, plus a parallel array `perm` of length
#  WT.total_capacity where `perm[slot_wt]` is the slot index in W.values
#  whose value belongs at WT.values[slot_wt]. Padding slots in WT get
#  perm = -1.
#
#  Use case: caching. The expensive part of transpose is the ~1 ms of
#  index-structure work (bincount, scatter of col_indices). The cheap
#  part is copying values. If W's topology is unchanged but values have
#  shifted (e.g., SGD updated W.values), we can skip the structure work
#  and refresh WT.values with a single O(nnz) scatter:
#
#      WT.values[:] = W.values[perm]   # with perm[pad]=-1 writing 0
#
#  See sparselab.ops._cached_transpose for the consumer.
#
#  This is a v0.1 optimization — the core transpose function above stays
#  untouched. If the cache proves broken we just stop calling this one
#  and fall back to the direct transpose.
# ─────────────────────────────────────────────────────────────────────

def transpose_with_perm(
    p: "_PaddedCSR", *, padding_ratio: float = 0.2
) -> tuple["_PaddedCSR", np.ndarray]:
    """
    Transpose `p` and also return the permutation map from WT slots to W slots.

    Returns:
        (WT, perm):
            WT — PaddedCSR transpose of p (same as transpose(p))
            perm — int64 array of length WT.total_capacity. For each
                   slot s in WT, perm[s] is the slot index in W that
                   holds the same numeric value (or -1 if s is padding).
    """
    M = p.nrows
    K = p.ncols
    nnz = p.nnz

    values_view = p.values
    col_view = p.col_indices
    row_start = p.row_start
    row_nnz = p.row_nnz

    # Build the entry stream AND track each entry's slot index in W.values.
    src_rows = np.empty(nnz, dtype=np.int32)
    src_cols = np.empty(nnz, dtype=np.int32)
    src_vals = np.empty(nnz, dtype=np.float32)
    src_slots = np.empty(nnz, dtype=np.int64)   # NEW: slot index in W.values

    offset = 0
    for i in range(M):
        n_live = int(row_nnz[i])
        if n_live == 0:
            continue
        start = int(row_start[i])
        src_rows[offset : offset + n_live] = i
        src_cols[offset : offset + n_live] = col_view[start : start + n_live]
        src_vals[offset : offset + n_live] = values_view[start : start + n_live]
        # Each live slot s in W sits at W.values[start + k] for k in [0, n_live).
        src_slots[offset : offset + n_live] = np.arange(start, start + n_live)
        offset += n_live
    assert offset == nnz

    # Count per-output-row nnz.
    out_row_nnz = np.bincount(src_cols, minlength=K).astype(np.int32)
    out_row_capacity = _compute_row_capacity(out_row_nnz, padding_ratio)
    out_row_start = np.concatenate(
        ([0], np.cumsum(out_row_capacity[:-1]))
    ).astype(np.int32)

    total_capacity = int(out_row_capacity.sum())
    out_values = np.zeros(total_capacity, dtype=np.float32)
    out_cols = np.full(total_capacity, -1, dtype=np.int32)
    # Permutation: default -1 (padding slot marker).
    perm = np.full(total_capacity, -1, dtype=np.int64)

    if nnz > 0:
        sort_order = np.argsort(src_cols, kind="stable")
        sorted_rows = src_cols[sort_order]
        first_occurrence = np.searchsorted(sorted_rows, sorted_rows)
        within_group_rank = np.arange(nnz, dtype=np.int32) - first_occurrence
        dest_slots = out_row_start[sorted_rows] + within_group_rank

        out_values[dest_slots] = src_vals[sort_order]
        out_cols[dest_slots] = src_rows[sort_order]
        # For each dest slot, record which W.values slot its value came from.
        perm[dest_slots] = src_slots[sort_order]

    WT = _core.PaddedCSR(
        nrows=int(K),
        ncols=int(M),
        values=out_values.tolist(),
        col_indices=out_cols.tolist(),
        row_start=out_row_start.tolist(),
        row_nnz=out_row_nnz.tolist(),
        row_capacity=out_row_capacity.tolist(),
    )
    return WT, perm
