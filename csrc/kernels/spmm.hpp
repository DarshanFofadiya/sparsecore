// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm.hpp
//
//  SpMM (Sparse-Dense Matrix Multiply) kernel declarations.
//
//  Computes Y = W @ X where W is a PaddedCSR sparse matrix and X is a
//  dense matrix. Y is dense, pre-allocated, pre-zeroed.
//
//  See docs/design/spmm.md for the full specification — this file
//  declares the C++ side of the API specified there.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include "padded_csr.hpp"


namespace sparselab {

// ─────────────────────────────────────────────────────────────────────────
//  spmm_scalar — scalar reference implementation of Y = W @ X.
//
//  Scalar, single-threaded, no SIMD. Exists as the Oracle reference
//  that the NEON SIMD version (spmm_simd_neon, Milestone 3d) is
//  verified against.
//
//  Parameters:
//    W  - sparse weight in PaddedCSR format, shape (M, K) = (W.nrows, W.ncols).
//         Not mutated.
//    X  - dense input pointer, row-major, shape (K, N). Length = K * N. Not mutated.
//    K  - inner dimension; must equal W.ncols.
//    N  - number of output columns.
//    Y  - dense output pointer, row-major, shape (M, N). Length = M * N.
//         The kernel zeros Y internally before accumulating, so callers
//         do not need to pre-zero. Any existing contents of Y are
//         overwritten.
//
//  Algorithmic note:
//    Loop order i → s → j (output row, live slot of that row, output column).
//    This exploits CSR's natural row-major storage and keeps Y's write
//    pattern sequential. See docs/design/spmm.md §3 for full rationale.
//
//  Complexity: O(nnz * N) floating-point ops. Memory-bandwidth bound for
//  typical transformer shapes.
// ─────────────────────────────────────────────────────────────────────────
void spmm_scalar(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
);

}  // namespace sparselab
