// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm.cpp
//
//  Scalar implementation of Y = W @ X for W in PaddedCSR format, X and Y dense.
//
//  This is the Oracle reference for the NEON SIMD version (spmm_simd_neon,
//  Milestone 3d). Both versions must produce results within rtol=atol=1e-5 of
//  each other and of torch.matmul(W.to_dense(), X).
// ═══════════════════════════════════════════════════════════════════════════

#include "spmm.hpp"
#include "parallel.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument


namespace sparselab {

void spmm_scalar(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
) {
    // ─── Shape check ───────────────────────────────────────────────────
    // We're strict here: the caller must pass K = W.ncols. The Python
    // binding verifies this, but we double-check in C++ as a safety net.
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_scalar: W.ncols (" + std::to_string(W.ncols) +
            ") does not equal K (" + std::to_string(K) + ")."
        );
    }

    const int64_t M = W.nrows;

    // ─── Zero the output buffer ────────────────────────────────────────
    // The kernel accumulates into Y (Y[i,j] += v * X[c,j]), so Y must
    // start at zero. We do this in the kernel itself rather than forcing
    // the caller to remember — making the function self-contained. The
    // memset is memory-bandwidth-bound just like the math, so the
    // relative cost is negligible.
    //
    // Originally the design called for the caller to pre-zero, but we
    // discovered during 3c-ii that pybind11's py::array_t<float>(shape)
    // constructor does NOT zero-initialize — it uses PyArray_SimpleNew,
    // which leaves memory in whatever state the allocator returned.
    // Zeroing inside the kernel removes that sharp edge.
    std::memset(Y, 0, static_cast<std::size_t>(M) *
                      static_cast<std::size_t>(N) * sizeof(float));

    // ─── Fast-path: empty matrices ─────────────────────────────────────
    // If either dimension is zero there's nothing to compute. Y is already
    // zeroed above (which for size 0 is a no-op).
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // ─── Main triple-nested loop: i → s → j ───────────────────────────
    //
    // For each output row i:
    //   Find W's row-i data slice: [row_start[i], row_start[i] + row_nnz[i])
    //   For each live slot s in row i:
    //     Let (c, v) = (col_indices[s], values[s]); c is a row index into X,
    //     v is the corresponding sparse weight.
    //     Accumulate v * X[c, :] into Y[i, :].
    //
    // Row-major indexing: Y[i, j] is at Y[i * N + j]; X[c, j] is at X[c * N + j].
    //
    // ─── Parallelism ──────────────────────────────────────────────────
    // The outer loop is embarrassingly parallel: each i writes only to
    // its own row of Y (y_row = Y + i * N), and all reads are from
    // W and X which are const. No locks, no atomics, no false sharing
    // across cachelines if rows are at least N*4 bytes long.
    //
    // SCORE_PARALLEL_FOR is a compile-time shim:
    //   - with OpenMP: expands to `#pragma omp parallel for schedule(static)`
    //   - without   : expands to nothing, loop stays sequential
    //
    // For M below SCORE_PARALLEL_ROW_THRESHOLD the fork/join overhead
    // dominates the work — `if(...)` clauses in the pragma gate the
    // parallel region without duplicating the loop body.
    // ───────────────────────────────────────────────────────────────────
    #if SCORE_HAVE_OPENMP
    #pragma omp parallel for schedule(static) \
        if(M >= SCORE_PARALLEL_ROW_THRESHOLD)
    #endif
    for (int64_t i = 0; i < M; ++i) {
        const int32_t row_ptr = W.row_start[i];
        const int32_t n_live  = W.row_nnz[i];
        float* y_row = Y + i * N;

        for (int32_t s = 0; s < n_live; ++s) {
            const int32_t c = W.col_indices[row_ptr + s];
            const float   v = W.values[row_ptr + s];
            const float* x_row = X + static_cast<int64_t>(c) * N;

            // Inner j loop — this is what NEON vectorizes in spmm_neon.cpp.
            // Plain scalar FMA: y[j] += v * x[j].
            for (int64_t j = 0; j < N; ++j) {
                y_row[j] += v * x_row[j];
            }
        }
    }
}

}  // namespace sparselab
