// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_grad.cpp
//
//  Scalar implementation of dL/dW at live slots of W.
//
//  The central loop is:
//    for each live slot s pointing at (i, k):
//        dW_values[s] = dot(dY[i, :], X[k, :])
//
//  That's exactly one N-length dot product per live entry. Total work is
//  O(nnz * N) FMAs, the same complexity as the forward pass — which is
//  the whole reason DST can train 10x faster than dense.
//
//  For 4a we ship only the scalar version. A NEON variant (identical
//  pattern to vector_dot_neon.cpp) would mechanically drop in as a
//  future optimization. Clang's auto-vectorization on -O3 may already
//  be producing NEON FMAs for this loop — we saw exactly this in 3d.
// ═══════════════════════════════════════════════════════════════════════════

#include "spmm_grad.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument


namespace sparsecore {

void spmm_grad_w(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
) {
    // ─── Shape check ───────────────────────────────────────────────────
    // W.ncols determines which X row each live slot reads from. If the
    // caller passes a mismatched K, we'd read past the end of X.
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_grad_w: W.ncols (" + std::to_string(W.ncols) +
            ") does not equal K (" + std::to_string(K) + ")."
        );
    }

    const int64_t M = W.nrows;
    const int32_t cap = W.total_capacity();

    // ─── Zero the output buffer ────────────────────────────────────────
    // Matches the self-zeroing contract of our other kernels: callers
    // don't need to pre-zero. For spmm_grad_w this also means padding
    // slots (which we never visit in the main loop) end up as 0.0 —
    // the correct neutral value for gradient-based updates.
    //
    // We zero the full capacity, not just nnz, because dW_values is
    // sized total_capacity to align with W.values (so optimizers can
    // do W.values -= lr * dW in one vectorized op).
    std::memset(dW_values, 0,
                static_cast<std::size_t>(cap) * sizeof(float));

    // ─── Fast-path: nothing to do ──────────────────────────────────────
    if (M == 0 || N == 0 || K == 0 || cap == 0) {
        return;
    }

    // ─── Main loop: one dot product per live slot ──────────────────────
    //
    // Outer walk: for each row i of W, find its live slots.
    // For each live slot s = (row_start[i] + offset), read
    // (c, _) = (col_indices[s], ignored — we don't use the weight here).
    //
    //   dW_values[s] = sum over j in [0, N) of dY[i, j] * X[c, j]
    //
    // Memory pattern per slot:
    //   - dY row pointer computed once: stays in L1 across the inner loop
    //   - X row pointer computed once: stays in L1
    //   - accumulator `acc` stays in a register
    //   - N FMAs with sequential reads → ideal SIMD/auto-vectorization target
    //
    // A plausible NEON hand-write would load 4 floats at a time from
    // dY_row and X_row, multiply lane-wise, accumulate in a 4-wide
    // register, then horizontal-sum at the end. Matches the structure
    // of vector_dot_neon.cpp exactly. Deferred to post-4a per the
    // design doc — get correctness end-to-end first.
    // ───────────────────────────────────────────────────────────────────
    for (int64_t i = 0; i < M; ++i) {
        const int32_t row_ptr = W.row_start[i];
        const int32_t n_live  = W.row_nnz[i];
        const float* dY_row = dY + i * N;

        for (int32_t s = 0; s < n_live; ++s) {
            const int32_t slot = row_ptr + s;
            const int32_t c = W.col_indices[slot];
            const float* X_row = X + static_cast<int64_t>(c) * N;

            // Scalar N-length dot product. Clang-17 at -O3 -mcpu=apple-m1
            // auto-vectorizes this to fmla.4s with 4-wide unrolling, so
            // the "scalar" label is a misnomer at the machine-code level.
            float acc = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                acc += dY_row[j] * X_row[j];
            }
            dW_values[slot] = acc;
        }
    }
}

}  // namespace sparsecore
