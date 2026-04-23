// ═══════════════════════════════════════════════════════════════════════════
//  kernels/dense_grad.cpp
//
//  Implementation of the full-dense gradient kernel G = dY @ X^T.
//
//  This is a vanilla dense matmul with NO sparse structure — just
//  M * K independent dot products, each of length N.
//
//  Structure is intentionally identical to spmm_grad_w so it
//  benefits from the same OpenMP parallelization pattern. The only
//  difference is we walk all (i, k) pairs instead of just the live
//  slots of a PaddedCSR.
// ═══════════════════════════════════════════════════════════════════════════

#include "dense_grad.hpp"
#include "parallel.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument


namespace sparselab {

void dense_grad(
    int64_t M, int64_t K, int64_t N,
    const float* dY,
    const float* X,
    float* G
) {
    if (M < 0 || K < 0 || N < 0) {
        throw std::invalid_argument(
            "dense_grad: shape params must be non-negative");
    }

    // ─── Zero the output buffer ────────────────────────────────────────
    // Self-zeroing contract like our other kernels: the caller doesn't
    // have to pre-zero.
    std::memset(G, 0,
                static_cast<std::size_t>(M) *
                static_cast<std::size_t>(K) * sizeof(float));

    // ─── Fast-path: empty matrices ─────────────────────────────────────
    if (M == 0 || K == 0 || N == 0) {
        return;
    }

    // ─── Main loop: G[i, k] = dot(dY[i, :], X[k, :]) ───────────────────
    //
    // Memory pattern:
    //   - dY_row computed once per i, stays in L1 across the k loop
    //   - X_row computed once per (i, k), but adjacent k values access
    //     contiguous memory regions, so the outer of (i, k) is i
    //   - accumulator `acc` lives in a register across the inner j loop
    //
    // Why not SIMD the inner loop by hand? Clang with -O3 -mcpu=apple-m1
    // auto-vectorizes this dot product loop into fmla.4s FMAs already.
    // We saw the same in spmm_grad_w in milestone 4a. If profiling at
    // transformer scale reveals a need, we hand-write NEON later —
    // same pattern as vector_dot_neon.cpp. Not needed for v0.1.
    //
    // ─── Parallelism ──────────────────────────────────────────────────
    // The i loop is embarrassingly parallel — each thread writes only to
    // G[i, :]. Same shape as our other parallel kernels. Below threshold
    // rows we stay sequential to avoid fork/join overhead.
    #if SCORE_HAVE_OPENMP
    #pragma omp parallel for schedule(static) \
        if(M >= SCORE_PARALLEL_ROW_THRESHOLD)
    #endif
    for (int64_t i = 0; i < M; ++i) {
        const float* dY_row = dY + i * N;
        float* G_row = G + i * K;

        for (int64_t k = 0; k < K; ++k) {
            const float* X_row = X + k * N;

            // N-length dot product. Auto-vectorized to NEON by clang.
            float acc = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                acc += dY_row[j] * X_row[j];
            }
            G_row[k] = acc;
        }
    }
}

}  // namespace sparselab
