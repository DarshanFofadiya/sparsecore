// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_neon.cpp
//
//  NEON SIMD implementation of Y = W @ X for W in PaddedCSR format.
//
//  Algorithm shape is identical to spmm_scalar:
//    for each output row i:
//      for each live slot s in that row:
//        for each output column j:
//          Y[i,j] += W.values[s] * X[W.col_indices[s], j]
//
//  The difference: the innermost j loop processes 4 floats per iteration
//  using 128-bit NEON intrinsics. Everything else is scalar.
//
//  Why vectorize only the inner loop?
//    - It's the only loop whose iterations are independent and with
//      contiguous memory access (Y and X rows are row-major float32).
//    - The outer i/s loops are branchy (empty rows, variable row_nnz)
//      and their trip counts are too small to amortize SIMD setup cost.
//    - This matches the well-known "SpMM scalar outer, SIMD inner" shape
//      used in intel/MKL, Eigen sparse, and every production CPU SpMM.
//
//  Borrow-Don't-Reinvent note:
//    This is the classic CSR-SpMM SIMD pattern. Implementations in MKL,
//    OpenBLAS, and Eigen::SparseMatrix all use approximately this
//    structure (outer scalar walk over (i, s), inner SIMD over j).
//    We are not inventing anything here — just porting the pattern to
//    ARM NEON intrinsics and our PaddedCSR layout.
// ═══════════════════════════════════════════════════════════════════════════

#include "spmm_neon.hpp"
#include "parallel.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument

#if defined(__ARM_NEON)
  #include <arm_neon.h>
#else
  #error "spmm_neon.cpp requires ARM NEON (compile for arm64)."
#endif


namespace sparsecore {

void spmm_simd_neon(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
) {
    // ─── Shape check (matches the scalar kernel) ───────────────────────
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_simd_neon: W.ncols (" + std::to_string(W.ncols) +
            ") does not equal K (" + std::to_string(K) + ")."
        );
    }

    const int64_t M = W.nrows;

    // ─── Zero the output buffer ────────────────────────────────────────
    // Same contract as the scalar kernel — the caller doesn't need to
    // pre-zero Y because we do it here.
    std::memset(Y, 0, static_cast<std::size_t>(M) *
                      static_cast<std::size_t>(N) * sizeof(float));

    // ─── Fast-path: empty matrices ─────────────────────────────────────
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // ─── Main triple-nested loop: i → s → (SIMD j) ─────────────────────
    //
    // ─── Parallelism ──────────────────────────────────────────────────
    // Identical shape to spmm_scalar: each i writes only to Y[i, :],
    // every read is const. The OpenMP `if` clause gates parallelism at
    // runtime — small matrices stay sequential and avoid fork/join cost.
    // See kernels/parallel.hpp.
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

            // ─── NEON inner loop ───────────────────────────────────────
            //
            // The scalar equivalent is:
            //     for (int64_t j = 0; j < N; ++j) y_row[j] += v * x_row[j];
            //
            // We replace it with three phases:
            //   Phase A: main SIMD loop, 8 floats (2 NEON regs) per iter
            //   Phase B: trail SIMD, 4 floats per iter for 0-7 leftover
            //   Phase C: scalar tail for the last 0-3 leftover floats
            //
            // Why 8-wide (2x unroll) instead of plain 4-wide?
            //   Modern Apple M-series cores can dispatch 2 FMAs per cycle.
            //   A single 4-wide NEON loop produces 1 FMA per iteration with
            //   a 4-cycle dependency chain on the same accumulator (can't
            //   issue the next FMA until the previous completes).
            //   By keeping two independent 4-wide vectors (y_a, y_b) we
            //   break the dependency chain and expose instruction-level
            //   parallelism — the CPU can execute both FMAs in the same
            //   cycle.
            //
            //   We tried a simpler 4-wide version first and found Apple
            //   Clang's auto-vectorization of the scalar loop actually
            //   beat our hand-written NEON, because the compiler already
            //   does unrolling + ILP for us when the code is simple.
            //   2x-unrolled NEON puts us ahead again.
            //
            // Broadcast the scalar weight `v` into all 4 lanes of a NEON
            // register once per live-slot. Amortized over N/8 iterations.
            // ─────────────────────────────────────────────────────────
            float32x4_t v_vec = vdupq_n_f32(v);

            // Phase A: main SIMD loop, 2x unrolled (8 floats per iter).
            //
            // Per iteration (~5-6 cycles on M-series):
            //   4 vld1q_f32  — 2 loads from x_row, 2 from y_row
            //   2 vfmaq_f32  — independent, can issue in the same cycle
            //   2 vst1q_f32  — back to y_row
            //
            // Throughput: ~8 floats processed per ~5 cycles ≈ 1.6 floats/cycle
            // vs the simple 4-wide NEON at ~1 float/cycle.
            int64_t j = 0;
            for (; j + 8 <= N; j += 8) {
                // Two independent SIMD FMAs. The CPU sees no data
                // dependency between them (y_a and y_b are disjoint
                // registers) so it schedules them in parallel.
                float32x4_t x_a = vld1q_f32(x_row + j);
                float32x4_t x_b = vld1q_f32(x_row + j + 4);
                float32x4_t y_a = vld1q_f32(y_row + j);
                float32x4_t y_b = vld1q_f32(y_row + j + 4);
                y_a = vfmaq_f32(y_a, v_vec, x_a);
                y_b = vfmaq_f32(y_b, v_vec, x_b);
                vst1q_f32(y_row + j, y_a);
                vst1q_f32(y_row + j + 4, y_b);
            }

            // Phase B: one more 4-wide SIMD iter if 4-7 floats remain.
            if (j + 4 <= N) {
                float32x4_t x_vec = vld1q_f32(x_row + j);
                float32x4_t y_vec = vld1q_f32(y_row + j);
                y_vec = vfmaq_f32(y_vec, v_vec, x_vec);
                vst1q_f32(y_row + j, y_vec);
                j += 4;
            }

            // Phase C: scalar tail.
            //
            // If N % 4 != 0, 1-3 elements remain after phases A and B.
            // Handle them with plain scalar FMA. `j` carries over from
            // whichever earlier phase ran last.
            //
            // Off-by-one alert: the Oracle tests include N=5, 13, 17, 33,
            // 34, 35 specifically to stress each residue of j mod 4 /
            // j mod 8. If this loop is buggy those cases blow up.
            for (; j < N; ++j) {
                y_row[j] += v * x_row[j];
            }
        }
    }
}

}  // namespace sparsecore
