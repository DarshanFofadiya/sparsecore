// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_grad_neon.cpp
//
//  NEON SIMD implementation of dL/dW at live slots of W.
//
//  Structure mirrors spmm_grad.cpp (the scalar version) at the outer
//  two loops: for each row i, for each live slot s pointing at column
//  c, compute dW_values[slot] = dot(dY[i, :], X[c, :]).
//
//  What's different here: the inner N-length dot product is vectorized
//  with the same 8-wide dual-accumulator NEON pattern we use in
//  spmm_neon.cpp — two independent 4-wide FMA chains unrolled 2×.
//  See §3 below for the detailed argument.
//
//  Correctness contract (identical to scalar):
//    - Self-zeros dW_values[0 .. total_capacity) at entry
//    - Padding slots remain 0.0 on exit (live slots overwrite; padding
//      untouched)
//    - Output aligned with W.values so optimizers can do a single
//      vectorized W.values -= lr * dW
//
//  Numerical note:
//    NEON's dual-accumulator changes the effective addition order of
//    the N dot-product terms vs the scalar single-accumulator version.
//    Float addition is non-associative, so per-slot outputs may differ
//    in the last 1-2 bits. This is the same looseness our test suite
//    already tolerates in spmm_simd vs spmm_scalar (rtol=atol=1e-5).
// ═══════════════════════════════════════════════════════════════════════════

#include "spmm_grad_neon.hpp"
#include "parallel.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument

#if defined(__ARM_NEON)
  #include <arm_neon.h>
#else
  #error "spmm_grad_neon.cpp requires ARM NEON (compile for arm64). "\
         "On x86 the bindings layer routes spmm_grad_w_simd to the scalar kernel."
#endif


namespace sparselab {

void spmm_grad_w_simd(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
) {
    // ─── Shape check (matches scalar) ──────────────────────────────────
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_grad_w_simd: W.ncols (" + std::to_string(W.ncols) +
            ") does not equal K (" + std::to_string(K) + ")."
        );
    }

    const int64_t M = W.nrows;
    const int32_t cap = W.total_capacity();

    // ─── Zero the output buffer ────────────────────────────────────────
    // Same self-zeroing contract as scalar: callers don't pre-zero,
    // padding slots stay at 0.0 so W.values -= lr * dW is safe.
    std::memset(dW_values, 0,
                static_cast<std::size_t>(cap) * sizeof(float));

    // ─── Fast-path: nothing to do ──────────────────────────────────────
    if (M == 0 || N == 0 || K == 0 || cap == 0) {
        return;
    }

    // ─── Main triple-nested loop: i → s → (SIMD j) ────────────────────
    //
    // Parallelism: each row i writes to dW_values[row_start[i] ..
    // row_start[i] + row_capacity[i]]. PaddedCSR invariants guarantee
    // these slices don't overlap across i, so parallel writes are
    // race-free without atomics — same pattern as spmm_neon.
    #if SCORE_HAVE_OPENMP
    #pragma omp parallel for schedule(static) \
        if(M >= SCORE_PARALLEL_ROW_THRESHOLD)
    #endif
    for (int64_t i = 0; i < M; ++i) {
        const int32_t row_ptr = W.row_start[i];
        const int32_t n_live  = W.row_nnz[i];
        const float* dY_row = dY + i * N;

        for (int32_t s = 0; s < n_live; ++s) {
            const int32_t slot = row_ptr + s;
            const int32_t c = W.col_indices[slot];
            const float* X_row = X + static_cast<int64_t>(c) * N;

            // ─── NEON inner dot product ───────────────────────────────
            //
            // Scalar equivalent (what we measured at ~14 GF/s in Gate 1):
            //     float acc = 0.0f;
            //     for (int64_t j = 0; j < N; ++j) acc += dY_row[j] * X_row[j];
            //
            // We replace the single-accumulator scalar loop with three
            // phases driven by two independent NEON FMA accumulators:
            //
            //   Phase A: main loop, 8 floats per iter (2x unrolled 4-wide)
            //   Phase B: trail, 4 floats per iter if 4-7 left after A
            //   Phase C: scalar 1-3 residue
            //
            // Finally one horizontal reduction: (acc_a + acc_b) → scalar.
            //
            // Why dual 4-wide accumulators (not plain 4-wide single)?
            //   A single NEON FMA into the same register has a 4-cycle
            //   dependency chain: the CPU cannot issue the next FMA
            //   until the prior one retires. Two independent
            //   accumulators break that chain — the M-series can
            //   dispatch 2 FMAs per cycle in parallel.
            //
            // Why inline here instead of calling vector_dot_simd_neon?
            //   At 40M-param FFN scale we have ~6.5M live slots per
            //   backward pass. A function-call-per-slot would be
            //   ~2ms of pure call overhead even at 0.3ns/call.
            //   Inlining also lets the compiler keep acc_a/acc_b in
            //   registers across phases without spilling.
            //
            // This is literally the same inner-loop shape as
            // spmm_neon.cpp — intentional pattern mirroring. If you
            // ever wonder whether to tune one, tune both.
            // ─────────────────────────────────────────────────────────
            float32x4_t acc_a = vdupq_n_f32(0.0f);
            float32x4_t acc_b = vdupq_n_f32(0.0f);

            // Phase A: 8 floats per iteration.
            // Per iteration (~5-6 cycles on M-series):
            //   4 × vld1q_f32   — 2 loads from dY_row, 2 from X_row
            //   2 × vfmaq_f32   — independent; can issue in same cycle
            //
            // Throughput: 8 floats / ~5 cycles ≈ 1.6 floats/cycle
            // vs scalar ~1 FMA per 4 cycles ≈ 0.25 floats/cycle.
            int64_t j = 0;
            for (; j + 8 <= N; j += 8) {
                float32x4_t dy_a = vld1q_f32(dY_row + j);
                float32x4_t dy_b = vld1q_f32(dY_row + j + 4);
                float32x4_t x_a  = vld1q_f32(X_row  + j);
                float32x4_t x_b  = vld1q_f32(X_row  + j + 4);
                // Each FMA: acc_{a,b}[k] += dy_{a,b}[k] * x_{a,b}[k]
                // for k = 0..3. One instruction, one rounding, four lanes.
                acc_a = vfmaq_f32(acc_a, dy_a, x_a);
                acc_b = vfmaq_f32(acc_b, dy_b, x_b);
            }

            // Phase B: one more 4-wide iteration if 4-7 floats remain.
            // At this point j is at a multiple of 8 and at most 7
            // elements are left. If ≥4 remain, absorb them here with a
            // single SIMD FMA into acc_a (acc_b is done).
            if (j + 4 <= N) {
                float32x4_t dy_vec = vld1q_f32(dY_row + j);
                float32x4_t x_vec  = vld1q_f32(X_row  + j);
                acc_a = vfmaq_f32(acc_a, dy_vec, x_vec);
                j += 4;
            }

            // Horizontal reduction: sum both accumulators' 4 lanes
            // into one scalar. vaddq_f32 fuses acc_a and acc_b
            // lane-wise first (1 instruction), then vaddvq_f32 sums
            // the 4 lanes (1 instruction). One reduction per live
            // slot, so amortized over the Phase-A iterations.
            float acc = vaddvq_f32(vaddq_f32(acc_a, acc_b));

            // Phase C: scalar residue for the final 0-3 elements.
            // N = 1, 2, 3 hit this directly (j starts at 0 and both
            // Phase A and Phase B skip). N = 5, 6, 7 land here with
            // 1-3 elements left. N = 13, 17, 33 exercise it after
            // one full 8-wide + possible 4-wide iter. Our tests
            // cover each of these residues explicitly.
            for (; j < N; ++j) {
                acc += dY_row[j] * X_row[j];
            }

            dW_values[slot] = acc;
        }
    }
}

}  // namespace sparselab
