// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_grad_avx2.cpp
//
//  AVX2 + FMA SIMD implementation of dL/dW at live slots of W.
//
//  Sibling of spmm_grad_neon.cpp for the Linux x86_64 build path.
//  Closes the Linux-parity gap documented in milestone 13 and
//  specified in docs/design/spmm_backward_avx2.md. The dual-
//  accumulator design was empirically validated on AMD Zen 4 via
//  the Gate A0 microbench in csrc/bench/avx2_dot_microbench.cpp
//  (see design §6.0): dual accumulator delivered 2.03× over single,
//  motivating the design revision to ship dual as v0.2 rather than
//  treating it as v0.3 headroom.
//
//  Structure mirrors spmm_grad.cpp (the scalar version) at the outer
//  two loops: for each row i, for each live slot s pointing at column
//  c, compute dW_values[slot] = dot(dY[i, :], X[c, :]).
//
//  What IS different here vs scalar (once Phase B lands): the inner
//  N-length dot product is vectorized with the 16-float-per-iteration
//  dual-accumulator AVX2 pattern from design §3.4 — two independent
//  256-bit __m256 accumulators chained with _mm256_fmadd_ps, a
//  trailing 8-wide Phase B iteration for 8-15 leftover floats, and
//  a scalar Phase C tail for the final 0-7 remainder.
//
//  Correctness contract (identical to scalar and NEON):
//    - Self-zeros dW_values[0 .. total_capacity) at entry
//    - Padding slots remain 0.0 on exit (live slots overwrite;
//      padding untouched)
//    - Output aligned with W.values so optimizers can do a single
//      vectorized W.values -= lr * dW
//
//  Numerical note:
//    AVX2's dual-accumulator changes the effective addition order
//    of the N dot-product terms vs the scalar single-accumulator
//    version. Float addition is non-associative, so per-slot outputs
//    may differ in the last 1-2 bits. This is the same looseness
//    our test suite already tolerates in the NEON variant
//    (rtol=atol=1e-5).
//
//  ┌─────────────────────────────────────────────────────────────────┐
//  │ PHASE B STATUS (current): REAL AVX2 DUAL-ACCUMULATOR KERNEL.    │
//  │                                                                 │
//  │ Inner dot loop is vectorized with the dual-accumulator AVX2     │
//  │ pattern from design §3.4. Outer loops, shape checks, memset,    │
//  │ and parallelism are unchanged from the scalar and Phase-A stub  │
//  │ versions — only the inner `for (j; j<N; ++j)` body was rewritten│
//  │ to use intrinsics.                                              │
//  └─────────────────────────────────────────────────────────────────┘
// ═══════════════════════════════════════════════════════════════════════════

#include "spmm_grad_avx2.hpp"
#include "parallel.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument

// AVX2 intrinsics. Conditionally included so that if this file is
// accidentally compiled on non-x86 (shouldn't happen thanks to
// setup.py's IS_X86_64 gate, but belt-and-suspenders), the compile
// fails early rather than silently producing a non-AVX2 binary.
#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
#else
  #error "spmm_grad_avx2.cpp requires AVX2 + FMA (compile for x86-64-v3). "\
         "On non-x86 the bindings layer routes spmm_grad_w_simd to the NEON "\
         "or scalar kernel as appropriate."
#endif


namespace sparselab {

void spmm_grad_w_simd(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
) {
    // ─── Shape check (matches scalar and NEON) ─────────────────────────
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_grad_w_simd: W.ncols (" + std::to_string(W.ncols) +
            ") does not equal K (" + std::to_string(K) + ")."
        );
    }

    const int64_t M = W.nrows;
    const int32_t cap = W.total_capacity();

    // ─── Zero the output buffer ────────────────────────────────────────
    // Same self-zeroing contract as scalar and NEON: callers don't
    // need to pre-zero. Ensures padding slots stay at 0.0 so
    // W.values -= lr * dW is safe without a mask pass.
    std::memset(dW_values, 0,
                static_cast<std::size_t>(cap) * sizeof(float));

    // ─── Fast-path: nothing to do ──────────────────────────────────────
    if (M == 0 || N == 0 || K == 0 || cap == 0) {
        return;
    }

    // ─── Main loop: one dot product per live slot ──────────────────────
    //
    // Each row i writes only to dW_values[row_start[i] : row_start[i]+n_live].
    // PaddedCSR invariants guarantee these slices don't overlap
    // across i, so parallel writes are race-free without atomics.
    // Same pattern as scalar and NEON.
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

            // ─── AVX2 inner dot product ───────────────────────────────
            //
            // Scalar equivalent (kept as the oracle in tests,
            // measured at ~3.8-4.3 GF/s on Zen 4 pre-AVX2 — see
            // Gate 1.5 / milestone 13):
            //     float acc = 0.0f;
            //     for (int64_t j = 0; j < N; ++j) acc += dY_row[j] * X_row[j];
            //
            // We replace the single-accumulator scalar loop with three
            // phases driven by two independent 256-bit AVX2 FMA
            // accumulators:
            //
            //   Phase A: main loop, 16 floats per iter (2x unrolled 8-wide)
            //   Phase B: trail, 8 floats per iter if 8-15 left after A
            //   Phase C: scalar 0-7 residue
            //
            // Finally one horizontal reduction: (acc_a + acc_b) → scalar.
            //
            // Why dual 8-wide accumulators (not a single 8-wide)?
            //   A single _mm256_fmadd_ps into the same register has a
            //   3-5 cycle dependency chain: on Zen 4 the CPU cannot
            //   issue the next FMA into `acc` until the prior one
            //   retires (~3 cycles). Two independent accumulators
            //   break that chain — Zen 4's scheduler dispatches both
            //   FMAs in the same cycle.
            //
            //   Empirically validated by Gate A0 microbench
            //   (.github/workflows/validate_avx2_microbench.yml):
            //   dual accumulator delivered 2.03× over single on the
            //   Zen 4 CI runner. Same shape applies to Haswell+ Intel
            //   parts (often 2.5×) and Zen 1 (~1.3×).
            //
            // Why inline here instead of a vector_dot_avx2 helper?
            //   At 40M FFN scale we have ~6.5M live slots per backward
            //   pass. A function-call-per-slot would be ~2 ms of pure
            //   call overhead. Inlining also lets the compiler keep
            //   acc_a/acc_b in ymm registers across phases without
            //   spilling.
            //
            // Alignment: both dY_row and X_row are 16-byte aligned
            // (std::vector<float>) but not guaranteed 32-byte aligned.
            // We use _mm256_loadu_ps (unaligned 256-bit load)
            // throughout — on every Zen+ and Haswell+ part, unaligned
            // 256-bit loads that stay within a 64-byte cache line
            // are 1 cycle / load port, same as aligned. Do NOT use
            // _mm256_load_ps here — any misalignment would #GP-fault.
            //
            // This is intentionally the same inner-loop shape as
            // spmm_grad_neon.cpp — pattern mirroring. If you tune one,
            // tune both.
            // ─────────────────────────────────────────────────────────
            __m256 acc_a = _mm256_setzero_ps();
            __m256 acc_b = _mm256_setzero_ps();

            // Phase A: 16 floats per iteration.
            // Per iteration (~5-6 cycles on Zen 4):
            //   4 × _mm256_loadu_ps  — 2 loads from dY_row, 2 from X_row
            //   2 × _mm256_fmadd_ps  — independent; scheduler issues
            //                          both in parallel
            //
            // Throughput: 16 floats / ~5 cycles ≈ 3.2 floats/cycle,
            // vs pre-AVX2 scalar ~1 FMA per 2-3 cycles ≈ 0.4 floats/cycle.
            int64_t j = 0;
            for (; j + 16 <= N; j += 16) {
                // Load two contiguous 8-float lanes from dY_row.
                // _mm256_loadu_ps fills a 256-bit ymm register with
                // 8 consecutive floats starting at the pointer.
                __m256 dy_a = _mm256_loadu_ps(dY_row + j);
                __m256 dy_b = _mm256_loadu_ps(dY_row + j + 8);
                // And two matching 8-float lanes from X_row.
                __m256 x_a  = _mm256_loadu_ps(X_row  + j);
                __m256 x_b  = _mm256_loadu_ps(X_row  + j + 8);
                // Two fused multiply-adds, one per accumulator.
                // _mm256_fmadd_ps(A, B, C) computes A*B + C with
                // single rounding (fused), 8 lanes in parallel.
                // Because acc_a and acc_b are distinct registers,
                // these two FMAs have NO data dependency on each
                // other — the out-of-order scheduler dispatches
                // them together, doubling effective throughput
                // over a single-accumulator loop.
                acc_a = _mm256_fmadd_ps(dy_a, x_a, acc_a);
                acc_b = _mm256_fmadd_ps(dy_b, x_b, acc_b);
            }

            // Fuse the two 8-wide accumulators into one before the
            // Phase B / Phase C tail. Lane-wise add — 1 cycle latency.
            // After this, `acc` holds 8 partial sums; acc_b is retired.
            __m256 acc = _mm256_add_ps(acc_a, acc_b);

            // Phase B: one more 8-wide iteration if 8-15 floats remain.
            // At this point j is a multiple of 16 and at most 15
            // elements are left. If ≥ 8 remain, absorb them here with
            // a single 8-wide FMA into `acc`.
            if (j + 8 <= N) {
                __m256 dy_vec = _mm256_loadu_ps(dY_row + j);
                __m256 x_vec  = _mm256_loadu_ps(X_row  + j);
                acc = _mm256_fmadd_ps(dy_vec, x_vec, acc);
                j += 8;
            }

            // Horizontal reduction: collapse the 8-lane acc into a
            // single scalar. AVX2 has no one-instruction "sum all
            // lanes" like NEON's vaddvq_f32, so we do the standard
            // 3-step reduction:
            //
            //   1. Split the 256-bit acc into two 128-bit halves
            //      (lo, hi) and add them lane-wise. Result: 4 lanes.
            //   2. Shuffle the upper 2 lanes down and add: 2 lanes.
            //   3. Shuffle lane 1 to lane 0 and add: 1 lane.
            //   4. Extract scalar.
            //
            // Three _mm_add + two shuffles, ~6 cycles total — one
            // reduction per live slot, amortized over Phase A's many
            // iterations.
            __m128 lo = _mm256_castps256_ps128(acc);       // [l0 l1 l2 l3]
            __m128 hi = _mm256_extractf128_ps(acc, 1);     // [h0 h1 h2 h3]
            __m128 sum4 = _mm_add_ps(lo, hi);              // [l0+h0 l1+h1 l2+h2 l3+h3]
            // movehl: move upper 2 lanes of sum4 into lower 2 of the result,
            // giving [l2+h2, l3+h3, _, _] in the low half.
            __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
            // shuffle selects lane 1 into lane 0: [lane1, _, _, _].
            __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
            float acc_scalar = _mm_cvtss_f32(sum1);

            // Phase C: scalar residue for the final 0-7 elements.
            // N = 1..7 hit this directly (j starts at 0 and both
            // Phase A and Phase B skip). N = 9..15 land here with
            // 1..7 elements left after Phase B. N = 17, 33, ...
            // exercise it after Phase A. Our tests explicitly cover
            // each of these residues (see design §5.2).
            for (; j < N; ++j) {
                acc_scalar += dY_row[j] * X_row[j];
            }

            dW_values[slot] = acc_scalar;
        }
    }
}

}  // namespace sparselab
