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
//  │ PHASE A STATUS (current): SCALAR-IDENTICAL STUB.                │
//  │                                                                 │
//  │ Body below is a verbatim copy of spmm_grad.cpp so we can land   │
//  │ the build + binding + dispatch plumbing in a separate commit    │
//  │ from the actual AVX2 kernel. CI's existing kernel_fn-parametrized│
//  │ oracle tests will pass against this stub (the stub IS the       │
//  │ scalar kernel), confirming dispatch correctness without mixing  │
//  │ in arithmetic risk from intrinsics.                             │
//  │                                                                 │
//  │ PHASE B WILL: replace the inner `for (j=0; j<N; ++j)` dot loop  │
//  │ with the dual-accumulator AVX2 pattern. No other code in this   │
//  │ file should need to change during Phase B — outer loops, shape  │
//  │ checks, memset, parallelism all stay identical. Spec §3.4 has   │
//  │ the exact intrinsic sequence.                                   │
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

            // ─── Phase A STUB: scalar inner dot loop ──────────────────
            // Verbatim copy of the scalar kernel's body. Phase B will
            // replace this with the dual-accumulator AVX2 pattern
            // from design §3.4 — two __m256 accumulators, 16
            // floats/iter via _mm256_fmadd_ps, scalar remainder tail,
            // 3-step horizontal reduction at end.
            float acc = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                acc += dY_row[j] * X_row[j];
            }
            dW_values[slot] = acc;
        }
    }
}

}  // namespace sparselab
