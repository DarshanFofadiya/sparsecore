// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_avx2.cpp
//
//  AVX2 + FMA SIMD implementation of the forward SpMM path
//  (Y = W @ X, W in PaddedCSR) for Linux x86_64.
//
//  Sibling of spmm_neon.cpp on the ARM side. Companion to the
//  dW AVX2 kernel in spmm_grad_avx2.cpp (milestone 14). This
//  file + the dW kernel together close the Linux x86 parity
//  gap documented in milestone 13.
//
//  See docs/design/spmm_forward_avx2.md for the full algorithm
//  specification, risk register, and projected speedup numbers.
//
//  Structure mirrors spmm.cpp (the scalar version) at the outer
//  two loops: for each row i, for each live slot s pointing at
//  column c with value v, accumulate v * X[c, :] into Y[i, :].
//
//  What IS different here vs scalar (once Phase B lands): the
//  inner N-length j-loop is vectorized with the dual-stream
//  write-through AVX2 pattern from design §3.4 — two independent
//  256-bit __m256 streams of (load Y, load X, fmadd, store Y),
//  each stream processing 8 floats per iteration, giving a total
//  of 16 floats per Phase-A iteration. Phase B handles the 8-15
//  residue after Phase A, Phase C handles the 0-7 scalar tail.
//
//  Correctness contract (identical to scalar and NEON):
//    - Self-zeros Y[0 .. M*N) at entry so callers don't need to
//      pre-zero. Required because py::array_t<float>(shape) does
//      not zero-initialize (see milestone 3c-ii).
//    - Per-slot arithmetic semantically equivalent to
//        Y[i, :] += v * X[c, :]
//      for every live slot (i, c, v) walking W.row_nnz[i].
//    - Output bit-identical to the scalar kernel for most
//      realistic problem sizes; within 1 ULP elementwise at
//      rtol=atol=1e-5 in all cases.
//
//  Numerical note:
//    Unlike the dW AVX2 kernel (which has a per-slot horizontal
//    reduction that changes associative ordering of float adds),
//    the forward kernel writes lanes straight back to memory
//    without any per-slot reduction. Reordering is therefore
//    minimal — most outputs are literally bit-identical to
//    scalar. When they differ, it is by 1 ULP due to the
//    difference between the single scalar accumulator
//    `y_row[j] += v * x_row[j]` and the SIMD lane order that
//    processes 8 j-values in parallel.
//
//  ┌─────────────────────────────────────────────────────────────────┐
//  │ PHASE A STATUS (current): SCALAR-IDENTICAL STUB.                │
//  │                                                                 │
//  │ Body below is a verbatim copy of spmm.cpp so we can land the    │
//  │ build + binding + dispatch plumbing in a separate commit from   │
//  │ the actual AVX2 kernel. CI's existing kernel_fn-parametrized    │
//  │ oracle tests (tests/test_spmm.py) pass against this stub (the   │
//  │ stub IS the scalar kernel), confirming dispatch correctness     │
//  │ without mixing in arithmetic risk from intrinsics.              │
//  │                                                                 │
//  │ PHASE B WILL: replace the inner `for (j=0; j<N; ++j)` loop with │
//  │ the dual-stream write-through AVX2 pattern from design §3.4.    │
//  │ No other code in this file should need to change during Phase B │
//  │ — outer loops, shape checks, memset, parallelism all stay       │
//  │ identical. See docs/design/spmm_forward_avx2.md §3.4 for the    │
//  │ exact intrinsic sequence.                                       │
//  └─────────────────────────────────────────────────────────────────┘
// ═══════════════════════════════════════════════════════════════════════════

#include "spmm_avx2.hpp"
#include "parallel.hpp"

#include <cstring>      // std::memset
#include <stdexcept>    // std::invalid_argument

// AVX2 intrinsics. Conditionally included so that if this file
// is accidentally compiled on non-x86 (shouldn't happen thanks
// to setup.py's IS_X86_64 gate, but belt-and-suspenders), the
// compile fails early rather than silently producing a non-AVX2
// binary. Addresses AC-6.1 from the requirements doc.
#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
#else
  #error "spmm_avx2.cpp requires AVX2 + FMA (compile for x86-64-v3). "\
         "On non-x86 the bindings layer routes spmm_simd to the NEON "\
         "or scalar kernel as appropriate."
#endif


namespace sparselab {

void spmm_simd_avx2(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
) {
    // ─── Shape check (matches scalar and NEON) ─────────────────────────
    // Strict: caller must pass K = W.ncols. The Python binding
    // verifies this, but we double-check in C++ as a safety net.
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_simd_avx2: W.ncols (" + std::to_string(W.ncols) +
            ") does not equal K (" + std::to_string(K) + ")."
        );
    }

    const int64_t M = W.nrows;

    // ─── Zero the output buffer ────────────────────────────────────────
    // Same self-zeroing contract as scalar and NEON: caller does
    // not pre-zero Y. Required because py::array_t<float>(shape)
    // uses PyArray_SimpleNew, which leaves memory in whatever
    // state the allocator returned. The write-through FMA below
    // assumes Y starts at zero.
    std::memset(Y, 0, static_cast<std::size_t>(M) *
                      static_cast<std::size_t>(N) * sizeof(float));

    // ─── Fast-path: empty matrices ─────────────────────────────────────
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // ─── Main triple-nested loop: i → s → j ───────────────────────────
    //
    // Outer loop is embarrassingly parallel: each row i writes
    // only to Y[i, :]; all reads are from const W and const X.
    // Same OpenMP pragma as scalar, NEON, and the dW AVX2 kernel.
    //
    // For rows below SCORE_PARALLEL_ROW_THRESHOLD the fork/join
    // overhead dominates the work — the `if(...)` clause gates
    // the parallel region without duplicating the loop body.
    // See kernels/parallel.hpp.
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

            // ─── Phase A STUB: scalar inner loop ──────────────────────
            // Verbatim copy of the scalar kernel's body. Phase B
            // will replace this with the dual-stream AVX2 pattern
            // from design §3.4 — hoisted _mm256_set1_ps(v), two
            // independent __m256 write-through streams processing
            // 16 floats/iter, 8-wide Phase B trail, scalar Phase C
            // tail. Plain FMA: y[j] += v * x[j].
            for (int64_t j = 0; j < N; ++j) {
                y_row[j] += v * x_row[j];
            }
        }
    }
}

}  // namespace sparselab
