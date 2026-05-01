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
//  │ PHASE B STATUS (current): REAL AVX2 DUAL-STREAM KERNEL.         │
//  │                                                                 │
//  │ Inner j-loop is vectorized with the dual-stream write-through   │
//  │ AVX2 pattern from design §3.4. Outer loops, shape check,        │
//  │ memset, and OpenMP parallelism are unchanged from the scalar    │
//  │ and Phase-A stub versions — only the inner `for (j; j<N; ++j)`  │
//  │ body was rewritten to use intrinsics.                           │
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

            // ─── AVX2 inner loop ──────────────────────────────────────
            //
            // Scalar equivalent (the stub body and the oracle that
            // tests/test_spmm.py compares against at rtol=atol=1e-5):
            //     for (int64_t j = 0; j < N; ++j) y_row[j] += v * x_row[j];
            //
            // We replace it with three phases driven by two independent
            // 256-bit AVX2 write-through streams:
            //
            //   Phase A: main loop, 16 floats per iter (2x unrolled 8-wide)
            //   Phase B: trail, 8 floats per iter if 8-15 left after A
            //   Phase C: scalar 0-7 residue
            //
            // Broadcast the scalar weight `v` into all 8 lanes of a
            // __m256 register once per live slot (one vbroadcastss
            // instruction, ~1 cycle). Hoisted out of the j-loop — doing
            // it per-iteration would repeat it N/16 times per slot for
            // no gain. Amortized over Phase A's many iterations.
            //
            // Why dual 8-wide streams (not plain 8-wide single)?
            //   Unlike the dW kernel (where dual accumulators broke an
            //   FMA data dependency on a single register), forward does
            //   write-through FMA — each SIMD iteration's result goes
            //   straight back to memory, not accumulated in a register.
            //   There is no per-register dependency chain to break.
            //
            //   BUT: two independent streams still win via memory-level
            //   parallelism. Zen 4 has 2 load ports + 1 store port per
            //   cycle. A single 8-wide stream uses one load port and
            //   one store port per iter; two streams keep both load
            //   ports busy and queue stores tightly. Expected dual-
            //   over-single ratio: 1.2–1.4× on Zen 4 (memory-bound),
            //   vs dW's 2× (compute-bound). See design §3.3.
            //
            // Why inline instead of a vector_axpy_avx2 helper?
            //   At 40M FFN scale we have ~6.5M live slots per forward
            //   pass. A per-slot call would cost ~2 ms of pure
            //   overhead. Inlining also lets the compiler keep `v_vec`
            //   hoisted in a ymm register across the entire j-loop.
            //
            // Alignment: x_row and y_row come from std::vector<float>
            // / py::array_t<float>; both are 16-byte aligned but not
            // guaranteed 32-byte. We use _mm256_loadu_ps /
            // _mm256_storeu_ps (unaligned) throughout — on Zen+ and
            // Haswell+, unaligned 256-bit accesses that don't cross a
            // 64-byte cache line are penalty-free at 1/cycle per port.
            // Do NOT use the aligned variants here — any misalignment
            // would #GP-fault.
            //
            // No horizontal reduction at end of the slot — unlike dW,
            // forward writes each lane straight back to y_row. This
            // is part of why forward is memory-bound while dW is
            // compute-bound (design §3.5).
            //
            // Pattern mirrors spmm_neon.cpp (8-wide lanes there, 16
            // here) and spmm_grad_avx2.cpp (dual-stream structure).
            // If you tune one, consider the others.
            // ─────────────────────────────────────────────────────────
            __m256 v_vec = _mm256_set1_ps(v);

            // Phase A: 16 floats per iteration.
            // Per iteration (~6-7 cycles on Zen 4, store-port bound):
            //   4 × _mm256_loadu_ps   — 2 from x_row, 2 from y_row
            //   2 × _mm256_fmadd_ps   — independent on disjoint lanes
            //   2 × _mm256_storeu_ps  — back to y_row
            //
            // Throughput: 16 floats / ~6 cycles ≈ 2.7 floats/cycle.
            // At 2-core OpenMP with L1 residency that translates to
            // ~15-20 GF/s realized kernel throughput, ~5× the scalar
            // ~3.8 GF/s baseline. See design §6.2 for the full
            // bandwidth-ceiling argument.
            int64_t j = 0;
            for (; j + 16 <= N; j += 16) {
                // Load 16 floats of x_row into two 8-lane registers.
                // _mm256_loadu_ps fills a ymm register with 8
                // consecutive float32 lanes starting at the pointer.
                __m256 x_a = _mm256_loadu_ps(x_row + j);
                __m256 x_b = _mm256_loadu_ps(x_row + j + 8);
                // Load the matching 16 floats of y_row so we can
                // accumulate v*x into them (y += v*x in-place).
                __m256 y_a = _mm256_loadu_ps(y_row + j);
                __m256 y_b = _mm256_loadu_ps(y_row + j + 8);
                // Two FMAs, one per stream. Because y_a and y_b are
                // disjoint register destinations AND write to disjoint
                // Y slots (j..j+7 vs j+8..j+15), the CPU sees no data
                // dependency between them and the out-of-order
                // scheduler issues both in the same cycle.
                //
                // _mm256_fmadd_ps(A, B, C) computes A*B + C with a
                // single rounding step (fused), 8 lanes in parallel.
                y_a = _mm256_fmadd_ps(v_vec, x_a, y_a);
                y_b = _mm256_fmadd_ps(v_vec, x_b, y_b);
                // Write both updated 8-lane results back to y_row.
                // The unaligned store is 1 cycle/issue on Zen+ /
                // Haswell+ when the 32-byte write stays within a
                // 64-byte cache line (which it does for our
                // contiguous row-major Y).
                _mm256_storeu_ps(y_row + j,     y_a);
                _mm256_storeu_ps(y_row + j + 8, y_b);
            }

            // Phase B: one more 8-wide iteration if 8-15 floats
            // remain. j is at a multiple of 16 after Phase A; if ≥ 8
            // elements are left we do a single-stream 8-wide
            // load-FMA-store and advance j by 8. Phase C then picks
            // up the last 0-7 residue.
            if (j + 8 <= N) {
                __m256 x_vec = _mm256_loadu_ps(x_row + j);
                __m256 y_vec = _mm256_loadu_ps(y_row + j);
                y_vec = _mm256_fmadd_ps(v_vec, x_vec, y_vec);
                _mm256_storeu_ps(y_row + j, y_vec);
                j += 8;
            }

            // Phase C: scalar tail for the final 0-7 elements.
            // N = 1..7 hits this directly (j stays 0 since both Phase
            // A and Phase B skip). N = 9..15 land here with 1..7
            // elements left after Phase B. N = 17, 33, 65 exercise it
            // after one or more full Phase A iters. The N-residue
            // sweep in tests/test_spmm_avx2.py (Phase C of the spec,
            // not yet committed) explicitly covers each of these.
            for (; j < N; ++j) {
                y_row[j] += v * x_row[j];
            }
        }
    }
}

}  // namespace sparselab
