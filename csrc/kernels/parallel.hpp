// ═══════════════════════════════════════════════════════════════════════════
//  kernels/parallel.hpp
//
//  Compile-time OpenMP shim for SparseCore kernels.
//
//  Rationale
//  ─────────
//  We parallelize the outer row loop of every SpMM-shaped kernel
//  (spmm_scalar, spmm_simd_neon, spmm_grad_w) because rows of W are
//  independent and their output slices (Y[i, :], dW_values[slot]) do
//  not alias across different i.
//
//  OpenMP is optional: setup.py looks for libomp on macOS and the
//  `-fopenmp` flag on Linux; if neither is available it builds without
//  `-fopenmp`, the `_OPENMP` macro stays undefined, and `SCORE_PARALLEL_FOR`
//  below reduces to nothing — the loop runs sequentially.
//
//  This lets us ship a single codebase that auto-parallelizes when
//  possible and is still correct in a minimal build environment.
//
//  Threshold (SCORE_PARALLEL_ROW_THRESHOLD)
//  ────────────────────────────────────────
//  Parallel regions have a fixed fork/join overhead (~3-5μs on M-series
//  Macs). For very small W (say 8x16) that overhead exceeds the work
//  we'd parallelize, so we fall back to sequential. 32 rows is a
//  conservative threshold — most real workloads (MNIST hidden layers at
//  512+, transformer FFN layers at thousands of rows) blow past this
//  easily.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

// Compile-time guard: OpenMP availability comes from whether the
// compiler saw -fopenmp (or on mac, -Xpreprocessor -fopenmp).
#if defined(_OPENMP)
  #include <omp.h>
  #define SCORE_HAVE_OPENMP 1
  // `_Pragma` is the C99-standard way to emit a pragma from a macro
  // expansion. The raw `#pragma` doesn't compose with macros.
  #define SCORE_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
  #define SCORE_HAVE_OPENMP 0
  #define SCORE_PARALLEL_FOR /* no-op when OpenMP is unavailable */
#endif

// Below this row count the outer loop runs sequentially even with OpenMP
// compiled in. Empirically chosen (see milestone_06.md); easy to tune
// later per-architecture if needed.
#define SCORE_PARALLEL_ROW_THRESHOLD 32
