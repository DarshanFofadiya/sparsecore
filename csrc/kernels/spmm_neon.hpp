// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_neon.hpp
//
//  NEON SIMD declaration for SpMM (sparse-dense matrix multiply).
//
//  Same contract as spmm_scalar (see spmm.hpp for the full signature).
//  The only difference is this version vectorizes the inner j loop with
//  128-bit ARM NEON intrinsics, processing 4 float32 elements per FMA.
//
//  Correctness: this version is verified against spmm_scalar within
//  rtol=atol=1e-5 across all shapes in the test suite. The two may
//  disagree in the last bit or two because NEON's 4-way accumulator
//  changes the effective addition order (float addition is non-
//  associative), but both are "correct" in the float32 sense.
//
//  Performance: theoretical 4x speedup on the inner loop. Real-world
//  wins depend on N (wider N = better SIMD utilization) and sparsity
//  (fewer live entries = fewer inner-loop invocations).
//
//  See docs/design/spmm.md for the full specification and benchmark
//  expectations.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include "padded_csr.hpp"


namespace sparselab {

void spmm_simd_neon(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
);

}  // namespace sparselab
