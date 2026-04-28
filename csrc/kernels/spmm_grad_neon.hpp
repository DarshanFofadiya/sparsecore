// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_grad_neon.hpp
//
//  NEON SIMD declaration for dL/dW at live slots of W (Phase 2 of
//  milestone 12 — issue #1).
//
//  Same contract as spmm_grad_w (see spmm_grad.hpp for the full
//  signature documentation). The only difference is this version
//  vectorizes the inner dot-product loop with 128-bit ARM NEON
//  intrinsics, processing 8 float32 elements per iteration using two
//  independent 4-wide accumulators.
//
//  Correctness: this version is verified against spmm_grad_w within
//  rtol=atol=1e-5 across all shapes in the test suite. The two may
//  disagree in the last bit or two because NEON's dual-accumulator
//  reduction changes the effective addition order (float addition is
//  non-associative), but both are "correct" in the float32 sense.
//
//  Performance: measured Gate 1 baseline on M3 Pro (scalar) is ~14
//  GF/s across FFN shapes (see examples/profile_dw_baseline.py).
//  Target after Phase 2 NEON: 3-5x local speedup → ~1.5-1.8x
//  end-to-end training step speedup at 40M scale.
//
//  See docs/design/spmm_backward_neon.md for the full specification
//  and benchmark expectations. See docs/design/spmm_backward.md for
//  the mathematical derivation (dL/dW = dot(dY[i,:], X[k,:]) at each
//  live slot (i, k)).
//
//  Status (Phase A — stub): this file is currently a scalar-identical
//  stub so we can commit the build + binding plumbing before
//  introducing SIMD logic. Gate 1 measurement showed scalar at 14 GF/s;
//  Phase B will replace the inner dot-product loop with NEON Phase
//  A/B/C (8-wide main + 4-wide trail + scalar remainder).
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include "padded_csr.hpp"


namespace sparselab {

// See spmm_grad.hpp for full parameter documentation. This is a
// drop-in replacement — same signature, same memory contracts,
// same self-zeroing behavior.
void spmm_grad_w_simd(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
);

}  // namespace sparselab
