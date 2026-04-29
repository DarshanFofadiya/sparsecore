// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_grad_avx2.hpp
//
//  AVX2 + FMA SIMD declaration for dL/dW at live slots of W (sibling
//  of spmm_grad_neon.hpp — see issue #2 and
//  docs/design/spmm_backward_avx2.md).
//
//  Same contract as spmm_grad_w (see spmm_grad.hpp for full signature
//  documentation). The only difference is this version vectorizes
//  the inner dot-product loop with 256-bit x86 AVX2 intrinsics,
//  processing 16 float32 elements per iteration using two independent
//  8-wide accumulators. The dual-accumulator choice was empirically
//  validated via Gate A0 microbench on AMD Zen 4 (see design
//  §3.3 / §6.0): dual delivers 2.03x over single on that CPU.
//
//  Correctness: this version is verified against spmm_grad_w within
//  rtol=atol=1e-5 across all shapes in the test suite. The two may
//  disagree in the last bit or two because AVX2's dual-accumulator
//  reduction changes the effective addition order (float addition is
//  non-associative), but both are "correct" in the float32 sense.
//  Empirical rel error on the Gate A0 microbench: 2.3e-7.
//
//  Performance: measured Gate 1 baseline on GitHub ubuntu-24.04 AMD
//  EPYC 9V74 is ~3.8 GF/s scalar pre -march, ~4.3 GF/s scalar post
//  -march=x86-64-v3 (see .scratch/gate_1_5_results.md / milestone 13).
//  Target after Phase B AVX2: ~30-40 GF/s on FFN shapes, matching
//  the ship floor / target from design §5.5.
//
//  Parallels with the NEON kernel
//  ──────────────────────────────
//  The C++ function signature is IDENTICAL to spmm_grad_neon.hpp so
//  bindings.cpp can call `sparselab::spmm_grad_w_simd` without
//  caring which architecture is underneath — the setup.py source
//  gating ensures exactly one of the two files is compiled per
//  build. See csrc/bindings.cpp for the dispatch in action.
//
//  Status (Phase A — stub): this file is currently a scalar-identical
//  stub so we can commit the build + binding plumbing before
//  introducing AVX2 intrinsics. Phase B will replace the inner
//  dot-product loop with the dual-accumulator AVX2 pattern from
//  design §3.4.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include "padded_csr.hpp"


namespace sparselab {

// See spmm_grad.hpp for full parameter documentation. This is a
// drop-in replacement — same signature, same memory contracts,
// same self-zeroing behavior. The declared symbol name matches
// spmm_grad_neon.hpp's so that bindings.cpp can call the same
// sparselab::spmm_grad_w_simd regardless of architecture.
void spmm_grad_w_simd(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
);

}  // namespace sparselab
