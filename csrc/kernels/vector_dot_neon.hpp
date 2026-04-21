// ═══════════════════════════════════════════════════════════════════════════
//  kernels/vector_dot_neon.hpp
//
//  Declaration of the NEON SIMD vector dot-product kernel.
//
//  Same signature as vector_dot_scalar, different implementation. The
//  two kernels are numerically close (within rtol=atol=1e-5) but not
//  bit-identical, because NEON accumulates into 4 parallel lanes and
//  sums them at the end, whereas the scalar version does one running
//  sum — float addition is non-associative, so the reordering causes
//  last-bit differences.
//
//  This is the Oracle-verified stand-in for the scalar kernel. Every
//  call site that uses vector_dot_scalar can switch to this and get
//  the same result with measurable speedup on Apple Silicon.
//
//  ARM-only. Enabled behind #ifdef __ARM_NEON; x86 AVX2 equivalent is
//  a post-v0.1 contribution opportunity.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstddef>


namespace sparsecore {

// ─────────────────────────────────────────────────────────────────────────
//  vector_dot_simd_neon
//
//  Computes sum over i of a[i] * b[i] using NEON 128-bit SIMD (4-wide
//  float32). Main loop processes 4 elements per iteration with fused
//  multiply-add (vfmaq_f32); a scalar remainder loop handles the final
//  1-3 elements when n is not a multiple of 4.
//
//  Parameters, returns, and empty-input behavior: identical to
//  vector_dot_scalar. See kernels/vector_dot.hpp for details.
//
//  Hardware requirement: ARM64 with NEON. Enabled only when compiled
//  with __ARM_NEON defined (Apple Silicon: always true with -mcpu=apple-m1).
// ─────────────────────────────────────────────────────────────────────────
float vector_dot_simd_neon(const float* a, const float* b, std::size_t n);

}  // namespace sparsecore
