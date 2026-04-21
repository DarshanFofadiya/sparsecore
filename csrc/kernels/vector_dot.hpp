// ═══════════════════════════════════════════════════════════════════════════
//  kernels/vector_dot.hpp
//
//  Declaration of the vector dot-product kernel.
//
//  Dot product is the fundamental primitive underneath almost every ML
//  operation: matmul is "dot product of every row pair", attention is
//  "dot product of queries and keys", and our future sparse SpMM
//  (Phase 3) has an inner dot-product kernel per row.
//
//  For this sub-milestone (2a) we provide a scalar implementation. The
//  NEON SIMD version lands in 2b and will be Oracle-verified against
//  this scalar version — they must produce bit-close results.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstddef>


namespace sparsecore {

// ─────────────────────────────────────────────────────────────────────────
//  vector_dot_scalar
//
//  Computes sum over i of a[i] * b[i], using float32 throughout.
//
//  Parameters:
//    a - pointer to n float32 values (not mutated)
//    b - pointer to n float32 values (not mutated)
//    n - element count; both arrays must have at least this many elements
//
//  Returns:
//    A single float32. If n == 0, returns 0.0f (the additive identity).
//
//  Accumulator choice:
//    We use a float (NOT double) accumulator. Rationale: this kernel is
//    the Oracle reference for the NEON SIMD version in 2b, which also
//    accumulates in float32 (NEON lanes are float32). Using a double
//    accumulator here would make scalar and SIMD disagree unnecessarily.
//    PyTorch's torch.dot also uses float32 accumulation for float32
//    inputs, so all three (scalar/SIMD/torch) agree within rtol=atol=1e-5.
//
//  Complexity: O(n), single-threaded, no SIMD.
// ─────────────────────────────────────────────────────────────────────────
float vector_dot_scalar(const float* a, const float* b, std::size_t n);

}  // namespace sparsecore
