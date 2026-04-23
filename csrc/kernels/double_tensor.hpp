// ═══════════════════════════════════════════════════════════════════════════
//  kernels/double_tensor.hpp
//
//  Declaration of the scalar double_tensor kernel.
//
//  Keeping the declaration separate from the implementation lets other
//  translation units (e.g., bindings.cpp, future benchmarks, other kernels)
//  call this function by just including the header — no C++ tooling magic.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once  // modern replacement for #ifndef/#define/#endif include guards

#include <cstddef>  // for std::size_t


namespace sparselab {

// ─────────────────────────────────────────────────────────────────────────
//  double_tensor_scalar
//
//  Multiplies every element of `input` by 2.0f and writes to `output`.
//  Both pointers must refer to at least `n` valid float32 elements.
//  `input` and `output` must not overlap — this function does not check.
//
//  Parameters:
//    input  - pointer to n float32 values (not mutated)
//    output - pointer to n float32 values (fully overwritten)
//    n      - element count
//
//  Complexity: O(n), single-threaded, scalar (no SIMD).
// ─────────────────────────────────────────────────────────────────────────
void double_tensor_scalar(const float* input, float* output, std::size_t n);

}  // namespace sparselab
