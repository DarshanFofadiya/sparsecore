// ═══════════════════════════════════════════════════════════════════════════
//  kernels/double_tensor.cpp
//
//  Scalar implementation of the double_tensor kernel.
//
//  This file knows nothing about Python, pybind11, or NumPy — it takes raw
//  float pointers, does math, writes raw float pointers. Bindings in
//  csrc/bindings.cpp are responsible for translating between Python tensors
//  and these pointers.
//
//  Milestone 1 (what this file currently does):
//    Plain scalar loop. No SIMD. Exists as the Oracle reference that the
//    future SIMD variants are tested against.
// ═══════════════════════════════════════════════════════════════════════════

#include "double_tensor.hpp"


namespace sparsecore {

void double_tensor_scalar(const float* input, float* output, std::size_t n) {
    // We use 2.0f (float literal), not 2.0 (double literal). The `f` suffix
    // keeps the multiply in float32 — without it, each element would be
    // promoted to double, multiplied in double, then demoted back to float.
    // Correct answer, but slower and less predictable for later SIMD work.
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = input[i] * 2.0f;
    }
}

}  // namespace sparsecore
