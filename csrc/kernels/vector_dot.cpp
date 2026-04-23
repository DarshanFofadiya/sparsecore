// ═══════════════════════════════════════════════════════════════════════════
//  kernels/vector_dot.cpp
//
//  Scalar implementation of the vector dot-product kernel.
//
//  This is the Oracle reference for the NEON SIMD version in sub-milestone
//  2b. Both versions must produce results within rtol=atol=1e-5 of each
//  other and of torch.dot.
// ═══════════════════════════════════════════════════════════════════════════

#include "vector_dot.hpp"


namespace sparselab {

float vector_dot_scalar(const float* a, const float* b, std::size_t n) {
    // Accumulator is float32 (not double) — see the rationale in
    // vector_dot.hpp. This matches NEON's native lane width and ensures
    // our scalar and SIMD implementations stay in numerical agreement.
    float sum = 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
    // Note on empty inputs: when n == 0, the loop executes zero times
    // and we return 0.0f — the additive identity, matching torch.dot's
    // behavior on empty tensors. No special-case code needed.
}

}  // namespace sparselab
