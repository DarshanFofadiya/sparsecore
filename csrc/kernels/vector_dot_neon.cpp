// ═══════════════════════════════════════════════════════════════════════════
//  kernels/vector_dot_neon.cpp
//
//  NEON SIMD implementation of the vector dot-product kernel.
//
//  The algorithm in 4 steps:
//    1. Initialize a 4-wide accumulator to [0, 0, 0, 0].
//    2. Main loop: load 4 elements from a and b, fused-multiply-add into
//       the accumulator, advance by 4. Repeat until fewer than 4 elements
//       remain.
//    3. Horizontal sum: add the 4 lanes of the accumulator into one float.
//    4. Remainder loop: handle the final 1-3 elements with scalar ops.
//
//  Numerical note:
//    This computes a DIFFERENT addition order than the scalar version —
//    4 parallel running sums instead of 1. Float addition is non-
//    associative, so the two results may differ in the last bit or two.
//    Both are "correct" in the float32-within-tolerance sense; neither
//    is more accurate than the other. Our test suite uses rtol=atol=1e-5
//    which handles this easily.
// ═══════════════════════════════════════════════════════════════════════════

#include "vector_dot_neon.hpp"

// arm_neon.h is the header where NEON intrinsics live. The compiler
// exposes them only when targeting ARM64 with NEON enabled; that's
// guaranteed on Apple Silicon with our -mcpu=apple-m1 flag.
#if defined(__ARM_NEON)
  #include <arm_neon.h>
#else
  #error "vector_dot_neon.cpp requires ARM NEON (compile for arm64)."
#endif


namespace sparselab {

float vector_dot_simd_neon(const float* a, const float* b, std::size_t n) {

    // ─────────────────────────────────────────────────────────────────
    //  Step 1: initialize the accumulator.
    //
    //  float32x4_t is NEON's name for "a 128-bit register holding 4
    //  consecutive float32 lanes." We'll be reading into and writing
    //  back from this register throughout the loop.
    //
    //  vdupq_n_f32(0.0f) broadcasts the scalar 0.0f to all 4 lanes,
    //  giving us [0.0, 0.0, 0.0, 0.0]. One instruction, ~1 cycle.
    // ─────────────────────────────────────────────────────────────────
    float32x4_t acc = vdupq_n_f32(0.0f);

    // ─────────────────────────────────────────────────────────────────
    //  Step 2: main SIMD loop.
    //
    //  Each iteration:
    //    - vld1q_f32 loads 4 floats from a[i..i+3] into a NEON register.
    //      The "1q" in the name means "1 register, 128-bit (quadword)."
    //      Loads from L1 cache in ~3 cycles (instruction latency).
    //    - vld1q_f32 loads the corresponding 4 floats from b[i..i+3].
    //    - vfmaq_f32(acc, a_vec, b_vec) computes:
    //        acc[k] = acc[k] + a_vec[k] * b_vec[k]  for k = 0,1,2,3
    //      all in one instruction, with a single rounding step (fused).
    //      Throughput: ~1 FMA every cycle on M-series cores.
    //
    //  The `i + 4 <= n` guard is how we avoid reading past the end of
    //  the arrays. It ensures we only enter the loop body when there
    //  are at least 4 remaining elements. Anything less goes to the
    //  scalar remainder loop in Step 4.
    // ─────────────────────────────────────────────────────────────────
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, a_vec, b_vec);
    }

    // ─────────────────────────────────────────────────────────────────
    //  Step 3: horizontal reduction.
    //
    //  After the main loop, acc holds 4 partial sums across its lanes:
    //      acc = [ p0, p1, p2, p3 ]
    //  where pk = sum over main-loop iterations j of a[4j+k] * b[4j+k].
    //
    //  We need one scalar result, so we sum across the 4 lanes:
    //      sum = p0 + p1 + p2 + p3
    //
    //  vaddvq_f32 does this horizontal reduction in a single
    //  instruction (takes ~3 cycles on Apple Silicon).
    // ─────────────────────────────────────────────────────────────────
    float sum = vaddvq_f32(acc);

    // ─────────────────────────────────────────────────────────────────
    //  Step 4: scalar remainder loop.
    //
    //  When n is not a multiple of 4 (e.g., n=17), 1-3 elements remain
    //  after the main loop. We process them with plain scalar ops.
    //
    //  Critical correctness note: `i` carries over from the main loop,
    //  pointing at the first unprocessed element. The loop adds the
    //  last few elementwise products to the running scalar sum. If
    //  this loop is buggy (off-by-one, wrong bound), sizes like 15
    //  and 17 in our test suite will immediately catch it — they were
    //  chosen specifically for this.
    // ─────────────────────────────────────────────────────────────────
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

}  // namespace sparselab
