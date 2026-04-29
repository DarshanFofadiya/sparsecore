// ═══════════════════════════════════════════════════════════════════════════
//  csrc/bench/avx2_dot_microbench.cpp
//
//  Gate A0 for the AVX2 dW kernel spec (issue #2).
//
//  Goal
//  ────
//  Validate the CORE design decisions of the AVX2 dW kernel on real Zen 4
//  hardware BEFORE we commit to writing the full kernel. Specifically:
//
//    1. Does a single-accumulator AVX2 inner dot loop hit the ~60-80 GF/s
//       target we projected for Zen 4?
//    2. Is a dual-accumulator variant meaningfully faster (justifying the
//       extra code complexity), or does Zen 4's single-FMA-pipe behavior
//       make it neutral?
//    3. Does _mm256_loadu_ps (unaligned load) carry a measurable penalty
//       vs aligned _mm256_load_ps?
//
//  This is pure C++ — no pybind11, no PaddedCSR, no Python. A standalone
//  binary that reports GF/s for several dot-loop variants. Run it via
//  .github/workflows/validate_avx2_microbench.yml on the same
//  ubuntu-24.04 x86_64 runner that Gate 1 / Gate 2 use, so the numbers
//  correlate with the full-kernel measurements to come.
//
//  What each variant measures
//  ──────────────────────────
//  scalar         — the reference baseline: a plain for-loop with
//                   float acc += a[j] * b[j]. Compiled with -march=x86-64-v3.
//                   If this hits >= 30 GF/s on its own, Clang auto-vec is
//                   much more aggressive than NEON experience suggested.
//  avx2_single    — what the spec proposes: one __m256 accumulator,
//                   _mm256_fmadd_ps main loop, scalar Phase B tail.
//  avx2_dual      — the v0.3 Intel-optimization variant: two independent
//                   __m256 accumulators issued in parallel. On Zen 4 we
//                   expect this to be about equal to avx2_single; on
//                   Intel Haswell it would win ~1.3-1.5×. Measuring
//                   gives the design §3.3 decision empirical ground.
//
//  Methodology
//  ───────────
//  For each variant:
//    • N = 2048 (matches demo_15 / demo_16 FFN shape sizes)
//    • Allocate two float32 arrays of length N, filled with known values
//    • Warmup: 100 iterations to get caches hot and CPU frequency stable
//    • Measure: 100k iterations of the dot loop, median of 10 runs of 10k each
//    • Compute GF/s = (2 * N * iterations) / elapsed_ns
//    • Also sanity-check each variant produces the same result to 1e-4
//
//  Why 100k iterations and N=2048
//  ──────────────────────────────
//  We want to stress the inner loop's throughput, not the outer overhead.
//  N=2048 means 256 Phase-A iterations per dot — enough to amortize any
//  per-call setup. 100k total iterations = ~210M FMAs = ~70ms @ 3 GF/s or
//  ~3ms @ 70 GF/s. Short enough for CI, long enough to dominate timer noise.
//
//  What a pass looks like
//  ──────────────────────
//    scalar:      3-8 GF/s   — if > 30, design re-evaluation needed
//    avx2_single: 50-80 GF/s — target; < 30 GF/s is a red flag
//    avx2_dual:   55-85 GF/s — if dual is > 1.3× single on Zen 4, we
//                              need to re-open the single-accumulator
//                              design decision (§3.3)
//
//  Expected output format: three lines per variant with variant name,
//  median ms, and computed GF/s. Easy to grep from CI artifacts.
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
#else
  #error "avx2_dot_microbench.cpp requires -march=x86-64-v3 (AVX2 + FMA)"
#endif


// ─────────────────────────────────────────────────────────────────────
//  Variant 1: scalar reference
//
//  Plain for-loop. Clang at -O3 -march=x86-64-v3 will likely auto-
//  vectorize this to AVX2 FMAs — that's actually what we want to
//  measure. If auto-vec hits 60 GF/s here, our spec's §7.3 risk
//  analysis is wrong and we should pivot to "just set -march".
//
//  Using __attribute__((noinline)) to prevent the compiler from
//  optimizing across variant boundaries (inlining + constant folding
//  would destroy the measurement).
// ─────────────────────────────────────────────────────────────────────

__attribute__((noinline))
float dot_scalar(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        acc += a[j] * b[j];
    }
    return acc;
}


// ─────────────────────────────────────────────────────────────────────
//  Variant 2: AVX2 single accumulator (what the spec proposes)
//
//  One __m256 acc. 8 floats/iter main loop via _mm256_fmadd_ps.
//  Scalar tail for N % 8 remainder. 3-step horizontal reduction at
//  end (256→128→64→32 bit).
// ─────────────────────────────────────────────────────────────────────

__attribute__((noinline))
float dot_avx2_single(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();

    int j = 0;
    // Phase A: 8 floats per iter
    for (; j + 8 <= n; j += 8) {
        __m256 av = _mm256_loadu_ps(a + j);
        __m256 bv = _mm256_loadu_ps(b + j);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }

    // Horizontal reduce: 256 → 128 → 64 → 32
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
    float result = _mm_cvtss_f32(s);

    // Phase B: scalar tail for 0-7 remainder
    for (; j < n; ++j) {
        result += a[j] * b[j];
    }
    return result;
}


// ─────────────────────────────────────────────────────────────────────
//  Variant 3: AVX2 dual accumulator (v0.3 Intel-optimization variant)
//
//  Two independent __m256 accumulators. On Intel Haswell/Skylake (2 FMA
//  pipes per cycle, 5-cycle latency) this breaks the dependency chain
//  and roughly doubles throughput. On Zen 4 (1 FMA pipe per cycle at
//  3-cycle latency) we expect it to be about the same.
//
//  The measurement here is the deciding vote on design §3.3: if avx2_dual
//  is ≤ 1.15× avx2_single on Zen 4, single-accumulator is the right call
//  for v0.2; if it's ≥ 1.3× we should rewrite the spec to use dual.
// ─────────────────────────────────────────────────────────────────────

__attribute__((noinline))
float dot_avx2_dual(const float* a, const float* b, int n) {
    __m256 acc_a = _mm256_setzero_ps();
    __m256 acc_b = _mm256_setzero_ps();

    int j = 0;
    // Phase A: 16 floats per iter (2 × 8-wide FMAs)
    for (; j + 16 <= n; j += 16) {
        __m256 av0 = _mm256_loadu_ps(a + j);
        __m256 bv0 = _mm256_loadu_ps(b + j);
        __m256 av1 = _mm256_loadu_ps(a + j + 8);
        __m256 bv1 = _mm256_loadu_ps(b + j + 8);
        acc_a = _mm256_fmadd_ps(av0, bv0, acc_a);
        acc_b = _mm256_fmadd_ps(av1, bv1, acc_b);
    }

    // Collapse the two accumulators into one
    __m256 acc = _mm256_add_ps(acc_a, acc_b);

    // Phase B: one more 8-wide iter if 8-15 floats remain
    if (j + 8 <= n) {
        __m256 av = _mm256_loadu_ps(a + j);
        __m256 bv = _mm256_loadu_ps(b + j);
        acc = _mm256_fmadd_ps(av, bv, acc);
        j += 8;
    }

    // Horizontal reduce: 256 → 128 → 64 → 32
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));
    float result = _mm_cvtss_f32(s);

    // Phase C: scalar tail for 0-7 remainder
    for (; j < n; ++j) {
        result += a[j] * b[j];
    }
    return result;
}


// ─────────────────────────────────────────────────────────────────────
//  Timing harness
//
//  Uses std::chrono::steady_clock with nanosecond resolution. For each
//  variant we run RUNS separate timing windows and take the median.
//  Running the loop ITERS times within each window amortizes timer
//  cost and catches CPU frequency transitions.
// ─────────────────────────────────────────────────────────────────────

constexpr int N = 2048;         // dot length — matches demo_15/16 FFN N
constexpr int ITERS = 10000;    // iters per timing window
constexpr int RUNS = 10;        // timing windows, median reported
constexpr int WARMUP = 1000;    // warmup iters, discarded

using clock_t = std::chrono::steady_clock;
using ns_t = std::chrono::nanoseconds;

template <typename Fn>
double time_variant_ms(Fn fn, const float* a, const float* b, int n) {
    // Use `accumulator` in a way the compiler cannot prove is dead.
    // Without this, the optimizer may decide the whole loop is unused
    // and eliminate it entirely — a classic microbench gotcha.
    volatile float sink = 0.0f;

    // Warmup
    for (int w = 0; w < WARMUP; ++w) {
        sink += fn(a, b, n);
    }

    std::vector<double> samples;
    samples.reserve(RUNS);
    for (int r = 0; r < RUNS; ++r) {
        auto t0 = clock_t::now();
        for (int i = 0; i < ITERS; ++i) {
            sink += fn(a, b, n);
        }
        auto t1 = clock_t::now();
        double ns = std::chrono::duration_cast<ns_t>(t1 - t0).count();
        samples.push_back(ns / 1e6);  // ms for the whole window
    }
    (void)sink;  // silence unused-variable warning — sink is side-effectful

    std::sort(samples.begin(), samples.end());
    return samples[RUNS / 2];
}


// ─────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────

int main() {
    std::printf("AVX2 dot-loop microbenchmark (Gate A0 for issue #2)\n");
    std::printf("=====================================================\n");
    std::printf("  N (dot length):       %d\n", N);
    std::printf("  iters per window:     %d\n", ITERS);
    std::printf("  timing windows:       %d (median)\n", RUNS);
    std::printf("  warmup iters:         %d\n", WARMUP);
    std::printf("  total FMAs per window: %lld\n",
                (long long)(ITERS) * (long long)(N));
    std::printf("\n");

    // Allocate inputs. Using posix_memalign to force 32-byte alignment
    // so we can rule out alignment as a confound. In the real kernel
    // the inputs come from std::vector<float> / numpy buffers which
    // are 16-byte aligned; but the comparison we want here is about
    // FMA throughput, not alignment. Alignment effect can be a
    // follow-up micro-experiment.
    float* a = nullptr;
    float* b = nullptr;
    if (posix_memalign((void**)&a, 32, N * sizeof(float)) != 0 ||
        posix_memalign((void**)&b, 32, N * sizeof(float)) != 0) {
        std::fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }

    // Deterministic input values. We avoid std::random_device so the
    // result is reproducible across CI runs — making regressions easier
    // to spot in the diff of artifact files.
    for (int j = 0; j < N; ++j) {
        a[j] = 1.0f + 0.001f * (j % 17);
        b[j] = 0.5f + 0.001f * (j % 13);
    }

    // Correctness sanity: all three variants should produce the same
    // result to within float32 accumulation tolerance. If they don't,
    // one of them has a bug (likely the dual-accumulator's Phase B
    // trailing iter) and the GF/s numbers are untrustworthy.
    float ref = dot_scalar(a, b, N);
    float v1 = dot_avx2_single(a, b, N);
    float v2 = dot_avx2_dual(a, b, N);
    auto rel_err = [ref](float v) {
        return std::fabs(v - ref) / std::fabs(ref);
    };
    std::printf("Correctness check (all three variants must agree):\n");
    std::printf("  scalar      = %f\n", (double)ref);
    std::printf("  avx2_single = %f  (rel err vs scalar: %.2e)\n",
                (double)v1, rel_err(v1));
    std::printf("  avx2_dual   = %f  (rel err vs scalar: %.2e)\n",
                (double)v2, rel_err(v2));
    if (rel_err(v1) > 1e-5 || rel_err(v2) > 1e-5) {
        std::fprintf(stderr, "FAIL: variants disagree — bug in one of them.\n");
        return 2;
    }
    std::printf("  ✓ all variants agree within 1e-5\n\n");

    // Timings. Two flops per FMA (one multiply, one add).
    auto flops_per_window = 2.0 * (double)ITERS * (double)N;

    std::printf("%-16s %10s %12s\n", "variant", "ms/window", "GF/s");
    std::printf("%-16s %10s %12s\n", "-------", "---------", "----");

    auto report = [&](const char* name, double ms) {
        double gflops = flops_per_window / (ms * 1e-3) / 1e9;
        std::printf("%-16s %10.3f %12.2f\n", name, ms, gflops);
    };

    double ms_scalar = time_variant_ms(dot_scalar, a, b, N);
    report("scalar", ms_scalar);

    double ms_single = time_variant_ms(dot_avx2_single, a, b, N);
    report("avx2_single", ms_single);

    double ms_dual = time_variant_ms(dot_avx2_dual, a, b, N);
    report("avx2_dual", ms_dual);

    std::printf("\n");
    std::printf("Ratios:\n");
    std::printf("  avx2_single / scalar:   %6.2fx speedup\n",
                ms_scalar / ms_single);
    std::printf("  avx2_dual   / scalar:   %6.2fx speedup\n",
                ms_scalar / ms_dual);
    std::printf("  avx2_dual   / avx2_single: %6.2fx\n",
                ms_single / ms_dual);

    std::printf("\n");
    std::printf("Design gate (§3.3 single-vs-dual accumulator decision):\n");
    double dual_over_single = ms_single / ms_dual;
    if (dual_over_single < 1.15) {
        std::printf("  dual/single < 1.15x  → single accumulator is correct for v0.2 (as designed).\n");
    } else if (dual_over_single < 1.30) {
        std::printf("  dual/single 1.15-1.30x → borderline; ship single, note dual as v0.3 tuning headroom.\n");
    } else {
        std::printf("  dual/single >= 1.30x → RESPEC: dual accumulator is the right v0.2 choice.\n");
    }

    std::printf("\n");
    std::printf("Target assessment:\n");
    double single_gflops = flops_per_window / (ms_single * 1e-3) / 1e9;
    if (single_gflops >= 60.0) {
        std::printf("  avx2_single @ %.1f GF/s ≥ 60 target → design validated.\n", single_gflops);
    } else if (single_gflops >= 30.0) {
        std::printf("  avx2_single @ %.1f GF/s (30-60) → design OK, ship threshold met.\n", single_gflops);
    } else {
        std::printf("  avx2_single @ %.1f GF/s < 30 → red flag, re-plan before full kernel.\n", single_gflops);
    }

    free(a);
    free(b);
    return 0;
}
