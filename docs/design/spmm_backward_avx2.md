# Design — AVX2 SIMD Kernel for Sparse Weight Gradient (`spmm_grad_w`, x86)

Sibling of [`spmm_backward_neon.md`](spmm_backward_neon.md).
Companion to [`spmm_backward.md`](spmm_backward.md), which covers
the math and the scalar kernel. This doc covers the AVX2
implementation shipped for Linux x86_64, closes [issue #2].

[issue #2]: https://github.com/DarshanFofadiya/sparselab/issues/2

---

## 1. The problem in one paragraph

On Linux x86 our scalar `spmm_grad_w` kernel runs at **~3.8 GF/s**
on GitHub's AMD EPYC 9V74 (Zen 4) runner — about 4× slower than the
pre-NEON ARM scalar baseline (14 GF/s) and ~25-30× slower per FFN
layer than post-NEON Apple Silicon. That is ~88 ms per FFN dW layer
at 40M-param transformer scale — call it **~1.4 seconds per backward
step just for dW** on a CPU whose theoretical f32 peak under AVX2 is
80-120 GF/s per core. Two compounding causes (documented in
[milestone 13](../demos/milestone_13.md)): (a) our build set no
`-march` on x86, so Clang defaulted to `x86-64-v1` — SSE2 only, no
AVX, no FMA; and (b) the dW inner loop's runtime-varying slot
indices defeated Clang's auto-vectorizer the same way they did on
ARM pre-NEON. We ship a hand-written AVX2 + FMA variant that mirrors
the dual-accumulator structure from `spmm_grad_neon.cpp`, route it
through the existing `_simd` dispatch surface, and bump the x86
baseline compile flag to `-march=x86-64-v3` so the scalar fallback
itself catches up.

**Platform scope:** this kernel serves Linux x86_64 only. Apple
Silicon macOS and Linux aarch64 already use the NEON kernel (see
[`spmm_backward_neon.md`](spmm_backward_neon.md)). Intel macOS and
Windows are explicitly out of scope for v0.2 — Intel macOS because
upstream PyTorch deprecated macOS x86_64 wheels in January 2024
(see the [v0.1.1 CHANGELOG entry][v011]), Windows because native
wheels are tracked as a separate effort.

[v011]: ../../CHANGELOG.md

## 2. What ships

### 2.1 New kernel

`csrc/kernels/spmm_grad_avx2.cpp` + `.hpp` — AVX2 + FMA
implementation, compile-gated on `__AVX2__ && __FMA__` in
`setup.py` so ARM64 builds skip the source entirely. The Python
binding `_core.spmm_grad_w_simd` routes to this kernel on x86
builds; on ARM64 it routes to the NEON kernel.

### 2.2 Build flag

`setup.py` adds `-march=x86-64-v3` on x86_64 builds (only). That
target implies AVX2 + FMA + BMI2 on top of SSE4, and has been the
Linux Foundation baseline since ~2021. Every x86 CPU from 2013+
(Intel Haswell+, AMD Zen 1+) meets this. CPUs older than 2013
(Nehalem, Sandy Bridge, Ivy Bridge, Bulldozer) are not supported
starting this release — documented in the changelog.

### 2.3 Dispatch

The autograd path in `sparselab/ops.py`'s `_SpMMFunction.backward`
is unchanged from the NEON release. It selects the kernel based on
the `kernel` argument stashed during forward:

```python
grad_w_fn = (
    _core.spmm_grad_w_simd
    if ctx.kernel in ("auto", "simd")
    else _core.spmm_grad_w
)
```

Because `kernel="auto"` is the default everywhere (`SparseLinear`,
`sparselab.spmm`), Linux x86 users pick up the speedup without any
code change.

### 2.4 Unchanged contracts

- **Public Python API:** no change. Existing training scripts
  continue to work bit-for-bit on every platform.
- **Autograd contract:** `backward` still returns
  `(dW_values, None, dX, None)`.
- **`PaddedCSR` memory layout:** unchanged.
- **Scalar kernel `spmm_grad_w`:** kept as the reference/tolerance
  oracle for AVX2 and as the fallback for pre-AVX2 x86 CPUs
  (unreachable once `-march=x86-64-v3` is set, kept for source
  builds with custom flags).
- **NEON kernel `spmm_grad_neon.cpp`:** literally not touched by
  this work. ARM64 performance is identical before and after.

## 3. Algorithm

### 3.1 Shape (identical to NEON)

Same triple-nested structure as `spmm_grad_neon.cpp`. For every
live slot `s` in row `i` pointing at column `c`, compute one
N-length dot product `dY[i, :] · X[c, :]` and store it to
`dW_values[slot]`:

```
for each row i in [0, M):                      // outer — OpenMP
  for each live slot s in row i:               // walk row_nnz[i]
    c = col_indices[row_start[i] + s]
    acc = dot_N(dY[i, :], X[c, :])             // inner — AVX2 target
    dW_values[row_start[i] + s] = acc
```

The inner `dot_N` is the hot loop. AVX2's 256-bit registers hold 8
float32 lanes — the **same effective width** as the NEON dual-4-wide
accumulator pattern — so the loop shape translates directly. What
changes is the intrinsic names, the horizontal-reduction sequence,
and the alignment story (§7.2).

### 3.2 SIMD strategy — Phase A/B/C, dual 8-wide accumulators

Per inner iteration, Phase A processes 16 floats using two
independent 256-bit FMAs:

```
  _mm256_loadu_ps(dy + j)     // load 8 floats, lane A
  _mm256_loadu_ps(dy + j + 8) // load 8 floats, lane B
  _mm256_loadu_ps(x  + j)     // load 8 floats, lane A
  _mm256_loadu_ps(x  + j + 8) // load 8 floats, lane B
  _mm256_fmadd_ps(a, b, acc_a) // FMA lane A, independent
  _mm256_fmadd_ps(a, b, acc_b) // FMA lane B, independent
```

Phase A (16 floats/iter, dual 256-bit accumulators) → Phase B
(trailing 8-wide iter if 8-15 floats remain) → Phase C (scalar
tail of 0-7 remainder floats).

At the very end, `_mm256_add_ps(acc_a, acc_b)` fuses both
accumulators lane-wise (1 instruction), then a horizontal reduction
collapses the single 256-bit accumulator to one scalar per live
slot.

### 3.3 Dual accumulator, validated empirically

The original design proposed a single accumulator here, reasoning
that Zen 4's FMA scheduling wouldn't benefit from two independent
accumulators. **A microbenchmark on the actual CI runner
disproved that before we wrote the kernel** — see §6.0 below.

Measured numbers (AMD EPYC 9V74, Zen 4, single-threaded, N=2048):

| Variant | ms/window | GF/s | vs scalar |
|---|---|---|---|
| scalar (auto-vec attempt) | 24.78 | 1.65 | 1.00× |
| avx2_single | 2.85 | 14.39 | 8.71× |
| **avx2_dual** | **1.40** | **29.19** | **17.66×** |

Dual accumulator delivered **2.03× over single accumulator** —
well above the 1.30× threshold that would motivate shipping dual
instead. Correctness check: both variants produce results bit-
identical to scalar within 2.3e-7 relative error.

**Interpretation:** Zen 4 scheduling effectively doubles throughput
when two independent accumulators are available. Whether the
mechanism is dual FMA pipes, out-of-order issue hiding FMA latency,
or some combination, the observable fact is 2× — and a single-
accumulator kernel would have left that on the table.

**Expected behavior on other x86 µarches:**
- **Intel Haswell/Skylake** (2 × 256-bit FMA pipes at 5-cycle
  latency): dual may give ~2.5× over single.
- **Zen 1** (AVX2 split into 2 × 128-bit micro-ops): dual's
  advantage shrinks to ~1.3× but still positive.
- **Zen 2+, Zen 3, Zen 4, Ice Lake+**: dual delivers ~2×.

Every supported µarch benefits from the dual variant; none is hurt.

### 3.4 Phase A inner loop

```cpp
__m256 acc_a = _mm256_setzero_ps();
__m256 acc_b = _mm256_setzero_ps();

int64_t j = 0;
// Phase A: 16 floats per iter (two independent 8-wide FMAs)
for (; j + 16 <= N; j += 16) {
    __m256 dy0 = _mm256_loadu_ps(dY_row + j);
    __m256 dy1 = _mm256_loadu_ps(dY_row + j + 8);
    __m256 x0  = _mm256_loadu_ps(X_row  + j);
    __m256 x1  = _mm256_loadu_ps(X_row  + j + 8);
    // Two FMAs, independent on disjoint accumulator registers.
    // Out-of-order scheduler dispatches them in parallel.
    acc_a = _mm256_fmadd_ps(dy0, x0, acc_a);
    acc_b = _mm256_fmadd_ps(dy1, x1, acc_b);
}

// Phase B: one more 8-wide iter if 8-15 floats remain
__m256 acc = _mm256_add_ps(acc_a, acc_b);
if (j + 8 <= N) {
    __m256 dy = _mm256_loadu_ps(dY_row + j);
    __m256 x  = _mm256_loadu_ps(X_row  + j);
    acc = _mm256_fmadd_ps(dy, x, acc);
    j += 8;
}
```

Per Phase A iteration (~5-6 cycles on Zen 4): 4 × 256-bit loads + 2
× 256-bit FMAs. Zen 4 has dual load ports capable of 2 × 256-bit
loads per cycle, and the FMAs are independent so the scheduler can
issue them both.

### 3.5 Horizontal reduction

AVX2 doesn't have NEON's `vaddvq_f32` one-instruction reduce. We
use the standard 3-step collapse:

```cpp
// Collapse 256-bit -> 128-bit by summing upper and lower halves.
__m128 lo = _mm256_castps256_ps128(acc);
__m128 hi = _mm256_extractf128_ps(acc, 1);
__m128 s  = _mm_add_ps(lo, hi);

// Collapse 128-bit -> 64-bit: [a b c d] -> [a+c b+d _ _]
s = _mm_add_ps(s, _mm_movehl_ps(s, s));

// Collapse 64-bit -> 32-bit: [a+c b+d _ _] -> scalar (a+c)+(b+d)
s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x1));

float result = _mm_cvtss_f32(s);
```

Three `_mm_add` + two shuffles, ~6 cycles latency. One reduction
per live slot, so amortized over Phase A's many iterations.
(Alternative: `_mm256_hadd_ps` + extract. Same latency, marginally
less readable; we pick the explicit shuffle version for teaching
value.)

### 3.6 Parallelism

Same row-level OpenMP `parallel for schedule(static)` with the
`SCORE_PARALLEL_ROW_THRESHOLD` gate used by the scalar and NEON
kernels. Zero changes to the parallelism layer. Race-freedom
argument is identical to NEON design §3.4: each row `i` writes only
to `dW_values[row_start[i] : row_start[i] + row_capacity[i]]`;
PaddedCSR invariants guarantee these slices don't overlap across
rows.

### 3.7 Self-zeroing contract

`memset(dW_values, 0, total_capacity * sizeof(float))` at entry,
matching scalar and NEON. Padding slots stay at 0.0 so
`W.values -= lr * dW` remains safe without a mask pass.

### 3.8 Why 8-wide AVX2 (not AVX-512)

Three reasons:
1. **CI reach.** GitHub's x86 runners are Zen 4 but do *not* expose
   AVX-512 via cpuid. An AVX-512 kernel we can't execute in CI is
   an unverified kernel. AVX-512 belongs in v0.3 and needs a
   self-hosted runner.
2. **Compile-target breadth.** AVX2 + FMA is universally available
   on every x86 CPU shipped from 2013 onward: every Haswell+, every
   Zen+ (Zen 1 from 2017). `-march=x86-64-v3` is the Linux
   Foundation baseline for "modern x86" and is supported as a
   manylinux wheel target.
3. **One win at a time.** 8-wide takes us from 3.8 GF/s to ≥ 30
   GF/s. That closes the x86 parity gap. AVX-512 is a follow-on
   doubling on top.

## 4. Decisions that matter

### 4.1 Loop order: (i, s, j) — same as NEON and scalar

`dY[i, :]` stays in L1 across all `j` iterations of the dot product.
`X[c, :]` is read as a streaming scan once per slot. Swapping to
`(i, j, s)` would re-scan `dY[i, :]` per slot — cache thrash. No
change from NEON.

### 4.2 Inline the dot loop — don't call a `vector_dot_avx2`

Same reasoning as NEON §4.2: at 40M scale we have ~6.5M live slots
per backward pass, and a function call per slot would add ~2 ms of
pure overhead. Inlining also keeps `acc_a` and `acc_b` in ymm
registers across the Phase A → Phase B transition. We do not build
a `vector_dot_avx2_simd` helper.

### 4.3 Dispatch surface — extend `__ARM_NEON` with `__AVX2__ && __FMA__`

In `bindings.cpp`'s `py_spmm_grad_w_simd`, the pre-AVX2 pattern
was:

```cpp
#if defined(__ARM_NEON)
    sparselab::spmm_grad_w_simd(W, ...);  // NEON kernel
#else
    sparselab::spmm_grad_w(W, ...);       // scalar fallback
#endif
```

Post-change:

```cpp
#if defined(__ARM_NEON)
    sparselab::spmm_grad_w_simd(W, ...);           // NEON kernel
#elif defined(__AVX2__) && defined(__FMA__)
    sparselab::spmm_grad_w_simd(W, ...);           // AVX2 kernel
#else
    sparselab::spmm_grad_w(W, ...);                // scalar fallback
#endif
```

Critical detail: **both branches call a C++ function named
`sparselab::spmm_grad_w_simd`.** On ARM64 that symbol is defined by
`spmm_grad_neon.cpp`; on x86_64 by `spmm_grad_avx2.cpp`. They are
never both compiled in the same build. The Python-facing name stays
`_core.spmm_grad_w_simd` in every case. This preserves the dispatch
surface set up by the NEON release — autograd does not change.

The same pattern applies to the `#include` block at the top of
`bindings.cpp`:

```cpp
#if defined(__ARM_NEON)
  #include "kernels/vector_dot_neon.hpp"
  #include "kernels/spmm_neon.hpp"
  #include "kernels/spmm_grad_neon.hpp"
#elif defined(__AVX2__) && defined(__FMA__)
  #include "kernels/spmm_grad_avx2.hpp"
#endif
```

### 4.4 `setup.py` source gating

Mirror the existing `IS_ARM64`-gated sources list:

```python
IS_X86_64 = platform.machine() in ("x86_64", "AMD64")

if IS_X86_64:
    extra_compile_args += ["-march=x86-64-v3"]

sources = [
    "csrc/bindings.cpp",
    "csrc/kernels/double_tensor.cpp",
    # ... scalar sources unchanged ...
]
if IS_ARM64:
    sources += [
        "csrc/kernels/vector_dot_neon.cpp",
        "csrc/kernels/spmm_neon.cpp",
        "csrc/kernels/spmm_grad_neon.cpp",
    ]
elif IS_X86_64:
    sources += [
        "csrc/kernels/spmm_grad_avx2.cpp",
    ]
```

The NEON sources and the AVX2 sources are mutually exclusive by
architecture — at no point does a single build compile both.

## 5. Testing strategy

### 5.1 Oracle tests — already parametrized

`tests/test_spmm_grad.py` was parametrized over `kernel_fn ∈
[scalar, simd]` in the NEON release. That work covers AVX2 on x86
CI automatically: the 23 shape/padding/edge-case tests × 2 kernels
= 46 passing cases per platform.

On x86 CI, the `simd` parameter now hits a real AVX2 kernel
instead of the scalar fallback. If AVX2 has a bug that scalar
doesn't, these tests catch it immediately.

### 5.2 AVX2-specific tests (`tests/test_spmm_grad_avx2.py`)

Mirrors `tests/test_spmm_grad_neon.py` case-for-case. Runs only on
x86_64. Coverage:

- **Scalar/AVX2 bit-tolerance agreement.** 20 random shapes with
  varied sparsity, `|scalar(x) - avx2(x)| < 1e-5 * max(|scalar|,
  1)` elementwise.
- **N-residue coverage.** N ∈ {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31,
  32, 33, 63, 64, 65}. Exercises every `N mod 16` residue, plus the
  boundary-plus-one cases where Phase A → Phase B transitions.
  Includes N=16, 32, 64 (exact multiples — Phase B never fires) and
  N=17, 33, 65 (one-over multiples — Phase B takes 1 scalar iter).
- **Empty-row interleaving.** W with empty rows between populated
  rows — ensures OpenMP's static schedule doesn't assume balanced
  per-thread work.
- **Single-slot-per-row.** All rows at `row_nnz=1`. Stresses
  Phase B when N is tiny (1-3 floats).
- **Determinism under parallelism.** Fixed seed, 10 repeated calls,
  `np.testing.assert_array_equal`. OpenMP static schedule must be
  bit-stable across invocations.

### 5.3 Autograd integration

`tests/test_spmm_autograd.py` already parametrizes
`torch.autograd.gradcheck` over `scalar` + `simd` (from NEON
release). On x86 CI the `simd` path now exercises AVX2 end-to-end
through autograd. No changes needed here.

### 5.4 Cross-platform regression

Three CI platforms must all stay green:
- `macos-14` (Apple Silicon arm64, NEON)
- `ubuntu-24.04-arm` (Linux aarch64, NEON)
- `ubuntu-24.04` (Linux x86_64, AVX2) — where the new kernel runs.

Milestone 13's measured numbers for NEON on Linux aarch64
(3.1-4.2× local on FFN shapes) must remain within ±5%. If they
regress, something unintentionally touched an ARM path.

### 5.5 Performance gates

`.github/workflows/profile_x86_baseline.yml` (the same workflow
milestone 13 used) runs `profile_dw_baseline.py` across platforms.
Post-implementation thresholds:

| Shape | Scalar before | AVX2 target | Expected si/sc |
|---|---|---|---|
| 384×1536, N=2048, s=0.90 | 62.8 ms / 3.8 GF/s | ≤ 6 ms / ≥ 40 GF/s | ~0.08-0.15 |
| 1536×384, N=2048, s=0.90 | 63.3 ms / 3.8 GF/s | ≤ 6 ms / ≥ 40 GF/s | ~0.08-0.15 |
| 640×2560, N=1024, s=0.90 | 85.6 ms / 3.9 GF/s | ≤ 8 ms / ≥ 40 GF/s | ~0.08-0.15 |
| 2560×640, N=1024, s=0.90 | 88.8 ms / 3.8 GF/s | ≤ 8 ms / ≥ 40 GF/s | ~0.08-0.15 |
| 64×64, N=128, s=0.80 | ~0.3 ms | no regression | any |

Target: **≥ 40 GF/s**. Ship floor: **≥ 30 GF/s**. Stretch: **50-80
GF/s**. Microbench dual-accumulator was 29.2 GF/s single-threaded;
2× OpenMP on the 2-core runner → ~50-55 GF/s inner loop, dropping
to ~40-45 GF/s once per-slot overhead is added.

## 6. Performance — measured numbers

### 6.0 Gate A0 — microbench validation

Before committing a full kernel implementation, we built a
standalone microbenchmark in `csrc/bench/avx2_dot_microbench.cpp`
and ran it on GitHub's ubuntu-24.04 runner (AMD EPYC 9V74, Zen 4,
single-threaded, N=2048, 10-run median of 10k iterations each) via
`.github/workflows/validate_avx2_microbench.yml`. The benchmark
measures three variants of the inner dot loop in isolation:

| Variant | ms/window | GF/s | vs scalar |
|---|---|---|---|
| scalar (auto-vec attempt) | 24.78 | 1.65 | 1.00× |
| avx2_single | 2.85 | 14.39 | 8.71× |
| avx2_dual | 1.40 | **29.19** | **17.66×** |

All three variants agree with scalar to 2.3e-7 relative error. The
benchmark is reproducible via
`gh workflow run "Validate AVX2 microbench"` and takes ~9 seconds.

Three findings drove design decisions:

1. **Scalar is even worse than milestone 13 suggested.** Milestone
   13's 3.8 GF/s included OpenMP parallelism across 2 threads.
   Single-thread pure dot loop is 1.65 GF/s — truly serial FMAs at
   ~2 cycles each, 15× below ceiling.

2. **Dual accumulator is 2× single on Zen 4.** This overturned the
   original "single is enough" reasoning. See §3.3 for the revised
   argument; the spec now ships dual.

3. **Target throughput is achievable.** Dual hit 29.2 GF/s
   single-threaded. With OpenMP across 2 threads in the full kernel
   we expect ~50-55 GF/s inner-loop throughput, dropping to ~40-45
   GF/s once per-slot overhead (col_indices lookup, row pointer
   arithmetic, per-slot horizontal reduce) is included — well above
   the 30 GF/s ship minimum.

### 6.1 Pre-implementation baseline (milestone 13)

For reference, the scalar dW kernel as published in v0.2.1:

| Shape | Scalar ms | simd ms | si/sc | Scalar GF/s |
|---|---|---|---|---|
| demo15 FFN up (384 × 1536, N=2048, s=0.90) | 62.76 | 62.78 | 1.00× | 3.82 |
| demo15 FFN down (1536 × 384, N=2048, s=0.90) | 63.26 | 62.68 | 1.00× | 3.79 |
| demo16 FFN up (640 × 2560, N=1024, s=0.90) | 85.55 | 85.44 | 1.00× | 3.91 |
| demo16 FFN down (2560 × 640, N=1024, s=0.90) | 88.81 | 88.73 | 1.00× | 3.76 |

`si/sc ≈ 1.00` across all shapes confirms the pre-spec `_simd`
binding was the scalar fallback.

### 6.2 Revised end-to-end projection

At 40M scale where dW is ~62% of a training step (milestone 10):

| AVX2 local speedup | dW share after | Step speedup on x86 |
|---|---|---|
| 8× (ship minimum) | ~17% | ~1.9× |
| 12× (mid-target) | ~12% | ~2.1× |
| 20× (stretch; matches microbench extrapolation) | ~7% | ~2.3× |

Because x86's starting point is 4× slower than ARM's was, a given
local speedup translates to a much larger *absolute* saving per
backward pass.

### 6.3 Precedent from NEON

The NEON port of the same kernel, on identical FFN shapes,
delivered **6.3-6.7× local** on Apple M3 and **3.1-4.2× local** on
Graviton ([milestone 12](../demos/milestone_12.md)). AVX2 on Zen 4
should land in or above that range because:

- Zen 4 has 512-bit internal FMA pipes split as 2 × 256-bit, so
  dual-accumulator code fully uses both pipes per cycle (confirmed
  by Gate A0's 2× single-vs-dual ratio).
- Zen 4's L2 bandwidth is ~50 GB/s per core, comparable to M3's.
- L1 is ~1 cycle latency, comparable.

Extrapolating from the microbench: **10-14× local speedup in the
full kernel** is realistic on Zen 4 (microbench inner loop hit
17.7×; real kernel loses some to per-slot overhead). On Intel
Skylake/Alder Lake the number could climb higher (up to ~15-20×)
thanks to 2 × 256-bit FMA-per-cycle issue.

### 6.4 Why this is the v0.2 blocker

[milestone 13](../demos/milestone_13.md) is explicit: the v0.2.1
release did not deliver the sparse-from-scratch speed story on
Linux x86. This spec's work is what makes v0.2 an honest release
for the majority of DL researchers worldwide who run on Linux x86.

## 7. Risk register

### 7.1 Low — Pre-2013 CPU compile-target breakage

**Risk:** `-march=x86-64-v3` produces a binary that segfaults with
`Illegal instruction` on any x86 CPU predating 2013 (pre-Haswell
Intel, pre-Zen AMD).

**Why this is low risk:** our platform matrix is

- Apple Silicon macOS (arm64) — unaffected
- Linux aarch64 — unaffected
- **Linux x86_64 only** — the platform this risk applies to

Every Linux distribution currently supported by PyTorch 2.8+
targets Haswell+ / Zen+ CPUs in practice. We do not ship Intel Mac
wheels (upstream PyTorch EOL'd that target in January 2024,
documented in v0.1.1). We do not ship Windows wheels for v0.2.
The intersection "user on our supported platforms × pre-2013 x86
CPU" is approximately empty.

**Mitigation:** document the requirement in `README.md` under
"Requirements" and note in CHANGELOG. manylinux_2_28 (our
cibuildwheel base image) supports `-march=x86-64-v3` as a build
target. The bindings still carry a scalar fallback branch,
unreachable once the flag is set but kept for source builds with
custom flags.

### 7.2 Low — Unaligned loads on AVX2

**Risk:** AVX2 256-bit loads prefer 32-byte alignment. Our
`PaddedCSR::values` and dense `dY`/`X` arrays come from
`std::vector<float>` / numpy and are guaranteed 16-byte aligned
but not 32-byte aligned.

**Mitigation:** use `_mm256_loadu_ps` (unaligned load) throughout.
On Zen 3+ and Haswell+ — every CPU we target — unaligned 256-bit
loads that do *not* cross a 64-byte cache-line boundary are
penalty-free at 1 load/cycle per port. Row data is contiguous and
well-behaved; boundary crosses happen statistically once per ~16
iterations and cost ~1 extra cycle — lost in the noise.

Do not use `_mm256_load_ps` (aligned load) — any misalignment
would `#GP` fault. This is called out prominently in the kernel's
block comments.

### 7.3 Low — Clang x86 auto-vec on dW inner loop

**Risk:** with `-march=x86-64-v3` set, Clang might auto-vectorize
the scalar `spmm_grad_w` well enough that our hand-written AVX2
kernel ties or loses.

**Counter-evidence:** the same outer structure
(`dW_values[row_ptr + s] = acc` with runtime-varying slot indices)
defeated Clang's auto-vectorizer on ARM at `-O3 -mcpu=apple-m1`.
Clang's x86 vectorizer operates on the same SSA representation.
Additionally, Gate A0 measured scalar under `-march=x86-64-v3` at
1.65 GF/s — firmly in "no auto-vec" territory. If Clang had been
auto-vectorizing the pattern, microbench scalar would have hit
10-15 GF/s.

### 7.4 Low — Intel vs AMD AVX2 behavior divergence

**Risk:** CI measures Zen 4; we ship wheels that run on Intel
Haswell+, Ice Lake, Alder Lake, Sapphire Rapids, Zen 1-4.
Different µarch behaviors could mean the 30 GF/s floor is met on
Zen 4 but missed elsewhere.

Relevant µarch points:
- **Zen 1** (2017): AVX2 internally as 2 × 128-bit. Effective
  throughput ~half of Zen 2+. A shape that lands at 40 GF/s on
  Zen 2 would land at ~20 GF/s on Zen 1 — still ~5-6× over scalar.
- **Haswell/Skylake** (Intel, 2013-2019): 2 × 256-bit FMA/cycle at
  5-cycle latency. Dual-accumulator code should hit peak here.
- **Zen 2+/3/4 and Intel Ice Lake+**: 1 × 256-bit FMA/cycle at 3-4
  cycle latency. Dual code close to peak.

**Mitigation:** CI measures Zen 4 (the middle of the distribution).
At 30 GF/s on Zen 4 in CI, ships should land between 15 and 50
GF/s elsewhere — always ≥ 4× the 3.8 GF/s pre-AVX2 baseline.
Documented honestly in the milestone writeup.

## 8. What we're explicitly not doing

- **No AVX-512 port.** v0.3 scope. GitHub's x86 runners don't
  expose AVX-512, so we can't CI it. Requires self-hosted runner.
- **No runtime CPU feature detection / dispatcher.** Compile-time
  `-march=x86-64-v3` is statically sufficient for 100% of our
  target users. Runtime dispatch (cpuid at import, pick best
  kernel) doubles binary size and code complexity; worth it when
  AVX-512 lands.
- **No forward SpMM AVX2 port.** dW is the larger backward
  bottleneck (milestone 10: dW is 62% of a step). Forward SpMM
  AVX2 is a separate follow-on, possibly v0.3.
- **No Intel macOS (x86_64) wheels.** Upstream PyTorch EOL'd
  macOS x86_64 in January 2024. Carve-out documented in v0.1.1
  CHANGELOG.
- **No Windows x86 wheels.** Tracked as a separate effort.
- **No SSE4 / AVX1 fallback for pre-2013 x86 CPUs.** Minimum
  requirement documented; users on older hardware stay on v0.2.1
  or build from source with custom flags.
- **No tolerance tightening.** Keep `rtol=atol=1e-5` same as NEON.
- **No new public Python API symbol.** `_core.spmm_grad_w_simd`
  already existed; it just starts being fast on x86.
- **No code changes to NEON kernels.** `csrc/kernels/spmm_grad_neon.{hpp,cpp}`
  are not touched.

---

## Appendix — Borrow-Don't-Reinvent references

**Scalar pattern mirrored:** `csrc/kernels/spmm_grad.cpp`.
Unchanged.

**SIMD pattern mirrored:** `csrc/kernels/spmm_grad_neon.cpp` — same
Phase A/B/C loop shape, same self-zeroing contract, same OpenMP
parallelism. The only per-lane differences are intrinsic names and
the reduction sequence (AVX2 lacks NEON's single-instruction
`vaddvq_f32`).

**Dispatch pattern extended:** `csrc/bindings.cpp`'s
`#if defined(__ARM_NEON)` gate in `py_spmm_grad_w_simd`. We add an
`#elif defined(__AVX2__) && defined(__FMA__)` arm to the same
switch. Scalar fallback arm stays.

**Build-flag pattern mirrored:** `setup.py`'s `IS_APPLE_SILICON`
and `IS_ARM64` checks. We add `IS_X86_64` and the
`-march=x86-64-v3` addition in the same style.

**External reference — Intel / AMD optimization guides:**
- Intel® 64 and IA-32 Architectures Optimization Reference Manual,
  Ch. 15 (AVX intrinsic usage and FMA patterns).
- AMD Software Optimization Guide for AMD Family 19h (Zen 3 /
  Zen 4), §2.7 (256-bit AVX2 pipe scheduling).
- Agner Fog's instruction tables (`https://agner.org/optimize/`)
  for per-µarch latency/throughput of `vfmadd231ps`,
  `vbroadcastss`, and the shuffle intrinsics used in the
  reduction.

None of these cover the "sparse dW per-slot dot product" pattern
specifically; our kernel is new, but every intrinsic we use is
well-documented off-the-shelf.

---

_Shipped as part of sparselab's Linux x86 parity work (v0.2.x). See
the CHANGELOG for the exact release version._
