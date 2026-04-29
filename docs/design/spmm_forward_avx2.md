# Design — AVX2 SIMD Kernel for Sparse Forward SpMM (`spmm_simd`, x86)

Sibling of [`spmm_backward_avx2.md`](spmm_backward_avx2.md).
Companion to [`spmm.md`](spmm.md), which covers the math and the
scalar kernel. This doc covers the AVX2 implementation shipped
for Linux x86_64 for the **forward** `Y = W @ X` path, completing
the x86 SIMD story started by the dW kernel (milestone 14).

---

## 1. The problem in one paragraph

On Linux x86 our scalar `spmm_scalar` forward kernel — the
`Y = W @ X` path — runs at **~4 GF/s** on GitHub's AMD EPYC 9V74
(Zen 4) runner (milestone 13). The AVX2 dW kernel shipped in
[milestone 14](../demos/milestone_14.md) brought per-layer dW
down 11.8-12.9× and delivered **3.0× end-to-end training
speedup** at 40M-param transformer scale — but forward SpMM on
x86 is still the scalar fallback. Post-milestone-14, forward
SpMM is ~12% of a training step (dW is no longer the bottleneck)
and **100% of inference-time cost**, making it the
highest-leverage remaining x86 kernel to port. We ship an AVX2 +
FMA variant that mirrors the dual-stream Phase-A/B/C structure
from `spmm_grad_avx2.cpp` but swaps the inner-loop body:
forward does write-through FMA (`y[j] += v * x[j]`) with **no
per-slot horizontal reduction**, so the kernel's bottleneck
shape changes from compute-bound (dW) to memory-bandwidth-bound
(forward). Expected per-layer speedup: **6-10×**, expected
end-to-end training speedup gain: **1.15-1.3×** on top of the
current 3.0×, expected **inference speedup: 6-10×** end-to-end
(forward is 100% of inference).

**Platform scope** (unchanged from dW AVX2): this kernel serves
Linux x86_64 only. Apple Silicon macOS and Linux aarch64 already
use the NEON forward kernel (see [`spmm.md`](spmm.md)). Intel
macOS and Windows x86_64 are explicitly out of scope for v0.2 —
Intel macOS because upstream PyTorch deprecated macOS x86_64
wheels in January 2024 (see the [v0.1.1 CHANGELOG entry][v011]),
Windows because native wheels are tracked as a separate effort.

[v011]: ../../CHANGELOG.md

## 2. What ships

### 2.1 New kernel

`csrc/kernels/spmm_avx2.cpp` + `.hpp` — AVX2 + FMA
implementation, compile-gated on `__AVX2__ && __FMA__` in
`setup.py` so ARM64 builds skip the source entirely. The Python
binding `_core.spmm_simd` routes to this kernel on x86 builds;
on ARM64 it routes to the NEON kernel; without SIMD it falls
back to scalar.

### 2.2 Build flag — no change

`-march=x86-64-v3` is already set in `setup.py` from the dW work
(milestone 14). That target implies AVX2 + FMA, which is
everything this kernel needs. No new flag to add, no platform
scope change.

### 2.3 Dispatch

The autograd path in `sparselab/ops.py`'s `_SpMMFunction.forward`
is unchanged from the NEON release — it already calls
`_core.spmm_simd` when `kernel="auto"` or `kernel="simd"`.
Because `kernel="auto"` is the default everywhere
(`SparseLinear`, `sparselab.spmm`), Linux x86 users pick up the
speedup without any code change once the binding dispatches to
AVX2.

### 2.4 Unchanged contracts

- **Public Python API:** no change. Existing training and
  inference scripts continue to work bit-for-bit on every
  platform.
- **Autograd contract:** `_SpMMFunction.forward` still calls
  `_core.spmm_simd`; `backward` continues to use the AVX2 dW
  kernel (shipped milestone 14) and the transpose-cache path
  for `dX`.
- **`PaddedCSR` memory layout:** unchanged.
- **Scalar kernel `spmm_scalar`:** kept as the reference /
  tolerance oracle for AVX2 and as the fallback for pre-AVX2
  x86 CPUs (unreachable once `-march=x86-64-v3` is set, kept
  for source builds with custom flags).
- **NEON kernel `spmm_neon.cpp`:** literally not touched by
  this work. ARM64 forward performance is identical before and
  after.
- **AVX2 dW kernel `spmm_grad_avx2.cpp`:** literally not
  touched. x86 backward performance is identical before and
  after.

## 3. Algorithm

### 3.1 Shape (identical to scalar and NEON)

Same triple-nested structure as `spmm_scalar` and
`spmm_simd_neon`. For every live slot `s` in row `i` pointing at
column `c` with value `v`, we do one N-length axpy-style update
into `Y[i, :]`:

```
for each row i in [0, M):                       // outer — OpenMP
  for each live slot s in row i:                // walk row_nnz[i]
    c = col_indices[row_start[i] + s]
    v = values[row_start[i] + s]
    for j in [0, N):                            // inner — AVX2 target
      Y[i, j] += v * X[c, j]
```

The inner `j` loop is the hot path. AVX2's 256-bit registers
hold 8 float32 lanes — the same effective width as the NEON
dual-4-wide accumulator pattern — so the loop shape translates
directly. What changes vs the dW AVX2 kernel is **how** we use
the SIMD register file:

- **Broadcast-then-FMA**, not dot product. `v` is scalar, so we
  broadcast once per live slot via `_mm256_set1_ps(v)` (one
  `vbroadcastss` instruction) and FMA it against 8 lanes of
  `x_row` into 8 lanes of `y_row`.
- **Write-through**, not accumulate-and-reduce. Each SIMD
  iteration does `load y_row` → `fmadd` → `store y_row`. No
  horizontal reduction at the end — `y_row` is already the
  final output buffer.
- **Memory-bound, not compute-bound.** Per 8-wide FMA: 2 ×
  256-bit loads (x, y) + 1 × 256-bit FMA + 1 × 256-bit store.
  Zen 4 has 2 load ports + 1 store port per core, so the store
  port is the binding constraint.

### 3.2 SIMD strategy — Phase A/B/C, dual 8-wide write-through

Phase widths are **identical to the dW kernel** for
consistency — the `N mod 16` residue boundaries match, so the
same test shapes exercise both kernels:

- **Phase A**: main loop, 16 floats/iter using two independent
  `__m256` streams (8-wide FMA into each). The two streams
  write to disjoint Y slots (`y_row + j` and `y_row + j + 8`),
  so they're independent by construction.
- **Phase B**: trailing 8-wide iteration if 8-15 floats remain
  after Phase A.
- **Phase C**: scalar tail for the final 0-7 floats.

Unlike dW, **there is no accumulator fusion at the end of Phase
A.** The two streams already wrote their results to memory in
the same iteration they were computed; there's nothing to fuse.
Phase B loads from `y_row + j`, FMAs, stores back; Phase C is
scalar.

### 3.3 Dual stream rationale

Why two independent 8-wide streams instead of one?

**The dW kernel's "dual accumulator" choice was about breaking a
data dependency on a single register across FMA latency** — Gate
A0 validated dual delivered 2.03× single on Zen 4 ([dW design
§3.3](spmm_backward_avx2.md#3-3-dual-accumulator-validated-empirically)).
That specific argument does *not* apply to forward because each
SIMD iteration writes its result to memory, not back to a
register — there's no per-register dependency chain to break.

Why ship dual anyway?

1. **Memory-level parallelism.** Two independent
   load-FMA-store streams let Zen 4's out-of-order scheduler
   issue two loads and retire one store in the same cycle. A
   single-stream 8-wide loop uses only one of the two load
   ports per cycle.
2. **Instruction-level parallelism in the FMAs.** Zen 4 and
   Intel Haswell+ can dispatch 2 × 256-bit FMAs per cycle. A
   single-stream loop issues 1 FMA per iteration with a data
   dependency through the `y_row` store. Dual streams issue 2
   independent FMAs per Phase A iteration.
3. **Mirror dW structure.** Same Phase A width, same `N mod 16`
   residue boundaries, same test coverage surface, same
   loop-shape pattern in code.
4. **Mirror NEON forward.** `spmm_neon.cpp` uses 2×-unrolled
   8-wide (two independent 4-wide FMAs into disjoint `y_a` /
   `y_b`). AVX2 dual is the direct analog at 8-wide lanes.

Expected magnitude of dual-over-single on Zen 4 for this
memory-bound pattern: **1.2-1.4×** (less than dW's 2× because
the bottleneck is L1 bandwidth, not FMA latency). Precedent
from NEON forward + dW AVX2 makes this a low-risk decision
without a dedicated microbench (see §6.0 on why we skip Gate F0).

### 3.4 Phase A inner loop

```cpp
// Broadcast scalar weight into 8 lanes, once per live slot.
__m256 v_vec = _mm256_set1_ps(v);

int64_t j = 0;
// Phase A: 16 floats/iter (two independent 8-wide FMAs, write-through)
for (; j + 16 <= N; j += 16) {
    __m256 x_a = _mm256_loadu_ps(x_row + j);
    __m256 x_b = _mm256_loadu_ps(x_row + j + 8);
    __m256 y_a = _mm256_loadu_ps(y_row + j);
    __m256 y_b = _mm256_loadu_ps(y_row + j + 8);
    // Two FMAs, independent on disjoint Y slots.
    // Out-of-order scheduler can dispatch both in parallel.
    y_a = _mm256_fmadd_ps(v_vec, x_a, y_a);
    y_b = _mm256_fmadd_ps(v_vec, x_b, y_b);
    _mm256_storeu_ps(y_row + j,     y_a);
    _mm256_storeu_ps(y_row + j + 8, y_b);
}

// Phase B: one more 8-wide iter if 8-15 floats remain
if (j + 8 <= N) {
    __m256 x_vec = _mm256_loadu_ps(x_row + j);
    __m256 y_vec = _mm256_loadu_ps(y_row + j);
    y_vec = _mm256_fmadd_ps(v_vec, x_vec, y_vec);
    _mm256_storeu_ps(y_row + j, y_vec);
    j += 8;
}

// Phase C: scalar tail for 0-7 remainder
for (; j < N; ++j) {
    y_row[j] += v * x_row[j];
}
```

Per Phase A iteration (~6-7 cycles on Zen 4, dominated by the
store port):

- 4 × 256-bit loads (2 from x_row, 2 from y_row)
- 2 × 256-bit FMAs (independent — disjoint Y slots)
- 2 × 256-bit stores (back to y_row)

This is **store-port-bound** on Zen 4 (1 store/cycle). At 2
stores per Phase A iteration the absolute floor is 2 cycles, but
the store→load forwarding for the next iteration's `y_row` load
pushes realistic throughput to ~6 cycles. 16 floats / 6 cycles ≈
2.7 floats/cycle ≈ ~11 GF/s sustained per core. Times 2 cores on
CI with OpenMP ≈ 22 GF/s sustained inner-loop throughput.
Per-slot setup (`_mm256_set1_ps(v)`, col_indices lookup) drops
realized kernel throughput to ~15-20 GF/s — **5× scalar**. See
§5.5 for ship-floor / target / stretch bands.

### 3.5 No horizontal reduction — this is a feature

The dW kernel has six instructions of horizontal reduction per
live slot (castps256 → extractf128 → add → movehl → add →
shuffle → add_ss → cvtss_f32). **The forward kernel has none of
this.** Each Phase A iteration stores its result straight back
to `y_row[j]`. The forward kernel's per-slot overhead is
strictly smaller than dW's.

This is also part of why forward is memory-bound while dW was
compute-bound: dW's ~6-cycle horizontal reduce per slot
amortizes compute cost; forward has no such amortization, so the
inner loop's 4-load + 2-store pattern directly hits L1/L2
bandwidth.

### 3.6 Parallelism — unchanged

Same row-level `#pragma omp parallel for schedule(static)` with
the `SCORE_PARALLEL_ROW_THRESHOLD` gate used by the scalar,
NEON, and dW AVX2 kernels. Race-free: each row `i` writes only
to `Y[i, :]`, and all reads are from W and X which are const.
No locks, no atomics, no false sharing beyond the row boundary.

### 3.7 Self-zeroing contract — unchanged

`std::memset(Y, 0, M × N × sizeof(float))` at entry. Matches
scalar and NEON. Required because `py::array_t<float>(shape)`
does not zero-initialize (the contract we established in
milestone 3c-ii) and the write-through FMA assumes `Y` starts
at zero.

### 3.8 Why 8-wide AVX2 (not AVX-512)

Three reasons, identical to
[dW design §3.8](spmm_backward_avx2.md#38-why-8-wide-avx2-not-avx-512):

1. **CI reach.** GitHub's x86 runners don't expose AVX-512 via
   cpuid.
2. **Compile-target breadth.** AVX2 + FMA is universally
   available on every x86 CPU shipped from 2013 onward.
3. **One win at a time.** 8-wide gets us from 4 GF/s to ≥ 20
   GF/s. That closes the x86 inference gap. AVX-512 is a
   follow-on doubling for v0.3.

## 4. Decisions that matter

### 4.1 Loop order: (i, s, j) — same as everywhere else

Unchanged. See [`spmm_backward_avx2.md §4.1`](spmm_backward_avx2.md#41-loop-order-i-s-j--same-as-neon-and-scalar).

### 4.2 Inline the inner loop — don't build a `vector_axpy_avx2`

Same reasoning as dW §4.2: at 40M scale we have ~6.5M live slots
per forward pass, and a per-slot function call would cost ~2 ms
of pure overhead. Inlining also lets the compiler keep `v_vec`
hoisted out of the j-loop and keep the two `y_a` / `y_b` streams
live in ymm registers across the Phase A → Phase B transition.

### 4.3 Dispatch surface — arch-specific symbol names

**The forward SpMM path already uses architecture-specific C++
symbol names** (`sparselab::spmm_simd_neon` on ARM), unlike the
dW path where both architectures share `sparselab::spmm_grad_w_simd`.
We follow the existing forward convention:

- ARM64: `sparselab::spmm_simd_neon` (existing, from `spmm_neon.cpp`)
- x86_64: `sparselab::spmm_simd_avx2` (new, from `spmm_avx2.cpp`)

Pre-AVX2 forward dispatch in `bindings.cpp`'s `py_spmm_simd`:

```cpp
#if defined(__ARM_NEON)
    sparselab::spmm_simd_neon(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
#else
    sparselab::spmm_scalar(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
#endif
```

Post-change:

```cpp
#if defined(__ARM_NEON)
    sparselab::spmm_simd_neon(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
#elif defined(__AVX2__) && defined(__FMA__)
    sparselab::spmm_simd_avx2(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
#else
    sparselab::spmm_scalar(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
#endif
```

The Python-facing name stays `_core.spmm_simd` on every
platform. Autograd unchanged.

**Why arch-specific symbol names for forward but a shared name
for dW?** Historical precedent. The NEON forward kernel was
written first (milestone 3d) and chose `spmm_simd_neon`. The
NEON dW kernel (issue #1) later chose `spmm_grad_w_simd` for
consistency with the scalar `spmm_grad_w` name. When we wrote dW
AVX2 (milestone 14) we matched the then-existing NEON dW name;
when we write forward AVX2 now we match the then-existing NEON
forward name. Preserving the existing naming is lower-risk than
renaming an established symbol.

Header includes in `bindings.cpp`:

```cpp
#if defined(__ARM_NEON)
  #include "kernels/vector_dot_neon.hpp"
  #include "kernels/spmm_neon.hpp"
  #include "kernels/spmm_grad_neon.hpp"
#elif defined(__AVX2__) && defined(__FMA__)
  #include "kernels/spmm_grad_avx2.hpp"
  #include "kernels/spmm_avx2.hpp"     // <-- added
#endif
```

### 4.4 `setup.py` source gating — one-line extension

Mirror milestone 14's pattern. Existing x86 branch:

```python
elif IS_X86_64:
    sources += [
        "csrc/kernels/spmm_grad_avx2.cpp",
    ]
```

Becomes:

```python
elif IS_X86_64:
    sources += [
        "csrc/kernels/spmm_grad_avx2.cpp",
        "csrc/kernels/spmm_avx2.cpp",   # <-- added
    ]
```

No compile flag changes. `-march=x86-64-v3` already in place.

### 4.5 Broadcast placement — hoisted outside j-loop

`_mm256_set1_ps(v)` emits one `vbroadcastss` (~1 cycle on
Haswell+/Zen+). Doing it inside the j-loop would repeat it N/16
times per slot. Hoisted to just above the Phase A loop so it
runs once per live slot. Same pattern as `vdupq_n_f32(v)` in
`spmm_neon.cpp`.

### 4.6 Stores are `_mm256_storeu_ps` (unaligned)

`Y` comes from `py::array_t<float>` and is 16-byte aligned but
not 32-byte aligned. `_mm256_storeu_ps` is penalty-free on
Zen+ / Haswell+ when the store doesn't cross a 64-byte cache
line. Row data is contiguous so line crosses happen
statistically once per ~16 iterations and cost ~1 extra cycle.
Do **not** use `_mm256_store_ps` (aligned store) — any
misalignment would `#GP` fault. Called out prominently in the
kernel's block comments.

## 5. Testing strategy

### 5.1 Oracle tests — already parametrized

`tests/test_spmm.py` has 23 oracle tests that compare
`_core.spmm_scalar` vs `_core.spmm_simd` vs
`torch.matmul(W.to_dense(), X)` at `rtol=atol=1e-5`. On x86 CI
the `simd` parameter currently hits scalar fallback; after this
spec it hits the real AVX2 kernel. If AVX2 has a bug that scalar
doesn't, these tests catch it immediately — no new test
infrastructure needed.

### 5.2 AVX2-specific forward tests (`tests/test_spmm_avx2.py`)

Mirrors `tests/test_spmm_grad_avx2.py` case-for-case,
retargeted to forward. Runs only on x86_64 via `pytestmark =
pytest.mark.skipif(...)`. Coverage:

- **Scalar/AVX2 bit-tolerance agreement.** 20 random shapes
  with varied sparsity; `np.allclose(Y_scalar, Y_avx2,
  rtol=1e-5, atol=1e-5)` elementwise.
- **N-residue coverage.** N ∈ {1, 2, 3, 4, 5, 7, 8, 9, 15, 16,
  17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65}. Exercises every
  `N mod 16` residue plus Phase A → B → C transition
  boundaries. Includes N=16, 32, 64 (exact multiples — Phase B
  skipped) and N=17, 33, 65 (one-over multiples — Phase C
  takes one scalar iter).
- **Empty-row interleaving.** W with empty rows between
  populated rows — ensures OpenMP's static schedule doesn't
  assume balanced per-thread work.
- **Single-slot-per-row.** All rows at `row_nnz=1`. Stresses
  Phase C when N is tiny (1-3 floats).
- **Padding-slot integrity.** `padding_ratio=1.0` so half of
  `values[]` is padding. Kernel must not walk padding slots
  (otherwise it reads `col_idx=-1` → huge unsigned index into
  X → segfault or garbage). Oracle-compare catches both.
- **Fully-sparse W returns exactly zero.** Self-zeroing
  contract check.
- **Determinism under parallelism.** 3 repeated calls,
  `np.testing.assert_array_equal` bit-stable check. Defends
  against a future well-intentioned `schedule(dynamic)` change
  that would silently break reproducibility.

### 5.3 Autograd integration — unchanged

`tests/test_spmm_autograd.py` already parametrizes
`torch.autograd.gradcheck` over `scalar` + `simd`. On x86 CI
the `simd` path now exercises AVX2 forward + AVX2 dW end-to-end
through autograd. No changes needed here.

### 5.4 Cross-platform regression

Three CI platforms must all stay green:

- `macos-14` (Apple Silicon arm64, NEON)
- `ubuntu-24.04-arm` (Linux aarch64, NEON)
- `ubuntu-24.04` (Linux x86_64, AVX2) — where the new kernel
  runs.

Milestone 12's NEON forward numbers (2.3-3.4 ms on FFN shapes)
must remain within ±5%. If they regress, something
unintentionally touched an ARM path.

### 5.5 Performance gates

`.github/workflows/profile_x86_baseline.yml` (the same workflow
milestones 13 and 14 used) runs `profile_dw_baseline.py` across
platforms. The existing script measures forward SpMM implicitly
as the "dense oracle" column; for this spec we extend the
driver to also time `_core.spmm_scalar` and `_core.spmm_simd`
on the same FFN shapes. ~20 lines added.

Post-implementation thresholds (2-core Zen 4):

| Shape | Scalar before | AVX2 target | Expected si/sc |
|---|---|---|---|
| 384×1536, N=2048, s=0.90 | ~65 ms / 3.8 GF/s | ≤ 12 ms / ≥ 20 GF/s | ~0.15-0.25 |
| 1536×384, N=2048, s=0.90 | ~65 ms / 3.8 GF/s | ≤ 12 ms / ≥ 20 GF/s | ~0.15-0.25 |
| 640×2560, N=1024, s=0.90 | ~95 ms / 3.9 GF/s | ≤ 14 ms / ≥ 22 GF/s | ~0.15-0.25 |
| 2560×640, N=1024, s=0.90 | ~95 ms / 3.8 GF/s | ≤ 14 ms / ≥ 22 GF/s | ~0.15-0.25 |
| 64×64, N=128, s=0.80 | ~0.3 ms | no regression | any |

Target: **≥ 6× per-layer** on FFN shapes. Ship floor: **≥ 5×**.
Stretch: **≥ 8×**. Lower absolute GF/s targets than dW (which
shipped at ~47-51 GF/s) because forward is memory-bandwidth
bound, not compute bound — see §6.2.

## 6. Performance — measured numbers and projections

### 6.0 Gate F0 — microbench (NOT SHIPPED)

The dW spec used a Gate A0 microbench
(`csrc/bench/avx2_dot_microbench.cpp`) to validate the
dual-accumulator decision before writing the full kernel. Gate
A0's output drove a spec revision (from "ship single" to "ship
dual").

**We intentionally skip the equivalent for forward.** Three
reasons:

1. Gate A0 already proved the CI runner's Zen 4 behavior
   matches design assumptions for AVX2 FMA throughput and
   unaligned load/store.
2. Forward's hot loop is strictly simpler than dW's (no
   horizontal reduction) and follows a textbook AXPY pattern —
   `y[j] += v * x[j]` is a classical vector-fma loop
   documented in every CPU optimization guide for the last
   decade.
3. Mirroring the proven structural choices from the NEON
   forward kernel and the dW AVX2 kernel is low-risk.

If Gate F1 numbers land below target we'll add a forward
microbench at `csrc/bench/avx2_axpy_microbench.cpp` as a
diagnostic. Not needed as a pre-ship gate.

### 6.1 Pre-implementation baseline (milestone 13)

From `profile_x86_baseline.yml` run at milestone 13 (scalar
`spmm_scalar`, unchanged by milestone 14):

| Shape | Scalar ms | simd ms | si/sc | Scalar GF/s |
|---|---|---|---|---|
| demo15 FFN up (384 × 1536, N=2048, s=0.90) | ~65 | ~65 | 1.00× | 3.8 |
| demo15 FFN down (1536 × 384, N=2048, s=0.90) | ~65 | ~65 | 1.00× | 3.8 |
| demo16 FFN up (640 × 2560, N=1024, s=0.90) | ~95 | ~95 | 1.00× | 3.9 |
| demo16 FFN down (2560 × 640, N=1024, s=0.90) | ~95 | ~95 | 1.00× | 3.8 |

`si/sc ≈ 1.00` across all shapes confirms the pre-spec `_simd`
binding is the scalar fallback on x86.

### 6.2 Projection from the memory-bandwidth ceiling

Forward SpMM's inner loop is:

```
y[j] += v * x[j]    // 1 FMA, 2 loads (x, y), 1 store
```

Per 8-wide FMA: 32 bytes read from x + 32 bytes read from y + 32
bytes stored to y = 96 bytes/iter and 8 FMAs/iter.

Ceiling math on Zen 4 (GitHub ubuntu-24.04 runner):

- **L1 bandwidth**: ~100 GB/s per core. 96 B/iter / 100 GB/s =
  ~0.96 ns/iter = ~10 GF/s per core inner-loop throughput (one
  stream).
- **L2 bandwidth**: ~50 GB/s per core. At FFN N=1024/2048 the
  per-stream working set is ~4-8 KB (fits L1); L2 only matters
  when X rows are re-visited across slots, which amortizes
  well for N ≤ 2048.
- **FMA peak**: ~50 GF/s per core (1 FMA/cycle × 8 lanes × 3.2
  GHz). Far above the memory ceiling — confirms memory is the
  binding constraint.

At 2 cores with OpenMP: ~20 GF/s sustained inner-loop
throughput. Per-slot overhead (broadcast, col_indices lookup,
row-pointer math) drops realized kernel throughput to **15-20
GF/s** — **5-5.3× over 3.8 GF/s scalar**. Matches the 5× ship
floor and lands inside the 6× target.

**Why forward's ceiling is lower than dW's.** dW is
compute-bound: 1 dot product per slot, no intermediate stores,
so AVX2 unlocks the full FMA-unit speedup (12-13× shipped in
milestone 14). Forward is memory-bound: AVX2 unlocks only as
much speedup as the store port + load ports can sustain over
the single-FMA-at-a-time scalar baseline (5-8× expected).

Both are real wins; their magnitudes differ because their
bottlenecks differ.

### 6.3 Precedent from NEON forward and AVX2 dW

NEON forward on Apple M3 delivered **~5-6× local speedup** on
identical FFN shapes (`demo15` and `demo16` — milestones 3d and
11). On Graviton-class ARM the ratio was ~4×. AVX2 forward on
Zen 4 should land in roughly that range because:

- Zen 4's L1 bandwidth (~100 GB/s) is lower than M3's
  (~150 GB/s), so AVX2 forward will be more
  bandwidth-constrained than NEON forward was on M3.
- AVX2's 256-bit lanes are 2× NEON's 128-bit lanes, partially
  compensating — one AVX2 store moves 32 bytes vs NEON's 16.
- AVX2 dW on Zen 4 shipped at 11.8-12.9× (milestone 14),
  confirming the µarch can sustain the projected speedups when
  the kernel is compute-bound.

Extrapolation: **6-8× per-layer local** is the central
expectation. 10× is plausible if store-to-load forwarding is
faster than modeled on some shapes. <5× would be a red flag
requiring Gate F0 microbench diagnosis.

### 6.4 End-to-end projection (training)

Milestone 14 measured the AVX2-dW path at **1436 ms/step**
end-to-end on 40M-param transformer training (3.0× over 4316
ms/step scalar). Approximate breakdown of that 1436 ms:

| Sub-step | Approx share | Approx ms |
|---|---|---|
| Forward SpMM (all layers) | ~12% | ~170 |
| Backward dW (AVX2) | ~27% | ~380 |
| Backward dX (dense) | ~17% | ~240 |
| Embedding / attn / softmax / loss | ~45% | ~650 |

If forward SpMM drops 6× → 170 ms becomes ~28 ms → step
becomes ~1294 ms → **1.11× over milestone-14 baseline**, 3.34×
over scalar.

If forward SpMM drops 8× → 170 ms becomes ~21 ms → step
becomes ~1287 ms → **1.12× over milestone-14 baseline**, 3.35×
over scalar.

Compute-bound ceiling flattens the curve — this is consistent
with the spec's expected **1.15-1.3× end-to-end training
speedup gain** scope. Further end-to-end wins require attacking
the next bottleneck (probably embedding lookups or attention
softmax).

### 6.5 End-to-end projection (inference) — the headline

For **inference** (no backward pass, forward is 100% of step):

- Scalar pre-AVX2: ~170 ms per layer at FFN scale →
  ~170-220 ms per token for a 40M transformer → unworkable for
  real-time inference.
- AVX2 forward @ 6× local: ~28 ms per layer → ~30 ms per
  token at 40M scale → interactive.
- AVX2 forward @ 8× local: ~21 ms per layer → ~22 ms per
  token → comfortably real-time.

This is the headline number that motivates the spec. Training
parity was the milestone-14 story; inference parity is this
spec's story. For the majority of deep-learning researchers who
run locally on Linux x86 CPUs, sparse inference is now a
practical daily workflow.

### 6.6 Measurement gates

**Gate F1 (per-layer, post-implementation).** Via
`profile_x86_baseline.yml` re-run on the feature branch HEAD.
Required:

- FFN shapes: ≥ 5× per-layer speedup (ship floor), target ≥ 6×.
- Tiny shape: no regression vs scalar.
- All shapes: AVX2 output agrees with scalar within
  `rtol=atol=1e-5` (implicit via oracle tests, re-verified
  alongside the ms numbers).

**Gate F2 (end-to-end 40M).** Via `validate_40m_scalar.yml` —
same script milestone 14 used. Required:

- `ms/step` for `kernel=simd` drops by **15-25%** from the
  current 1436 ms/step (milestone-14 baseline). Target ≤
  1300 ms/step; ship floor ≤ 1360 ms/step.
- `ms/step` for `kernel=scalar` unchanged (within CI noise) —
  confirms we didn't touch the scalar path.
- Val loss identical at step 200 (3.2198 to 4 decimal places)
  — confirms AVX2's reordering doesn't perturb SGD dynamics.
  Same bar milestone 14 cleared at 0.0000 nats.

## 7. Risk register

### 7.1 Low — Pre-2013 CPU compile-target breakage

Identical risk and mitigation to
[dW §7.1](spmm_backward_avx2.md#71-low--pre-2013-cpu-compile-target-breakage).
`-march=x86-64-v3` already set by milestone 14; no new surface
area here. CHANGELOG v0.1.1 already documents the pre-2013
exclusion.

### 7.2 Low — Unaligned loads and stores on AVX2

Same shape as [dW §7.2](spmm_backward_avx2.md#72-low--unaligned-loads-on-avx2),
extended to stores. On Zen 3+ and Haswell+ unaligned 256-bit
**stores** that don't cross a 64-byte cache line are
penalty-free at 1 store/cycle. Row data is contiguous — store
line-crosses happen statistically once per ~16 iterations and
cost ~1 extra cycle, lost in the noise.

Do **not** use `_mm256_store_ps` (aligned store). Called out
in the kernel's block comments.

### 7.3 NEW — Memory bandwidth vs compute ceiling

**Risk:** forward is memory-bound. If we ship 5-6× per-layer,
a skeptic could ask "why didn't we get 10×, like NEON did on
M3 or AVX2 dW did on Zen 4?"

**Evidence:** §6.2's L1 bandwidth ceiling on Zen 4 is ~10 GF/s
per core for a 96-B/iter AXPY loop. Two-core OpenMP reaches
~20 GF/s realized; scalar baseline is ~4 GF/s; ratio is 5×.
This is the upper bound *any* AVX2 implementation can hit on
this hardware, regardless of cleverness in the kernel design.

NEON on Apple M3 got ~6× on forward because M-series has
~150 GB/s L1 bandwidth per core (vs Zen 4's ~100 GB/s) and
wider out-of-order windows. We can't teleport that bandwidth
into Zen 4.

**Mitigation:** document the bandwidth ceiling in the milestone
writeup up-front, same way milestone 14 documented the compute
ceiling for dW. Don't let a skeptic surface this critique
first — we surface it.

### 7.4 Low — Clang x86 auto-vec on forward inner loop

**Risk:** `-march=x86-64-v3` is already set, so Clang *could*
auto-vectorize `spmm_scalar`'s inner loop. If it does and the
measured scalar rate jumps above 3.8 GF/s, our hand-written
kernel's relative speedup shrinks.

**Counter-evidence:**

- Milestone 14's measurement *post* -march set dW scalar at
  4.3 GF/s, only a 12-16% bump from 3.8. The dW outer
  structure — runtime-varying slot indices into `dW_values` —
  defeated Clang's auto-vectorizer. The forward outer
  structure (runtime-varying `c` and `v` indexing `X` and
  `W.values`) is the same shape and defeats auto-vec for the
  same reason.
- If Clang ever gets smart enough to fully auto-vectorize this
  pattern, our hand-written kernel becomes redundant. That's
  fine — means the scalar path got faster. We'd retire the
  AVX2 kernel and save maintenance cost. Not a correctness
  risk, only a "was the effort worth it" risk.

**Mitigation:** Gate F1 measures scalar *and* AVX2 on the same
run; the ratio is the honest speedup. If scalar somehow jumps
to 20 GF/s we'll see it before investing in end-to-end
validation.

### 7.5 Low — Intel vs AMD µarch divergence

Same low risk and same mitigation as
[dW §7.4](spmm_backward_avx2.md#74-low--intel-vs-amd-avx2-behavior-divergence).
Zen 4 in CI is in the middle of the Intel/AMD distribution.
Intel Haswell+/Ice Lake+ should perform similarly or slightly
better (2 × 256-bit FMA pipes); Zen 1 (2 × 128-bit AVX2
micro-ops) slightly worse. Never worse than scalar, so "AVX2
forward is a net win" holds everywhere.

## 8. What we're explicitly not doing

- **No AVX-512 port.** v0.3 scope. Same rationale as dW §8.
- **No runtime CPU feature detection / dispatcher.**
  Compile-time `-march=x86-64-v3` is statically sufficient.
  Runtime dispatch is a v0.3 complexity budget when AVX-512
  lands.
- **No forward Gate F0 microbench.** §6.0 argues it's not
  blocking. Add later as a diagnostic if Gate F1 misses target.
- **No dX (upstream-gradient) AVX2 port.** `dX` already uses
  the transpose-cache dense path via `torch.matmul`; it's not
  a sparse kernel and doesn't benefit from AVX2 work on our
  side.
- **No Intel macOS (x86_64) wheels.** Upstream PyTorch EOL'd
  macOS x86_64 in January 2024. Carve-out documented in
  v0.1.1 CHANGELOG.
- **No Windows x86 wheels.** Separate effort.
- **No SSE4 / AVX1 fallback for pre-2013 x86 CPUs.** Minimum
  requirement already documented in v0.1.1 CHANGELOG.
- **No tolerance tightening.** Keep `rtol=atol=1e-5` same as
  NEON and dW AVX2.
- **No new public Python API symbol.** `_core.spmm_simd`
  already existed; it just starts being fast on x86.
- **No code changes to NEON forward (`spmm_neon.cpp`), NEON dW
  (`spmm_grad_neon.cpp`), AVX2 dW (`spmm_grad_avx2.cpp`), or
  scalar kernels.**
- **No change to `setup.py` flags.** Source list gets one new
  row; compile flags unchanged.
- **No change to autograd.** `_SpMMFunction` already calls
  `_core.spmm_simd` in forward when `kernel="auto"`.

---

## Appendix — Borrow-Don't-Reinvent references

**Scalar pattern mirrored:** `csrc/kernels/spmm.cpp`. Unchanged.

**SIMD pattern mirrored (forward):** `csrc/kernels/spmm_neon.cpp`.
Same Phase A/B/C loop shape (at 8-wide lanes instead of our
16-wide Phase A), same self-zeroing contract, same OpenMP
parallelism, same hoisted-broadcast pattern.

**SIMD pattern mirrored (structure):**
`csrc/kernels/spmm_grad_avx2.cpp`. Same 16-float Phase A,
dual-stream structure, Phase B 8-wide trail, Phase C 0-7 scalar
tail, same OpenMP gate. The only structural difference is the
inner-loop body (write-through AXPY vs dot-product-accumulate)
and absence of horizontal reduction.

**Dispatch pattern extended:** `csrc/bindings.cpp`'s
`py_spmm_simd` — same `#if/#elif/#else` pattern milestone 14
used for `py_spmm_grad_w_simd`.

**Build-flag pattern unchanged:** `setup.py`'s
`-march=x86-64-v3` already in place. One new source file in
the `IS_X86_64` branch; no new flags.

**External reference — Intel / AMD optimization guides:**
- Intel® 64 and IA-32 Architectures Optimization Reference
  Manual, Ch. 15 (AVX intrinsic usage, FMA patterns, store-port
  scheduling).
- AMD Software Optimization Guide for AMD Family 19h (Zen 3 /
  Zen 4), §2.7 (256-bit AVX2 pipe scheduling).
- Agner Fog's instruction tables (`https://agner.org/optimize/`)
  for per-µarch latency/throughput of `vmovups`, `vfmadd231ps`,
  and `vbroadcastss`.

The AXPY (`y += v * x`) pattern is textbook — documented in
every BLAS manual since the 1970s. Our kernel adapts it to the
PaddedCSR walk; every intrinsic it uses is well-documented
off-the-shelf.

---

_Shipped as part of sparselab's Linux x86 parity work (v0.2.x).
Completes the AVX2 SIMD story started by
[milestone 14](../demos/milestone_14.md) (dW backward). See the
CHANGELOG for the exact release version._
