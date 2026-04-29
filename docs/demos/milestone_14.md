# Milestone 14 — AVX2-accelerated dW kernel (Linux x86 parity)

**Closes issue #2. Shipped as part of sparselab's Linux x86 parity
work (v0.2.x).**

## What this milestone proves

The AVX2 port of `spmm_grad_w_simd` — the sparse weight-gradient
kernel — makes sparse-from-scratch training on **Linux x86_64**
**3.0× faster end-to-end** at 40M-param transformer scale.
Per-step wallclock drops from **4316 ms → 1436 ms** on
GitHub's AMD EPYC 9V74 (Zen 4) CI runner. Training dynamics are
unchanged: same seed produces identical val loss to four decimal
places. No public API change; Linux x86 users get this automatically
via the default `kernel="auto"` routing in `SparseLinear`.

This closes the Linux-parity gap flagged in
[milestone 13](milestone_13.md):

- Pre-work, Linux x86 scalar dW ran at ~3.8 GF/s — **~4× slower than
  the pre-NEON ARM scalar baseline** and ~25-30× slower per FFN
  layer than post-NEON Apple Silicon. Clang on x86 with our pre-fix
  flags emitted truly-scalar FMAs.
- After Gate 1.5 (Phase 0 of this work), adding `-march=x86-64-v3`
  to `setup.py` raised scalar to ~4.3 GF/s — an incremental 12-16%
  win purely from letting the compiler see AVX2 as a valid target.
  Auto-vectorization did not kick in for the dW inner loop (same
  pattern as ARM pre-NEON).
- After this milestone, scalar stays at 4.3 GF/s (unchanged — we did
  not touch `spmm_grad.cpp`), but `_core.spmm_grad_w_simd` on x86
  now dispatches to the hand-written dual-accumulator AVX2 kernel
  in `csrc/kernels/spmm_grad_avx2.cpp`.

For the majority of deep-learning researchers worldwide who run on
Linux x86, sparse-from-scratch training is now a practical daily
workflow on CPU, not a macOS-only story.

## Headline numbers

### End-to-end training step at 40M scale (demo_16 architecture)

Same transformer as [milestone 11](milestone_11.md) and
[milestone 12](milestone_12.md) — 8 layers, d_model=640, d_ff=2560,
attention 70% sparse, FFN 90% sparse, ~40M dense params / ~6.5M
live sparse weights. 200 steps per kernel, seed=42, identical data
and code paths apart from the dW kernel dispatch. Both runs were
executed on the same `ubuntu-24.04` runner class (AMD EPYC 9V74)
via `.github/workflows/validate_40m_scalar.yml`.

| Path                              | Per-step wallclock | Final val loss @ 200 |
|-----------------------------------|--------------------|-----------------------|
| Sparse-all, kernel="scalar"       | **4295 ms/step**   | 3.2198                |
| Sparse-all, kernel="simd" (AVX2)  | **1436 ms/step**   | 3.2198                |
| **End-to-end speedup**            | **2.99×**          | gap: **0.0000 nats**  |

"Before" baseline (Phase-A stub, measured on HEAD's parent):

| Path                              | Per-step wallclock | Final val loss @ 200 |
|-----------------------------------|--------------------|-----------------------|
| Sparse-all, kernel="scalar"       | 4316 ms/step       | 3.2198                |
| Sparse-all, kernel="simd" (stub)  | 4303 ms/step       | 3.2198                |
| Ratio                             | 1.00×              | gap: 0.0000 nats      |

The scalar column is unchanged across both runs (4316 → 4295, within
CI noise), verifying we did not regress the scalar path. The simd
column dropped 4303 → 1436 ms/step — a **3.0× reduction** entirely
attributable to the new AVX2 kernel.

Val-loss identity between scalar and AVX2 (3.2198 on both) verifies
that AVX2's dual-accumulator reordering does not perturb SGD
dynamics — same observation we made for NEON in milestone 12.
Across ~3200 dW kernel invocations per 200-step run (16 FFN layers ×
200 steps), accumulator reordering didn't introduce a single bit of
drift in the final val loss.

### Per-layer dW throughput (Zen 4, 2 vCPUs, via profile_x86_baseline)

Measured on the same runner class as the end-to-end validation,
via `.github/workflows/profile_x86_baseline.yml` triggered on the
same commit (`e8ac45a`).

| Shape                                       | Scalar ms | AVX2 ms | Speedup | AVX2 GF/s |
|---------------------------------------------|-----------|---------|---------|-----------|
| demo15 FFN up   (384 × 1536, N=2048, s=0.90) | 62.4      | 4.90    | **12.7×** | ~49       |
| demo15 FFN down (1536 × 384, N=2048, s=0.90) | 62.2      | 4.91    | **12.7×** | ~49       |
| demo16 FFN up   (640 × 2560, N=1024, s=0.90) | 84.8      | 7.16    | **11.8×** | ~47       |
| demo16 FFN down (2560 × 640, N=1024, s=0.90) | 84.7      | 6.54    | **12.9×** | ~51       |
| tiny (64 × 64, N=128, s=0.80)                | 0.04      | 0.01    | 4.0×      | ~6        |

Scalar column reproduces [milestone 13](milestone_13.md)'s
pre-AVX2 measurement unchanged. AVX2 GF/s is computed from the
measured ms × 2 FMAs per multiply-accumulate × (M × n_live × N)
arithmetic intensity. All FFN shapes land comfortably within the
spec's **30 / 40-45 / 50-55** ship-floor / target / stretch bands
from [design §5.5](../design/spmm_backward_avx2.md).

The tiny shape's more modest 4× reflects that 64-element dot
products barely exercise the Phase A loop (one iteration) — most
time is per-slot overhead (horizontal reduction, col_indices
lookup, row pointer arithmetic) which AVX2 cannot accelerate.
Critically, it did not regress — the spec's explicit non-regression
threshold for tiny shapes is maintained.

## What we measured — Gate A0, Gate 1.5, and Gate 2

The design had three explicit measurement gates (see
[`docs/design/spmm_backward_avx2.md`](../design/spmm_backward_avx2.md)
§5.5 and §6.0):

**Gate A0 (pre-implementation, design-validation).** Standalone
microbenchmark at `csrc/bench/avx2_dot_microbench.cpp` run via
`.github/workflows/validate_avx2_microbench.yml`. Tested the
single-accumulator vs dual-accumulator AVX2 pattern on the actual
CI runner before we committed a week of kernel work to either.
Result: dual delivered **2.03× over single** on Zen 4 — enough of a
win to revise the spec from "single is sufficient" to "dual ships".

**Gate 1.5 (post-flag, pre-kernel).** After adding
`-march=x86-64-v3` to `setup.py`, re-measured scalar to confirm
Clang hadn't auto-vectorized the inner loop. Scalar went from 3.8
GF/s → 4.3 GF/s, a 12-16% flag-induced win that did not erase the
case for a hand-written kernel.

**Gate 2 (ship decision).** Required ≥ 1.5× end-to-end speedup at
40M-param scale with val-loss delta < 0.01 nats. Measured
**2.99× speedup** with **0.0000 nats** val-loss delta — nearly
**2× above the ship threshold** and clean of any training-
dynamics drift.

All three gates are reproducible via manual-dispatch GitHub Actions
workflows, all three produce machine-readable artifacts.

## The kernel — what it does

`csrc/kernels/spmm_grad_avx2.cpp` mirrors the 16-wide
dual-accumulator pattern from
[`csrc/kernels/spmm_grad_neon.cpp`](../../csrc/kernels/spmm_grad_neon.cpp):

- **Phase A**: main loop, 16 floats/iter using two independent
  8-wide `__m256` accumulators chained by `_mm256_fmadd_ps`. The
  two FMAs are issued into disjoint registers so Zen 4's
  out-of-order scheduler dispatches them in the same cycle.
- **Phase B**: trailing 8-wide FMA if 8-15 floats remain after
  Phase A.
- **Horizontal reduce**: standard 3-step collapse
  (`_mm256_castps256_ps128` + `_mm256_extractf128_ps` +
  `_mm_movehl_ps` + `_mm_shuffle_ps`) to a single scalar per live
  slot.
- **Phase C**: scalar cleanup for the final 0–7 floats.

The inner loop is inlined (not a call to a hypothetical
`vector_dot_avx2_simd`) — at 40M scale that would be ~6.5M function
calls per backward pass, worth ~2 ms of pure overhead. OpenMP
parallelizes over the outer row dimension exactly like the scalar
and NEON variants, with the same `SCORE_PARALLEL_ROW_THRESHOLD`
gate.

Loads use `_mm256_loadu_ps` (unaligned) throughout: row pointers
come from `std::vector<float>` at 16-byte alignment but not 32-byte.
On every Zen+ and Haswell+ part, unaligned 256-bit loads that don't
cross a 64-byte cache line are penalty-free at 1 cycle / load port.

## Dispatch — no public API change

On x86_64, a new C++ symbol `sparselab::spmm_grad_w_simd` is exposed
through pybind11 as `_core.spmm_grad_w_simd`. The name is the same
as the NEON binding; `setup.py`'s `IS_ARM64` vs `IS_X86_64` gating
ensures only one of the two kernels is compiled per build. Python-
facing the API is identical on every platform.

The `bindings.cpp` dispatch surface gained one new `#elif` branch:

```cpp
#if defined(__ARM_NEON)
    sparselab::spmm_grad_w_simd(W, ...);          // NEON kernel
#elif defined(__AVX2__) && defined(__FMA__)
    sparselab::spmm_grad_w_simd(W, ...);          // AVX2 kernel
#else
    sparselab::spmm_grad_w(W, ...);               // scalar fallback
#endif
```

The autograd path in `sparselab/ops.py` is unchanged from v0.2.1 —
it still chooses between scalar and simd based on the `kernel`
argument stashed in forward. Since `kernel="auto"` is the default,
Linux x86 users pick up the speedup with no code change.

## Correctness

This milestone did not add new *types* of tests — the oracle
parametrization over scalar + simd from milestone 12
(`kernel_fn ∈ [scalar, simd]`) automatically exercised the AVX2
kernel on x86 CI. Full suite remains **442 passed, 2 skipped**
across supported Linux platforms.

On x86 CI (`ubuntu-24.04`), the 23 shape / padding / edge-case
tests in `test_spmm_grad.py` now hit the real AVX2 kernel via
`kernel_fn=simd`. The 18 N-residue cases in
`test_spmm_grad_neon.py` — which were originally written for the
NEON Phase-A/B/C boundaries — are structurally identical to AVX2's
Phase-A/B/C boundaries (just 16/8/1 widths instead of 8/4/1), so
the existing tests cover Phase B → Phase C transitions on AVX2
with no new test files needed.

Numerical note: AVX2's dual-accumulator reorders the dot-product
summation similarly to NEON. Per-slot results may differ from
scalar by 1–2 ULPs over N ~ 1000-2000. All tests pass at
`rtol=atol=1e-5`, the same tolerance applied to NEON in milestone
12.

## What this milestone does not claim

- ❌ **AVX-512.** GitHub's x86 runners are Zen 4 but don't expose
  AVX-512 via cpuid. An AVX-512 kernel we can't CI is an unverified
  kernel. AVX-512 belongs in v0.3 and needs a self-hosted runner.
- ❌ **Intel-specific numbers.** Our CI is AMD Zen 4. Intel
  Haswell+ / Ice Lake+ should perform similarly or better (dual
  256-bit FMA pipes), but we have not measured on Intel silicon.
- ❌ **Windows x86.** Tracked as a separate effort; not in v0.2
  scope.
- ❌ **Pre-2013 x86 CPUs.** The `-march=x86-64-v3` flag requires
  Haswell+ (Intel) / Zen 1+ (AMD). Older CPUs are not supported
  starting the next release — will be documented in the changelog
  when that release is tagged.
- ❌ **All backward paths got 3× faster.** This is a
  `dW`-only speedup. Forward SpMM on x86 remains scalar; `dX`
  already used the transpose cache. dW was the largest share of a
  step (62% at 10M, measured in milestone 10) so fixing it
  translated to the headline end-to-end number.

## Reproduce

```bash
# The full "before/after" workflow (ubuntu-24.04 CI, ~30 min):
gh workflow run "Validate 40M scalar baseline" --ref main
# Wait, then:
gh run download <run-id> --dir ./validate-artifacts
cat ./validate-artifacts/validate-40m-x86_64/validate_40m.txt

# Per-layer AVX2 throughput (ubuntu-24.04 CI, ~2 min):
gh workflow run "Profile x86 baseline" --ref main

# Design-gate microbenchmark (ubuntu-24.04 CI, ~9 sec):
gh workflow run "Validate AVX2 microbench" --ref main

# Local (Apple Silicon / Linux aarch64) — unchanged:
python examples/demo_17_dw_neon.py
python examples/profile_dw_baseline.py
pytest tests/ -q
```

## Files changed

- **New:** `csrc/kernels/spmm_grad_avx2.{hpp,cpp}` (Phase A stub +
  Phase B body landed across 2 commits)
- **New:** `csrc/bench/avx2_dot_microbench.cpp` (Gate A0)
- **New:** `.github/workflows/validate_avx2_microbench.yml`
- **New:** `.github/workflows/validate_40m_scalar.yml`
- **New:** `docs/design/spmm_backward_avx2.md`
- **New:** `examples/validate_40m_dw.py`
- **Modified:** `setup.py` (compile `spmm_grad_avx2.cpp` when
  `IS_X86_64`, `-march=x86-64-v3`)
- **Modified:** `csrc/bindings.cpp` (AVX2 dispatch branch)

## What this unblocks

With Linux x86 parity closed:

1. **The upcoming v0.2 minor release can honestly claim "sparse
   training at transformer scale on CPU" across all supported
   platforms.** Milestone 12's message was Apple-Silicon-only.
   This milestone makes it a cross-platform claim.
2. **Community Quiet-Open share** — the headline
   "3× faster 40M sparse training on a free CI runner" screenshot
   is now true and reproducible by anyone with a GitHub account.
3. **100M+ validation becomes realistic on CPU.** Before AVX2, a
   single training step of a 100M-param sparse model on x86 would
   have been ~10 seconds; after, it's within ~5 seconds — enough
   that a community-contributed overnight run fits.
4. **AVX-512 follow-on (v0.3)** is now a pure additive change: the
   dispatch branch is in `bindings.cpp`, the CI microbenchmark
   harness is in place, the numerical-tolerance story is
   established.

---

_Measured: 2026-04-29 via `validate_40m_scalar.yml` GitHub Actions
runs 25093014403 (before, commit `2e57e63`) and 25094103954 (after,
commit `e8ac45a`). Per-layer numbers via `profile_x86_baseline.yml`
run 25094911114._
