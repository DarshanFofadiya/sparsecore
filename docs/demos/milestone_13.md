# Milestone 13 — Cross-platform dW baseline (x86 vs ARM64)

**This is a data-point milestone, not a ship milestone.** No code
shipped; we ran the `profile_dw_baseline.py` script on three
platforms and captured the numbers so future SIMD work on x86 has
a motivated baseline to close.

This mirrors the Gate-1 discipline we used for the ARM NEON dW
kernel ([milestone 12](milestone_12.md), issue #1): measure scalar
baseline before committing to hand-written SIMD, so we invest in
the right place for the right reasons.

Reproduce via the `Profile x86 baseline` GitHub Actions workflow —
manual trigger, ~2 minutes per matrix row.

## Headlines

- ✅ **Linux aarch64 (Graviton-class): NEON works.** The hand-written
  NEON kernel is 3.1-4.2× faster than scalar on FFN shapes, so the
  ARM64 wheel we ship is genuinely accelerated on non-Apple ARM too.
- ⚠️ **Linux x86_64 is badly under-served.** Scalar dW runs at
  **~3.8 GF/s**, about 4× slower than the pre-NEON scalar baseline
  on Apple Silicon, and **~25-30× slower per FFN dW layer** than
  post-NEON Apple Silicon.
- 🎯 **AVX2 hand-written kernel is the highest-leverage v0.3 item.**
  Clang-17 on x86 with our default flags emits truly-scalar FMAs —
  not even SSE2. Even a plain 8-wide AVX2 FMA kernel should give us
  4-6× local speedup, on par with what NEON did for ARM.

## Hardware measured

| Platform | Runner | CPU | Cores | SIMD flags |
|----------|--------|-----|-------|------------|
| Apple Silicon | local | M3 Pro | 6 (perf) | NEON (ASIMD) |
| Linux aarch64 | GitHub Actions `ubuntu-24.04-arm` | Graviton-class | 4 | NEON |
| Linux x86_64 | GitHub Actions `ubuntu-24.04` | AMD EPYC 9V74 (Zen 4) | 2 | AVX, AVX2, FMA (no AVX-512) |

Runner notes:
- GitHub's default x86 runners are AMD Zen 4 (Genoa-class) at the
  time of writing. The CPU supports AVX-512 architecturally but the
  runner doesn't expose it via `cpuid` — GitHub likely gates it to
  avoid per-runner capability drift.
- Torch threads varies (2 on x86, 4 on aarch64, 6 on M3). The GFLOPS
  numbers below are per-process throughput, which folds in parallel
  scaling. Per-core GFLOPS would be similar across all three but
  that's not what researchers care about.

## dW throughput per platform

All at identical FFN shapes (the ones actually used in demos 15 and
16's mini-GPT workloads). `scalar` = our `spmm_grad_w` kernel;
`simd` = `spmm_grad_w_simd` (NEON on ARM, scalar fallback on x86).

### Apple Silicon M3 Pro (milestone 12, reproduced)

| Shape | Scalar ms | NEON ms | si/sc | Scalar GF/s |
|-------|-----------|---------|-------|-------------|
| demo15 FFN up | 15.55 | 2.38 | 0.15× | 14.4 |
| demo15 FFN down | 15.40 | 2.31 | 0.15× | 14.8 |
| demo16 FFN up | 20.71 | 3.25 | 0.16× | 14.0 |
| demo16 FFN down | 21.42 | 3.40 | 0.16× | 14.8 |

NEON speedup: **~6.5× local**, measured end-to-end **1.96×** at
40M-param transformer scale.

### Linux aarch64 (Graviton-class, GitHub Actions)

| Shape | Scalar ms | NEON ms | si/sc | Scalar GF/s |
|-------|-----------|---------|-------|-------------|
| demo15 FFN up | 18.97 | 4.55 | **0.24×** | 12.65 |
| demo15 FFN down | 19.00 | 4.69 | 0.25× | 12.63 |
| demo16 FFN up | 27.11 | 8.04 | 0.30× | 12.33 |
| demo16 FFN down | 27.04 | 8.25 | 0.31× | 12.37 |

NEON speedup: **~3.1-4.2× local**, slightly less than Apple Silicon.
Almost certainly because Graviton's NEON microarchitecture can't
dispatch two independent FMAs per cycle as aggressively as Apple's
wide-issue cores — but it still delivers a meaningful win. **The
NEON wheel ships correctly for Graviton.**

### Linux x86_64 (AMD EPYC 9V74, GitHub Actions)

| Shape | Scalar ms | simd ms | si/sc | Scalar GF/s |
|-------|-----------|---------|-------|-------------|
| demo15 FFN up | 62.76 | 62.78 | 1.00× | **3.82** |
| demo15 FFN down | 63.26 | 62.68 | 1.00× | 3.79 |
| demo16 FFN up | 85.55 | 85.44 | 1.00× | 3.91 |
| demo16 FFN down | 88.81 | 88.73 | 1.00× | 3.76 |

si/sc ≈ 1.00 is expected because the `_simd` binding falls back to
the scalar kernel on x86 (no hand-written AVX path exists yet).
But the scalar number itself is alarming: **3.8 GF/s is ~4× slower
than the 14 GF/s we see on ARM scalar**. That's the signal Clang-17
is not emitting AVX FMAs at all on this code — probably not even
SSE2.

## Why the x86 scalar is this bad

Two compounding reasons:

1. **No platform-specific compile flags.** Our `setup.py` sets
   `-O3 -std=c++17` on x86 but no `-march` or `-mtune`. Clang's
   default target for `x86-64` is x86-64-v1 (baseline, which is
   SSE2). Without `-mavx2` or `-march=x86-64-v3` the compiler
   won't emit AVX instructions even when the hardware has them,
   because the resulting binary wouldn't run on older CPUs.

2. **The dW inner loop defeats auto-vectorization on x86 the same
   way it did on ARM.** The outer `dW_values[row_ptr + s] = acc`
   structure with runtime-varying slot indices is opaque to the
   vectorizer; Clang plays it safe and emits serial scalar FMAs.

On M-series the scalar loop at least runs at 14 GF/s because Apple
Clang applies `-mcpu=apple-m1` by default, which implies NEON/FMA
are always available. No equivalent default exists on generic x86
builds.

## Researcher-visible impact of the x86 gap

Per-FFN-layer dW time on the 40M-param transformer:

| Platform | dW per FFN layer | 16 FFN layers per step |
|----------|-----------------:|-----------------------:|
| Apple Silicon M3 (NEON) | ~3.4 ms | **~54 ms per step** |
| Linux aarch64 (NEON) | ~8.2 ms | ~131 ms per step |
| Linux x86_64 (scalar) | ~88 ms | **~1.4 seconds per step** |

Put differently: a 1000-step training run of demo_16's all-sparse
path takes

- ~13 minutes on Apple Silicon post-NEON (the v0.2.1 headline)
- ~20-30 minutes on Graviton (still a real workflow)
- **~25+ minutes *just for dW backward*** on GitHub's x86 runner,
  with total step time pushed well past what makes for a usable
  interactive research loop

For a researcher on a Linux x86 workstation — which is the majority
of deep-learning researchers globally — the v0.2.1 library as
packaged does not deliver the sparse-from-scratch speed story we
wrote about in milestone 12. That's the scope we need to close
before tagging a public v0.2 release.

## What this motivates (v0.2.x or v0.3 roadmap)

**Immediate (this is why the milestone exists):**

- **AVX2 hand-written dW kernel** (issue #2, spec to follow). Same
  structural template as `spmm_grad_neon.cpp`: 8-wide dual-accumulator
  FMA with Phase A/B/C. The AVX2 `_mm256_fmadd_ps` intrinsic is the
  direct analog of `vfmaq_f32`. Expected local speedup: 4-6× based
  on the scalar baseline being ~4× below even a minimal AVX2
  implementation's ceiling.

- **Add `-march=x86-64-v3` to x86 compile flags** for our wheels.
  This raises the baseline to AVX2 at zero code cost. Any x86 CPU
  from 2013+ has AVX2; sparse training on a pre-2013 CPU is not a
  realistic use case. Standalone this fix will probably get us
  1.5-2× on scalar dW from compiler auto-vec alone, before any
  hand-written SIMD.

**Later (v0.3):**

- **Runtime AVX-512 dispatch** for Ice Lake / Zen 4 / Sapphire
  Rapids users who have it. Not worth the dispatch-surface
  complexity for v0.2 given GitHub's runners don't have AVX-512
  enabled; we'd ship an unverified path.

- **ARM forward SpMM tiling** to reclaim the per-FFN Python
  overhead visible once dW is fast. Less urgent than the x86 work.

## Reproduce

```
# Manual trigger via GitHub CLI:
gh workflow run "Profile x86 baseline" --ref main

# Or via the GitHub UI:
#   Actions → "Profile x86 baseline" → Run workflow → main

# Wait ~2 minutes, then:
gh run list --workflow="profile_x86_baseline.yml" --limit 1
gh run download <run-id> --dir ./profile-artifacts
cat ./profile-artifacts/profile-Linux-x86_64/profile_output.txt
```

Two matrix rows run in parallel (`ubuntu-24.04` and
`ubuntu-24.04-arm`). Output artifacts are uploaded to the run.

## Honest disclaimers

- **These are GitHub Actions runners, not bare metal.** Virtualized
  runners can show higher variance and lower absolute throughput
  than what a user's real machine hits. We care about the *ratio*
  between scalar and NEON, and about the *class* of x86 throughput
  (scalar-only vs AVX), both of which are preserved in VMs.
- **We did not test Intel x86 runners.** GitHub currently provisions
  AMD Zen 4 for its x86 pool. An Intel-specific AVX-512 test would
  need a self-hosted runner or a different CI provider. Not a
  blocker for v0.2 planning but worth noting.
- **Our scalar kernel's low x86 GFLOPS is partially our own fault**
  (missing `-march` flag). Adding that flag alone — before any
  hand-written AVX kernel — will probably close some of the gap.
  The v0.2 fix should do both: set the march flag AND ship an AVX2
  hand-kernel, for the same reason we did both `-mcpu=apple-m1` and
  hand-written NEON on ARM.

---

_Measured: 2026-04-28 via `profile_x86_baseline.yml` GitHub Actions
run 25035167835._
