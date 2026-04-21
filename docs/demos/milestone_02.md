# Milestone 2 Demo — NEON SIMD Vector Dot Product

**Script:** `examples/demo_02_dot.py`
**Phase:** 2 (Dense SIMD warmup)
**Status:** ✅ Complete

## What this milestone proves

1. **Hand-rolled NEON intrinsics work end-to-end on Apple Silicon.** Our kernel uses `vld1q_f32`, `vfmaq_f32`, `vdupq_n_f32`, `vaddvq_f32` — the four intrinsics we'll need for every SIMD kernel in the project.
2. **SIMD gives a real speedup over scalar.** Roughly **3x at large sizes** (on M3 Pro), which is 75% of the theoretical 4x ceiling for 4-wide float32 lanes.
3. **Our remainder-loop handles all size classes correctly.** The 50-test Oracle suite includes sizes 15, 17 (where remainder bugs hide) — all pass.
4. **The scaffolding for adding more kernels is proven.** `csrc/kernels/` pattern works; the x86 AVX contribution path is a clear `#ifdef` away.

## How to run it

```bash
conda activate sparsecore
python examples/demo_02_dot.py
```

Takes ~10 seconds.

## Sample output (M3 Pro)

```
      size     scalar (µs)       NEON (µs)      torch (µs)    NEON vs scalar
────────────────────────────────────────────────────────────────────────────
       128           0.734           0.663           0.616            1.11x
      1024           1.399           0.801           0.591            1.75x
     16384          13.516           4.914           1.108            2.75x
    131072         102.361          33.275           2.490            3.08x
   1048576         815.120         272.269          15.640            2.99x
```

Your numbers will vary by hardware and thermal state.

## What's happening at each size

- **n=128** — FFI overhead dominates. All three implementations look similar; you're measuring Python → C++ → Python call cost, not kernel cost. This is honest reality; we report it rather than hiding it.
- **n=1024** — Cache-resident, SIMD starts to win. ~1.75x over scalar.
- **n=16_384** — Steady-state SIMD, ~2.75x speedup.
- **n=131_072, 1_048_576** — Memory-bandwidth-bound, NEON at ~3x over scalar (near the 4x ceiling).

## The torch.dot comparison

At n=1M, `torch.dot` is ~17x faster than our NEON. This is not a defect — it's a realistic view of the gap between "a clean-room SIMD implementation" and "Apple's multi-year hand-tuned Accelerate framework." `torch.dot` on Apple Silicon uses:

- **Apple AMX** — a dedicated matrix coprocessor we don't touch
- **Multi-core parallelism** — our kernel is single-threaded
- **Explicit prefetching** — we rely on the CPU's hardware prefetcher

The goal of Milestone 2 is not to beat Apple Accelerate. It's to prove our SIMD infrastructure works, our tests catch remainder-loop bugs, and our `kernels/` + `bindings/` pattern scales. That goal is met.

We revisit the performance gap in v0.2, after the sparse work (Phase 3+4) is done — by which point single-core NEON may be irrelevant anyway, because the sparse SpMM kernel has different bottlenecks than dense dot.

## What to observe in the demo output

- ✓ The `NEON vs scalar` column grows from ~1x (tiny sizes, FFI-bound) to ~3x (large sizes, SIMD-bound). That monotonic rise is the SIMD-win signature.
- ✓ The correctness block at the bottom shows absolute diff ~1e-3 but relative diff ~1e-6 — this is expected. Reductions of 1M floats accumulate per-element rounding errors; relative tolerance is what matters, and we're safely under rtol=1e-5.
- ✓ The compiled `.so` is reported as `arm64 native` — no Rosetta emulation, which is a silent failure mode we've designed against.

## Failure modes to recognize

If you ever see:

- **NEON slower than scalar** — likely a compiler flag regression. Check `setup.py` for `-mcpu=apple-m1` and `-O3`.
- **NEON speedup at n=10_000 but not at n=1_000_000** — possibly thermal throttling; rerun after a few minutes of cool-down.
- **Large numerical diff in the correctness block** — don't panic yet; check relative, not absolute. Absolute grows with size; relative doesn't.

## Public-artifact status

Per `.kiro/steering/demo-driven.md`, Milestone 2's demo is **shareable in Quiet Open community threads** (r/cpp, r/simd) once the repo goes public. The 16-line NEON kernel with its teaching comments makes it a clean artifact for "I learned NEON building SparseCore" content.
