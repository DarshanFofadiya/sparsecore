# Milestone 3 — PaddedCSR + SpMM (scalar + NEON)

## What ships

- **PaddedCSR** (3a–3b): sparse matrix format with per-row padding for O(1) insertion during future topology mutation
- **`sparsecore.spmm(W, X)`** (3c–3d): forward-pass Y = W @ X with a scalar reference + NEON SIMD implementation
- **198 passing tests** including 43 NEON-specific correctness tests covering all 4 tail residues, empty rows, padding slots, and extreme values

## Demo to run

```bash
python examples/demo_03_spmm.py
```

Expected output shape:

```
SPARSITY         NNZ    torch (ms)    ours (ms)    SPEEDUP   ORACLE DIFF
      0.0%   1,048,576         0.8         45.5      0.02x ×    0.00e+00
     50.0%     524,690         0.9         24.5      0.04x ×    0.00e+00
     80.0%     209,781         0.8         10.0      0.08x ×    0.00e+00
     90.0%     104,782         0.8          5.1      0.16x ×    0.00e+00
     95.0%      52,411         0.8          2.6      0.31x ×    0.00e+00
     98.0%      21,026         0.8          1.2      0.68x ×    0.00e+00
     99.0%      10,485         0.8          0.7      1.17x ✓    0.00e+00
```

The exact numbers vary by machine (M3 Pro used here); the shape of the
curve is the story.

## What this proves

**Correctness.** `0.00e+00` max oracle diff across every sparsity — our
sparse multiply produces identical results to `torch.matmul` to the
float32 bit. That's the non-negotiable layer we have to nail before
anything else matters.

**Scaling.** Our wall-clock time scales with `nnz`, not matrix size.
As sparsity rises from 0% to 99%, our time drops from 45ms → 0.7ms (a
65× reduction tracking the 100× drop in non-zero entries). The constant
factor is what the NEON work is spending on each live entry.

**Crossover.** On this shape with M3 Pro, NEON-sparse beats AMX-dense
starting at **99% sparsity**. Below that, Apple's AMX coprocessor
outruns general-purpose NEON on dense-equivalent work.

## What we learned about performance

When we measured our hand-written NEON against our scalar kernel,
they came out **almost exactly equal** (1.01× — statistical noise).

Investigation with `clang++ -S` showed why: Apple Clang with
`-O3 -mcpu=apple-m1` already auto-vectorizes the scalar inner loop to
`fmla.4s` instructions with 4× unrolled independent accumulators —
effectively writing our NEON kernel for us. We verified by inspecting
the emitted assembly.

The honest conclusion: **on Apple Silicon, hand-written NEON intrinsics
don't beat compiler auto-vectorization for simple FMA loops.** Our
NEON kernel is valuable as:
1. Documentation — it expresses the SIMD intent without relying on
   compiler heuristics staying helpful on future releases
2. A portability anchor — the same file can serve as a starting point
   for AVX2/AVX512/RVV variants where auto-vectorization is less
   reliable
3. A performance floor — if a future compiler update regresses the
   scalar path, our explicit NEON stays fast

But the headline number (1.17× over AMX at 99%) comes almost entirely
from skipping zero entries, not from vectorization cleverness. That's
the right story for our positioning: **sparsity is the moat, not SIMD**.

## What this means for the project narrative

We initially framed SpMM as "beating dense via aggressive SIMD." The
data says it's actually "beating dense via skipping zeros, at the
sparsity levels where DST research already lives (90-99%)." The
technical work is the same either way, but the story is more honest
and more defensible.

One silver lining for the positioning: **Apple AMX is an Apple-only
superpower.** On Intel/AMD/ARM-server CPUs (Graviton, Ampere),
`torch.matmul` falls back to OpenBLAS or MKL, which are *also* just
NEON or AVX FMA loops. Our crossover on those platforms will likely
appear at much lower sparsity (perhaps 75-85%), matching published
x86 SpMM benchmarks. Apple Silicon is actually our *worst* case for
this comparison, not our best. That's a strong talking point.

## What to try next

- Run with larger `N` (`N = 2048`) — wider activations amortize
  overhead better, crossover holds at ~99% on Apple Silicon
- Force scalar vs SIMD: `sparsecore.spmm(W, X, kernel="scalar"|"simd")`
- Inspect assembly: `clang++ -O3 -std=c++17 -mcpu=apple-m1 -S
  csrc/kernels/spmm.cpp -Icsrc | grep fmla`

## Commits

- `bed197b` — feat(spmm): scalar Y = W @ X for PaddedCSR
- (this milestone's commit) — feat(spmm): NEON SIMD + benchmark demo

## What's next (Milestone 4a)

Add autograd support: `torch.autograd.Function` so `sparsecore.spmm`
participates in backprop. Without this, users can only run forward
passes — no training. That's the next required piece.
