# Design — NEON SIMD Kernel for Sparse Weight Gradient (`spmm_grad_w`)

Companion to [`spmm_backward.md`](spmm_backward.md). That doc covers
the math and the scalar kernel; this one covers the NEON-accelerated
variant (`spmm_grad_w_simd`) shipped in v0.2.1, closes [issue #1].

[issue #1]: https://github.com/DarshanFofadiya/sparselab/issues/1

---

## 1. The problem in one paragraph

The backward pass for `Y = W @ X` at sparse `W` requires one N-length dot
product per live slot: `dW_values[s] = dY[i, :] · X[k, :]`. At 10M-param
scale `dW` was 62% of a training step
([milestone 10](../demos/milestone_10.md)); at 40M scale the
sparse/dense slowdown ratio narrowed from 4.6× to 4.1× — direct evidence
that `dW` dominates and grows linearly with model size. The previous
scalar kernel **measured at ~14 GF/s** on M3 Pro across demo_15 and
demo_16 FFN shapes (§6.0) — Apple Silicon's f32 peak is 150-200 GF/s
per core, so there was ~10× ceiling above the scalar. We ship a
hand-written NEON version that mirrors the 8-wide dual-accumulator
pattern from `spmm_neon.cpp` and routes through runtime dispatch with
a scalar fallback on x86.

## 2. What ships

### 2.1 New kernel

`csrc/kernels/spmm_grad_neon.cpp` + `.hpp` — NEON implementation,
compile-gated on `__ARM_NEON` in `setup.py` so x86 builds skip the
source entirely. The Python binding `_core.spmm_grad_w_simd` routes to
the scalar kernel on non-ARM hardware so the public API surface stays
portable.

### 2.2 Dispatch

The autograd path in `sparselab/ops.py`'s `_SpMMFunction.backward`
picks the kernel based on the argument stashed during forward:

```python
grad_w_fn = (
    _core.spmm_grad_w_simd
    if ctx.kernel in ("auto", "simd")
    else _core.spmm_grad_w
)
dW_np = grad_w_fn(W, dY_f32.numpy(), X_f32.numpy())
```

Since `kernel="auto"` is the default everywhere — `SparseLinear`,
`sparselab.spmm` — Apple Silicon users see the speedup without
changing any code.

### 2.3 Unchanged contracts

- Public Python API: **no change.**
- Autograd contract: `backward` still returns `(dW_values, None, dX, None)`.
- `PaddedCSR` memory layout: unchanged.
- Scalar kernel `spmm_grad_w` kept as reference + x86/no-NEON fallback.

## 3. Algorithm

### 3.1 Shape

Same triple-nested structure as the forward NEON kernel, with one
critical difference in the inner math. Forward is **scatter-to-row**
(each live slot contributes to a whole row of `Y`); dW is
**reduce-to-slot** (each live slot produces one scalar via a dot
product over `N`).

```
for each row i in [0, M):                       // outer — parallelizable
  for each live slot s in row i:                // walk row_nnz[i] live entries
    c = col_indices[row_start[i] + s]
    acc = dot_N(dY[i, :], X[c, :])              // inner — SIMD target
    dW_values[row_start[i] + s] = acc
```

The inner `dot_N` is the hot loop — same shape as `vector_dot_neon.cpp`
but called `nnz` times instead of once. Per-call dispatch overhead
therefore matters (see §4.2).

### 3.2 SIMD strategy (mirrors `spmm_neon.cpp`)

Two independent 4-wide accumulators, loop unrolled 2×. Per inner
iteration:

```
  vld1q_f32(dy + j)       // load  lane A
  vld1q_f32(dy + j + 4)   // load  lane B
  vld1q_f32(x  + j)       // load  lane A
  vld1q_f32(x  + j + 4)   // load  lane B
  vfmaq_f32(acc_a, ...)   // FMA   lane A — independent
  vfmaq_f32(acc_b, ...)   // FMA   lane B — independent
```

- **Phase A** — 8 floats/iter main loop.
- **Phase B** — trailing 4-wide iter if 4–7 floats remain.
- **Phase C** — scalar cleanup for 1–3 residue floats.
- **Reduce** — `vaddvq_f32(vaddq_f32(acc_a, acc_b))` → one scalar per
  live slot.

### 3.3 Why 8-wide dual-accumulator (not plain 4-wide)

1. Single 4-wide loop produces 1 FMA/iter with a **4-cycle dependency
   chain** on the same accumulator — CPU can't issue the next FMA
   until the prior retires.
2. Two independent accumulators let the out-of-order scheduler dispatch
   2 FMAs per cycle (M-series can issue 2 FMAs/cycle).
3. We verified on the forward path (`spmm_neon`) that naive 4-wide
   NEON actually **loses** to Clang auto-vectorized scalar. Only the
   unrolled version beats the compiler.

### 3.4 Parallelism

Same row-level OpenMP as `spmm_neon.cpp`:

```cpp
#pragma omp parallel for schedule(static) if(M >= SCORE_PARALLEL_ROW_THRESHOLD)
```

Race-freedom: each row `i` writes only to
`dW_values[row_start[i] : row_start[i] + row_capacity[i]]`. PaddedCSR's
invariants guarantee these slices don't overlap across `i`. `dY`, `X`,
and `W`'s index arrays are read-only.

### 3.5 Self-zeroing contract

`memset(dW_values, 0, total_capacity * sizeof(float))` at entry.
Padding slots stay at 0.0 so optimizers can do
`W.values -= lr * dW` as a single vectorized op without a mask pass.

## 4. Decisions that matter

### 4.1 Loop order: (i, s, j) not (i, j, s)

`(i, s, j)` keeps `dY[i, :]` in L1 across all `j` in the dot product,
and `X[c, :]` is read as a single streaming scan per slot. Swapping
to `(i, j, s)` would re-scan `dY[i, :]` for every slot — cache thrash.

### 4.2 Inline the dot loop — don't call `vector_dot_simd_neon`

Sparse FFN at 40M scale has `nnz ≈ 6.5M` — that's 6.5M function calls
per backward pass if we reused `vector_dot_simd_neon`. Even at
~0.3 ns/call that's ~2 ms just in call overhead. Inlining saves the
overhead and lets the compiler keep `acc_a`/`acc_b` in registers
across phases. Code duplication (the dot loop appears in both
`vector_dot_neon.cpp` and here) is accepted — both kernels carry
teach-the-reader comments.

### 4.3 Dispatch surface

- C++: `sparselab::spmm_grad_w_simd(...)` with same signature as scalar.
- Python: `_core.spmm_grad_w_simd`; scalar kept as `_core.spmm_grad_w`.
- x86 builds: `_simd` binding routes to scalar. Same pattern as
  `py_spmm_simd` and `py_vector_dot_simd`.

## 5. Testing strategy

### 5.1 Oracle parametrization

All tests in `tests/test_spmm_grad.py` are parametrized over
`[scalar, simd]` via a `kernel_fn` fixture. 15 oracle tests × 2 kernels
= 46 cases running against real shapes up to 64×64, all at
`rtol=atol=1e-5` against a dense-PyTorch oracle.

### 5.2 NEON-specific tests

`tests/test_spmm_grad_neon.py` (41 cases) adds:

- **Scalar/NEON bit-tolerance agreement.** 20 random shapes, varied
  sparsity, `np.allclose(scalar, neon, rtol=atol=1e-5)`.
- **N-residue coverage.** N ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17,
  31, 32, 33, 63, 64, 65} — every `N % 8` residue and every
  Phase A→B→C transition.
- **Empty-row interleaving.** W with empty rows between populated rows
  — stresses OpenMP's static schedule under unbalanced per-thread
  work.
- **Single-slot-per-row with tiny N.** Every row has `nnz=1` and
  `N=3` — exercises the scalar-only Phase-C cleanup path.
- **Determinism.** Fixed seed, 10 repeated calls,
  `np.testing.assert_array_equal` — OpenMP static schedule must be
  bit-stable across invocations.

### 5.3 Autograd integration

`tests/test_spmm_autograd.py` adds a parametrized
`test_gradcheck_dW_values_per_kernel[scalar|simd]` that runs
`torch.autograd.gradcheck` against each dispatch path. Finite-differences
agreement on both kernels guards the Phase D dispatch from silent
regressions.

## 6. Performance — measured numbers

### 6.0 Gate 1 — scalar baseline (pre-implementation)

Run before committing any NEON code, via
`examples/profile_dw_baseline.py` on M3 Pro (6 threads, 30-run median):

| Shape | Scalar ms | Dense ms | Ratio | Scalar GF/s |
|---|---|---|---|---|
| demo15 FFN up (384 × 1536, N=2048, s=0.90) | 16.62 | 2.66 | 6.24× | 14.4 |
| demo15 FFN down (1536 × 384, N=2048, s=0.90) | 16.20 | 2.74 | 5.91× | 14.8 |
| demo16 FFN up (640 × 2560, N=1024, s=0.90) | 23.89 | 3.95 | 6.05× | 14.0 |
| demo16 FFN down (2560 × 640, N=1024, s=0.90) | 22.60 | 3.80 | 5.95× | 14.8 |
| tiny (64 × 64, N=128, s=0.80) | 0.03 | 0.00 | 5.53× | 8.3 |

**Three takeaways from this table:**

1. **Scalar is NOT auto-vectorized well.** ~14 GF/s across every
   realistic shape is consistent with Clang emitting sequential scalar
   FMAs with no unrolling (single FMA/cycle latency ≈ 14 GF/s on
   M-series). Plain `vfmaq_f32` at 1 op/cycle with 4 lanes = ~56 GF/s;
   an 8-wide dual-accumulator target is ~90–120 GF/s; f32 theoretical
   peak per core is ~150-200 GF/s. We were ~10× below ceiling.

2. **6× slower than dense BLAS despite doing 10× less arithmetic.**
   At 90% sparsity our kernel does (1-s) × dense_FLOPs = 10% of dense's
   arithmetic — should be ~2-3× faster than dense, not 6× slower. The
   net gap (~12-18× headroom) is what NEON unlocks.

3. **Absolute per-layer time is large.** 16-24 ms per FFN dW call.
   At 40M scale with 8 layers × 2 FFN projections × dW ≈ 320-400 ms
   of dW per backward pass, matching milestone 10's "dW is 62% of a
   training step" finding.

### 6.1 Gate 2 — post-NEON measurement

After implementation, same machine, same shapes, same methodology:

| Shape | Scalar ms | NEON ms | Speedup |
|---|---|---|---|
| demo15 FFN up (384 × 1536, N=2048, s=0.90) | 15.55 | 2.38 | **6.53×** |
| demo15 FFN down (1536 × 384, N=2048, s=0.90) | 15.40 | 2.31 | **6.67×** |
| demo16 FFN up (640 × 2560, N=1024, s=0.90) | 20.71 | 3.25 | **6.37×** |
| demo16 FFN down (2560 × 640, N=1024, s=0.90) | 21.42 | 3.40 | **6.30×** |
| tiny (64 × 64, N=128, s=0.80) | 0.03 | 0.02 | 1.39× |

End-to-end on a 3-layer sparse MLP training step: **1.38×** faster
(scalar 2.46 ms → NEON 1.79 ms). The smaller end-to-end ratio comes
from the forward SpMM, `dX = Wᵀ @ dY` transpose path, Python
overhead, and optimizer step — all of which are not `dW`.

### 6.2 Precedent from `spmm_neon`

The forward `spmm_simd` NEON version measured ~1.37-1.40× speedup
vs scalar on comparable shapes (milestone 6). The `dW` speedup is
larger because forward's scalar kernel benefited from partial
auto-vectorization (outer scatter-to-row keeps the inner store simple
for Clang), while dW's outer `dW_values[row_ptr + s] = acc` structure
with runtime-varying slot indices defeated the vectorizer's analysis.
Larger baseline gap → larger NEON-vs-scalar ratio.

## 7. Numerical notes

NEON's dual-accumulator changes the addition order of the
dot-product terms vs the scalar single-accumulator version. Float
addition is non-associative → per-slot outputs differ from scalar by
1–2 ULPs on N ≥ 1000. This is the same looseness already tolerated in
`spmm_simd` vs `spmm_scalar`; tests use `rtol=atol=1e-5`.

## 8. What this does not do

- **No AVX-512 port.** That's [issue #2]. This spec establishes the
  dispatch surface; AVX becomes a pure additive change on x86.
- **No backward NEON for `dX`.** `dX = Wᵀ @ dY` already uses
  `spmm_simd` (NEON on the forward kernel) via the transpose cache.
- **No buffer reuse / arena.** That's [issue #7].
- **No tolerance tightening.** Training-grade numerics don't need
  tighter than 1e-5 and our accumulator reorder wouldn't fit.
- **No public API change.** Users call `.backward()` the same way.

[issue #2]: https://github.com/DarshanFofadiya/sparselab/issues/2
[issue #7]: https://github.com/DarshanFofadiya/sparselab/issues/7

## 9. Appendix — Borrow-Don't-Reinvent

**Scalar pattern mirrored:** `csrc/kernels/spmm_grad.cpp`. Already
parallelized by row; its comment explicitly anticipated this NEON port
("A NEON variant [...] would mechanically drop in as a future
optimization").

**SIMD pattern mirrored:** `csrc/kernels/spmm_neon.cpp`'s 8-wide
dual-accumulator FMA with Phase A/B/C. Verbatim inner-loop structure.

**Dispatch pattern mirrored:** `csrc/bindings.cpp`'s `py_spmm_simd`
under `#if defined(__ARM_NEON)` with scalar fallback.

**No external reference for the dW pattern specifically.** cuSPARSE
doesn't expose a `spmm_grad` public API (it's internal to their
autograd); MKL doesn't do sparse training at all. Our scalar kernel
and its NEON port are both new, but they follow off-the-shelf SIMD
patterns.

---

_Shipped in v0.2.1 — see [milestone 12](../demos/milestone_12.md)
for measured numbers and honest framing._
