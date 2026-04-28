# Milestone 12 — NEON-accelerated dW kernel (v0.2.1)

**Closes issue #1. Shipped in v0.2.1.**

## What this milestone proves

The NEON SIMD port of `spmm_grad_w` — the sparse weight gradient
kernel — makes sparse-from-scratch training on Apple Silicon roughly
**1.4× faster end-to-end** through one of the kernel's hottest paths.
Per-layer the kernel itself is **~6× faster** than the scalar version
across every realistic FFN shape. No public API change; Apple Silicon
users get this automatically via the default `kernel="auto"` routing
in `SparseLinear`.

This was the highest-ROI v0.2 item by a wide margin:

- `dW` was 62% of a training step at 10M-param scale
  ([milestone 10](milestone_10.md)).
- The scalar kernel was running at ~14 GF/s — ~10× below the
  theoretical ceiling of NEON on M-series silicon. Clang-17 at
  `-O3 -mcpu=apple-m1` did not auto-vectorize our inner dot loop.

After this milestone, `dW` is no longer the largest cost in a sparse
backward pass. Future optimization work now targets forward SpMM,
`dX` via the transpose cache, and per-layer Python overhead.

## Headline numbers

### Per-layer dW throughput (M3 Pro, 6 threads, 30-run median)

| Shape                                  | Scalar ms | NEON ms | Speedup |
|----------------------------------------|-----------|---------|---------|
| demo15 FFN up   (384 × 1536, N=2048, s=0.90) | 15.55     | 2.38    | **6.53×** |
| demo15 FFN down (1536 × 384, N=2048, s=0.90) | 15.40     | 2.31    | **6.67×** |
| demo16 FFN up   (640 × 2560, N=1024, s=0.90) | 20.71     | 3.25    | **6.37×** |
| demo16 FFN down (2560 × 640, N=1024, s=0.90) | 21.42     | 3.40    | **6.30×** |
| tiny (64 × 64, N=128, s=0.80)                | 0.03      | 0.02    | 1.39×     |

All four FFN shapes exceed the ship-threshold of 3.0× from the spec;
tiny shape did not regress. The speedup is remarkably uniform across
FFN shapes — the inner dot loop's cost is now close to memory-bound
rather than compute-bound.

### End-to-end training step (3-layer sparse MLP @ 90% sparsity)

Identical model, identical inputs, identical seeds — only the
backward dW kernel differs:

| Backward kernel | Per-step wallclock |
|-----------------|--------------------|
| Scalar          | 2.46 ms            |
| NEON            | 1.79 ms            |
| **Speedup**     | **1.38×**          |

Why the end-to-end speedup is smaller than the per-layer speedup:
sparse backward is not all `dW`. A full training step also includes
the forward SpMM, the `dX = Wᵀ @ dY` step (which already uses NEON
via the transpose cache), Python overhead, loss, optimizer. In this
3-layer MLP those non-dW paths diluted the dW-only speedup from 6.5×
down to 1.4×. On larger models where dW is a bigger fraction of the
step (e.g., demo 16's 40M transformer where dW was 62% of the step),
the end-to-end speedup will be closer to 2×.

## What we measured — Gate 1 and Gate 2

The spec had two explicit decision gates (see
`.kiro/specs/dw-neon-kernel/design.md` §8):

**Gate 1 (pre-implementation).** Measure the scalar baseline to
confirm there was real SIMD headroom. Scalar sat at **~14 GF/s**
across all FFN shapes — deep in the scalar regime — proving Clang
hadn't already auto-vectorized. This resolved the single largest
risk in the spec (that hand-written NEON might tie or lose to the
compiler) and justified committing ~3 days to the implementation.

**Gate 2 (ship decision).** Required ≥ 3.0× local speedup on all
FFN shapes. Measured 6.3-6.7× — a clean margin above the threshold.

Both gates live in the reproducible profiler at
[`examples/profile_dw_baseline.py`](../../examples/profile_dw_baseline.py).

## The kernel — what it does

`csrc/kernels/spmm_grad_neon.cpp` mirrors the 8-wide
dual-accumulator pattern from `spmm_neon.cpp`:

- **Phase A**: main loop, 8 floats/iter using two independent 4-wide
  NEON FMA accumulators. This breaks the 4-cycle dependency chain of
  a single accumulator and lets M-series cores issue 2 FMAs per cycle.
- **Phase B**: trailing 4-wide iteration if 4–7 floats remain.
- **Phase C**: scalar cleanup for the final 0–3 floats.
- **Horizontal reduce**: `vaddq_f32(acc_a, acc_b)` → `vaddvq_f32` to
  collapse both accumulators' 8 lanes into one scalar per live slot.

The inner loop is inlined (not a call to `vector_dot_simd_neon`)
because at 40M scale that would be ~6.5M function calls per backward
pass. OpenMP parallelizes over the outer row dimension with the same
`SCORE_PARALLEL_ROW_THRESHOLD` gate as the other kernels.

## Dispatch — no public API change

On ARM64, a new C++ symbol `spmm_grad_w_simd` is exposed through
pybind11 as `_core.spmm_grad_w_simd`. On x86 the binding routes to
the scalar kernel — the public API stays identical.

The autograd path in `sparselab/ops.py`'s `_SpMMFunction.backward`
chooses between scalar and NEON based on the `kernel` argument
stashed in forward:

```python
grad_w_fn = (
    _core.spmm_grad_w_simd
    if ctx.kernel in ("auto", "simd")
    else _core.spmm_grad_w
)
```

Since `kernel="auto"` is the default everywhere (`SparseLinear`
forward, `sparselab.spmm`), Apple Silicon users pick up the speedup
with no code change.

## Correctness

This milestone added 56 new tests; full suite is now **442 passed,
2 skipped**.

- **Oracle parametrization** over scalar + simd covers all 23 existing
  shape / padding / edge-case tests in `test_spmm_grad.py`.
- **NEON-specific tests** (`test_spmm_grad_neon.py`, 41 new cases):
  20 random-shape agreement checks, 18 N-residues exercising every
  Phase A/B/C transition, interleaved empty rows (OpenMP static
  schedule), single-slot-per-row with tiny N, determinism across
  repeated calls (bit-identical, not approximate).
- **Autograd `gradcheck`** (`test_spmm_autograd.py`): finite-diff
  gradient verification on both scalar and simd dispatch paths,
  so a silent default-kernel change can't regress the NEON path.

Numerical note: NEON's dual-accumulator reorders the dot-product
summation, so per-slot results may differ from scalar by 1–2 ULPs
over N ~ 1000–2000. This is float32 non-associativity, not a bug.
All tests pass at `rtol=atol=1e-5`, the standard we apply to
`spmm_simd` versus `spmm_scalar`.

## What this demo does not claim

- ❌ **Single-digit-ms inference on huge models.** We still do one
  kernel call per SparseLinear per forward or backward; Python and
  pybind11 overhead is ~0.5–1 ms per layer and dominates for tiny
  layers. Buffer reuse / arena allocation (v0.2 issue #7) is the
  next target for that.
- ❌ **Beats cuBLAS/cuSPARSE.** This is a CPU-only kernel for Apple
  Silicon. GPU ports are a v0.3 scope (issue #3).
- ❌ **All backward paths got 6× faster.** This is a `dW`-only
  speedup. `dX` already used NEON via the transpose cache; forward
  SpMM is already NEON.
- ❌ **Works on Intel Macs or Linux x86.** The NEON binding routes
  to the scalar kernel on x86 so nothing breaks, but there's no
  AVX-512 port yet (v0.2 issue #2).

## Reproduce

```bash
# The headline demo:
python examples/demo_17_dw_neon.py

# Raw per-layer numbers (what Gate 1 and Gate 2 used):
python examples/profile_dw_baseline.py

# The full test suite:
pytest tests/ -q
```

Expected runtime:
- `demo_17_dw_neon.py`: ~30 seconds
- `profile_dw_baseline.py`: ~30 seconds
- Full test suite: ~4 seconds

## Files changed

- **New:** `csrc/kernels/spmm_grad_neon.{hpp,cpp}`
- **New:** `tests/test_spmm_grad_neon.py`
- **New:** `examples/demo_17_dw_neon.py`
- **Modified:** `csrc/bindings.cpp` (new `spmm_grad_w_simd` binding,
  refactored prepare-validate helper)
- **Modified:** `setup.py` (compile `spmm_grad_neon.cpp` when
  `IS_ARM64`)
- **Modified:** `sparselab/ops.py` (autograd dispatch)
- **Modified:** `tests/test_spmm_grad.py` (kernel_fn parametrization)
- **Modified:** `tests/test_spmm_autograd.py` (per-kernel gradcheck)
- **Modified:** `examples/profile_dw_baseline.py` (NEON column)

## What we'll do next (v0.2 priorities this unblocks)

1. **AVX-512 port (issue #2).** The dispatch surface is now in place —
   AVX is a pure additive change on x86. With dW now at 6.5× on ARM,
   AVX parity becomes the gating item for Linux x86 training speed.
2. **Buffer reuse / arena (issue #7).** Now that dW is fast, the
   per-backward allocator churn — we create two fresh numpy arrays
   per layer — is a visible fraction of the remaining time. An arena
   saves 5-15% of the step.
3. **100M+ scale validation (issue #10).** Sparse-all at 40M was
   4.1× slower than dense in milestone 11. With this milestone that
   narrows to roughly 2.0× — close enough that a community-
   contributed 100M+ run becomes newsworthy on its own.

---

_Written: 2026-04-27._
_Shipped in v0.2.1._
