# Milestone 2 Demo — Vector Dot Product on NEON (in progress)

**Script:** `examples/demo_02_dot.py` *(written in sub-milestone 2c)*
**Phase:** 2 (Dense SIMD warmup)
**Status:** 🚧 In progress

## What this milestone will prove

The NEON 128-bit SIMD path works end-to-end on Apple Silicon. A vector dot product written by hand with `arm_neon.h` intrinsics:
- produces numerically identical results to the scalar version (within `rtol=atol=1e-5`)
- runs measurably faster than the scalar version on our M3 Pro
- stays within a sensible factor of `torch.dot` (Apple's Accelerate framework)

This validates that our SIMD plumbing, compiler flags, and lane arithmetic are all correct — which is the prerequisite for writing the sparse SpMM kernel in Phase 3.

## Sub-milestone progress

- [x] **2a-i** — Refactor `bindings.cpp` into `kernels/*.cpp` + bindings split
- [x] **2a-ii** — Scalar `vector_dot` kernel implementation
- [x] **2a-iii** — Python wrapper + registration with shape validation
- [x] **2a-iv** — Oracle test suite (23 tests, full coverage of SIMD boundary sizes)
- [x] **2a-v** — Wrap-up and commit (this sub-milestone)
- [ ] **2b**   — NEON SIMD rewrite of `vector_dot`, Oracle-verified identically
- [ ] **2c**   — Benchmark harness + demo (`demo_02_dot.py`) + this doc fills in

## Expected output at Milestone 2 completion

When 2c ships, running `python examples/demo_02_dot.py` should produce something like:

```
┌─────────────────────────────────────────────────────────────────┐
│  SparseCore — Milestone 2: NEON SIMD Vector Dot                 │
├─────────────────────────────────────────────────────────────────┤
│  Size       scalar (ms)    NEON (ms)    torch.dot (ms)   speedup│
│  ─────      ───────────    ──────────   ──────────────   ───────│
│     1024       0.0042        0.0012        0.0008          3.5x │
│    16384       0.068         0.022         0.014           3.1x │
│   131072       0.53          0.18          0.11            2.9x │
│                                                                 │
│  ✓ All sizes Oracle-verified: |NEON - torch.dot| < 1e-5         │
└─────────────────────────────────────────────────────────────────┘
```

Exact numbers TBD once we benchmark. What matters: NEON is measurably faster than scalar, and everything is Oracle-correct.

## What 2a already delivered (before the demo)

At end of 2a, even without a runnable demo, these hard things are in place:
- A clean `kernels/` + `bindings.cpp` split — every future C++ kernel slots in naturally
- 23 Oracle-verified test cases specifically designed to trip NEON remainder-loop bugs
- Dtype coercion, shape validation, and error paths all tested
- `vector_dot` is callable from Python *right now* — just not SIMD-accelerated yet

Run them:
```bash
pytest tests/test_vector_dot.py -v
```
