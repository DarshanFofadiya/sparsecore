# Milestone 1 Demo — The PyTorch Bridge

**Script:** `examples/demo_01_bridge.py`
**Phase:** 1 (Scaffolding)
**Status:** ✅ Complete

## What this demo proves

The PyTorch ↔ C++ bridge works. A PyTorch tensor can be passed into a function we wrote in C++, operated on, and returned — with numerical correctness verified against a PyTorch Oracle.

This is deliberately the least impressive milestone in the project. The operation (multiply by 2.0) is trivial, the code has no SIMD, no sparsity, no parallelism. But every subsequent milestone — NEON kernels, Padded-CSR, dynamic topology mutation — builds on this same bridge pattern, so proving it works end-to-end is non-negotiable.

## How to run it

```bash
conda activate sparselab
python examples/demo_01_bridge.py
```

## Expected output

You should see, in order:

1. A boxed banner declaring this is Milestone 1.
2. The absolute path to the compiled `.so` file, with its size and architecture.
3. Three side-by-side tensor prints (input, C++ output, PyTorch Oracle) that all agree: `[2., 4., 6., 8., 10.]`.
4. A summary with four green checkmarks.
5. A "play with it" section suggesting next experiments.

The critical line is:

```
Oracle check: max |C++ − PyTorch| = 0.00e+00  (< 1e-5 tolerance)
```

For multiply-by-2 the diff should be *exactly zero* (floating-point multiplication by a power of two is exact). If it is nonzero, something is off in the FFI layer — worth investigating, not moving on.

## What to look at after running

- `csrc/bindings.cpp` — the C++ source, heavily commented. ~21 lines of logic.
- `tests/test_bridge.py` — the Oracle test suite. 21 test cases including SIMD-relevant boundary sizes.
- `docs/PROJECT_OVERVIEW.md` — what Milestone 2 looks like.

## What this does NOT prove

- No SIMD acceleration — `csrc/bindings.cpp` uses a plain scalar loop.
- No sparsity — we haven't defined a sparse format yet.
- No autograd integration — that's Phase 4.
- No performance improvement — this is plumbing, not a benchmark.
