# Changelog

All notable changes to SparseLab are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

**AVX2 SIMD kernel for `dW` on Linux x86_64 — closes the Linux-parity
gap from milestone 13. Sparse-from-scratch training on x86 is now
**~3× faster end-to-end** at 40M-param transformer scale (per-step
wallclock 4316 → 1436 ms on a free GitHub Actions Zen 4 runner). Per
FFN layer the dW kernel itself is 12-13× faster than scalar. Training
dynamics unchanged — same seed produces identical val loss to four
decimal places.**

Closes #2.

### Added
- AVX2 + FMA implementation of `spmm_grad_w` for x86_64 builds.
  Mirrors the dual-accumulator pattern from `spmm_grad_neon.cpp`,
  adapted to 256-bit lanes (16 floats/iter Phase A, 8 floats/iter
  Phase B, scalar 0-7 Phase C, 3-step horizontal reduction). Gate A0
  microbench on the CI runner confirmed dual-accumulator beats
  single-accumulator by 2.03× on Zen 4 before the kernel was written.
  Per-layer measured speedup: 12.7× / 12.7× / 11.8× / 12.9× on the
  four canonical FFN shapes from demo_15 / demo_16, delivering ~47-51
  GF/s sustained — within the design's 40-45 GF/s target band.
- `csrc/kernels/spmm_grad_avx2.{hpp,cpp}` — new AVX2 kernel, compile-
  gated on `__AVX2__ && __FMA__`.
- `csrc/bench/avx2_dot_microbench.cpp` — standalone Gate A0
  microbenchmark for single-vs-dual accumulator validation.
- `.github/workflows/validate_avx2_microbench.yml` — runs Gate A0
  microbench on the CI runner (~9 seconds, manual dispatch).
- `.github/workflows/validate_40m_scalar.yml` — runs the end-to-end
  40M-transformer before/after comparison (~30 minutes, manual
  dispatch). Used to capture milestone 14's headline 3× ratio.
- `examples/validate_40m_dw.py` — the script that workflow drives.
  Monkey-patches `_SpMMFunction.forward` to force the kernel choice
  on a full 200-step training run, then reports the scalar-vs-simd
  wallclock ratio and val-loss delta.
- `tests/test_spmm_grad_avx2.py` — 44 AVX2-specific tests targeting
  every Phase A/B/C boundary (N mod 16 residues 1-65), random-shape
  agreement with scalar, empty-row interleaving, single-slot-per-row,
  determinism under OpenMP. Skipped on non-x86_64 platforms via
  `pytest.mark.skipif`.
- `docs/design/spmm_backward_avx2.md` — committable design doc for
  the AVX2 kernel, including the Gate A0 measured numbers that
  revised the spec from single-accumulator to dual-accumulator.
- `docs/demos/milestone_14.md` — measured 3.0× end-to-end speedup,
  12-13× per-layer, val-loss delta = 0.0000 nats across 200 SGD
  steps.

### Changed
- `setup.py` on x86_64 builds now compiles with `-march=x86-64-v3`.
  That target implies AVX2 + FMA + BMI2 and has been the Linux
  Foundation baseline for "modern x86" since ~2021. Standalone (pre
  the new kernel) this flag alone raised scalar dW from ~3.8 GF/s
  to ~4.3 GF/s on Zen 4. Full detail in milestone 14's Gate 1.5
  section.
- `csrc/bindings.cpp` dispatch for `spmm_grad_w_simd` gained one
  new `#elif defined(__AVX2__) && defined(__FMA__)` branch to route
  to the AVX2 kernel on x86. The Python-facing symbol
  `_core.spmm_grad_w_simd` is unchanged; on ARM64 it still routes
  to NEON, on pre-AVX2 x86 to scalar.
- Test suite: 442 → **486 passed** on Linux x86_64 (44 new AVX2
  tests). On ARM64 stays at 442 passed + 44 skipped (AVX2 tests
  skip cleanly).

### Breaking
- **Minimum x86 CPU requirement: Haswell (Intel, 2013+) or Zen 1
  (AMD, 2017+).** The `-march=x86-64-v3` flag emits AVX2 + FMA
  instructions; pre-2013 x86 CPUs (Nehalem, Sandy Bridge, Ivy
  Bridge, Bulldozer) will hit `Illegal instruction` at import. Every
  Linux distribution currently supported by PyTorch 2.8+ targets
  Haswell+ / Zen+ CPUs in practice. Users on older hardware can
  stay on v0.2.1 or build from source with custom flags.

### Internal
- Platform scope reconfirmed: Apple Silicon macOS, Linux aarch64,
  Linux x86_64. Intel Mac and Windows remain out of scope. See
  v0.1.1 for the upstream-PyTorch-deprecation context on Intel Mac.
- CI Test workflow (`.github/workflows/test.yml`) now exercises
  the AVX2 kernel on every push. Linux aarch64's NEON numbers
  verified unchanged vs milestone 13 (si/sc within 5%).

## [0.2.1] — 2026-04-27

**NEON SIMD kernel for `dW` (sparse weight gradient) — the single
largest cost of sparse-from-scratch training on Apple Silicon drops
by ~6.5× per layer. End-to-end training at 40M-param transformer
scale is ~2× faster (measured 1.96×), narrowing sparse-all's slowdown
vs dense from 4.1× to ~2.4×. Training dynamics unchanged — same seed
produces identical val loss.**

Closes #1.

### Added
- NEON SIMD implementation of `spmm_grad_w` (the sparse weight
  gradient kernel). Mirrors the 8-wide dual-accumulator pattern from
  `spmm_neon.cpp`. On M-series silicon all four tested FFN shapes hit
  6.3-6.7× speedup vs the scalar kernel. End-to-end training step
  speedup measured at **1.96× on demo_16's 40M-param transformer**
  (sparse-all path, 200 steps, same seed, identical final val loss).
- `examples/demo_17_dw_neon.py` — user-facing demo with per-layer
  and end-to-end speedup tables.
- `examples/profile_dw_baseline.py` — reproducible benchmark for
  dW kernel throughput (scalar vs NEON vs dense-BLAS oracle).
- `docs/demos/milestone_12.md` — measured numbers and honest
  limitations.
- `tests/test_spmm_grad_neon.py` — 41 NEON-specific tests covering
  every inner-loop phase boundary (N residues 1-65), random-shape
  agreement with scalar, empty-row interleaving, single-slot-per-row,
  determinism under OpenMP.

### Fixed
- `SparseLinear` init: Kaiming-uniform bound is now computed against
  `effective_fan_in = in_features * (1 - sparsity)` instead of the
  dense fan-in. The previous dense bound under-scaled live weights by
  `sqrt(1 - sparsity)` per layer, causing signal collapse in stacked
  sparse MLPs. Matches Cerebras's "sparsity-compensated init". Safe
  at `sparsity=0` (reduces to dense bound). Surfaced while debugging
  demo 18's MNIST stack.

### Internal
- `csrc/kernels/spmm_grad_neon.{hpp,cpp}` — new NEON kernel, gated
  on `__ARM_NEON` with scalar fallback on x86.
- `csrc/bindings.cpp` — new `spmm_grad_w_simd` Python symbol; shared
  prepare-validate helper across scalar/NEON bindings.
- `sparselab/ops.py` — `_SpMMFunction.backward` now dispatches dW to
  `spmm_grad_w_simd` when `ctx.kernel in {"auto", "simd"}`. Public
  API unchanged — `SparseLinear(kernel="auto")` is still the default.
- `tests/test_spmm_grad.py` — all 15 oracle tests parametrized over
  both kernels via a `kernel_fn` fixture (46 test cases).
- `tests/test_spmm_autograd.py` — new `gradcheck` case explicitly
  parametrized over scalar + simd dispatch.
- Full test suite: **442 passed, 2 skipped** (was 376 pre-milestone).

### Research artifacts (not launch demos)
- `examples/demo_18_global_skip_mnist.py` — 4-model MNIST MLP
  comparison of sparse-sequential vs sparse-global-skip at matched
  live-param budget. Null result: global-skip did not beat sparse-
  sequential on this workload.
- `examples/demo_20_global_skip_transformer.py` — transformer FFN
  global-skip at demo 16's 40M-param shape. Three near-bias settings
  (uniform, stratified 0.5, stratified 0.8) all within 0.003 nats of
  each other at 1000 steps — connection distribution pattern does
  not meaningfully affect outcome at this scale.

## [0.2.0] — 2026-04-23

**Renamed from `sparsecore` to `sparselab`.** No functional code changes.

The original name collided with Google's TPU SparseCore hardware block
(documented since 2020 across OpenXLA, Keras, and Google Cloud). Sharing
a name with a well-established Google product is bad ergonomics for a
library that aims to become the canonical community platform for
dynamic sparse training — every mention would require a disambiguation
paragraph, and search rankings would perpetually sit in Google's orbit.
Renaming now, with zero external adopters, is cheap; renaming later
would get progressively more expensive.

### Changed
- Package name: `sparsecore` → `sparselab`
- Import statement: `import sparselab` (was `import sparsecore`)
- GitHub repo: `DarshanFofadiya/sparsecore` → `DarshanFofadiya/sparselab`
  (old URLs auto-redirect via GitHub)
- PyPI project: new project `sparselab` on pypi.org. The old
  `sparsecore` project on PyPI stays live for pinned installs; one
  final `sparsecore` `0.1.2` release raises `ImportError` pointing at
  the new name.
- Environment variables (advanced opt-outs in setup.py):
  `SPARSECORE_NO_OPENMP` → `SPARSELAB_NO_OPENMP`,
  `SPARSECORE_LIBOMP_PREFIX` → `SPARSELAB_LIBOMP_PREFIX`

### Fixed
- Editable installs (`pip install -e .`) on macOS no longer abort with
  `OMP: Error #15` when importing. The C++ extension's libomp
  install name is now rewritten post-build (same approach
  `scripts/repair_wheel_macos.sh` uses for published wheels) via a
  `BuildExtWithRepair` class in `setup.py`. Two libomps in a process
  used to abort OpenMP's runtime; now only one is loaded.

### Migration
- `pip install sparselab` and `import sparselab` everywhere you had
  `sparsecore`. All public API names (`PaddedCSR`, `spmm`,
  `SparseLinear`, `SparsityAlgorithm`, `SET`, `RigL`, `Static`,
  `DynamicSparsityAlgorithm`) are unchanged.
- Pinned to `sparsecore==0.1.1`? Your install keeps working. Future
  development happens in `sparselab`.

## [0.1.1] — 2026-04-23

Maintenance release — notebook fix, documentation improvements, and
a clearer story on Intel Mac support.

### Fixed
- Colab notebook (`examples/colab_try_sparselab.ipynb`): the
  `KeepTopK` custom-algorithm example raised
  `ValueError: drop_fraction must be in (0.0, 1.0], got 0.0` because
  it subclassed `DynamicSparsityAlgorithm` (which is designed for
  drop+regrow DST methods like SET/RigL) and passed
  `drop_fraction=0.0`. It now subclasses `SparsityAlgorithm` directly,
  which is the right base for pruner-only algorithms that don't
  regrow. The example docstring explains the distinction.
- Colab notebook: the toy-regression training loop ran for only 200
  steps, which didn't show meaningful convergence. Bumped to 2000.

### Added
- "Open in Colab" badge in both the README badge row and the notebook's
  first markdown cell, so new users have a one-click path from the
  project page to a runnable environment.

### Investigated but not shipped
- **Intel Mac (macOS x86_64) wheels.** We added the new
  `macos-15-intel` GitHub Actions runner to the cibuildwheel matrix
  and confirmed the wheel builds cleanly. The smoke test then fails
  because `pip install` cannot resolve `torch>=2.8` on that platform:
  [upstream PyTorch deprecated macOS x86_64 wheels in January 2024](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)
  and the last torch macOS x86_64 wheel published is 2.2.2. Shipping
  an Intel Mac sparselab wheel would therefore be unusable — the
  dependency cannot be installed from PyPI. Intel Mac users who need
  sparselab can still build from sdist with `torch<=2.2.2` pinned,
  but we don't ship a pre-built wheel for this platform. See the
  workflow header comment in `.github/workflows/wheels.yml` for the
  full reasoning. This replaces the vaguer "Intel runner retired"
  note in the v0.1.0 changelog, which is now outdated — the runner
  exists; it's the upstream torch wheel that doesn't.

## [0.1.0] — 2026-04-22 (first public release)

First public release. The pluggable DST foundation.

### Added
- `sparselab.PaddedCSR` — sparse matrix storage with O(1) slot
  insert, cached transpose, round-trip with `torch.sparse_csr`.
  Eight structural invariants checked by the C++ constructor.
- `sparselab.spmm(W, X)` — sparse-dense matmul, autograd-aware. NEON
  path on ARM64, scalar path on x86. OpenMP parallelized across the
  outer row loop.
- `sparselab.SparseLinear(nn.Module)` — drop-in `nn.Linear`
  replacement. Standard `nn.Parameter`, standard `state_dict`,
  standard `torch.optim` compatibility.
- `sparselab.SparsityAlgorithm` — pluggable DST base class. Inspired
  by Cerebras's `cstorch.sparse.SparsityAlgorithm`.
- `sparselab.Static` — no-op reference sparsity algorithm.
- `sparselab.SET` — Sparse Evolutionary Training (Mocanu et al.,
  2018) with magnitude-based drop and random regrow.
- `sparselab.RigL` — Rigging the Lottery (Evci et al., 2020) with
  gradient-based regrow.
- 372 tests including gradcheck against PyTorch autograd and
  dense-equivalence oracles at 1e-5 tolerance.
- 15 example demos, end-to-end from "hello pybind" to "10M-param
  mini-GPT trained on Tiny Shakespeare at 90% FFN sparsity."
- Pre-built PyPI wheels for macOS arm64 (Apple Silicon), Linux x86_64,
  and Linux aarch64 across Python 3.11, 3.12, 3.13. libomp bundled
  inside the wheels — no `brew install libomp` needed for
  `pip install` users.
- Colab notebook (`examples/colab_try_sparselab.ipynb`) for zero-
  setup exploration.
- Docker-based fresh-install test (`scripts/test_fresh_install.sh`)
  and SageMaker recipe (`scripts/test_on_sagemaker.md`).

### Known limitations
- CPU-only. No GPU backend.
- Single-machine. No distributed / DDP training.
- Native Windows wheels not available (use WSL2 with the Linux wheel
  in the meantime). Planned v0.2.
- Intel Mac wheels not available — upstream PyTorch deprecated macOS
  x86_64 wheels in January 2024. See the 0.1.1 "Investigated but not
  shipped" note above for the full story.
- The `dW` kernel relies on Clang auto-vectorization, not hand-
  tuned NEON. A dedicated NEON `dW` kernel is the top v0.2 speedup
  target (~1.3–1.5× end-to-end at FFN scale).
- Transpose cache uses `id(W)` as its key; there's a theoretical
  collision risk if Python GC reuses an id for a new same-shape
  PaddedCSR. Documented in `sparselab/ops.py`.
- Sparse attention works but is not a first-class API (see
  `examples/demo_14_sparse_attention.py`).

### Acknowledgments
- `sparselab.SparsityAlgorithm` API shape is adopted from Cerebras's
  `cstorch.sparse.SparsityAlgorithm` (see `docs/LANDSCAPE.md` for the
  full comparison).
- The Padded-CSR layout and the NEON SpMM / dW / dense-grad kernels
  are original work.
- `scripts/test_fresh_install.sh` pattern inspired by scientific-
  Python wheel-release conventions.
