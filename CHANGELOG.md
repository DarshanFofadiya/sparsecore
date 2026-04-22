# Changelog

All notable changes to SparseCore are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — TBD (first public release)

First public release. The pluggable DST foundation.

### Added
- `sparsecore.PaddedCSR` — sparse matrix storage with O(1) slot
  insert, cached transpose, round-trip with `torch.sparse_csr`.
  Eight structural invariants checked by the C++ constructor.
- `sparsecore.spmm(W, X)` — sparse-dense matmul, autograd-aware. NEON
  path on ARM64, scalar path on x86. OpenMP parallelized across the
  outer row loop.
- `sparsecore.SparseLinear(nn.Module)` — drop-in `nn.Linear`
  replacement. Standard `nn.Parameter`, standard `state_dict`,
  standard `torch.optim` compatibility.
- `sparsecore.SparsityAlgorithm` — pluggable DST base class. Inspired
  by Cerebras's `cstorch.sparse.SparsityAlgorithm`.
- `sparsecore.Static` — no-op reference sparsity algorithm.
- `sparsecore.SET` — Sparse Evolutionary Training (Mocanu et al.,
  2018) with magnitude-based drop and random regrow.
- `sparsecore.RigL` — Rigging the Lottery (Evci et al., 2020) with
  gradient-based regrow.
- 372 tests including gradcheck against PyTorch autograd and
  dense-equivalence oracles at 1e-5 tolerance.
- 15 example demos, end-to-end from "hello pybind" to "10M-param
  mini-GPT trained on Tiny Shakespeare at 90% FFN sparsity."
- Pre-built PyPI wheels for macOS arm64, macOS x86_64, Linux x86_64,
  and Linux aarch64 across Python 3.11, 3.12, 3.13. libomp bundled
  inside the wheels — no `brew install libomp` needed for
  `pip install` users.
- Colab notebook (`examples/colab_try_sparsecore.ipynb`) for zero-
  setup exploration.
- Docker-based fresh-install test (`scripts/test_fresh_install.sh`)
  and SageMaker recipe (`scripts/test_on_sagemaker.md`).

### Known limitations
- CPU-only. No GPU backend.
- Single-machine. No distributed / DDP training.
- Native Windows wheels not available (use WSL2 with the Linux wheel
  in the meantime). Planned v0.2.
- The `dW` kernel relies on Clang auto-vectorization, not hand-
  tuned NEON. A dedicated NEON `dW` kernel is the top v0.2 speedup
  target (~1.3–1.5× end-to-end at FFN scale).
- Transpose cache uses `id(W)` as its key; there's a theoretical
  collision risk if Python GC reuses an id for a new same-shape
  PaddedCSR. Documented in `sparsecore/ops.py`.
- Sparse attention works but is not a first-class API (see
  `examples/demo_14_sparse_attention.py`).

### Acknowledgments
- `sparsecore.SparsityAlgorithm` API shape is adopted from Cerebras's
  `cstorch.sparse.SparsityAlgorithm` (see `docs/LANDSCAPE.md` for the
  full comparison).
- The Padded-CSR layout and the NEON SpMM / dW / dense-grad kernels
  are original work.
- `scripts/test_fresh_install.sh` pattern inspired by scientific-
  Python wheel-release conventions.
