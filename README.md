# SparseCore

> 🚧 **v0.1 in progress.** Core kernels (Padded-CSR SpMM with NEON SIMD +
> OpenMP), autograd, and `SparseLinear(nn.Module)` are landed and
> tested. DST algorithms (SET, RigL) are the next milestones. See
> `docs/PROJECT_OVERVIEW.md` for the roadmap.

**SparseCore** is a dynamic-sparse-training engine for PyTorch. Unlike
mask-on-dense libraries, we store weights as actually-sparse
(Padded-CSR) and dispatch to hand-tuned NEON kernels. Built CPU-first
and Apple-Silicon-first so you can experiment with sparse topologies
locally on a MacBook.

The goal: **make training sparse-from-scratch the default choice for
the open-source ML community, not an afterthought.**

---

## Quick look

```python
import torch, sparsecore

# One-line swap: nn.Linear → sparsecore.SparseLinear.
fc1 = sparsecore.SparseLinear(784, 512, sparsity=0.9)
fc2 = torch.nn.Linear(512, 10)
opt = torch.optim.SGD([*fc1.parameters(), *fc2.parameters()], lr=0.01)

# Rest of your training loop is normal PyTorch.
x = torch.randn(128, 784)
logits = fc2(torch.relu(fc1(x)))
loss = logits.sum()
loss.backward()
opt.step()
```

At 90% sparsity on MNIST this reaches **97.45% accuracy** — within
0.61 pp of the equivalent dense network — for **18% of dense memory**.
Full numbers in `docs/demos/milestone_05.md`.

---

## Install

SparseCore is **not yet on PyPI**. For now you install from source.

### Prerequisites

| | Requirement | How to get it |
|-|-|-|
| **Python** | 3.11+ | miniconda, pyenv, or system Python |
| **PyTorch** | ≥2.8 | `pip install torch` or via `environment.yml` |
| **C++17 compiler** | clang 14+ or gcc 9+ | macOS: Xcode command-line tools; Linux: distro package |
| **libomp** (macOS only) | any version | `brew install libomp` |

On Linux, `gcc` / `clang` ship with OpenMP built in — no extra install.

On macOS, Apple's default Clang does NOT include OpenMP. Installing
`libomp` via Homebrew is strongly recommended — without it you'll
build without parallel kernels and your sparse training will be 4-6×
slower than it should be. The build will warn loudly if libomp is
missing.

### Install steps

With conda (recommended, replicable across machines):

```bash
git clone https://github.com/DarshanFofadiya/sparsecore.git
cd sparsecore
conda env create -f environment.yml
conda activate sparsecore
brew install libomp          # macOS only
pip install -e .
```

With plain pip into an existing environment:

```bash
git clone https://github.com/DarshanFofadiya/sparsecore.git
cd sparsecore
brew install libomp          # macOS only
pip install -e .
```

Either way, the last step invokes `setup.py`, which compiles our C++
kernels via pybind11. First build takes ~45 seconds.

### Verify install

```bash
pytest
# 299 passed, 2 skipped in ~3s
```

If the tests pass, you have a working sparsecore with OpenMP parallel
kernels enabled.

---

## Demos

Runnable examples, each a single file with a banner explaining what it
proves:

```bash
python examples/demo_01_bridge.py                  # pybind11 "hello world"
python examples/demo_02_dot.py                     # NEON SIMD dot product
python examples/demo_03_spmm.py                    # sparse matmul benchmark
python examples/demo_04_autograd.py                # sparse backward pass
python examples/demo_05_mnist.py                   # MNIST at 7 sparsities
python examples/demo_08_sparse_full_convergence.py # converged-vs-converged @ 90%
python examples/demo_09_parallel_speedup.py        # OpenMP speedup numbers
```

Some demos (5, 6, 7, 8) need `torchvision` and `matplotlib`:

```bash
pip install 'sparsecore[demos]'
```

---

## What works today (as of milestone 4c)

- ✅ `sparsecore.PaddedCSR` — sparse storage format with O(1) slot insert
- ✅ `sparsecore.spmm(W, X)` — SpMM with NEON + OpenMP, autograd-aware
- ✅ `sparsecore.SparseLinear(nn.Module)` — drop-in `nn.Linear` replacement
- ✅ Standard `torch.optim.{SGD, Adam, ...}` support — our params are
     regular `nn.Parameter`s via a zero-copy numpy↔torch alias
- ✅ 299 tests, including autograd `gradcheck` and dense-equivalence
     oracles
- ✅ OpenMP parallelization of forward/backward kernels (~5× speedup)

## Not yet (next milestones)

- ⏳ Milestone 4d: `Router` / `SparsityAlgorithm` base class (Cerebras-style pluggable DST API)
- ⏳ Milestone 4e: SET (random drop + regrow)
- ⏳ Milestone 4f: RigL (gradient-regrow — closes the MNIST accuracy gap)
- ⏳ Milestone 4g: tiny-transformer launch demo
- ⏳ PyPI wheels (`pip install sparsecore` without a compiler)

---

## Documentation

- **`docs/PROJECT_OVERVIEW.md`** — the full project thesis and roadmap
- **`docs/LANDSCAPE.md`** — how we fit into the existing sparse-ML ecosystem (Cerebras, Neural Magic, rigl-torch, torchao…)
- **`docs/design/*.md`** — design docs written *before* the code they describe
- **`docs/demos/*.md`** — per-milestone write-ups with measured results

---

## License

MIT. Built by [Darshan Fofadiya](https://github.com/DarshanFofadiya) as
an open-source contribution; pull requests welcome once v0.1 is cut.
