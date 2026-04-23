# SparseCore

A PyTorch-native library for dynamic sparse training on CPU. Store
weights as actually-sparse (Padded-CSR), dispatch to hand-tuned NEON
kernels, plug in your own DST algorithm as a short Python subclass.

Built CPU-first and Apple-Silicon-first so you can prototype sparse
topologies on a MacBook without a GPU.

> **Status: v0.1.** Core kernels, autograd, `SparseLinear`, the
> `SparsityAlgorithm` API, SET, RigL, and end-to-end transformer
> training all work and are tested. See the roadmap below for what's
> next.

---

## Who this is for

You should try SparseCore if you are:

- **A DST researcher** tired of reinventing the scaffolding around
  every new algorithm. Write your next drop/grow rule as a ~50-line
  subclass of `SparsityAlgorithm` and it plugs into real sparse
  storage and real sparse kernels, not a mask-on-dense simulation.
- **A PyTorch user without a GPU** who wants to train small-to-medium
  sparse models (10K – 10M parameters) on a MacBook or workstation
  and iterate quickly.
- **Someone curious what "actually sparse" looks like** under the
  hood — the Padded-CSR layout and the NEON inner loop are
  documented and commented to teach, not just to run.

You should probably *not* pick SparseCore if you want to:

- Pretrain a 7B-param LLM on your laptop. We are not a magic speedup
  over GPU training for large models; the CPU memory bandwidth wall
  is real.
- Fine-tune a dense pretrained model via pruning. That's
  [Neural Magic's SparseML](https://github.com/neuralmagic/sparseml)
  territory.
- Run on GPU. v0.1 is CPU-only. GPU support is a v0.3+ contribution
  opportunity.

If you're trying to do something we don't cover, open an issue — we'd
rather point you at the right tool than pretend we're it.

---

## The honest performance picture

What we measure, on an M3 Pro, at the end of v0.1:

| | Dense PyTorch (CPU) | SparseCore (90% sparse, CPU) |
|-|-|-|
| **MNIST 2-layer MLP, test accuracy** | 98.06% | 97.45% |
| **Weight memory** | 1.60 MB | 0.29 MB (18%) |
| **Per-step wallclock, FFN-mid scale** | ~10 ms | ~24 ms |
| **Max trainable model size, single machine** | limited by RAM | limited by RAM |

Two takeaways we want to be upfront about:

1. **Per-step, we are ~2× slower than GPU-trained dense on the same
   small model**, and modestly slower than dense on CPU. This is a
   structural property of sparse memory access on CPU (it doesn't
   cache as well as dense matmul), not something we can fully
   optimize away. Planned v0.2 work narrows the gap; it does not
   close it.

2. **The win is memory, not raw speed.** At 90% sparsity the weight
   footprint is ~18% of the dense equivalent. On a workstation CPU
   with 256 GB DDR5 you can train sparse models whose dense
   equivalents wouldn't fit on a single consumer GPU. Every DST
   paper can be expressed as a ~50-line plugin, so researchers spend
   time on algorithms, not plumbing.

If your bottleneck is "my model doesn't fit in GPU VRAM and I don't
want to rent an H100," SparseCore on a well-specced CPU box is
genuinely a path forward. If your bottleneck is "my training loop is
slow," it probably isn't.

---

## Quick look

```python
import torch
import sparsecore

# One-line swap: nn.Linear → sparsecore.SparseLinear.
model = torch.nn.Sequential(
    sparsecore.SparseLinear(784, 512, sparsity=0.9),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
)
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Pluggable DST: add SET topology mutation in 2 lines.
algo = sparsecore.SET(sparsity=0.9, drop_fraction=0.3, update_freq=100)
model.apply(algo)        # attaches to every SparseLinear in the tree

# Rest of your training loop is normal PyTorch.
for step in range(1000):
    x = torch.randn(128, 784)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    opt.step()
    algo.step()          # drives topology mutation on the schedule
    opt.zero_grad()
```

`SparseLinear` is a standard `nn.Module`. Its parameters are standard
`nn.Parameter`s. It loads into standard `torch.optim` optimizers. The
only thing different is that under the hood, the weight tensor is
stored as a Padded-CSR and the forward/backward calls go through our
sparse kernels.

---

## Install

```bash
pip install sparsecore
```

Want to try it without installing anything locally? Open our [Colab
notebook](examples/colab_try_sparsecore.ipynb) — it installs
SparseCore from PyPI, runs a smoke test, and walks through a small
training loop with SET. One click, no setup.

That's it. Pre-built wheels are published for the following platforms,
with OpenMP and the NEON/scalar kernels bundled inside — no system
libraries to install, no compiler required:

| Platform | Arch | Python versions | Kernel |
|-|-|-|-|
| macOS | arm64 (Apple Silicon) | 3.11, 3.12, 3.13 | NEON + OpenMP |
| Linux | x86_64 (manylinux) | 3.11, 3.12, 3.13 | scalar + OpenMP |
| Linux | aarch64 (manylinux) | 3.11, 3.12, 3.13 | NEON + OpenMP |

**Windows & Intel Mac:** not yet. Native Windows wheels are planned
for v0.2. Intel Mac wheels are paused while we wait for GitHub's
replacement Intel CI runner — we don't want to ship wheels we can
only test under Rosetta emulation. In the meantime:
- **Windows users:** use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
  with our Linux wheel — that path works today.
- **Intel Mac users:** build from source (see "Development install"
  below). It takes ~45 seconds on any modern Mac.

If you're on a platform not in the table above, pip falls back to
compiling from source. For that you'll need:

- **Python** 3.11+
- **PyTorch** ≥ 2.8 (pulled in automatically)
- **C++17 compiler** — clang 14+ or gcc 9+
- **libomp** on macOS: `brew install libomp`. On Linux it ships with
  gcc/clang. Without it the build still succeeds but runs sequentially
  (4–6× slower).

### Development install

For hacking on SparseCore itself:

```bash
git clone https://github.com/DarshanFofadiya/sparsecore.git
cd sparsecore
brew install libomp        # macOS only
pip install -e '.[dev]'
```

The editable install rebuilds the C++ kernels whenever you touch a
file in `csrc/`. First build takes ~45 seconds.

### Verify install

```python
import sparsecore
print(sparsecore.__version__)          # should print 0.1.0 or newer

# Quick smoke test — this should run in under a second
import torch
W = sparsecore.PaddedCSR.random(256, 128, sparsity=0.9, seed=0)
X = torch.randn(128, 32)
Y = sparsecore.spmm(W, X)
print(Y.shape)                          # torch.Size([256, 32])
```

If you installed from source, the full test suite is also available:

```bash
pytest
# 372 passed in ~3s
```

If something doesn't work, please [open an issue with the output](https://github.com/DarshanFofadiya/sparsecore/issues)
— v0.1 is the first time other people are installing this, so we
want to hear about failures.

---

## Demos

Runnable examples, each a single file with a banner explaining what it
proves. Run them top to bottom; each adds one more concept:

```bash
python examples/demo_01_bridge.py                  # pybind11 "hello world"
python examples/demo_02_dot.py                     # NEON SIMD dot product
python examples/demo_03_spmm.py                    # sparse matmul benchmark
python examples/demo_04_autograd.py                # sparse backward pass
python examples/demo_05_mnist.py                   # MNIST at 7 sparsity levels
python examples/demo_08_sparse_full_convergence.py # dense vs sparse @ 90%, converged
python examples/demo_09_parallel_speedup.py        # OpenMP thread scaling
python examples/demo_11_rigl_vs_set_vs_static.py   # RigL vs SET vs Static
python examples/demo_13_tiny_transformer.py        # 200-step char transformer
python examples/demo_14_sparse_attention.py        # sparse attention (not promoted to API)
python examples/demo_15_mini_gpt.py                # 10M-param GPT on tiny-shakespeare
```

Demos that need visualization or datasets (MNIST, transformer) pull in
matplotlib and torchvision:

```bash
pip install -e '.[demos]'
```

---

## What works today

- `sparsecore.PaddedCSR` — sparse storage with O(1) slot insert, cached
  transpose, round-trip with `torch.sparse_csr`.
- `sparsecore.spmm(W, X)` — sparse-dense matmul with NEON + OpenMP,
  autograd-aware.
- `sparsecore.SparseLinear(nn.Module)` — drop-in `nn.Linear`
  replacement. Standard `nn.Parameter`, standard `state_dict`.
- `sparsecore.SparsityAlgorithm`, `Static`, `SET`, `RigL` — pluggable
  DST API. Inspired by Cerebras's `cstorch.sparse.SparsityAlgorithm`;
  see `docs/LANDSCAPE.md`.
- 372 tests, including gradcheck against PyTorch autograd and
  dense-equivalence oracles at 1e-5 tolerance.
- 15 demos, end-to-end from "hello pybind" through "10M-param
  mini-GPT trained on Shakespeare at 90% sparsity."

## Known limitations (we'd rather tell you upfront)

- **Single machine only.** No distributed / DDP support in v0.1.
  Planned for v0.3.
- **CPU only.** No CUDA. Planned as a v0.3+ contribution opportunity.
- **Slower per-step than dense on small models.** Structural CPU
  limitation; see performance table above.
- **Transpose cache has a theoretical `id()` collision risk** when a
  `PaddedCSR` is garbage-collected and Python reuses its id for a new
  same-shape, same-topology-version CSR. Documented in
  `sparsecore/ops.py`; has not been observed in practice but is real.
- **`dW` kernel isn't NEON-vectorized yet** — it relies on Clang's
  auto-vectorization at `-O3`. A hand-tuned NEON dW is the biggest
  outstanding performance win (~1.3–1.5× end-to-end at FFN scale)
  and is the top v0.2 item.
- **Sparse attention is not a primitive.** We showed it works (see
  `demo_14_sparse_attention.py`) but didn't promote it to a
  first-class API in v0.1.

---

## Roadmap

**v0.1 (this release).** The pluggable DST foundation. Kernels,
storage, autograd, `SparseLinear`, `SparsityAlgorithm` base,
`Static` / `SET` / `RigL`, end-to-end transformer demo, pre-built
PyPI wheels for macOS and Linux.

**v0.2 (next ~4–6 weeks).**
- **Windows native wheels** — removes the WSL2 workaround.
- **Intel Mac wheels** — once GitHub's replacement Intel CI runner
  ships so we can build + test natively.
- NEON `dW` kernel — the main outstanding speedup target
  (~1.3–1.5× end-to-end at FFN scale).
- Buffer reuse / arena in the backward path.
- Richer parallelism tuning (`schedule(dynamic)` experiments for
  uneven nnz distributions).
- AVX-512 kernels for x86 (good community contribution target).
- More DST algorithms — adaptive-sparsity variants, Sparse Momentum,
  etc. — from community PRs.

**v0.3 (post-launch community phase).**
- PyTorch DDP compatibility for data-parallel training across CPU
  nodes (the plumbing is mostly there; needs validation).
- Memory-mapped weights for models that push single-node RAM.
- Sparse attention as a first-class primitive if there's demand.
- GPU backend as a community-led contribution opportunity.

**Explicitly not in the roadmap:**
- Beating GPU training on large dense models. Different use case.
- Post-training pruning / quantization (see SparseML).
- Structured sparsity (2:4 etc.). See torchao.

---

## Positioning vs other projects

| | What it is | How we relate |
|-|-|-|
| **Cerebras `cstorch.sparse`** | Production sparse training on wafer-scale chips | We adopt their `SparsityAlgorithm` API shape. They use dense+mask; we use Padded-CSR. Complementary. |
| **Neural Magic SparseML** | Post-training pruning for inference | Different workflow. They compress trained models; we train from scratch. |
| **rigl-torch** | Community PyTorch port of RigL | Single-algorithm, mask-simulated. We're the pluggable multi-algorithm version with real sparse storage. |
| **torchao.sparsity** | GPU structured (2:4) sparsity | Different axis: structured-GPU-posttraining vs unstructured-CPU-fromscratch. |

Full details: `docs/LANDSCAPE.md`.

---

## Documentation

- `docs/PROJECT_OVERVIEW.md` — project thesis and architecture.
- `docs/LANDSCAPE.md` — honest survey of the sparse-ML ecosystem.
- `docs/design/*.md` — design docs written before the code they
  describe (Padded-CSR, SpMM, SparseLinear, Router, RigL).
- `docs/demos/milestone_*.md` — per-milestone writeups with measured
  results and text samples.

---

## Contributing

Pull requests are welcome. The codebase is intentionally small and
readable; we'd rather merge a thoughtful 50-line PR than a
1,000-line refactor. See `docs/design/` for the design philosophy,
`.kiro/steering/` for the conventions we hold ourselves to, and
`tests/` for how we oracle-test every kernel.

If you're thinking about a new DST algorithm, start with
`sparsecore/router.py` — `SET` and `RigL` are both ~200 lines
including doc, ~50 lines of real logic, and would be good templates
for a new subclass.

---

## License

MIT. Built by [Darshan Fofadiya](https://github.com/DarshanFofadiya).
