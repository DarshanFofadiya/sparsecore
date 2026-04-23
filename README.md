# SparseCore

![v0.1](https://img.shields.io/badge/version-0.1.0-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyPI](https://img.shields.io/pypi/v/sparsecore)
![tests](https://img.shields.io/badge/tests-372%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

**Masking is not sparsity.**

---

## TL;DR — the 60-second read

SparseCore is a PyTorch library for training sparse neural networks
*from scratch*, with real sparse storage and real sparse kernels.
Not mask-on-dense. Not post-training pruning. Actual sparsity
at training time, on commodity hardware.

**Why it matters:**

- **Most neural networks are mostly unnecessary.** 90%-sparse MNIST
  reaches **97.45% vs 98.06% dense — 0.61 pp gap for 82% memory
  reduction**. A 10M-param Tiny Shakespeare transformer tracks
  dense val loss within 0.055 nats at **17.5% of dense parameters**.

- **Nobody else is doing this.** Cerebras trains sparse but uses
  dense storage + a binary mask on wafer-scale chips. Neural
  Magic does post-training pruning for CPU inference (not
  training). torchao is GPU-structured-2:4. rigl-torch is a
  single algorithm and mask-simulated. Nobody ships an
  actually-sparse training stack you can `pip install` on a
  laptop. (See the table below for specifics.)

- **v0.1 proves the paradigm on a laptop.** 10M-param transformer
  on a MacBook CPU: 37% of dense memory, quality tracking dense,
  real sparse storage end-to-end. Not a simulation.

- **v0.2 scales it to clusters.** CPU-cluster data parallelism
  turns "1B dense model → 100M live sparse" into a realistic
  workload on commodity CPU infrastructure. A 10-machine CPU
  cluster with 128 GB RAM each is a few thousand dollars; an
  8×H100 DGX node that handles an equivalent dense workload
  runs $300K+ upfront or $20K+/month in the cloud.

- **It's also a hardware problem, not just software.** GPUs are
  built for dense; sparse accelerators have no training stack to
  target. SparseCore is the software the hardware ecosystem has
  been waiting for.

**For whom:** DST researchers. PyTorch users without GPU access.
Contributors who care about low-level CPU performance. Anyone
building toward purpose-built sparse hardware.

**Get it:** `pip install sparsecore`. Pre-built wheels for macOS
arm64 and Linux x86_64/aarch64, Python 3.11–3.13. MIT license.
372 tests including autograd gradcheck.

---

## What no one else is shipping

We audited the ecosystem before building this. Here's the concrete
gap:

| Project | Storage | Training | Hardware | `pip install` on a laptop |
|-|-|-|-|-|
| **SparseCore** (us) | **Real sparse (Padded-CSR)** | **From scratch, pluggable DST** | **CPU (NEON + OpenMP)** | **Yes** |
| Cerebras `cstorch.sparse` | Dense + mask | From scratch, pluggable DST | Wafer-scale only | No |
| Neural Magic SparseML | Dense + mask | Post-training pruning | CPU (inference) | Yes (inference only) |
| rigl-torch | Dense + mask | From scratch, RigL only | CPU/GPU | Yes (mask-simulated) |
| torchao.sparsity | Structured 2:4 | Post-training | GPU | Yes (GPU, structured) |
| `torch.sparse` | Real sparse | Not really supported | CPU/GPU | Yes (no training support) |

**The corner we're in that nobody else occupies: actually-sparse,
unstructured, training-from-scratch, CPU-native, with a pluggable
DST algorithm interface.** That's the specific claim; `docs/LANDSCAPE.md`
walks through each project in detail with what we learned from
them and what we explicitly diverge on.

If you know of a library doing the same thing we're doing,
please file an issue — keeping this comparison honest is part of
how we want to operate.

---

## Most neural networks are mostly unnecessary

That's not a claim we're making in the abstract. It's what our
measurements show, and what the DST literature has been pointing
at for years.

From our own demos in this repo:

- **MNIST (2-layer MLP), trained to convergence:** 90%-sparse
  reaches **97.45%** accuracy vs dense **98.06%** — a **0.61 pp
  gap for 82% memory reduction**. The caveat: sparse needs ~1.8×
  more epochs to reach its plateau (the cost of a random-and-
  frozen mask). Smarter DST routers (RigL / SET) close that
  gap in fewer epochs; the v0.1 demo uses random masks to
  establish the floor. See `docs/demos/milestone_05.md` and
  `docs/demos/milestone_08.md`.
- **10M-parameter transformer on Tiny Shakespeare (10k steps):**
  Keeping only **17.5% of weights** (attention 70% + FFN 90%
  sparse) tracks dense val-loss to within 0.055 nats — within
  run-to-run noise for char-level LM. Memory footprint: **37%
  of dense** at inference. See `docs/demos/milestone_10.md`.

The pattern is consistent across every model size and task we've
tried: you can keep comparable quality at roughly 10–20% of dense
parameters, and competitive quality at 30%. The extra 80-90% of
weights in a trained dense model are mostly noise around the
small fraction that does the actual work.

**Sparsity isn't a lossy compression; it's the actual information
structure of the learned model.** Dense training can't cheaply
tell the difference because it has to compute the whole matrix
regardless of what's doing the work. The reason you can't just
run sparse training as a drop-in is that nobody's shipped a
software stack that treats sparsity as first-class at training
time.

That's the problem SparseCore is built to solve.

---

## This is a software AND hardware problem

Two things have to be true for sparse to win:

1. **The software stack has to treat sparsity as first-class.** Not
   a mask over a dense tensor. Not a post-training step. The
   storage format, kernels, autograd integration, and training
   loop all have to work with live weights directly. That's what
   SparseCore is.

2. **The hardware has to be built for it.** Current GPUs are
   engineered for the dense-matmul workload and they optimize it
   ruthlessly. They handle sparse poorly because nobody's asked
   them to; vendor roadmaps don't prioritize what researchers
   don't actually run. Purpose-built sparse accelerators — the
   neuromorphic-style chips the industry has been theorizing
   about for a decade — have no training software to target, so
   they stay theoretical.

SparseCore's bet is that the software stack has to exist first,
so the hardware has something real to optimize for.

The brain runs on roughly **20 watts** — about a dim light bulb.
The GPUs that approximate a fraction of what it does draw
kilowatts each, scaled into datacenters that draw gigawatts.
That gap isn't fundamental. Part is software: the mask-on-dense
paradigm wastes most of the compute. Part is hardware: we built
silicon for the wrong workload. Both have to be fixed, and the
software is the one researchers can move first.

---

## What v0.1 delivers

**v0.1 is the proof that actually-sparse training is viable on
commodity hardware.** A 10-million-parameter transformer on a
MacBook CPU, trained from scratch at 37% of dense memory, quality
tracking dense. Not a simulation — real sparse kernels, real
at-rest memory.

What becomes possible on top of this foundation:

- **Researchers without GPU access can run real experiments.** A
  Mac and a weekend replaces a cloud bill for a lot of research.
- **Community kernel optimization over time.** The SpMM, dW, and
  transpose kernels are all contributor-shaped problems. Every
  optimization PR is a speedup everyone inherits.
- **Purpose-built sparse hardware has a software stack to target.**
  Sparse accelerators have been a research topic for 10+ years;
  SparseCore is the first real end-to-end sparse training stack
  they can plug into.

---

## Current results (v0.1)

10M-parameter decoder-only transformer (6 layers, d_model=384),
trained from scratch on Tiny Shakespeare for 10,000 steps on an
M3 Pro MacBook:

| | Dense | Sparse FFN 90% | Sparse all (attn 70% + FFN 90%) |
|-|-|-|-|
| **Parameters** | 10.7M | 4.4M live | **1.9M live** |
| **Inference memory** | 41.0 MB | 19.9 MB (48%) | **15.3 MB (37%)** |
| **Training memory (weight+grad+padding)** | 81.8 MB | 35.9 MB | **25.2 MB (31%)** |
| **Final validation loss** | 2.534 | 2.582 | 2.589 |

**Memory footprint of the all-sparse model: 37% of dense at
inference, 31% at training.** Real, at-rest, not simulated.

**Quality tracks dense to within 0.055 nats** after 10,000 steps
— within noise for char-level language modeling at this scale. No
sparse-specific pathology. Full writeup: [`docs/demos/milestone_10.md`](docs/demos/milestone_10.md).

### On speed: honest, not apologetic

We're slower per step than dense on CPU today — 2.4× for FFN-only
sparsity, 4.6× for all-sparse. This is a real cost and we don't
hide it.

It's also a solvable problem. Three things will change it:

1. **The NEON `dW` kernel is not yet hand-tuned** (Clang
   auto-vectorization only). Planned v0.2 work: ~1.3–1.5×
   end-to-end speedup at FFN scale.
2. **Sparse kernels have fixed per-layer overhead.** At the matrix
   sizes in this demo, that overhead dominates. It doesn't at
   larger scale — the break-even point is when weight matrices
   become memory-bandwidth bound, typically ~1024+ hidden size
   and above.
3. **Per-step speed is not the right frame.** CPU-cluster data
   parallelism (v0.2 roadmap) changes the scaling story entirely.
   See "The trajectory" below.

---

## The trajectory

v0.1 runs on one machine. That's the proof-of-concept phase — show
that actually-sparse training works, make it a real library, ship
it with wheels, tests, and demos.

**v0.2 adds data parallelism across CPU cores and machines.** This
is the scaling story that matters:

- 1B dense parameters at 90% sparsity = 100M live weights.
- 100M live weights ≈ 400 MB at training-time precision. Fits in
  RAM on any modern laptop.
- With CPU-cluster DDP, training it across 10 machines with 128 GB
  RAM each is a realistic configuration. Total hardware cost:
  a few thousand dollars of commodity workstations, or pennies-
  on-the-dollar compared to their GPU equivalent in the cloud.
- The GPU equivalent for a 1B-dense workload today is an 8×H100
  DGX node: roughly **$300K–$400K to purchase outright**, or
  ~$20K/month sustained in cloud at mid-market rates. Not
  available to most researchers at any university lab, lab-
  adjacent startup, or geography without GPU allocation.

CPU clusters are accessible to nearly any researcher, any
university lab, any startup without GPU allocation. H100 nodes
aren't. That's the asymmetry we're building toward.

**v0.3 and beyond: the hardware question.** If CPU-native actually-
sparse training works at scale, the next step is hardware that's
purpose-built for it. Not general-purpose GPUs doing sparse poorly,
not wafer-scale chips with dense-mask simulation — actual sparse
accelerators that match the brain's efficiency profile. The
neuromorphic industry wants this. The problem is nobody has a
training stack to target. SparseCore intends to be that stack.

We're not claiming to beat GPUs today. We are claiming the
paradigm is wrong, and that CPU-native actually-sparse training
deserves to exist as a serious research platform that can
eventually scale into specialized hardware.

---

## Who this is for

- **DST researchers** tired of reinventing scaffolding for every
  algorithm. Write your next drop/grow rule as a ~50-line subclass
  of `SparsityAlgorithm` on top of real sparse storage. No more
  mask-on-dense simulation.
- **Researchers without GPU access.** A MacBook or workstation CPU
  is enough to run real experiments on 10K – 10M parameter models
  today, and larger with v0.2 DDP.
- **Contributors who care about low-level performance.** The SpMM
  and dW kernels are the moats; every optimization compounds
  forever. NEON today, AVX-512 and ARM server-class tomorrow.
- **Anyone curious about sparse-first ML.** The code is
  intentionally readable and well-commented. A grad student can
  read the NEON inner loop and understand it.

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
`nn.Parameter`s. It loads into standard `torch.optim` optimizers.
The only thing different is that under the hood, the weight tensor
is stored as a Padded-CSR and the forward/backward go through our
sparse kernels.

### Prove the memory claim yourself

```python
import torch
import sparsecore

# A 784 × 512 layer, dense vs 90% sparse.
dense  = torch.nn.Linear(784, 512, bias=False)
sparse = sparsecore.SparseLinear(784, 512, sparsity=0.9, bias=False)

# Dense: 4 bytes per weight (float32).
dense_bytes = dense.weight.numel() * 4

# Sparse: 4 bytes per live value + 4 bytes per column index = 8 bytes per live.
# (Plus O(nrows) for the tiny row-metadata arrays — negligible at this scale.)
sparse_bytes = sparse.nnz * 8

print(f"Dense:  {dense_bytes / 1024:.1f} KB")
print(f"Sparse: {sparse_bytes / 1024:.1f} KB  ({100 * sparse_bytes / dense_bytes:.0f}% of dense)")
# Dense:  1568.0 KB
# Sparse: 310.5 KB  (20% of dense)
```

That's 20% of dense memory for the same 784 × 512 Linear layer at
90% sparsity. Real bytes, not a mask. The column-index array is
what makes it 20% rather than the naive "10% of dense" — every
live weight carries a 4-byte index so the kernel knows which
column it belongs to. That index overhead is the cost of being
actually sparse; it's also why the break-even point is around
50% sparsity (below that, dense storage is smaller).

---

## Install

```bash
pip install sparsecore
```

Pre-built wheels are published for the following platforms, with
OpenMP and the NEON/scalar kernels bundled inside — no system
libraries to install, no compiler required:

| Platform | Arch | Python versions | Kernel |
|-|-|-|-|
| macOS | arm64 (Apple Silicon) | 3.11, 3.12, 3.13 | NEON + OpenMP |
| Linux | x86_64 (manylinux) | 3.11, 3.12, 3.13 | scalar + OpenMP |
| Linux | aarch64 (manylinux) | 3.11, 3.12, 3.13 | NEON + OpenMP |

**Windows & Intel Mac:** not yet. Native Windows wheels are planned
for v0.2. Intel Mac wheels are paused while we wait for GitHub's
replacement Intel CI runner — we don't ship what we can't test on
real hardware. In the meantime:
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

Runnable examples, each a single file with a banner explaining what
it proves. Run them top to bottom; each adds one more concept:

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
python examples/demo_15_mini_gpt.py                # 10M-param GPT, 3-way comparison
```

Demos that need visualization or datasets (MNIST, transformer) pull
in matplotlib and torchvision:

```bash
pip install -e '.[demos]'
```

---

## What works today

- `sparsecore.PaddedCSR` — sparse storage with O(1) slot insert,
  cached transpose, round-trip with `torch.sparse_csr`.
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

- **Single machine only** in v0.1. Multi-machine DDP is the
  v0.2 roadmap's headline item.
- **CPU only.** GPU is a v0.3+ contribution target.
- **Slower per-step than dense on small models.** Explained above;
  v0.2 dW kernel work narrows the gap significantly.
- **Transpose cache has a theoretical `id()` collision risk** when a
  `PaddedCSR` is garbage-collected and Python reuses its id for a new
  same-shape, same-topology-version CSR. Documented in
  `sparsecore/ops.py`; has not been observed in practice but is real.
- **`dW` kernel isn't NEON-vectorized yet** — it relies on Clang's
  auto-vectorization at `-O3`. A hand-tuned NEON dW is the top v0.2
  speedup target.
- **Sparse attention is not a primitive** in v0.1. We verified it
  works end-to-end (see demo_14 and demo_15 all-sparse) but didn't
  promote it to a first-class API.
- **Fixed row capacity in Padded-CSR.** Each row's capacity is frozen
  at layer construction (initial `nnz × 1.2`). This gives us O(1)
  insertion during topology mutation. Algorithms that grow a row's
  live count beyond initial capacity will fail — SET and RigL work
  fine because they keep per-row `nnz` constant. Adaptive-density
  DST would need a `compact_all()` primitive. Planned v0.2.

---

## Roadmap

**v0.1 (this release).** The pluggable DST foundation. Kernels,
storage, autograd, `SparseLinear`, `SparsityAlgorithm` base,
`Static` / `SET` / `RigL`, end-to-end 10M-param transformer demo,
pre-built PyPI wheels for macOS and Linux.

**v0.2 (next ~4–6 weeks).** The scaling and optimization phase.
- **CPU-cluster data parallelism via PyTorch DDP.** This is the
  one that changes what's possible: training 100M-param-live
  sparse models (1B dense equivalent) across commodity CPU
  clusters that cost a few thousand dollars.
- **Hand-tuned NEON `dW` kernel.** 1.3–1.5× end-to-end speedup
  at FFN scale.
- **Buffer reuse / arena in the backward path.**
- **Parallelism tuning** (`schedule(dynamic)` experiments for
  uneven nnz distributions).
- **AVX-512 kernels for x86** (excellent community contribution
  target).
- **`PaddedCSR.compact_all()` primitive** for adaptive-density
  DST algorithms.
- **Windows native wheels.**
- **Intel Mac wheels** (once GitHub's replacement runner ships).
- **More DST algorithms** — Sparse Momentum, adaptive-sparsity
  variants — from community PRs.

**v0.3 (post-launch community phase).**
- **Memory-mapped weights** for models that exceed node RAM.
- **Sparse attention as a first-class primitive.**
- **GPU backend** as a community-led contribution opportunity.
- **Hardware-vendor partnerships** for sparse accelerators once
  the software stack proves itself at scale.

---

## Positioning vs other projects

| | What it is | How we relate |
|-|-|-|
| **Cerebras `cstorch.sparse`** | Production sparse training on wafer-scale chips | We adopt their `SparsityAlgorithm` API shape. They use dense+mask simulation; we use Padded-CSR. Complementary. |
| **Neural Magic SparseML** | Post-training pruning for inference | Different workflow. They compress trained models; we train sparse from scratch. |
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
  results and text samples. The v0.1 launch artifact is
  [`milestone_10.md`](docs/demos/milestone_10.md).

---

## Contributing

Pull requests are welcome. The codebase is intentionally small and
readable; we'd rather merge a thoughtful 50-line PR than a
1,000-line refactor. See `docs/design/` for the design philosophy
and `tests/` for how we oracle-test every kernel.

If you're thinking about a new DST algorithm, start with
`sparsecore/router.py` — `SET` and `RigL` are both ~50 lines of
real logic, good templates for a new subclass.

If you're thinking about kernel optimization (NEON / AVX / GPU),
the moats are in `csrc/kernels/`. Every improvement compounds for
every user forever.

---

## License

MIT. Built by [Darshan Fofadiya](https://github.com/DarshanFofadiya).
