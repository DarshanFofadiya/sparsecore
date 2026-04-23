# SparseLab: Architecture & Project Overview

## Why SparseLab Exists

Current dynamic sparse training (DST) research forces a painful
tradeoff: either use dense-simulated masks (wasting compute and memory
on weights that are supposed to be zero) or spend weeks writing custom
C++ kernels. SparseLab is a **PyTorch-native, actually-sparse DST
library** built for researchers in this space.

The concrete offer:

- Write any DST algorithm (RigL, SET, Condensed, a new one you're
  inventing) as a ~50-line Python subclass of `SparsityAlgorithm`.
- Hand it a `SparseLinear(nn.Module)` and it plugs into standard
  PyTorch — real `nn.Parameter`s, real `torch.optim`, real
  `state_dict`.
- Under the hood, weights are stored as Padded-CSR and forward/backward
  go through NEON + OpenMP kernels. Not mask-on-dense simulation.

This is an *actually sparse training library*, not a dense training
library that pretends to be sparse. We are CPU-first and
Apple-Silicon-first so the iteration loop stays on a researcher's
laptop instead of a rented GPU.

*For the full audit of the existing ecosystem and where we fit, see
`LANDSCAPE.md`.*

---

## 1. The Target Application: Tiny-but-Credible Transformer

SparseLab is built to train Transformers from scratch using dynamic
sparsity on commodity hardware.

- **v0.1 target:** 10M-parameter character-level decoder-only
  transformer (6 layers, `d_model=384`, `d_ff=1536`) trained on
  Tiny-Shakespeare at 90% FFN sparsity. Runs on a MacBook.
- **Scope:** We sparsify the **weight matrices** (linear projections,
  FFN). Attention stays dense in v0.1 — we proved sparse attention
  works (see `demos/demo_14_sparse_attention.py`) but didn't promote
  it to a first-class API.
- **Success criterion (v0.1):** Loss curves track dense training
  within expected tolerances; topology mutation via SET/RigL works
  end-to-end; 369-test suite passes; all demos run.

Larger-scale training (100M+ parameters, distributed, GPU) is
explicitly out of scope for v0.1 and lives on the v0.2/v0.3 roadmap.

## 2. The Hardware Target: CPU, Apple-Silicon-First

- **Primary:** ARM64 (Apple M-series) with 128-bit NEON, OpenMP
  parallelism across the performance cores.
- **Secondary:** x86_64 Linux (scalar kernels + OpenMP — still fast,
  no NEON). AVX-512 kernels are a good v0.2 community contribution.
- **ARM Linux (Graviton, Raspberry Pi 5, Ampere):** same NEON path
  as macOS. Wheels ship for Linux aarch64.
- **Not supported in v0.1:** Windows (planned for v0.2 via clang-cl),
  GPU (a v0.3+ contribution opportunity), distributed training
  (v0.3).

## 3. The API Strategy: The Middle Path

We are building a PyTorch extension, not a standalone framework.

- **The UX:** `sparselab.SparseLinear(d_in, d_out, sparsity=0.9)` is a 100% drop-in replacement for `nn.Linear`. Same initialization surface, same forward/backward contract, same `state_dict` integration.
- **The Integration:** `torch.autograd.Function` bridges our C++ SIMD kernels into PyTorch's autograd graph. No custom autograd engine.
- **The Matrix Shape:** Transformers process batched sequences. Our core C++ kernel optimizes for **Batched SpMM**: `(d_out, d_in) sparse × (B·S, d_in)` dense → `(B·S, d_out)` dense.

## 4. The Data Layout: Padded-CSR

Standard CSR is fast to read but brittle under mutation — inserting a new nonzero requires shifting arrays. Dynamic sparse training *requires* cheap insertions because connections are born every few hundred steps.

**Padded-CSR** over-allocates each row with empty slots, giving us **O(1) insertion** during the grow phase of backprop. The cost is a small memory overhead (typically 10-20% of the row capacity).

**Mutation model.** `row_capacity[i]` is set at layer construction from the initial per-row `nnz × (1 + padding_ratio)` and stays frozen for the life of the layer. Topology mutation happens through a single primitive, `rewrite_row(i, new_cols, new_values)`, which overwrites row `i`'s slot range in place and fills trailing slots with a `col=-1, value=0` sentinel. There's no free-list, no compaction, no garbage collection — the padding slots are simply recycled when `nnz` drops and re-used when `nnz` grows.

This is sufficient for algorithms where per-row `nnz` stays constant over training (SET, RigL as shipped). It's *not* sufficient for algorithms where `nnz` can drift — those would need a `compact_all()` primitive that redistributes capacity across rows, which is planned for v0.2.

## 5. The Pluggable Router API

This is the product researchers interact with.

```python
class SparsityAlgorithm:
    def __init__(self, sparsity: float): ...
    def step(self): ...            # called per training step
    def update(self): ...          # override to mutate topology
```

Any DST algorithm (RigL, SET, Sparse Momentum, future papers) becomes
a ~50-line Python subclass with docs and ~200 lines total. The C++
engine doesn't know which algorithm is running. This is the
scaffolding we offer the community.

v0.1 ships with:
- `Static` (reference implementation: random mask at init, never changes)
- `SET` (Sparse Evolutionary Training — magnitude-based drop, random regrow)
- `RigL` (Rigging the Lottery — magnitude-based drop, gradient-based regrow)

## 6. Execution Status (milestones delivered in v0.1)

**Phase 1 — PyTorch bridge.** CMake/setup.py + pybind11 + trivial C++
function callable from Python. Delivered; `demo_01_bridge.py`.

**Phase 2 — Dense NEON warmup.** Scalar matmul → SIMD matmul with NEON
intrinsics. Oracle-verified against `torch.matmul`. Delivered;
`demo_02_dot.py`.

**Phase 3 — Static sparse inference.** Padded-CSR struct, PyTorch ↔
Padded-CSR round-trip, SIMD SpMM forward. Oracle-verified. Delivered;
`demo_03_spmm.py`.

**Phase 4 — Dynamic sparse training.** Sparse backward pass, autograd
integration, `SparseLinear` nn.Module, pluggable `SparsityAlgorithm`
API, SET, RigL, end-to-end transformer training. Delivered;
`demo_04_autograd.py` through `demo_15_mini_gpt.py`.

## 7. Definition of Done for v0.1

The project ships to the community when all of these are true:

1. A standard PyTorch `nn.Module` can swap `nn.Linear` →
   `sparselab.SparseLinear` with a one-keyword change.
2. A 10M-param decoder-only transformer trains end-to-end on a
   character-level task at 90% FFN sparsity, tracking the dense
   baseline's loss curve within reasonable tolerance.
3. `Static`, `SET`, `RigL` all work as `SparsityAlgorithm` subclasses.
4. Kernels pass the Oracle suite at 1e-5 tolerance (372 tests today).
5. `pip install sparselab` works on macOS arm64, macOS x86_64,
   Linux x86_64, Linux aarch64 — wheels published to PyPI with
   libomp bundled inside.
6. README, LANDSCAPE, and demo writeups are polished enough that a
   grad student can follow them start-to-finish.

Not part of v0.1 DoD (and called out as such):
- Faster than `torch.matmul` on CPU. (We are not — the pitch is
  memory and pluggability, not raw speed. See README "honest
  performance picture.")
- Windows support. (Planned v0.2.)
- GPU backend. (v0.3+ contribution opportunity.)
- Distributed training. (v0.3.)

## 8. Directory Layout (as shipped)

```
sparselab/
├── docs/
│   ├── PROJECT_OVERVIEW.md         # this file
│   ├── LANDSCAPE.md                # ecosystem audit
│   ├── design/                     # design docs written before the code
│   │   ├── padded_csr.md
│   │   ├── spmm.md
│   │   ├── spmm_backward.md
│   │   ├── sparse_linear.md
│   │   ├── router.md
│   │   ├── rigl.md
│   │   └── tiny_transformer.md
│   └── demos/                      # per-milestone writeups + screenshots
│       ├── milestone_01.md         ... milestone_10.md
│       └── demo_*.png / .txt artifacts
├── csrc/                           # C++ kernels and pybind11 glue
│   ├── bindings.cpp
│   └── kernels/
│       ├── padded_csr.{hpp,cpp}
│       ├── spmm.{hpp,cpp}          # scalar reference SpMM
│       ├── spmm_neon.{hpp,cpp}     # NEON-vectorized SpMM
│       ├── spmm_grad.{hpp,cpp}     # dW kernel
│       ├── dense_grad.{hpp,cpp}    # RigL's dense-grad stall kernel
│       ├── vector_dot{_neon}.{hpp,cpp}
│       └── parallel.hpp            # OpenMP shim
├── sparselab/                     # Python package
│   ├── __init__.py                 # attaches Python factories to C++ class
│   ├── layout.py                   # PaddedCSR factories + transpose
│   ├── ops.py                      # spmm + autograd Function
│   ├── nn.py                       # SparseLinear
│   └── router.py                   # SparsityAlgorithm base + Static/SET/RigL
├── tests/                          # 369 Oracle tests (pytest)
├── examples/                       # 15 runnable demos
├── .github/workflows/              # cibuildwheel CI for PyPI wheels
├── environment.yml
├── pyproject.toml
├── setup.py
└── README.md
```

## 9. What's next after v0.1

See the v0.2/v0.3 roadmap in the README for the full list. Highest-
priority items:

- **dW kernel optimization.** At FFN-mid scale, `dW` is 62% of a
  training step. A NEON-vectorized dW is measured at ~1.3–1.5×
  end-to-end speedup and is the headline v0.2 item.
- **`PaddedCSR.compact_all()` primitive** (v0.2). Redistributes
  row capacity based on current `nnz`, enabling adaptive-density
  DST algorithms where per-row live count drifts across training.
- **Windows native wheels** (v0.2). Removes the WSL2 workaround.
- **Intel Mac wheels** (v0.2). Once GitHub's replacement runner ships.
- **PyTorch DDP compatibility** (v0.3). The plumbing mostly works;
  needs end-to-end validation and a multi-node demo.
- **Sparse attention as a first-class primitive.** We proved it
  works at 70% attention sparsity + 90% FFN sparsity in
  `demo_14_sparse_attention.py` and at 10k-step scale in the
  launch demo; v0.2 or v0.3 promotes the pattern to a documented
  API.
