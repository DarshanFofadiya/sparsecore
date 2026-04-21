# SparseCore: Architecture & Project Overview

## Why SparseCore Exists

> Current dynamic sparse training (DST) research forces a false choice: either use dense-simulated masks (wasting massive compute and memory) or spend six months writing custom C++ kernels. SparseCore is the first actually-sparse, PyTorch-native training framework built for the researcher. It provides a Pluggable Router API so you can implement any DST algorithm (RigL, SET, Condensed) in pure Python, backed by a native Padded-CSR C++ engine. Optimized heavily for Apple Silicon, SparseCore allows you to aggressively prototype and mutate billion-parameter sparse topologies locally on your MacBook before scaling to the GPU cluster.

This is our elevator pitch, our README headline, and our guiding narrative. Every design choice ladders up to it.

*For the full audit of existing projects and where we fit, see `LANDSCAPE.md`.*

---

## 1. The Target Application: "Tiny but Credible" Transformer

SparseCore is built to train Transformers from scratch using dynamic sparsity on commodity hardware.

- **v0.1 Goal:** Train a 2-layer, 128-hidden decoder-only Transformer on a character-level language modeling task.
- **Scope Definition:** We sparsify the **Weights** (linear projections, FFN), NOT the Attention mechanism. Attention remains dense for v0.1.
- **Success Criterion:** The demo transformer trains from scratch to reasonable loss at ≥90% unstructured weight sparsity on a MacBook CPU, end-to-end, with dynamic topology mutation during training.

## 2. The Hardware Target: Apple Silicon (NEON)

- **Processor:** ARM64 (Apple M-series).
- **SIMD:** 128-bit NEON intrinsics (`<arm_neon.h>`), 4 `float32` elements per lane.
- **No x86 emulation.** Rosetta would defeat the entire point. x86/AVX kernels are an explicit v0.2+ contribution opportunity, gated behind `#ifdef __x86_64__`.
- **Threading:** OpenMP across the 6 performance cores of Apple Silicon.

## 3. The API Strategy: The Middle Path

We are building a PyTorch extension, not a standalone framework.

- **The UX:** `sparsecore.SparseLinear(d_in, d_out, sparsity=0.9)` is a 100% drop-in replacement for `nn.Linear`. Same initialization surface, same forward/backward contract, same `state_dict` integration.
- **The Integration:** `torch.autograd.Function` bridges our C++ SIMD kernels into PyTorch's autograd graph. No custom autograd engine.
- **The Matrix Shape:** Transformers process batched sequences. Our core C++ kernel optimizes for **Batched SpMM**: `(d_out, d_in) sparse × (B·S, d_in)` dense → `(B·S, d_out)` dense.

## 4. The Data Layout: Padded-CSR

Standard CSR is fast to read but brittle under mutation — inserting a new nonzero requires shifting arrays. Dynamic sparse training *requires* cheap insertions because connections are born every few hundred steps.

**Padded-CSR** over-allocates each row with empty slots, giving us **O(1) insertion** during the grow phase of backprop. The cost is a small memory overhead (typically 10-20% of the row capacity).

## 5. The Pluggable Router API

This is the product researchers interact with.

```python
class Router(ABC):
    def pick_to_drop(self, weights: PaddedCSR) -> list[Index]: ...
    def pick_to_grow(self, weights: PaddedCSR, grads: PaddedCSR) -> list[Index]: ...
```

Any DST algorithm (RigL, SET, Sparse Momentum, future papers) becomes a ~100-line Python subclass. The C++ engine doesn't know or care which algorithm is running. This is the scaffolding we give the community.

v0.1 ships with:
- `SETRouter` (random growth — fast, no dense math, most training-stable)
- `MagnitudeRouter` (pruning only, no growth — for debugging and baseline comparisons)

RigL (gradient-based growth with periodic dense stalls) is v0.2.

## 6. Execution Roadmap

**Phase 1 — PyTorch Trojan Horse (scaffolding).** CMake/setup.py + pybind11 + a trivial C++ function callable from Python. Proves the bridge works.

**Phase 2 — Dense Baseline & NEON Warmup.** Naive C++ dense matmul → SIMD matmul with NEON intrinsics. Oracle-verified against `torch.matmul`.

**Phase 3 — Static Sparse Inference.** Padded-CSR struct. PyTorch → Padded-CSR converter. SIMD SpMM forward pass. Oracle-verified.

**Phase 4 — Dynamic Sparse Training.** Sparse backward pass. Router API. Topology mutation mid-training. End-to-end transformer demo.

## 7. Definition of Done for v0.1

The project ships to the community when:
1. A standard PyTorch `nn.Module` can swap `nn.Linear` → `sparsecore.SparseLinear` with zero other changes.
2. A 2-layer transformer trains end-to-end on a character-level task at ≥90% sparsity.
3. SET and Magnitude routers both work as pluggable classes.
4. All kernels pass the Oracle suite at 1e-5 tolerance.
5. Sparse SpMM measurably outperforms dense `torch.matmul` at ≥90% sparsity on M-series hardware.
6. README, LANDSCAPE, and example notebooks are polished enough that a grad student can follow them start-to-finish.

## 8. Planned Directory Layout

```
SparseCore/
├── docs/                       # PROJECT_OVERVIEW, LANDSCAPE, SYSTEM_PROMPT
├── csrc/                       # C++ kernels + pybind11 glue
│   ├── padded_csr.hpp
│   ├── spmm_neon.cpp
│   └── bindings.cpp
├── sparsecore/                 # Python package
│   ├── __init__.py
│   ├── layers.py               # SparseLinear
│   ├── routers.py              # SETRouter, MagnitudeRouter
│   └── trainer.py              # DST training loop helpers
├── tests/                      # Oracle tests (pytest)
├── examples/                   # Transformer demo notebook
├── environment.yml
├── setup.py
└── README.md
```
