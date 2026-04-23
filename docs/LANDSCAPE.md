# The Sparse Neural Network Landscape

*A survey of what exists, what doesn't, and where SparseLab fits.*

This document exists so that contributors, users, and skeptics can quickly understand why SparseLab was built and what it is not. **We read the landscape honestly.** Existing work is respected, credited, and positioned relative to our mission.

If you find this catalog incomplete or inaccurate, please open a PR. Keeping this document current is a first-class community responsibility.

---

## The Four Quadrants of Sparse ML

We split the ecosystem along two axes:

|                          | **Post-training (prune then deploy)** | **Training-from-scratch**     |
| ------------------------ | ------------------------------------- | ----------------------------- |
| **Structured sparsity**  | torchao, TensorRT, BLaST              | Condensed Sparsity, EcoSpa    |
| **Unstructured sparsity**| SparseML, DeepSparse                  | rigl, rigl-torch, SparseLab  |

SparseLab lives in the bottom-right cell — **unstructured, training-from-scratch** — and this quadrant is the least served by production-quality tooling today.

---

## Projects Worth Knowing

### Neural Magic — SparseML + DeepSparse
- **Repos:** [neuralmagic/sparseml](https://github.com/neuralmagic/sparseml), [neuralmagic/deepsparse](https://github.com/neuralmagic/deepsparse)
- **What it does:** Post-training pruning, quantization, distillation (SparseML). CPU inference runtime that exploits unstructured sparsity (DeepSparse).
- **Why it's not us:** Inference-first, pruning-after-training workflow. The company was acquired by Red Hat in early 2025 and the team migrated to vLLM; these repos are now in maintenance mode.
- **What we learn from it:** They proved unstructured sparse inference on CPU can beat dense. Their recipe format, HuggingFace integration patterns, and configuration-as-YAML design are worth studying.
- **The opening they leave:** The training-from-scratch story was never their focus, and they've vacated the CPU-sparse narrative.

### Google Research — RigL
- **Repo:** [google-research/rigl](https://github.com/google-research/rigl)
- **What it does:** Reference implementation of the RigL algorithm (Rigging the Lottery, ICML 2020). TensorFlow/JAX.
- **Why it's not us:** Research artifact, not a library. Uses dense-mask simulation — measures what sparse training *could* do, not what it *costs*.
- **What we learn from it:** The RigL algorithm itself. Our Router interface must express this cleanly so it plugs into SparseLab without rewriting.

### rigl-torch
- **Repo:** [nollied/rigl-torch](https://github.com/nollied/rigl-torch)
- **What it does:** Community PyTorch port of RigL. "2 lines of code to add RigL to your project."
- **Why it's not us:** Still dense-simulated under the hood. Designed for a single algorithm rather than a pluggable router. Low maintenance activity.
- **What we learn from it:** Minimal user code footprint is the right ergonomic target. We should match or beat it.

### Cerebras — cerebras.pytorch.sparse
- **Docs:** [Cerebras Sparsity](https://training-api.cerebras.ai/en/latest/wsc/tutorials/sparsity.html)
- **What it does:** Cerebras's PyTorch sparse training API for their wafer-scale CS-2/CS-3 hardware. Ships the same algorithm catalog we plan — Static, GMP, SET, RigL — with a clean `SparsityAlgorithm` base class and composable per-parameter schedules. The most mature sparse-training API in the industry today.
- **Why it's not us (the subtle but important difference):** Cerebras stores weights *densely* and applies a binary mask before forward and after backward (from their docs: "mask tensor is multiplied inplace to the original dense parameter before forward and to the gradients after backward"). This is the correct choice for their wafer-scale chip, which schedules dense matmuls in SRAM extremely fast. **But it means their "sparsity" is simulated** — every forward pass still does the full dense matmul, every backward still allocates the full dense gradient. Memory and FLOPs are dense even when the math is zero.
- **Why that matters for us:** On CPU hardware, the dense+mask approach is actively wrong. A 7B-parameter dense model needs ~14GB just to hold weights at float16 — impossible on a 32GB MacBook. At 90% sparsity with *true* sparse storage (what we build), that's ~1.4GB. The same approach that works for Cerebras on their own chip would make the library unusable on commodity hardware.
- **What we adopt (with credit):** Their API is the industrial-quality reference. We will borrow the `SparsityAlgorithm` base class shape, their algorithm catalog, their per-parameter composability, and their schedule abstraction — see `borrow-dont-reinvent.md`. The credit will be explicit in docstrings.
- **What we diverge on:** The storage substrate. Cerebras: dense tensor + binary mask (right for WSE). SparseLab: Padded-CSR genuinely sparse (right for CPU/MacBook).
- **Positioning:** Cerebras and SparseLab are complementary, not competitive. A researcher prototypes a new DST algorithm on SparseLab on their laptop; when they're ready for a production run at 100B+ parameters, they deploy to Cerebras. We are the sandbox; they are the scale system.

### HuggingFace — pytorch_block_sparse
- **Repo:** [huggingface/pytorch_block_sparse](https://github.com/huggingface/pytorch_block_sparse)
- **What it does:** Block-structured sparse linear layer, drop-in replacement for `nn.Linear`.
- **Why it's not us:** Block-structured, CUDA-only, last meaningful commit years ago.
- **What we learn from it:** The `BlockSparseLinear(nn.Module)` API pattern. Also an object lesson: sparsity libraries without an active community stall.

### PyTorch core — `torch.sparse` + torchao
- **Docs:** [torch.sparse](https://pytorch.org/docs/main/sparse.html), [pytorch/ao](https://github.com/pytorch/ao)
- **What it does:** Native CSR/COO tensor types in PyTorch; `torchao.sparsity` provides semi-structured (2:4) GPU sparsity.
- **Why it's not us:** `torch.sparse` training support is incomplete; torchao is structured-2:4, GPU-only, post-training.
- **What we learn from it:** We should be interoperable with `torch.sparse` formats where sensible. Not every sparse representation needs to be custom.
- **What we complement:** We explicitly own the unstructured + CPU + training-from-scratch corner they don't address.

### Tim Dettmers — sparse_learning
- **Repo:** [TimDettmers/sparse_learning](https://github.com/TimDettmers/sparse_learning)
- **What it does:** Research code for sparse momentum and related DST algorithms.
- **Why it's not us:** Research artifact, low maintenance.
- **What we learn from it:** Historical context; Dettmers' writing on LLM sparsity is essential background reading.

### Active Research Papers (2024–2025)

The sparse-transformer-pretraining space is unusually active. Papers published in the last 18 months:

- **BLaST** ([arxiv:2507.03117](https://arxiv.org/html/2507.03117v2), July 2025) — Block and Sparse Transformers, iterative sparsification for pretraining
- **MST** ([arxiv:2408.11746](https://arxiv.org/html/2408.11746v1), August 2024) — 4× FLOP reduction via DST with sparsity variation
- **Condensed Sparsity** ([ICLR 2024](https://github.com/calgaryml/condensed-sparsity)) — DST with structured sparsity
- **EcoSpa** ([arxiv:2511.11641](https://arxiv.org/html/2511.11641v1), November 2025) — Coupled structured sparse training
- **SLTrain** ([NeurIPS 2024](https://github.com/andyjm3/SLTrain)) — Sparse + low-rank pretraining
- **Learned Shuffles for DST** ([arxiv:2510.14812](https://arxiv.org/html/2510.14812v1), October 2025)
- **DST for Deep RL** ([arxiv:2510.12096](https://arxiv.org/html/2510.12096v1), October 2025)

Each paper reinvents DST scaffolding from scratch. Most don't ship usable libraries. **SparseLab exists to break this reinvention loop** by providing the scaffolding every DST paper needs so researchers can focus on their algorithm, not their plumbing.

---

## Adjacent but Not Us

- **facebookresearch/SparseConvNet** — Submanifold sparse convolutions (geometric sparsity, not weight sparsity). Archived 2025.
- **rusty1s/pytorch_sparse** — Sparse ops for graph neural networks. Different domain.
- **dvlab-research/SparseTransformer** — Sparse attention for variable-length inputs. Token-sparsity, not weight-sparsity.
- **LibXSMM, Intel oneMKL, OpenBLAS sparse** — HPC-grade SpMM kernels. Fast, but no autograd, no ML integration, no training loop.

---

## Where SparseLab Fits

Reading the landscape honestly:

- **"First sparse-from-scratch training library"** — ❌ False. rigl-torch, Tim Dettmers' sparse_learning, and several research repos predate us.
- **"First CPU sparse training engine"** — ⚠️ Partial. Neural Magic owned CPU sparse for years (inference); nobody has productionized CPU sparse *training*.
- **"First drop-in `nn.Linear` replacement"** — ❌ False. HuggingFace did it in 2020 (block-structured).
- **"First actually-sparse, unstructured, training-from-scratch, CPU-native engine with a pluggable router interface for DST algorithms"** — this is the specific corner we're claiming. We're not aware of an equivalent library at launch, but we'd welcome being corrected via a GitHub issue — keeping this catalog honest is part of the community hygiene we're trying to model.

**The real opening we're filling:** Sparse-from-scratch is an active 2024-2025 research area with a dozen published methods but no canonical, production-quality, hackable library that researchers default to. Every paper reinvents the DST scaffolding. The community is starving for a clean "PyTorch for sparse training" that all these methods can plug into.

## Our Moats (what makes us actually work)

1. **Padded-CSR data layout** — O(1) insertion of grown connections during training. Standard CSR can't do this.
2. **Native Apple Silicon NEON kernels** — the best-tuned sparse training kernels for Apple Silicon we're aware of. Researchers prototype on MacBooks; we make that fast.
3. **Pluggable `SparsityAlgorithm` API** — RigL, SET, Sparse Momentum, and future papers can be expressed as short Python subclasses (typically ~50 lines of real logic, ~200 with docs). No C++ contributions required from users.
4. **True sparse storage** — we don't compute dense gradients and mask them. We compute only what's needed. This is the actual FLOP and memory saving; the dense+mask approach that works on wafer-scale chips is actively the wrong choice on CPU.

## SparseLab vs Cerebras — The One Table

The most useful comparison for readers already familiar with Cerebras:

| Dimension                      | Cerebras `cstorch.sparse`         | SparseLab                      |
| ------------------------------ | --------------------------------- | ------------------------------- |
| Algorithm catalog              | Static, GMP, SET, RigL            | Static, SET, RigL (v0.1)        |
| API design quality             | Production-hardened, composable   | We adopt theirs                 |
| Storage substrate              | Dense tensor + binary mask        | Padded-CSR (genuinely sparse)   |
| FLOPs at 90% sparsity          | 100% (mask after dense matmul)    | ~10% (only live connections)    |
| Memory at 90% sparsity         | 100% (dense weight tensor)        | ~10% (sparse storage)           |
| Target hardware                | Cerebras CS-2/CS-3 wafer          | Apple Silicon, ARM, x86         |
| Installable on a MacBook       | No                                | `pip install sparselab`        |
| Primary audience               | Customers with Cerebras contracts | Open-source research community  |

**The mental model:** Cerebras is the right choice if you have a wafer and need to productionize a trained model. SparseLab is the right choice if you're a researcher iterating on a new DST algorithm on your laptop.

## Why a Researcher in 2026 Chooses SparseLab

SparseLab is a PyTorch-native, actually-sparse DST library. Write
your next drop/grow rule as a short `SparsityAlgorithm` subclass;
it runs against real Padded-CSR storage and real sparse kernels, not
a mask-on-dense simulation. Built CPU-first and Apple-Silicon-first,
so the iteration loop stays on your laptop instead of a rented GPU.

It's not faster than dense on CPU — honest about that in the README.
The value is **memory footprint** (~18% of dense at 90% sparsity),
**pluggability** (no need to reinvent the DST scaffolding for each new
algorithm), and **accessibility** (no GPU required to prototype).

---

## Our Policy: Borrow, Don't Reinvent

Every project listed above has already solved problems we will hit. Before designing any public API, schedule format, checkpoint layout, or algorithm signature, **we study the equivalent in Cerebras / Neural Magic / torchao / rigl-torch first**, and either adopt their design (with credit) or write a one-sentence justification for why we diverge.

This is engineering humility, not laziness. Our moats are the Padded-CSR layout and the Apple Silicon NEON kernels — everything else should match community conventions.

Contributions that propose new API patterns are welcome, but expected to include a "prior art" section explaining what existing project they considered and why this design is preferred.

---

## Open questions & deferred experiments

Things we've noticed but haven't answered yet. Each is a candidate for a
future demo or benchmark in its own right.

### Sparse 90% vs narrow dense at matched parameter count

Our demo 8 shows sparse 784→512→10 @ 90% hits 97.45% on MNIST for
~40k live params in the first layer. A "narrow dense" version —
784→51→10 with ~40k params and every neuron fully connected — would
have the same parameter budget but a fundamentally different
computational graph:

- Sparse: 512 output neurons, each seeing ~10% of input features
  (a *random subset*, different per neuron).
- Narrow dense: 51 output neurons, each seeing *all* input features.

The hypothesis (backed by the Lottery Ticket Hypothesis and Liu et al.
2019, "Sparse Networks from Scratch") is that sparse beats narrow
dense at matched params because 512 sparse specialists can cover
more hypothesis space than 51 dense generalists — even though both
have the same FLOP budget. That hypothesis has good theoretical support
but deserves a concrete SparseLab-native measurement before we rely
on it in the launch blog.

Estimated cost: ~10 minutes of wallclock (one extra training run of
the narrow model) plus a few lines of demo code. Deferred for now so
we can focus on shipping `SparseLinear` (milestone 4b) and OpenMP
parallelization (milestone 4c). Tracked as a follow-up; will get its
own demo when it lands.

### Adaptive-sparsity DST (no fixed nnz budget)

RigL and SET both hold `nnz` constant — the total number of live
connections per layer never changes. That's a modeling choice, not
a truth. Different layers probably want different sparsity levels
(attention heads might want 50% sparsity; FFN blocks might want 95%).
A fixed-budget approach can't discover this.

A natural SparseLab-native variant: at each update, drop
low-magnitude weights as usual, but **grow every position where
|dL/dW| exceeds a learned or scheduled threshold**, rather than a
fixed top-K. Let `nnz` drift up when the loss landscape asks for
more connections, down when existing ones are wasted.

Two ways to parameterize this, both viable plugins of our Router API:

- **Threshold-driven:** grow wherever |G| > τ, where τ is scheduled
  or learned.
- **Regularizer-driven:** add a small L1 penalty on the weight values
  that pushes the optimizer to naturally zero-out unused connections;
  grow at top-|G| positions. The L1 penalty and RigL's grow criterion
  combine into self-balancing sparsity.

Neither is in the DST literature under our exact formulation — there
are adjacent ideas (SWAT, movement pruning) but none map cleanly
onto the "pluggable DST algorithm" shape.

This is exactly the kind of v0.2+ experiment our Router API is
designed to serve: a researcher implements it as a ~100-line
`SparsityAlgorithm` subclass without touching kernels. Great
candidate for the first community contribution after launch.

Estimated cost: ~1 day to prototype, maybe a week to tune. Out of
scope for v0.1 but tracked so it doesn't slip.
