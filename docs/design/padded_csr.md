# Design: Padded-CSR Storage Format

**Status:** Draft — design for Milestone 3
**Owner:** SparseCore project
**Audience:** Contributors, reviewers, future maintainers

---

## 1. What Problem This Solves

Dynamic Sparse Training (DST) requires the sparsity pattern — *which* weights are alive and which are zero — to change every few hundred training steps. Pruning and regrowth.

Standard CSR (Compressed Sparse Row) is the industry-default storage for sparse matrices. PyTorch uses it ([`torch.sparse_csr_tensor`](https://docs.pytorch.org/docs/stable/sparse.html)), SciPy uses it, oneMKL uses it, cuSPARSE uses it. It's a triple: `(crow_indices, col_indices, values)`.

**Standard CSR fails for DST** because inserting a single new non-zero at position `(row=5, col=37)` requires:
- shifting every entry of `col_indices` and `values` past row 5 forward by one slot,
- incrementing every entry of `crow_indices` past index 5 by one.

For a million-parameter layer, each regrowth cycle would shift gigabytes of memory. This is why every existing DST library (rigl, rigl-torch, Cerebras sparsity, SparseML) uses **mask-based dense storage** instead: a full dense weight tensor plus a binary mask. They pay the full dense memory and compute cost to get O(1) mutation.

**Padded-CSR is our attempt to keep CSR's compact representation and fast SpMM kernel while also supporting O(1) insertion within a preallocated capacity budget.**

### 1.1 Expected SpMM performance vs dense

Prior work on x86 CPUs ([Masked Matrix Multiplication, arxiv 2402.14118](https://arxiv.org/html/2402.14118); [SparseDNN, arxiv 2101.07948](https://arxiv.org/abs/2101.07948); [Transformer SpMM, arxiv 2306.16601](https://ar5iv.labs.arxiv.org/html/2306.16601)) has established that a well-written unstructured sparse SpMM beats Intel MKL's dense GEMM at 70%–95% sparsity, with speedups up to 5x. This is the baseline our Apple Silicon port is aiming to replicate in NEON terms.

**Honest caveat:** Apple Silicon's AMX coprocessor (called by `torch.matmul` via Accelerate) delivers approximately 14x the GFLOPS of a pure NEON kernel on dense matmul ([Apple Silicon GEMM benchmarks](https://zhen8838.github.io/posts/mac-amx_en.html)). Our FLOP-count advantage at 90% sparsity is 10x. The arithmetic means our expected crossover sparsity — where sparse SpMM starts beating `torch.matmul` wall-clock time — is probably closer to **95% than 90%** on M-series hardware, higher than the x86 literature suggests.

This is not a failure of the architecture; it's a consequence of Apple's dense coprocessor being exceptionally good. DST research targets sparsity in the 90%–99% range, so our useful operating window (where we beat the dense baseline) is still large and real — but the exact crossover point is an empirical question we answer in Milestone 3d's benchmark. We report it honestly whatever it turns out to be.

## 2. What Padded-CSR Looks Like

At the highest level, a Padded-CSR matrix is a standard CSR structure where **each row has reserved empty slots** for future insertions.

### 2.1 The fields

A Padded-CSR matrix with shape `(nrows, ncols)` stores six fields:

| Field                | Type      | Length                | What it holds |
| -------------------- | --------- | --------------------- | ------------- |
| `values`             | `float32` | `total_capacity`      | Non-zero values + padding slots (zero-valued) |
| `col_indices`        | `int32`   | `total_capacity`      | Column index of each slot; `-1` for padding slots |
| `row_start`          | `int32`   | `nrows`               | Index into `values`/`col_indices` where each row begins |
| `row_nnz`            | `int32`   | `nrows`               | Actual live (non-padding) count in each row |
| `row_capacity`       | `int32`   | `nrows`               | Total allocated slots per row (live + padding) |
| `shape`              | scalar    | —                     | `(nrows, ncols)` |

Layout in memory (row `i` occupies slots `[row_start[i], row_start[i] + row_capacity[i])`):

```
row 0: [v₀₀][v₀₁][v₀₂][ -- ][ -- ]   ← row_nnz=3, row_capacity=5
row 1: [v₁₀][ -- ][ -- ][ -- ]       ← row_nnz=1, row_capacity=4
row 2: [v₂₀][v₂₁][ -- ][ -- ][ -- ]  ← row_nnz=2, row_capacity=5
          ↑                  ↑
          live (col_idx ≥ 0)  padding (col_idx = -1, value = 0)
```

A padding slot is always `(col_idx = -1, value = 0.0)`. This is the sentinel convention.

### 2.2 The invariants

These MUST hold at all times:

1. **Shape:** `nrows ≥ 0`, `ncols ≥ 0`.
2. **Consistency:** `len(values) == len(col_indices) == total_capacity == sum(row_capacity)`.
3. **Row bounds:** `row_start[i+1] == row_start[i] + row_capacity[i]` for `i = 0..nrows-2`; `row_start[0] == 0`.
4. **Row counts:** `0 ≤ row_nnz[i] ≤ row_capacity[i]` for all `i`.
5. **Live slots first:** Within each row, the first `row_nnz[i]` slots are live (valid `col_idx`, real `value`), and the remaining `row_capacity[i] - row_nnz[i]` slots are padding (`col_idx = -1`, `value = 0.0`).
6. **Sorted live columns:** `col_indices[row_start[i] .. row_start[i] + row_nnz[i] - 1]` is strictly sorted ascending and all values are in `[0, ncols)`. (Matches PyTorch CSR convention 5.6 — a `torch.sparse_csr_tensor` invariant required by cuSparse.)
7. **Padding sentinel:** Padding slots' `col_idx == -1` and `value == 0.0`. The SpMM kernel relies on this to skip padding without branching.
8. **Dtype:** values are `float32` (v0.1 scope). Indices are `int32` (enough for 2 billion slots; `int64` is Phase 5+ work for billion-parameter layers).

### 2.3 Design choices explained

**Why `int32` for indices and not `int64`?**
- Our target v0.1 transformer has ≤1M parameters per layer. `int32` spans 2.1 billion. Comfortably sufficient.
- `int32` halves the index-array memory vs `int64`, improving cache utilization for SpMM.
- PyTorch's CSR default is `int32` too ([invariant 1.3](https://pearu.github.io/csr_tensor_invariants.html)). Borrow, don't reinvent.
- When v1.0 supports billion-parameter models, we'll add an `int64` variant behind a template parameter. Not today.

**Why store `row_nnz` and `row_capacity` separately instead of two pointer arrays?**
An alternative standard-CSR-adjacent layout would be `(row_start, row_end_live, row_end_capacity)` — three pointer arrays, derivable by addition. We chose `(row_start, row_nnz, row_capacity)` because:
- `row_nnz` is directly what a user/router asks for ("how many live connections in row i?") — no subtraction needed.
- The most common mutation, "insert one new connection at the end," increments `row_nnz` by 1. Plainly obvious.
- It costs one extra `int32` array of size `nrows`, which is tiny for realistic transformer widths (nrows ≤ 4096).

**Why sentinel `col_idx = -1` for padding instead of, say, `col_idx = ncols`?**
- `-1` is immediately visually distinct from any valid index in code review.
- `< 0` is a single branch check; `>= ncols` requires passing `ncols` everywhere.
- NEON masking is easier: `vcltzq_s32` (compare less-than-zero) is one intrinsic. v0.1 uses the simpler `col_idx >= 0` check; the SIMD optimization comes later.

**Why allocate capacity PER ROW instead of a global padding pool?**
- Per-row capacity means every row has predictable, bounded insertion cost.
- A global pool would need a free-list or compaction pass; both add complexity we don't need.
- The downside: some rows may become "full" (nnz == capacity) and insertion fails. Phase 4's mutation API handles this via row-level resize (see §6).

**Why not use PyTorch's native CSR directly?**
- PyTorch CSR has no capacity concept — every row is tight. We'd have to repack on every insertion. That's exactly the problem we're solving.
- We accept `torch.sparse_csr_tensor` as an *input format* (§5.2) and can emit it as an *output format*. But our internal representation is our own.

## 3. Example Walkthrough

A `3×4` matrix with two non-zeros per row, 1 padding slot per row:

```python
# Logical matrix (what PyTorch would see):
#   [3.0, 0.0, 0.0, 1.0]
#   [0.0, 0.0, 2.0, 5.0]
#   [4.0, 0.0, 0.0, 0.0]

PaddedCSR(
    values       = [3.0, 1.0, 0.0,   2.0, 5.0, 0.0,   4.0, 0.0, 0.0],
    col_indices  = [  0,   3,  -1,     2,   3,  -1,     0,  -1,  -1],
    row_start    = [0, 3, 6],
    row_nnz      = [2, 2, 1],
    row_capacity = [3, 3, 3],
    shape        = (3, 4),
)

# Total capacity    : 9
# Total live nnz    : 5
# Padding overhead  : 4 slots = 44%  (intentionally high for illustration;
#                     typical v0.1 setting is 10-20% padding)
```

## 4. The Public Python API

### 4.1 Class skeleton

```python
class PaddedCSR:
    """Sparse matrix with padded-row CSR storage for O(1) insertion."""

    # ── Construction ──
    @classmethod
    def from_torch_sparse_csr(cls, tensor: torch.Tensor, *,
                              padding_ratio: float = 0.2) -> "PaddedCSR": ...

    @classmethod
    def from_dense(cls, dense: torch.Tensor, *,
                   threshold: float = 0.0,
                   padding_ratio: float = 0.2) -> "PaddedCSR":
        """Live = |w| > threshold. Padding = ceil(nnz * padding_ratio)."""
        ...

    @classmethod
    def random(cls, nrows: int, ncols: int, *,
               sparsity: float, padding_ratio: float = 0.2,
               seed: int | None = None) -> "PaddedCSR": ...

    # ── Read-only accessors ──
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def nnz(self) -> int:
        """Total live non-zeros across all rows."""
        ...
    @property
    def sparsity(self) -> float:
        """Fraction of logical cells that are zero (equiv. 1 - nnz/(nrows*ncols))."""
        ...
    @property
    def total_capacity(self) -> int: ...
    @property
    def padding_slots(self) -> int: ...
    @property
    def padding_ratio(self) -> float: ...

    # ── Interop ──
    def to_dense(self) -> torch.Tensor: ...
    def to_torch_sparse_csr(self) -> torch.Tensor: ...
    def values_view(self) -> torch.Tensor:
        """1-D view of the `values` buffer. For optimizer updates only;
        shape and semantics may change between steps if topology mutates."""
        ...

    # ── Mutation (Phase 4 preview — NOT SHIPPED IN v0.1 MILESTONE 3) ──
    def drop(self, indices: list[tuple[int, int]]) -> None: ...
    def grow(self, indices: list[tuple[int, int]],
             initial_values: list[float] | None = None) -> None: ...
    def resize_row(self, row: int, new_capacity: int) -> None: ...
```

### 4.2 What ships in Milestone 3

Milestone 3 ships construction, accessors, and interop. **Mutation methods are reserved for Phase 4** — the design is already shaped so they plug in naturally, but they're not part of the v0.1 Milestone 3 deliverable.

## 5. Python ↔ C++ FFI Strategy

### 5.1 Ownership

The **C++ side owns** the six fields. The Python `PaddedCSR` class is a thin wrapper (a `py::class_<PaddedCSRImpl>`) that holds a pointer to the C++ object. When the Python object is garbage-collected, the C++ destructor runs and `std::vector` frees everything.

Rationale:
- No ambiguity about lifecycle. Standard RAII.
- Can expose zero-copy `values_view()` as a NumPy array backed by the C++ `std::vector<float>` (stable pointer for the object's lifetime).
- Matches pybind11's canonical class-binding pattern.

### 5.2 Construction from `torch.sparse_csr_tensor`

PyTorch CSR has `(crow_indices, col_indices, values, shape)` with tight rows (no padding). Conversion procedure:

```
Given: crow_indices (len=nrows+1), col_indices (len=nnz), values (len=nnz)

1. Compute: for each row i, tight_nnz_i = crow_indices[i+1] - crow_indices[i]
2. Compute row_capacity[i] = ceil(tight_nnz_i * (1 + padding_ratio)), minimum 1.
3. Compute row_start[i] = sum(row_capacity[0..i-1]).
4. total_capacity = sum(row_capacity).
5. Allocate padded values[total_capacity] = 0.0, col_indices[total_capacity] = -1.
6. For each row i, copy the tight_nnz_i live entries from PyTorch into the first
   tight_nnz_i slots of the i-th padded row. Set row_nnz[i] = tight_nnz_i.
```

This is an O(nnz + nrows) operation, done in Python using NumPy indexing — no C++ kernel needed. Lives in `sparsecore/layout.py`.

### 5.3 Construction from dense `torch.Tensor`

Given a dense float32 tensor `W`:

```
1. Compute mask = |W| > threshold.
2. Derive (crow_indices, col_indices, values) from the mask.
3. Call from_torch_sparse_csr(W.to_sparse_csr(), padding_ratio=...).
```

Implementation note: rather than build CSR ourselves in Python, delegate to `W.to_sparse_csr()` and let PyTorch do the dense→CSR conversion. Then the rest is our code.

## 6. Mutation Semantics (Phase 4 preview)

Documenting these now so §2 invariants cover them; implementation deferred.

**`drop(indices)`**: for each `(row, col)` in `indices`, find the live slot matching that column in that row, mark it padding. Then compact the row: move all remaining live slots to the front so the "live slots first" invariant (§2.2 rule 5) still holds. `row_nnz[row]` decremented by 1 per drop.

**`grow(indices, initial_values)`**: for each `(row, col)`, if the row has padding capacity (`row_nnz[row] < row_capacity[row]`):
- Insert the new entry in sorted position within the live slots.
- Shift subsequent live slots right by one; the rightmost live slot steps into the first padding slot.
- Increment `row_nnz[row]`.

If the row is at capacity (`row_nnz[row] == row_capacity[row]`), the caller must call `resize_row` first. This is an explicit contract choice — we don't auto-resize — because the router code in Phase 4 can batch resize decisions at the end of a mutation cycle rather than one-at-a-time.

**`resize_row(row, new_capacity)`**: reallocate the full matrix (new `values`, `col_indices`, etc.) with updated capacity for the target row. O(total_capacity). Expected to be called at most a few times per training step across all rows, not per-insertion. Phase 4's router design will batch these.

## 7. Testing Strategy

Milestone 3 Oracle tests:

1. **Round-trip identity**: dense → PaddedCSR → dense preserves values exactly.
2. **Round-trip with PyTorch**: `torch.sparse_csr_tensor → PaddedCSR → torch.sparse_csr_tensor` preserves the same logical matrix.
3. **Invariant-holding**: after construction from various inputs, all 8 invariants from §2.2 hold. This is a property-style test running on many seeds.
4. **Padding-ratio handling**: `padding_ratio=0.0` gives a tight CSR (no padding). `padding_ratio=1.0` doubles capacity. `padding_ratio=0.5` gives 1.5x capacity. All produce invariant-correct structures.
5. **Edge cases**: zero-row matrix (`nrows=0`), zero-column matrix, all-zero matrix (`nnz=0`), all-dense row, single-element matrix.
6. **Large inputs**: 4096×4096 at 95% sparsity — confirms no memory corruption, no integer overflow.

Milestone 3 does NOT test:
- SpMM math correctness (that's Milestone 3's later sub-milestone, 3c).
- Mutation correctness (that's Phase 4).
- Performance (that's the Milestone 3 demo, 3d).

## 8. Open Questions

Design questions not resolved in this document, deferred to implementation:

- **Q1:** Should `row_capacity` be a per-row field, or could we assume uniform capacity for all rows (single scalar)? Uniform is simpler but wastes memory when row densities vary. **Leaning: per-row, as specified in §2.1.** Uniform can be recovered by a helper factory that sets all rows to the same value.

- **Q2:** Should `values_view()` return a PyTorch tensor or a NumPy array? **Leaning: PyTorch tensor** — it's what the optimizer expects. NumPy array is available via `.numpy()` on the tensor anyway.

- **Q3:** Do we need a `__repr__` that's useful but doesn't dump all nnz values for large matrices? **Leaning: yes**, modeled on PyTorch's sparse tensor repr.

- **Q4:** What happens if a user asks for `PaddedCSR(..., padding_ratio=0)` and then Phase 4 tries to grow a row? **Decision: raise an informative error.** "Row is at capacity; call `resize_row` or construct with padding_ratio > 0." No silent reallocation.

## 9. Dependencies on Later Phases

The design of this format is directly shaped by Phase 4's needs:
- The existence of padding (§2.1) is FOR Phase 4's grow operation.
- The `row_capacity` field is FOR Phase 4's resize decisions.
- The "sorted live columns" invariant (§2.2 rule 6) is FOR efficient SpMM (Milestone 3c) and for binary search during mutation (Phase 4).

This means: **any change to §2 invariants in the future is a breaking change that affects all of Phase 3 and Phase 4**. Expect this document to be updated carefully, with migration notes for any revision.

## 10. Prior Art Acknowledged

- **PyTorch `torch.sparse_csr_tensor`** ([docs](https://docs.pytorch.org/docs/stable/sparse.html), [invariants](https://pearu.github.io/csr_tensor_invariants.html)) — our base format. We adopt their naming (`col_indices`, `values`) and their index-sorting and column-bound invariants. Differences: they use `crow_indices` (cumulative), we use `row_start + row_capacity` (to express capacity separately from row count).
- **SciPy CSR** ([docs](https://lectures.scientific-python.org/advanced/scipy_sparse/csr_array.html)) — the canonical reference for CSR. Same three-array triple. No padding concept.
- **Cerebras sparsity** ([docs](https://training-api.cerebras.ai/en/latest/wsc/tutorials/sparsity.html)) — mask-based storage internally. We diverge: our actual-sparse engine doesn't materialize masks.
- **Standard academic CSR+padding** (sometimes called "ELLPACK" or "Sliced-ELLPACK" in HPC literature) uses uniform padding across rows for SIMD alignment. We use per-row variable capacity to prioritize insertion flexibility over SIMD alignment — our SpMM kernel handles irregular row lengths via row-wise parallelism (Phase 3d).

---

*Last updated: Milestone 3a, pre-implementation draft. Update this document whenever §2 invariants change.*
