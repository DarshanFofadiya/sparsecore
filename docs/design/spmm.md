# Design: SpMM (Sparse × Dense Matrix Multiplication)

**Status:** Draft — design for Milestone 3c (scalar) and 3d (NEON)
**Owner:** SparseLab project
**Prerequisites:** `docs/design/padded_csr.md` (the storage format this kernel reads)

---

## 1. What This Kernel Does

SpMM computes `Y = W @ X` where:
- `W` is a sparse weight matrix in PaddedCSR format, shape `(M, K)`
- `X` is a dense activations matrix, shape `(K, N)`
- `Y` is a dense output matrix, shape `(M, N)`

This is the **forward pass of every linear layer in a neural network**. When Phase 4 wraps SparseLab into `SparseLinear(nn.Module)`, its `forward()` will call this kernel.

### 1.1 Why not sparse-sparse?

We explicitly target sparse-weight × dense-activations, not sparse-sparse. Reasons:
- **Transformer weights are sparse; activations are dense.** That's the training workload we care about.
- **Sparse-sparse matmul (SpGEMM)** is a substantially harder algorithm with different performance characteristics. Out of v0.1 scope.
- **Dense output.** Downstream layers expect dense input; materializing a dense Y avoids a second conversion.

### 1.2 Why not dense-dense?

We deliberately don't ship a dense matmul. Our competitive claim is at the sparse end; dense is PyTorch's job (it has Accelerate + AMX on Apple Silicon). Ship only what we're uniquely positioned to do well.

## 2. The Python API

```python
from sparselab import PaddedCSR, spmm

W = PaddedCSR.from_dense(torch.randn(1024, 512) * mask)  # sparse (M, K)
X = torch.randn(512, 128, dtype=torch.float32)           # dense   (K, N)
Y = spmm(W, X)                                           # dense   (M, N)
```

**Contract:**
- `W.shape == (M, K)`
- `X.shape == (K, N)` and `X.dtype == float32` and `X` is contiguous
- Returns a new `torch.Tensor` of shape `(M, N)`, dtype `float32`, CPU
- Input `W` and `X` are NOT mutated

**Error conditions:**
- Shape mismatch (`W.ncols != X.shape[0]`) → `ValueError`
- `X` not 2-D → `ValueError`
- `X` not float32 → auto-converted via pybind11 `forcecast` (consistent with vector_dot)
- `W` not a `PaddedCSR` → Python's usual `TypeError` (no special handling needed)

## 3. The Core Algorithm (Scalar Version, Milestone 3c)

In plain pseudocode:

```
Y[i, :] = 0  for all i
for i in 0..M-1:                                  # iterate output rows
    start = W.row_start[i]
    n_live = W.row_nnz[i]
    for s in 0..n_live-1:                         # iterate LIVE slots of row i
        c = W.col_indices[start + s]
        v = W.values[start + s]
        for j in 0..N-1:                          # accumulate contribution to Y[i]
            Y[i, j] += v * X[c, j]
```

### 3.1 Loop ordering: why i → s → j?

Three possible loop orders give the same mathematical answer but radically different performance:

| Order | Memory access for W | Memory access for X | Memory access for Y |
| ----- | ------------------- | ------------------- | ------------------- |
| **i → s → j** (chosen) | sequential reads of one row of W | gathered reads of one row of X per slot | sequential writes of one row of Y |
| i → j → s | same-row W reads repeated N times | sequential writes, reread per j | sequential writes |
| j → i → s | gathered access to W | sequential access by column | gathered access |

**We pick i → s → j** because:
1. **W is read row-by-row**, matching how it's stored in PaddedCSR (rows are contiguous).
2. **Y is written row-by-row**, matching how a caller will consume the output dense matrix.
3. **X is accessed row-by-row** (once per live slot), which means **each row of X gets reused across all `j` values for that slot**. This is the cache-friendly key: an `X` row of 128 floats fits in a single cache line region and stays hot through the inner loop.

This is the standard CSR SpMM loop order. Every reference implementation (oneMKL, cuSPARSE, SparseDNN) uses this or a tile-blocked variant of it.

### 3.2 What Milestone 3d's NEON version changes

The inner `j` loop becomes SIMD-vectorized. Instead of scalar `Y[i, j] += v * X[c, j]`:

```
for j in 0..N-1 step 4:
    y_vec = vld1q_f32(&Y[i, j])            # load 4 floats of Y
    x_vec = vld1q_f32(&X[c, j])            # load 4 floats of X[c]
    y_vec = vfmaq_f32(y_vec, vdupq_n_f32(v), x_vec)   # Y += v * X
    vst1q_f32(&Y[i, j], y_vec)             # store 4 floats of Y
# scalar remainder loop for j not divisible by 4
```

The outer `i` and middle `s` loops stay scalar. Only `j` gets vectorized, because:
- `j` is the longest loop (N could be thousands; nnz per row is dozens).
- `j` accesses contiguous memory in X and Y — perfect for SIMD loads.
- `s` accesses scattered `c = W.col_indices[s]` — NEON scatter/gather is expensive, not worth it for v0.1.

## 4. Data Types and Layout

- **Input `X`:** `float32`, C-contiguous (row-major), 2-D, shape `(K, N)`
- **Output `Y`:** `float32`, C-contiguous, 2-D, shape `(M, N)` — allocated by the kernel
- **Weight `W`:** `PaddedCSR` already constrains dtypes (values=float32, indices=int32)

Alignment: float32 arrays are naturally 4-byte aligned by `std::vector<float>` / NumPy's default allocator. NEON's `vld1q_f32` requires only 4-byte alignment (not 16-byte like some AVX paths), so we're fine with no special allocator work.

## 5. C++ Function Signature

```cpp
namespace sparselab {

// Scalar reference implementation (Milestone 3c).
// Output must be pre-allocated with shape (M, N) and pre-zeroed.
void spmm_scalar(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,    // X is (K, N), row-major
    float* Y                                  // Y is (M, N), row-major
);

// NEON SIMD implementation (Milestone 3d).
void spmm_simd_neon(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
);

}  // namespace sparselab
```

Note the kernel takes pre-zeroed output memory — the Python binding is responsible for allocation. This keeps the kernel focused on math.

## 6. Python Binding Strategy

The binding in `csrc/bindings.cpp` will:

1. Accept `PaddedCSR` and a `torch.Tensor`-compatible `X` (unwrapped via pybind11's `py::array_t<float>` with `c_style | forcecast`).
2. Validate shapes (`W.ncols == X.shape[0]`, `X.ndim == 2`).
3. Allocate output `Y` as a fresh numpy array of shape `(M, N)`, zero-initialized.
4. Call `sparselab::spmm_scalar(...)` or `sparselab::spmm_simd_neon(...)`.
5. Return `Y` wrapped as a NumPy array. Python-side, the user converts back to torch via `torch.from_numpy(Y)` if desired — or we can auto-convert at the public API level.

**Public Python surface** (in `sparselab/__init__.py`):

```python
def spmm(W: PaddedCSR, X: torch.Tensor, *, kernel: str = "auto") -> torch.Tensor:
    """
    Sparse-dense matrix multiply: Y = W @ X.

    Args:
        W: sparse weight, shape (M, K), PaddedCSR
        X: dense input, shape (K, N), float32 torch.Tensor
        kernel: "scalar" | "neon" | "auto" (default)
                "auto" picks the fastest available for the current platform.

    Returns:
        Y: dense output, shape (M, N), float32 torch.Tensor
    """
```

The `kernel` parameter is our forward-compatible way to expose both implementations. Users and benchmarks can pick; default is `auto` which picks NEON on Apple Silicon and scalar elsewhere.

## 7. Testing Strategy

Milestone 3c Oracle tests (`tests/test_spmm.py`):

### Group 1: Oracle correctness (the core claim)

Parametrized over:
- **Shapes (M, K, N):** `(1,1,1)`, `(8,8,8)`, `(15,31,47)` (primes, awkward sizes), `(64,64,64)`, `(256,128,64)` (realistic mini-transformer shapes)
- **Sparsities:** `0.5`, `0.8`, `0.9`, `0.95`, `0.99`
- **Seeds:** a few seeds per combination

For each combination, assert `spmm(W, X)` ≈ `torch.matmul(W.to_dense(), X)` within `rtol=atol=1e-5`.

### Group 2: Edge cases

- **Empty sparsity:** `spmm(all_dense_W, X)` still works (nnz = M*K, no padding).
- **Fully zero W:** `spmm(zero_W, X) == zeros(M, N)`.
- **Fully zero X:** `spmm(W, zero_X) == zeros(M, N)`.
- **Fully zero row of W:** That row of Y is all zeros.
- **Shape mismatch:** raises ValueError.
- **1-D X:** raises ValueError.

### Group 3: Scalar vs NEON equivalence (when 3d lands)

Same @pytest.mark.parametrize("kernel", ["scalar", "neon"]) pattern as vector_dot tests. Each test runs against both kernels; both must match Oracle.

### Group 4: Zero-copy behavior

- `spmm` does not mutate `W.values`, `W.col_indices`, or `X`.
- Returned `Y` is a fresh tensor, independent of inputs.

### Group 5: Dtype coercion

- `X` passed as `float64` is silently downcast to `float32` (consistent with `vector_dot`).
- `X` passed as `float16` also auto-cast (matches pybind11's `forcecast` behavior).

## 8. Performance Expectations

### Scalar (Milestone 3c)

Not optimized. Will be **slower than `torch.matmul`** at all sparsities. That's fine — correctness is 3c's sole deliverable.

Rough envelope for `M=1024, K=512, N=128` at 90% sparsity:
- `torch.matmul(W.to_dense(), X)`: ~0.1 ms (Apple AMX)
- Our `spmm_scalar`: ~5–15 ms (single core, unvectorized)
- Difference: ~50×–150×. Big but unsurprising for a scalar reference.

### NEON (Milestone 3d)

The interesting benchmark. See `docs/design/padded_csr.md §1.1` for our honest crossover expectations: around **95% sparsity** we expect to match or beat `torch.matmul` on Apple Silicon. At 99% sparsity, we should win decisively.

If 3d comes in substantially worse than this (say, losing at 99% sparsity), that's a bug in the kernel, not a failure of the architectural bet.

## 9. Open Questions Deferred to Implementation

- **Q1: Zero-initialize Y outside or inside the kernel?** Outside is cleaner (kernel does only math). Inside is slightly faster (one fewer pass over memory). Leaning: **outside, done by the Python binding via numpy's default-zero allocation.**

- **Q2: Should we parallelize the `i` loop with OpenMP in 3c or wait until 3d?** OpenMP setup on macOS needs `brew install libomp`. Nothing about 3c's scalar algorithm is thread-hostile — the `i` loop is embarrassingly parallel. Leaning: **add OpenMP in 3d alongside NEON, not in 3c.** Keep 3c minimal.

- **Q3: Support `float16` / `bfloat16`?** Both are widely requested for ML workloads. NEON has float16 intrinsics on Apple Silicon. Leaning: **v0.1 is float32-only** to keep scope tight; float16/bfloat16 is a clean post-launch contribution.

- **Q4: Is the output tensor ever expected to be non-zero on input (accumulate-into)?** Some BLAS GEMM APIs take `Y = alpha * W @ X + beta * Y`. Leaning: **v0.1 ships `Y = W @ X` only**; the `alpha/beta` variant is a simple extension if needed later.

## 10. Dependencies on Later Phases

- **Phase 4 router** will use SpMM's output shape and semantics. Our `spmm(W, X)` contract must remain stable once Phase 3 ships.
- **Backward pass** (Phase 4) also needs a transposed variant: given output gradients, compute gradients w.r.t. `X`. That's `spmm_transpose` (`Y = Wᵀ @ G`). Different algorithm, different loop order; separate kernel file, scheduled for Phase 4.

## 11. Prior Art Acknowledged

- **Intel MKL `mkl_sparse_s_mm`** — the x86 reference for unstructured CSR × dense. Same i→s→j loop order.
- **cuSPARSE `cusparseSpMM`** — GPU equivalent. Different tiling but same algorithmic structure.
- **LIBXSMM sparse kernels** — hand-tuned code-generated SpMM for small matrices.
- **Masked Matrix Multiplication (arxiv 2402.14118)** — our direct competitive reference for unstructured CPU SpMM at 60-95% sparsity.
- **Transformer SpMM (arxiv 2306.16601)** — shows 5x speedup over oneDNN dense GEMM at 70-90% sparsity on x86. The claim we're trying to replicate in NEON terms.

---

*Last updated: Milestone 3c pre-implementation draft.*
