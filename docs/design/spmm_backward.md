# Design Doc — SpMM Backward Pass (Milestone 4a)

## Purpose

This document specifies how SparseLab computes gradients for the
sparse-dense matrix multiply `Y = W @ X`, where `W` is a `PaddedCSR`
and `X` is dense. Everything in this doc is math + API spec; kernels
and tests reference this doc, not each other.

Milestone 4a turns our forward-only `sparselab.spmm` into a full
autograd-compatible operation. After 4a lands, a user can call
`loss.backward()` on a computation that includes a sparse layer.

---

## 1. The Math

### 1.1 Forward

Let `W ∈ R^{M×K}` be a sparse weight, `X ∈ R^{K×N}` be a dense input,
and `Y ∈ R^{M×N}` be the dense output:

```
Y = W @ X
Y[i, j] = sum over k of W[i, k] * X[k, j]
```

In our storage, `W` has exactly `nnz` non-zero entries, each at some
`(i, k)` pair. The forward kernel walks only those `nnz` entries.

### 1.2 Upstream gradient

During backprop, PyTorch hands us `dL/dY ∈ R^{M×N}` (always dense).
This is a full matrix of partial derivatives of the loss with respect
to every entry of `Y`.

We need to produce two outputs:

1. **`dL/dX ∈ R^{K×N}`** — gradient with respect to the dense input.
   Same shape as `X`, dense.
2. **`dL/dW`** — gradient with respect to the sparse weight. Here we
   have a choice of representation, and this choice is the central
   design decision of this milestone.

### 1.3 Chain rule — dL/dX

Applying the chain rule (see any autograd reference, e.g. Goodfellow
*Deep Learning* §6.5):

```
dL/dX = Wᵀ @ dL/dY
```

This is a matrix-matrix product:
- `Wᵀ` is `K×M`, still sparse (transposing a sparse matrix keeps its
  non-zero count identical).
- `dL/dY` is `M×N`, dense.
- `dL/dX` is `K×N`, dense.

**This is another SpMM.** We reuse our existing forward kernel — no
new math. All we need is the ability to transpose a `PaddedCSR`.

### 1.4 Chain rule — dL/dW

The "textbook" derivation gives:

```
dL/dW = dL/dY @ Xᵀ
dL/dW[i, k] = sum over j of dL/dY[i, j] * X[k, j]
```

This is an `M×N @ N×K = M×K` matmul. Materialized densely, this
produces a full `M×K` tensor, which is exactly the dense-simulated
approach Cerebras uses and that we explicitly reject.

**The critical observation:** in our setting, `W` has only `nnz` live
parameters. The optimizer only updates those `nnz` weights. The zero
slots of `W` are not parameters in the calculus sense — they have no
associated `requires_grad`, they are not trainable. So we only need
`dL/dW` at positions where `W` is live.

Concretely, for each live slot `s` of `W` — pointing at `(i, k)` with
value `v` — we compute:

```
dL/dW_values[s] = dL/dY[i, :] · X[k, :]     # scalar: dot product of two N-vectors
```

**This is the whole insight.** One dot product per live slot. Total
work: `O(nnz * N)`. Total memory: `O(nnz)` floats for the gradient.

Returned shape: a 1-D array of length `nnz`, aligned with the
`values` array of `W`.

---

## 2. Why "Grad at live slots only" is correct

Someone reading this might worry: "what about gradients that *would*
exist at dormant positions — don't we need those for the DST regrow
step?"

**Yes, we do — but at a different time and through a different API.**

Two levels of gradient in DST:

| Gradient                         | When it's needed          | How we compute it            |
|----------------------------------|---------------------------|------------------------------|
| `dL/dW` at live slots            | Every training step       | Our new `spmm_grad_W` kernel |
| `dL/dW` at dormant slots         | Only during regrow (e.g. every 100 steps) | Dense backward, computed on demand by the Router |

The `torch.autograd.Function.backward` call only cares about gradients
for the *current parameters being tracked*. For a `SparseLinear` layer
in our design, the parameters are exactly the `nnz` live values —
the dormant cells are not registered as parameters, so autograd
correctly asks only for `dL/dW_values`.

The regrow-time dense gradient is a **separate operation** we expose
on the Router, not something autograd handles (Milestone 4d/4e).

This separation is the single most important design decision in this
doc. It's what separates SparseLab from dense-simulated libraries.

---

## 3. Representation choices

### 3.1 How `dL/dW` flows back to the user

PyTorch's autograd expects `ctx.backward()` to return gradients that
match the saved forward inputs by position. Our forward was:

```python
Y = SparseSpMM.apply(W_values, col_indices, row_start, row_nnz, X)
```

— where we deliberately pass `W_values` as a `torch.Tensor` (the
thing autograd tracks) rather than the whole `PaddedCSR` object (which
it doesn't know how to differentiate). The index arrays go through as
non-differentiable metadata.

Then `backward` returns:
- `dL/dW_values` — 1-D Tensor, length nnz, same layout as `W.values`
- `None` for each index array (non-differentiable)
- `dL/dX` — 2-D Tensor, shape `(K, N)`

This pattern is directly inspired by PyTorch's own
`torch.sparse.mm` backward and by torchao's
`SparseSemiStructuredTensor` — see
[pytorch/ao/sparsity](https://github.com/pytorch/ao/tree/main/torchao/sparsity)
for their implementation of the same idea for 2:4 sparsity.

### 3.2 What PaddedCSR.transpose produces

Transposing `W: (M, K)` into `Wᵀ: (K, M)` is a well-known CSR
operation. Algorithm (standard, from every sparse-matrix textbook):

1. Count per-column nnz in `W` (becomes per-row nnz in `Wᵀ`).
2. Prefix-sum the counts to get `Wᵀ.row_start`.
3. Walk `W`'s entries in order, placing each `(i, k, v)` into `Wᵀ` at
   position `(k, i, v)` using the prefix-sum offsets.

This takes `O(nnz + M + K)` time and allocates a new PaddedCSR. We
expose it as a method: `Wt = W.transpose()`.

For `dL/dX`, we can actually avoid materializing `Wᵀ` entirely by
writing a specialized kernel that iterates over `W`'s rows and
accumulates into `dL/dX` transposed. But that's a post-4a
optimization — for the first cut, materialize `Wᵀ` and reuse our
forward kernel. Same math, clearer code, easier to verify.

---

## 4. API spec

### 4.1 C++ kernel signatures

```cpp
namespace sparselab {

// dL/dW at live slots only.
//   W          — the sparse weight (only col_indices, row_start,
//                row_nnz are read; values are NOT needed)
//   dY         — upstream gradient, dense (M, N) row-major
//   X          — original forward input, dense (K, N) row-major
//   K, N       — shape params (K must equal W.ncols)
//   dW_values  — output buffer, length W.nnz(). Written in live-slot order.
void spmm_grad_W(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
);

// dL/dX. Computed as Wᵀ @ dY, but reuses spmm_simd internally.
//   WT    — the already-transposed sparse weight
//   dY    — upstream gradient (M, N)
//   M, N  — shape params
//   dX    — output buffer, dense (K, N), M*N floats
void spmm_grad_X(
    const PaddedCSR& WT,
    const float* dY, int64_t M, int64_t N,
    float* dX
);

}  // namespace sparselab
```

Both kernels self-zero their output buffers (same contract as
`spmm_scalar` — the caller doesn't need to pre-zero).

### 4.2 Python autograd wrapper

```python
class SpMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W_values, W, X):
        # W is the PaddedCSR as a non-differentiable object
        # W_values is the same data as W.values, but as a Tensor
        #   that autograd tracks
        ctx.save_for_backward(W_values, X)
        ctx.W = W
        return sparselab.spmm(W, X)  # our existing forward

    @staticmethod
    def backward(ctx, dY):
        W_values, X = ctx.saved_tensors
        W = ctx.W
        # Compute dX = Wᵀ @ dY
        WT = W.transpose()
        dX = spmm_grad_X_impl(WT, dY)
        # Compute dW_values = per-live-slot dot products
        dW_values = spmm_grad_W_impl(W, dY, X)
        return dW_values, None, dX  # None for W (non-diff)
```

### 4.3 Public API

No new public function for 4a — users never call the autograd function
directly. Instead, the dispatch in `sparselab.spmm` gets a check:
"if either input has `requires_grad`, route through the autograd
wrapper." This is how `torch.sparse.mm` works; we match it.

Milestone 4b wraps this in a `SparseLinear(nn.Module)` class, and
that's the surface users actually see.

---

## 5. Correctness strategy

### 5.1 Oracle 1 — analytical gradcheck

`torch.autograd.gradcheck` is the gold standard: it compares our
backward against a finite-differences numerical gradient. If we
implement `backward` wrong, gradcheck fails within `rtol=1e-3` on
small random inputs.

**This is the most important test in all of milestone 4a.** Other
tests are supplementary.

### 5.2 Oracle 2 — reference computation vs dense

For small problem sizes (e.g. 16×16 @ 16×8 at 50% sparsity), we can
compute the "what dense autograd would have given us" reference and
sanity-check our sparse backward against it:

```python
# Dense reference
W_dense = W.to_dense().requires_grad_(True)
Y_ref = W_dense @ X
loss = (Y_ref * upstream_grad).sum()
loss.backward()
# W_dense.grad is the full dense gradient

# Our sparse backward produces dL/dW_values aligned with W.values
# For each live slot (i, k):
#     dW_values[s] should match W_dense.grad[i, k]
# For each dormant position (i, k):
#     W_dense.grad[i, k] may be non-zero, but we don't track it
```

This test verifies we're computing the right gradient at live
positions. It does NOT tell us whether dormant gradients are "correct"
— they're not our concern until regrow.

### 5.3 Edge cases to cover

- Empty `W` (nnz = 0): `dW_values` should be length-0, `dX` should be zero
- Empty rows in `W`: contribute zero to `dX` for their `i`
- `N = 1` (column vector upstream): no SIMD in inner loop
- `N` not a multiple of 4: exercises tail paths in any SIMD version
- Very high sparsity (99%): most of `dX` rows come from very few FMAs
- `dY` contains zeros mixed with nonzeros: our kernel shouldn't special-case

---

## 6. Performance expectations

| Operation                 | FLOPs                | Memory accesses                              |
|---------------------------|----------------------|----------------------------------------------|
| Forward (existing)        | `O(nnz * N)`         | `O(nnz + nnz * N)` reads, `O(M * N)` writes  |
| `dL/dX`                   | `O(nnz * N)`         | same as forward (it's the same kernel)       |
| `dL/dW_values`            | `O(nnz * N)`         | `O(nnz + M * N + K * N)` reads, `O(nnz)` writes |
| `PaddedCSR.transpose()`   | `O(nnz + M + K)`     | one pass over nnz entries                    |

A full forward+backward step is about 3× a forward step. For a 90%
sparse FFN layer, that's roughly 0.3× of a dense forward+backward —
which is exactly the ~3× FLOP reduction the DST literature promises.

For 4a we don't need to write a NEON version of `spmm_grad_W`. The
scalar implementation is enough to verify correctness end-to-end and
run the demo. NEON optimization for backward is Milestone 4a-opt
(post-4a).

---

## 7. What we're explicitly NOT doing in 4a

- No dense `dL/dW` materialization (the anti-pattern)
- No NEON backward kernel (scalar only for 4a; speed comes in 4a-opt)
- No SparseLinear class (that's 4b)
- No Router / DST algorithm support (that's 4c-e)
- No gradient accumulation optimization (post-v0.1)
- No gradient sparsity (the idea that dL/dY itself might be sparse
  for ReLU layers — interesting, but out of scope)

---

## 8. Sequencing of implementation

Order to follow, each step tested before the next:

1. **4a-i** — this design doc (done)
2. **4a-ii** — `PaddedCSR.transpose()` method + tests
3. **4a-iii** — `spmm_grad_X` kernel (wraps transpose + existing SpMM)
4. **4a-iv** — `spmm_grad_W` kernel (new: per-slot dot product)
5. **4a-v** — `SpMMFunction` autograd wrapper
6. **4a-vi** — `torch.autograd.gradcheck` test
7. **4a-vii** — demo: manual one-step "training" loop that runs forward, backward, optimizer step on a tiny sparse network

Each of these steps is self-contained and testable. Step 2 (transpose)
is low-risk, pure plumbing — a good warmup before the kernel work in
steps 3 and 4.

---

## 9. References

- Goodfellow, Bengio, Courville. *Deep Learning*, §6.5 (backprop derivation)
- PyTorch autograd docs: https://pytorch.org/docs/stable/notes/extending.html
- torchao sparse backward implementation:
  https://github.com/pytorch/ao/tree/main/torchao/sparsity
- Evci et al., *Rigging the Lottery* (ICML 2020) — for the "grow
  at top-K |dL/dW|" motivation that forces us to think about dormant
  gradients

---

## 10. Open questions (to resolve during implementation)

1. Where exactly does the "requires_grad routing" live? Probably in
   `sparselab/ops.py`'s `spmm()`; we'll confirm during 4a-v.
2. Do we need to save `W_values` for backward, or is `W` enough?
   Technically backward only reads `W.col_indices`, `W.row_start`,
   `W.row_nnz`, so `W_values` is NOT used in backward. But autograd
   requires saving at least one Tensor that "represents" W for the
   gradient-return contract. We'll save `W.values` as a placeholder.
3. How does `transpose()` interact with `padding_ratio`? Default: use
   the same ratio as the input. Custom: future enhancement.
