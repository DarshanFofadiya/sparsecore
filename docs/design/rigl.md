# Design: RigL (Milestone 4f)

## What RigL is

**Rigging the Lottery: Making All Tickets Winners** (Evci et al., 2020,
https://arxiv.org/abs/1911.11134). A DST algorithm that mutates topology
every N steps, like SET, but makes smarter choices about *where to grow*:
instead of random empty positions, grow where the dense gradient is
largest.

Our user-facing API matches SET exactly — swap `SET(...)` for
`RigL(...)` and you're done:

```python
layer = sparsecore.SparseLinear(784, 512, sparsity=0.9)

algo = sparsecore.RigL(
    sparsity=0.9,
    drop_fraction=0.3,
    update_freq=100,
    seed=42,
)
layer.apply(algo)

# training loop: same as SET, same as Static
for x, y in loader:
    opt.step()
    algo.step()
```

## The algorithm in detail

Every `update_freq` steps, for each attached `SparseLinear` layer:

1. **Drop** — same as SET: drop the `drop_fraction × nnz` smallest
   magnitude live weights (global threshold, not per-row).

2. **Grow** — **different from SET**:
   a. Compute the full dense gradient of the loss w.r.t. the weight
      matrix, `G[i, j] = dL/dW[i, j]`. This is the gradient the model
      *would* feel at every position, including positions where
      `W[i, j]` is currently zero.
   b. Mask out positions that are currently live (survived the drop)
      — we're only interested in grow candidates.
   c. Take the top-K magnitudes `|G[i, j]|` across the remaining
      positions, where K equals the number we dropped. Those
      `(i, j)` are the grow positions.
   d. New weights at grown positions: zero (like SET).

3. **Rewrite** — for each row, combine surviving + grown slots,
   sort by column, call `rewrite_row`.

## Why this works (in one paragraph)

The dense gradient `dL/dW[i, j]` is the signal the model would use if
the connection existed. A large `|dL/dW[i, j]|` at a dead position
says "if you add this connection, I'll immediately start training it
to reduce loss." Conversely, a small magnitude says "even if you
added this connection, it would just sit at zero." So growing at
high-magnitude-dead positions is like pre-selecting "connections
that would have been wanted if they existed." This is the smarter
version of SET's random grow.

## The new C++ kernel we need

### Problem statement

Given:
- `W : PaddedCSR` — sparse weight matrix, shape `(M, K)`
- `dY : dense` — upstream gradient, shape `(M, N)`
- `X  : dense` — forward-pass input, shape `(K, N)`

Compute:
- `G : dense` — the full dense gradient `dL/dW`, shape `(M, K)`

Where `G[i, k] = sum over j of dY[i, j] * X[k, j]`. That's just
`dY @ Xᵀ` — a dense matmul that ignores the sparse structure of W
entirely.

### Why we can't reuse `spmm_grad_w`

Our existing backward kernel walks only the *live* slots of W and
computes their per-slot dot products. It deliberately skips empty
positions because during normal training we don't need their
gradients. For RigL we need all positions, including dead ones.

### Options considered

| Option | Memory | Compute | Code |
|--------|--------|---------|------|
| A. Materialize full dense G every update | M·K floats | M·K·N FMAs | Simple: just a dense matmul |
| B. Top-K candidates via approximate scan | O(K) per row | O(M·K·N) | Medium complexity, needs per-row heap |
| C. Stream G, accumulate top-K globally | O(top-K) | O(M·K·N) | Cache-friendly, custom kernel |

**Picking option A for v0.1.** At 784×512 f32 that's 1.6 MB, allocated
only at update events (every 100 steps). The full matmul is 200 GFLOPs
at transformer FFN scale but happens at 1% frequency, so amortized
cost is negligible compared to training. Options B/C are "nice to
have" post-v0.1 optimizations that we shouldn't block on.

### Kernel shape

```cpp
void dense_grad(
    int64_t M, int64_t K, int64_t N,
    const float* dY,        // (M, N), row-major
    const float* X,         // (K, N), row-major
    float* G                // (M, K), row-major  — output
);
```

Implementation: the standard `Y = A @ Bᵀ` matmul. `G[i, k] = dot(dY[i, :], X[k, :])`.
We can parallelize the outer `i` loop (same shape as `spmm_grad_w`).

### What it does NOT need

- Does not look at W at all — W's sparsity is irrelevant to computing G.
  W is only consulted by the Python side *after* G is computed, to mask
  out live positions.

- Does not output `int32` top-K indices. That's a Python-side argpartition
  on the returned dense array. If we discover the materialization cost
  becomes a bottleneck at transformer scale (possible at nrows × ncols
  above ~10M), we add a fused top-K kernel. Not in v0.1.

## Algorithm implementation plan

Structurally, `RigL._update_layer` is very similar to `SET._update_layer`.
The differences, inline:

```python
def _update_layer(self, layer, rng):
    csr = layer._csr

    # ── PASS 1 (same as SET): global magnitude threshold
    # Gather all live values, find k-th smallest |value|, that's the
    # drop threshold.
    # ...

    # ── NEW (RigL): get the dense gradient
    # Requires access to dY and X from the most recent backward pass.
    # We'll stash these on the layer at forward/backward time — see
    # "State we need to track" below.
    G = _core.dense_grad(dY=..., X=...)   # shape (nrows, ncols)

    # ── PASS 2 (same structure as SET): per-row drop + grow
    for each row i:
        # Drop as before (magnitude below threshold).
        # ...

        # For RigL, mask out currently-live positions from G[i, :].
        # Then pick the top n_drop from the remaining.
        live_mask_row = <bool array over ncols>
        candidates = abs(G[i, :])
        candidates[live_mask_row] = -inf  # exclude currently-live
        grow_cols = top_k_indices(candidates, k=n_drop)
        # ...

        # Rewrite row (same as SET)
        csr.rewrite_row(i, merged_cols, merged_vals)
```

## State we need to track

RigL needs the most recent `dY` and `X` from the forward/backward
pass. Two approaches:

**Option 1: Capture at backward time.**
Modify `_SpMMFunction.backward` to stash `dY` and `X` on the layer
object. Clean-ish but forces coupling: the autograd Function knows
about RigL's needs.

**Option 2: Let RigL capture via hooks.**
PyTorch supports `register_forward_hook` and `register_full_backward_hook`
on nn.Modules. RigL.attach() registers a hook that captures `dY` and
`X` into layer state. Doesn't modify our core ops — RigL is truly
a bolt-on. This is the Cerebras pattern.

**Going with Option 2.** Keeps the core layer and autograd pure; RigL
is a plugin even in its state-tracking.

One wrinkle: `register_full_backward_hook` fires with the gradient
w.r.t. the *output* of the layer (= our `dY`), not the input. We
also need `X` (the forward input). So we need a forward hook to
capture `X` and a backward hook to capture `dY`. Both are one-liners
to attach; both release the captured tensors when we consume them
on the next `update()`.

## What RigL does not do

- **No dense optimizer.** Our `dY` and `X` come from sparse forward
  kernels; the dense gradient is constructed only at update time
  (every 100 steps). Unlike some RigL implementations that compute
  `dW_dense` every batch as part of a mask-on-dense approach, we keep
  backward sparse and only go dense at mutation time. Much cheaper.

- **No cosine drop-fraction decay.** The RigL paper has a cosine
  schedule for `drop_fraction` that decreases over training (more
  exploration early, less later). We ship with a constant
  `drop_fraction`. Schedule is a nice-to-have for v0.2.

- **No adaptive `update_freq`.** Static frequency.

## Tests to write

1. `dense_grad` kernel: output matches a naive `dY @ Xᵀ` reference.
2. `dense_grad` shape: (M, K) for all reasonable (M, K, N).
3. RigL construction + attach works like SET.
4. After one `update()`: nnz preserved, invariants preserved.
5. New grown positions correspond to highest-magnitude cells of G
   (not random like SET).
6. End-to-end: training step after RigL update still works.
7. Sanity: two RigL runs with same seed produce identical topology
   after update (requires the forward/backward hooks to capture
   deterministically too).

## Demo plan

Extend `demo_10_set_vs_static.py` into `demo_11_rigl_vs_set_vs_static.py`:
same MNIST setup, three training runs, one plot with three accuracy
curves. If RigL beats Static by any margin, that's the launch narrative.

## What's next after 4f

**Milestone 4g: the tiny transformer demo.** With Static / SET /
RigL all plumbed, we pick the best-performing algorithm (almost
certainly RigL) and train a small character-level transformer with
it. That's the v0.1 launch artifact.
