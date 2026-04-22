# Design: Router API (Milestone 4d)

## What it is

The Router is our pluggable sparsity algorithm API. It is the contract
that lets a community researcher express SET, RigL, or a brand-new
DST algorithm as a short Python class without touching SparseCore's
kernels, storage, or autograd internals.

**By the end of this milestone, an empty `SparsityAlgorithm` subclass
plus one implementation (`Static` — no-op) is testable end-to-end.**
SET (4e) and RigL (4f) will be ~40 and ~60 line subclasses.

## Why "Router"

Because the algorithm routes the training process: it decides which
connections are live, when to drop them, where to grow new ones. The
metaphor matches how mixture-of-experts "routers" work — an auxiliary
policy head that dynamically reshapes the compute graph.

Public class name is **`SparsityAlgorithm`** (matches Cerebras), but
the module is `sparsecore.router` to match our narrative.

## Prior art — Cerebras `cstorch.sparse.SparsityAlgorithm`

Reference: https://training-api.cerebras.ai/en/latest/wsc/tutorials/sparsity.html

Key patterns we adopt verbatim:
- User-facing API: `model.apply(sparsity)` (one line to attach).
- Algorithm has a single abstract method: `update()` — called once per
  scheduled step, iterates over sparse params and assigns new masks.
- A `DynamicSparsityAlgorithm` subclass handles schedule bookkeeping
  so SET/RigL stay tiny.
- `Static(sparsity=0.9)` as a no-op reference implementation.
- Optional: `Group({"fc1.*": alg1, "fc2.*": alg2})` for per-param algorithms.
  **We defer `Group` to milestone 4d-ii** (incremental).

Key patterns we diverge on:
- Cerebras stores dense weights + a separate mask tensor. We store
  Padded-CSR. So our "mask" is implicit in `csr.col_indices` /
  `csr.row_nnz`. An algorithm's `update()` mutates the CSR indices,
  not a mask buffer. See "Topology mutation primitives" below.
- Cerebras's `optimizer.apply(sparsity)` step is unnecessary for us
  because our weight values are already per-live-slot; there are no
  "dead" optimizer state positions to zero. One less step for the user.

## User story (where this should end up)

Before 4d (today):

```python
fc1 = sparsecore.SparseLinear(784, 512, sparsity=0.9)  # mask frozen at init
# ... training loop, no topology mutation ...
```

After 4d:

```python
fc1 = sparsecore.SparseLinear(784, 512, sparsity=0.9)

sparsity = sparsecore.sparse.Static(sparsity=0.9)   # no-op: mask stays frozen
fc1.apply(sparsity)                                  # attaches algorithm

# Training loop unchanged. At the end of every opt.step():
#   sparsity.step()    # user calls this; algorithm.update() runs if schedule says so
```

After 4e/4f (the payoff):

```python
fc1 = sparsecore.SparseLinear(784, 512, sparsity=0.9)

sparsity = sparsecore.sparse.RigL(
    sparsity=0.9,
    drop_fraction=0.3,     # 30% of live connections churn per update
    update_freq=100,        # every 100 steps
)
fc1.apply(sparsity)

# Same training loop. sparsity.step() after each opt.step() handles
# the gradient-aware topology mutation automatically.
```

## Public API shape

```python
class SparsityAlgorithm:
    """Base class for all sparsity policies."""

    def __init__(self, sparsity: float): ...

    @property
    def layers(self) -> list[SparseLinear]:
        """Layers this algorithm was attached to (via model.apply)."""

    def attach(self, layer: SparseLinear) -> None:
        """Called by SparseLinear.apply(). Registers this algorithm
        as the owner of that layer's topology. Most users never call
        this directly."""

    def step(self) -> None:
        """Invoked by the user from their training loop AFTER
        optimizer.step(). Increments the algorithm's internal step
        counter and, if the schedule says so, calls update()."""

    def update(self) -> None:
        """ABSTRACT. Override in subclasses to mutate topology.

        Inside update() you have access to self.layers. For each
        layer you can:
          - read `layer._csr.col_indices`, `row_nnz`, `values`
          - call `layer._drop_slot(row, local_slot_idx)` to mark a
            connection dead
          - call `layer._grow_slot(row, new_col)` to add a connection
            to an empty slot
        See 'Topology mutation primitives' below.
        """
```

```python
class Static(SparsityAlgorithm):
    """No-op: holds the init-time random mask, never mutates it.
    Serves as the reference implementation."""
    def update(self) -> None:
        pass
```

```python
class DynamicSparsityAlgorithm(SparsityAlgorithm):
    """Base for algorithms that mutate topology on a schedule.
    SET and RigL subclass this. Handles step counting and the
    'should we update this step' decision."""
    def __init__(self, sparsity, drop_fraction, update_freq): ...
    def step(self) -> None:
        self._step_idx += 1
        if self._step_idx % self.update_freq == 0:
            self.update()
```

## Attaching the algorithm to a layer

Two ways, both one-liner:

```python
# Per-layer
fc1.apply(sparsity)

# Whole-model: torch.nn.Module.apply recursively applies a callable
# to each submodule. We exploit this: calling model.apply(sparsity)
# invokes sparsity.__call__(module) on every submodule. Our __call__
# filters for SparseLinear and registers.
model.apply(sparsity)
```

Wait — `nn.Module.apply` takes a **callable**, not an algorithm
object. We make `SparsityAlgorithm` callable (`__call__(module)`) so
the standard PyTorch idiom works. Inside the `__call__` we
selectively attach only to SparseLinear modules.

## Topology mutation primitives (on SparseLinear)

These are new methods we add to `SparseLinear` that the algorithm
calls. They abstract over CSR index mutation so algorithm authors
don't need to know about our storage format.

```python
class SparseLinear(nn.Module):
    # existing methods: forward, reset_parameters, ...

    def live_slots(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (row_of_slot, col_of_slot) as parallel arrays. Each
        array has length nnz. Used by DST algorithms to iterate over
        all currently-live connections."""

    def drop_slots(self, slot_indices: np.ndarray) -> None:
        """Mark these slots dead (col_indices[s] = -1, values[s] = 0,
        decrement row_nnz). Does not reclaim memory — the slot stays
        in place, just empty."""

    def grow_slots(self, row_indices: np.ndarray, col_indices: np.ndarray,
                    values: np.ndarray | None = None) -> int:
        """Add new live slots at (row, col) positions. Returns the
        number successfully added (some may fail if row_capacity is
        full). If values is None, new slots start at zero."""

    def can_grow(self, row: int) -> bool:
        """True if row has an empty slot available for growth."""
```

These three primitives — `live_slots`, `drop_slots`, `grow_slots` —
are the full surface a DST algorithm needs. RigL adds one more
requirement: access to the dense gradient at dead positions. We
handle that via `SparseLinear.dense_grad_sketch()` which will be
introduced alongside RigL itself in 4f (not needed for 4d/4e).

## What 4d does NOT do

- **No `Group({"fc1.*": ...})` yet.** That's 4d-ii, additive. Our
  4d-i just accepts one algorithm per layer (the common case).
- **No dense-gradient sketching for RigL.** 4f territory.
- **No schedule abstraction beyond "every N steps".** Cerebras has
  cosine/polynomial decay of `drop_fraction` over training. We'll
  add `Schedule` as its own small milestone if the demos show need.
- **No `state_dict` serialization of algorithm state.** We'll cross
  that bridge when a user asks for checkpoint portability of DST
  runs. Scoped for 4f+.

## Tests to write

1. Empty `SparsityAlgorithm` subclass can be instantiated and attached
   to a layer via `layer.apply(alg)` — no error.
2. `model.apply(sparsity)` on a 3-layer MLP attaches only to the
   `SparseLinear` submodules (skips the dense ones).
3. `Static.update()` is a no-op: nnz and col_indices unchanged before
   and after `sparsity.step()`.
4. `Static.step()` advances internal counter.
5. A mock dynamic algorithm can use `live_slots`, `drop_slots`,
   `grow_slots` to shuffle the topology. After the shuffle:
   - `layer.nnz` equals the pre-shuffle nnz (drop count == grow count)
   - The weight values at newly-grown slots are zero
   - The weight matrix has moved: some previously-live (i,j) are now
     zero, and some previously-zero (i,j) are now live
6. Full MNIST end-to-end training with `Static` produces identical
   accuracy to the no-algorithm baseline. Sanity check that the
   attachment overhead is really zero.

## Naming one-off discussion

Cerebras calls their module `cstorch.sparse`. We're going to call
ours `sparsecore.router` (module) but expose the classes under
`sparsecore.sparse` too for Cerebras-familiar users. So all of
these work:

```python
import sparsecore
sparsecore.Static(sparsity=0.9)         # top-level alias (most discoverable)
sparsecore.sparse.Static(sparsity=0.9)  # Cerebras-compatible path
sparsecore.router.Static(sparsity=0.9)  # our canonical module location
```

We pick `sparsecore.router` for the file because *router* is a more
evocative name for "the thing that decides connection topology" than
*sparse*, and we already have `sparsecore/nn.py` and `sparsecore/ops.py`
so adding `sparsecore/router.py` keeps the source tree narratively
coherent. The aliases are free.

## What's next after 4d

Milestone 4e: `SET` subclass of `DynamicSparsityAlgorithm`. Drops
lowest-magnitude weights, grows random empty slots. ~40 lines.
Demonstrated against `Static` on MNIST @ 90%: expected to narrow
the 0.61pp gap toward zero.
