# Milestone 7 — The Router API (4d)

## What landed

The `SparsityAlgorithm` base class + `Static` reference implementation.
This is the pluggable contract every DST algorithm will fit into:

```python
import sparselab

layer = sparselab.SparseLinear(784, 512, sparsity=0.9)
algo = sparselab.Static(sparsity=0.9)
layer.apply(algo)                 # or model.apply(algo) on a whole MLP
```

`algo.step()` (called from the user's training loop after
`optimizer.step()`) increments the internal step counter and, for
dynamic algorithms, optionally calls `update()`. `Static.update()`
is a no-op — the mask stays frozen for the whole run.

## Why this matters

Before 4d, any DST algorithm would have had to live inside
`SparseLinear.forward()`, bloating the layer and coupling layer
internals to mutation policy. With 4d, algorithms are first-class,
pluggable objects. SET will be ~40 lines in a new file (4e). RigL
will be ~60 lines (4f). The exact same `model.apply(...)` API swaps
one for the other at the call site.

This is the "researcher-friendliness" bet the whole project is built
around. The Router is the API surface a community contributor writes
against when they have a new idea.

## API walkthrough

### `SparsityAlgorithm` (base class)

```python
class SparsityAlgorithm:
    sparsity: float                     # target sparsity in [0, 1)
    layers: list[SparseLinear]          # what this algorithm governs

    def __init__(self, sparsity: float): ...
    def attach(self, layer) -> None:    # register a layer
    def __call__(self, module) -> None: # dispatch for model.apply
    def step(self) -> None:             # advance counter, maybe update
    def update(self) -> None:           # ABSTRACT: override in subclasses
```

### `Static` (reference implementation)

```python
class Static(SparsityAlgorithm):
    def update(self) -> None:
        pass  # literally nothing
```

### How `model.apply(algo)` works

PyTorch's `nn.Module.apply(fn)` recursively calls `fn(submodule)` on
every submodule. By making `SparsityAlgorithm.__call__(module)` a
method that filters for `SparseLinear` instances, the Cerebras
ergonomics come for free:

```python
class MyMLP(nn.Module):
    def __init__(self):
        self.fc1 = sparselab.SparseLinear(784, 512, sparsity=0.9)
        self.fc2 = nn.Linear(512, 10)  # dense — skipped
        self.fc3 = sparselab.SparseLinear(10, 1, sparsity=0.5)

model = MyMLP()
algo = sparselab.Static(sparsity=0.9)
model.apply(algo)
# algo.layers is now [model.fc1, model.fc3] — dense layers skipped.
```

## What's in this milestone

| File | What | Lines |
|------|------|-------|
| `docs/design/router.md` | Design doc (Cerebras prior art analysis, class hierarchy, decisions deferred) | ~220 |
| `sparselab/router.py`  | `SparsityAlgorithm` + `Static` | ~180 (half comments) |
| `sparselab/__init__.py` | Public re-exports | +2 lines |
| `tests/test_router.py`  | 14 tests covering the API contract | ~210 |

Full test suite now: **313 passed, 2 skipped** (was 299 before 4d).

## What is NOT in 4d

Deliberate scope cuts to keep this milestone small and reviewable:

- **No C++ topology mutation primitives** (`drop_slots`, `grow_slots`
  on `PaddedCSR`). These are what SET/RigL will actually call to
  change the live-connection pattern. Moved to milestone 4e because
  they're only meaningful when used by a real DST algorithm.
- **No `DynamicSparsityAlgorithm` convenience base**. Cerebras has it
  as a common parent for SET/RigL (handles schedule / step-gating
  boilerplate). We'll add it in 4e alongside SET itself, since SET
  is what first exercises the "update every N steps" behaviour.
- **No `Group({"fc1.*": Static(0.3), "fc2.*": SET(0.9, ...)})`**
  pattern matching. Useful — Cerebras has it — but additive later.
  Our current per-layer `apply` already supports the "different algo
  per layer" use case, just without the glob convenience.
- **No `state_dict` serialization of algorithm state**. DST runs
  might want to checkpoint "step_idx" etc. Scoped for post-4f.

## The `Static` no-op proves the API end-to-end

Static's whole reason for existing is to be the minimal concrete
subclass that exercises every part of the plumbing without actually
doing DST. The test `test_training_loop_with_static_matches_no_attach`
is the proof: two identical MLPs trained on the same inputs with the
same seed, one with `Static` attached and one without. After 10
training steps, their weights are **bit-identical**. That means:

1. The `attach` / back-pointer plumbing doesn't change anything.
2. `algo.step()` has zero side effects for `Static`.
3. Nothing in the forward or backward path reads `layer._sparsity_algorithm`.

These are all the invariants DST algorithms in 4e/4f will assume.

## What's next

Milestone 4e: **SET** — the simplest DST algorithm. Drops the lowest-
magnitude K% of weights and regrows K% random new connections every N
steps. Will be ~40 lines of `DynamicSparsityAlgorithm` subclass + the
C++ mutation primitives + a demo showing the MNIST convergence gap
narrow vs. `Static`.
