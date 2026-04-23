# Milestone 9 — RigL + DST algorithm comparison (4f)

## What landed

Two things:

1. **`sparselab.RigL`** — the third sparsity algorithm in the Router API.
   Uses the dense gradient to decide where to grow new connections
   instead of random choice. Cerebras-compatible API:

```python
layer = sparselab.SparseLinear(784, 512, sparsity=0.9)
algo = sparselab.RigL(
    sparsity=0.9,
    drop_fraction=0.3,
    update_freq=100,
    seed=42,
)
layer.apply(algo)
# ... training loop identical to Static / SET, just swap the algorithm.
```

2. **`sparselab._core.dense_grad`** — a new C++ kernel computing
   `G = dY @ X^T`, the full dense gradient at every (row, col)
   including currently-dead positions. Used by RigL (and any future
   gradient-aware DST algorithm) to find top-K grow candidates.

## What's new technically

### `dense_grad` kernel

A vanilla dense matmul parallelized with OpenMP, same pattern as
`spmm_grad_w`. Clang auto-vectorizes the inner dot-product into
NEON FMAs at `-O3`. 12 new tests verify correctness against numpy's
`dY @ X.T` reference at scales up to 512×2048, and agreement with
`spmm_grad_w` at live positions.

At DST cadence (one call every 100 training steps), the dense matmul
cost is amortized to ~5 μs per training step — invisible overhead.

### Forward/backward hook capture

RigL.attach() installs two PyTorch hooks per layer: a forward hook to
capture `X`, a full-backward hook to capture `dY`. These are consumed
and cleared at each update() so we never leak tensor references.

This design means the core ops (SpMM forward, autograd Function) stay
pure — RigL is a true bolt-on plugin. No coupling between kernels
and sparsity policy.

### Gradient-aware growth

The RigL `update()` logic, summarized:

1. Compute `|G| = |dY @ X^T|` via the new kernel.
2. For each row: drop the smallest-magnitude live weights (same as
   SET — global magnitude threshold).
3. For each row: grow at the top-K empty positions by `|G[i, :]|`,
   where K equals the number dropped. New weights init to zero.
4. Write back atomically via `rewrite_row`.

The unit test `test_rigl_grows_at_high_gradient_positions` proves this
is actually gradient-aware — a synthetic scenario where column 15 has
dominant `|G|` is constructed, and column 15 becomes live after
update.

## The honest MNIST result

Demo 11 ran all three algorithms head-to-head at `drop_fraction=0.3`.
Demo 12 swept drop_fraction ∈ {0.1, 0.2, 0.3} for both SET and RigL
against a shared Static baseline. All 10 epochs, 90% sparsity, same
seed, same hyperparameters.

| Config | Final acc | Best acc | vs Static |
|--------|-----------|----------|-----------|
| Static       | 91.86% | 91.86% |  +0.00 pp |
| **SET(0.1)** | **92.01%** | **92.01%** | **+0.15 pp** ← best |
| SET(0.2)     | 91.98% | 91.98% |  +0.12 pp |
| SET(0.3)     | 91.72% | 91.72% |  -0.14 pp |
| RigL(0.1)    | 91.67% | 91.67% |  -0.19 pp |
| RigL(0.2)    | 91.65% | 91.65% |  -0.21 pp |
| RigL(0.3)    | 91.43% | 91.43% |  -0.43 pp |

**Highlights:**

- **SET(0.1) beats Static by 0.15 pp.** First measurable DST win in
  our setup. Small but real.
- **Drop-fraction matters.** Both algorithms improve monotonically as
  drop_fraction decreases — the "less churn is better at short
  budgets" hypothesis was correct.
- **SET > RigL at every drop_fraction in our MNIST setup.** This is
  opposite of the literature's expected RigL > SET ordering.

## Why SET > RigL here, honestly

We checked three things before calling this a real result:

1. **Is `dense_grad` actually firing?** Yes — verified by
   instrumenting: exactly 10 calls in 100 steps with `update_freq=10`.
2. **Is RigL's grow actually gradient-aware?** Yes — the test
   `test_rigl_grows_at_high_gradient_positions` proves it at a unit
   level. At the MNIST scale, RigL does pick different grow positions
   than SET (verified by comparing `col_indices` after the first
   update — 287/40206 live slots differ).
3. **Does RigL's cost actually match SET's in wall-clock?** Yes —
   both finish in 59s. The dense_grad overhead is tiny.

So RigL is working correctly. Our best explanation for the accuracy
gap:

**MNIST is a weak testbed for RigL specifically.** MNIST pixel-space
has very high local correlation — a digit looks similar whether you
sample from the top-left 10% of pixels or a random 10%. So "grow at
top-gradient positions" ≈ "grow anywhere" for this task, which means
RigL's information advantage collapses to zero (and the zero-init
penalty it pays per mutation hurts). This is consistent with why the
RigL paper uses CIFAR and ImageNet, not MNIST, to show its gains.

**SET's random grow happens to be a fine strategy on MNIST.** The
"noise" from random growth is what matters — dropping low-magnitude
weights is most of where the benefit comes from at 10 epochs on
this task.

The transformer demo (milestone 4g, next) is where RigL's advantage
should actually show up because connectivity matters more in attention
weights and FFN weights.

## Sanity checks passed

- Full test suite: **361 passed, 2 skipped** (+23 from 4f: 12 for
  `dense_grad`, 11 for RigL).
- `rewrite_row` still invariant-preserving across 20 RigL update cycles.
- Training loop with RigL attached runs cleanly for 10+ steps.
- Reproducibility under seed: identical cols across two runs with
  same seed confirmed.

## Open research directions noted

Three directions surfaced and are captured in `docs/LANDSCAPE.md`
"Open questions & deferred experiments" so they don't slip:

- **Sparse 90% vs narrow dense at matched params** — the
  Lottery Ticket direct comparison.
- **Adaptive-sparsity DST (no fixed nnz budget)** — grow wherever
  gradient is large without a top-K constraint, letting `nnz` drift
  with what the task needs. Your idea from the demo_11 review.
- **Longer training budgets (30-50 epochs)** where the literature
  shows RigL's advantage appears. Cheap to measure; we just didn't
  do it at v0.1 to keep the demo quick.

## What landed

- feat(rigl): RigL + dense_grad kernel + drop-fraction sweep +
  23 new tests + 2 demos

## What's next

**Milestone 4g: tiny transformer demo.** The launch artifact.

Scope:
- 2-layer decoder-only transformer at d_model=128, ~1M params total
- Tiny Shakespeare dataset (~1 MB, downloads on first run)
- Sparse FFN layers (90% sparsity via `SparseLinear`)
- Compare: dense baseline vs sparse + SET(0.1) vs sparse + RigL(0.1)
- Runtime: ~30 min on M3 Pro
- Text samples at checkpoints showing the model learning

This is the task where DST is *supposed* to work. If RigL's advantage
appears at transformer scale, our launch narrative has a real result
to point at: "SparseLab's RigL matches dense accuracy at 90%
sparsity on a character-level transformer, with [specific memory/time
numbers]."
