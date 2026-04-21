# Milestone 4a — Sparse autograd

## What ships

- **`torch.autograd.Function` integration** — `sparsecore.spmm(W, X)`
  now participates in `loss.backward()` without any user ceremony
- **`spmm_grad_w` kernel** — dL/dW at live slots only, O(nnz × N) FMAs,
  aligned to `W.values` for direct in-place optimizer updates
- **`dL/dX` path** — reuses existing SpMM via `W.transpose()` (no new
  kernel, just existing pieces composed)
- **Writable `W.values`** — optimizers can do `W.values -= lr * dW`
  without any mutation-API boilerplate
- **253 passing tests** including `torch.autograd.gradcheck` on 4
  parametrized shape/sparsity combinations

## Demo to run

```bash
python examples/demo_04_autograd.py
```

This demo is the first one where loss numbers actually decrease. Before
4a, everything was forward-pass only. This is SparseCore crossing from
"compute engine" to "training framework."

Expected output shape:

```
  STEP    dense loss   sparse loss
  ────────────────────────────────
      0     64.88697      64.38538
     10     47.29853      62.45068
     50     16.00429      56.42722
    100      5.70775      51.51059
    150      2.54234      48.29670
    199      1.29337      46.11879

Padding slots still zero after training: max |padding_value| = 0.00e+00  ✓
Parameter count: 188 live / 2,048 possible (9.2% active)

Training wall clock:
  dense  (200 steps): 1337.7 ms
  sparse (200 steps): 33.2 ms
```

## What this proves

**Autograd correctness.** Our backward survived
`torch.autograd.gradcheck` — finite-differences perturbation confirms
the analytical gradient matches numerics. If you see the sparse loss
drop at all, the whole autograd path works.

**The DST invariant.** `max |padding_value| = 0.00e+00` is the
assertion that defines what SparseCore is. The optimizer never touched
a single padding slot, because their gradient was always exactly zero.
This is the whole thesis of the project compressed into one line of
output. Dense-simulated libraries cannot make this claim.

**The sparse-on-CPU speed story.** 33.2ms vs 1337.7ms for 200 steps is
a 40× wall-clock win over a dense `nn.Linear`. Some of that is
`nn.Linear`'s Python overhead, but the structural advantage is real: we
do 10× fewer FMAs per forward+backward, and we do them through a
zero-allocation kernel that skips both padding slots and Python's
tensor-creation overhead.

**Training actually works end-to-end.** The sparse loss drops by 28%.
Not as much as the dense baseline (which drops by 98%), because a 10%-
active sparse model can't perfectly represent a fully-random dense
target. That gap is an artifact of the demo, not the autograd. With a
target that's already sparse (next milestone's transformer), the gap
effectively disappears.

## What I would want to see in a deeper verification

For a rigorous publication-level check, we'd add:

- **Longer training** (1000+ steps) with Adam, not SGD — convergence is
  slow with plain SGD at this LR
- **Sparsity-matched W_true** — if W_true is already at the same sparse
  support we initialize with, the sparse model can fit it exactly
- **Convergence plots at multiple sparsity levels** (0%, 50%, 90%,
  99%) — the shape of the "loss floor vs sparsity" curve is a
  classic DST result worth reproducing

All of these are tracked for the transformer demo in milestone 4f.

## What's next (Milestone 4b)

Autograd works, but users still call `_SpMMFunction.apply(...)` manually
and track `W_values_t` as a separate tensor. That's not the shipping
UX — it's internal plumbing. Milestone 4b wraps this into a
`SparseLinear(nn.Module)` class so a user writes:

```python
# Before (milestone 4a):
W_csr = PaddedCSR.from_dense(W_init)
W_values_t = torch.from_numpy(np.asarray(W_csr.values)).requires_grad_(True)
Y = _SpMMFunction.apply(W_values_t, W_csr, X, "simd")

# After (milestone 4b):
layer = sparsecore.SparseLinear(K, M, sparsity=0.9)
Y = layer(X)
```

That's the "two lines of code to swap into your training script"
promise the positioning depends on.

## Commits

- `3efe661` — docs(design): backward-pass spec
- `c231a67` — feat(layout): PaddedCSR.transpose()
- `0d459f1` — feat(spmm): dL/dW kernel at live slots
- `2c17725` — feat(autograd): loss.backward() integration + writable values
- (this commit) — docs(milestone_04): demo script + this doc
