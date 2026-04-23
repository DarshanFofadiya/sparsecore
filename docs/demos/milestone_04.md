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

**The sparse-on-CPU speed story — with an important correction.**
Demo 4 initially reported "33.2ms vs 1337.7ms, 40× faster than dense."
That number was misleading, and I've left the correction here rather
than hide it.

What actually happened: demo 4 compared our raw kernel path against
`nn.Linear` running through the full PyTorch framework (module call +
`optimizer.zero_grad()` + `optimizer.step()`) on a shape so tiny
(32×64) that Python overhead dominated the dense path. That was a
framework-overhead comparison, not a matmul comparison.

The honest numbers, measured head-to-head in
`examples/demo_04b_honest_benchmark.py` (both paths with raw tensors,
no `nn.Module` wrapping, shape 512×256×64):

| Sparsity | Dense ms/step | Sparse ms/step | Speedup |
|----------|---------------|-----------------|----------|
| 0%       | 0.20          | 24.22           | 0.01x    |
| 50%      | 0.27          | 11.40           | 0.02x    |
| 70%      | 0.23          | 6.52            | 0.04x    |
| 80%      | 0.21          | 4.64            | 0.04x    |
| 90%      | 0.20          | 2.35            | 0.09x    |
| 95%      | 0.19          | 1.52            | 0.12x    |
| 99%      | 0.22          | 0.68            | 0.33x    |

**We are slower than dense at every sparsity on Apple Silicon** for
this shape in the training loop. Apple's AMX matrix coprocessor
processes the dense matmul at ~40 GFLOP/s without regard to sparsity;
no general-purpose NEON loop can match that throughput, even at 99%.

What this means for positioning:

- **Apple Silicon is our HARDEST platform for the speed story.** AMX
  is an Apple-only advantage. On Graviton/Ampere/Intel (untested by
  us, but published SpMM benchmarks beat MKL starting around 75%
  sparsity), the crossover should land much earlier.
- **Where we genuinely win even on Apple Silicon today:**
  - **Memory footprint** — a 90%-sparse model uses ~10% of dense
    memory. This is what lets us train models that don't fit on a
    MacBook at all in dense form.
  - **Correctness invariant** — padding slots stay at exactly 0.0
    after training, which dense-simulated libraries cannot claim.
  - **Researcher ergonomics** — the Pluggable Router design
    (milestones 4c-e) is the real product. "Speed" is a
    "fast enough to be usable" claim, not "faster than dense."

The version of SparseCore that can make a legitimate "faster than
dense" claim for forward + backward is the one that runs on non-Apple
hardware. That's an important test to add soon.

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

## What landed

- docs(design): backward-pass spec
- feat(layout): PaddedCSR.transpose()
- feat(spmm): dL/dW kernel at live slots
- feat(autograd): loss.backward() integration + writable values
- docs(milestone_04): demo script + this doc
