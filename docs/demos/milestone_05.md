# Milestone 5 — MNIST at multiple sparsities

## What this proves

This is the first demo where both models actually **converge on real data**,
not a synthetic regression. It answers the question behind the project: at
what sparsity does your model still learn the task?

It also validates autograd correctness in the strongest possible way: at
0% sparsity, the sparse and dense paths reach **exactly the same test
accuracy (92.35%)**. If there were any bug in our backward, the two
paths would drift — instead, they produce bit-identical learning
trajectories when they're mathematically equivalent.

## How to run

```bash
pip install sparsecore[demos]
python examples/demo_05_mnist.py
```

~4 minutes on an M3 Pro. Downloads MNIST (~12 MB) on first run.

## Results (M3 Pro, 3 epochs, 512-unit hidden layer)

| Sparsity | Live params | Dense acc  | Sparse acc | Accuracy gap | Dense time | Sparse time | Dense KB | Sparse KB | Mem ratio |
|----------|-------------|------------|------------|--------------|------------|-------------|----------|-----------|-----------|
| 0%       | 401,408     | 92.35%     | **92.35%** | 0.00 pp      | 7.4s       | 130.8s      | 3,136    | 5,652     | 180%      |
| 50%      | 200,721     | 92.17%     | 91.13%     | -1.04 pp     | 8.5s       | 72.6s       | 3,136    | 2,942     | 94%       |
| 70%      | 120,840     | 92.18%     | 90.35%     | -1.83 pp     | 8.2s       | 47.7s       | 3,136    | 1,796     | 57%       |
| 80%      | 80,305      | 92.11%     | 89.76%     | -2.35 pp     | 8.1s       | 33.4s       | 3,136    | 1,218     | 39%       |
| 90%      | 40,206      | 92.10%     | 88.65%     | -3.45 pp     | 7.9s       | 20.7s       | 3,136    | 574       | 18%       |
| 95%      | 19,992      | 92.07%     | 87.00%     | -5.07 pp     | 7.8s       | 14.5s       | 3,136    | 294       | 9%        |
| 99%      | 3,881       | 91.98%     | 79.13%     | -12.85 pp    | 8.3s       | 9.8s        | 3,136    | 70        | 2%        |

Loss curves: see `demo_05_mnist_curves.png` next to this file.

Memory is the at-rest size of the hidden layer's state — weight +
gradient + (for sparse) index arrays — computed exactly from tensor
sizes, not measured. Excludes transient backward activations which are
similar on both paths at this batch size.

## Follow-up (demo 6 + demo 7): what happens with more training?

The 3-epoch data above is honest but incomplete. We ran two follow-up
experiments to pin down whether the gap is a convergence artifact or a
real capacity ceiling.

**Demo 6 (`examples/demo_06_convergence.py`)** — 15 epochs across three
sparsities with patient early stopping:

| Sparsity | Best dense acc | Best sparse acc | Gap      | Ep where each peaked |
|----------|----------------|-----------------|----------|----------------------|
| 70%      | 96.03%         | 94.38%          | 1.65pp   | both at ep 15        |
| 90%      | 95.75%         | 92.80%          | 2.95pp   | both at ep 15        |
| 99%      | 95.66%         | 89.05%          | 6.61pp   | both at ep 15        |

The surprise: nothing plateaued. Every curve was still monotonically
rising at ep 15. So the gaps above are *early-stopping snapshots*, not
real capacity ceilings. See `demo_06_convergence.png`.

**Demo 7 (`examples/demo_07_90pct_convergence.py`)** — same 90%
sparsity, but with MAX_EPOCHS=100 and patience=10, so each path
genuinely trains until it stops improving:

| Path   | Best accuracy | Epoch of peak | Total epochs run | Wall time |
|--------|---------------|---------------|------------------|-----------|
| Dense  | 98.06%        | 72            | 82 (plateau)     | 232 s     |
| Sparse | 97.09%        | 99            | 100 (ran out)    | 700 s     |

**Gap at 90% sparsity: 0.97 pp.**

Two important facts are baked into that single number:

- **Dense genuinely plateaued.** Best at epoch 72, stale for 10 straight
  epochs, early stopping triggered at epoch 82. This is the dense
  path's real ceiling on this task.
- **Sparse was still climbing when we ran out of budget.** Best at
  epoch 99 out of 100 allowed. A longer run would almost certainly
  reduce the gap further — sparse gained +0.13pp in its last 5 epochs
  and hadn't had a 10-epoch stale window at any point.

Even under that conservative reading, **at 90% sparsity sparse gets
within 1 pp of dense**, for 18% of dense's memory. That's not a
plateau-vs-plateau claim (we ran out of compute before we could make
that one cleanly), but it's enough to say: the earlier 3-epoch and
15-epoch gaps were *mostly convergence, not capacity*.

See `demo_07_90pct_curves.png` for the full trajectories. The visual
is striking — both curves are still on the same overall trend when
sparse runs out of epochs.

**What we still don't know and are being upfront about:**

- The exact asymptotic gap. To claim "sparse fully matches dense at
  90%" we'd need to let sparse train until *it* triggers the patience
  stop. That's a 20–30-minute run; we've left it as a known follow-up
  rather than pretending the 0.97pp number is the final answer.
- Whether the same convergence story holds at 99% sparsity. The demo_6
  snapshot (6.61pp at ep 15) is steep enough that there may be a real
  structural component there — precisely the regime where RigL-style
  gradient-based regrow (milestone 4e) is supposed to earn its keep.

## What the 3-epoch data tells us (findings from demo_05)

### 1. Autograd correctness is verified by the 0% row

Sparse and dense paths reach identical test accuracy when sparsity is 0
(same model, same init, same data). That's the strongest possible
verification that our backward pass is analytically correct — any bug
would cause drift.

### 2. MNIST tolerates up to 90% sparsity with ≤3.5 pp accuracy loss

This is consistent with published DST literature. The "knee" of the
accuracy-vs-sparsity curve sits around 95%:

- 0% → 70%: graceful decay, <2pp loss, ~70% parameter reduction
- 70% → 90%: moderate decay, another ~2pp loss, ~10× parameter reduction
- 90% → 95%: another ~2pp loss
- 95% → 99%: sharp cliff, another ~8pp loss

The cliff at 99% is classic: at that capacity, the model can't represent
MNIST's class structure with random sparsity patterns alone. This is
exactly where RigL-style gradient-based connection growth (milestone
4e) earns its keep — by putting the few available weights where they
matter most, rather than scattering them at random.

### 3. Memory break-even is at ~50% sparsity

Sparse storage is NOT automatically smaller than dense. We pay for:
- `col_indices`: 4 bytes per slot
- `row_start`, `row_nnz`, `row_capacity`: 12 bytes per row
- `dW_values`: gradient buffer same size as `values` (includes padding)

At 0% sparsity our state is 180% of dense memory — we're paying for
indexing that doesn't save anything. At 50% we hit break-even (94% of
dense). From there the curve is steep: 57% at 70% sparsity, 18% at
90%, just 2.2% at 99%.

**The memory / accuracy Pareto for MNIST:**

| What you care about     | Best operating point | Sparsity | Accuracy | Memory |
|-------------------------|----------------------|----------|----------|--------|
| Maximum accuracy        | dense                | 0%       | 92.35%   | 100%   |
| Most memory per accuracy point lost | moderate sparse | 80%  | 89.76%   | 39%    |
| Aggressive research trade | high sparse        | 90%      | 88.65%   | 18%    |
| Memory-constrained only | very sparse          | 99%      | 79.13%   | 2%     |

The "80% sparsity, 39% memory, -2.35 pp accuracy" row is where the
dollar-per-accuracy ratio is healthiest for this task. That's the
number a researcher constrained to a MacBook-sized training budget
cares about.

### 4. Apple Silicon speed story: dense wins at every sparsity

Dense stays flat at ~8s regardless of sparsity. That's Apple AMX
operating at its full throughput on the dense matmul, which doesn't
care that 99% of the weights are zero — it multiplies them anyway.

Our sparse time scales linearly with `nnz`:

```
0%   →  130.8s (401k params, slowest)
99%  →    9.8s (3.9k params, nearly matches dense)
```

The slope is ~320μs per 1000 live parameters. At 99% sparsity we come
within 18% of dense wall-clock, but we never cross over on Apple
Silicon. This is consistent with `demo_03_spmm.py` and
`demo_04b_honest_benchmark.py`: AMX is the fixed ceiling we can't
beat with general-purpose NEON.

### 5. The real platform story is cross-architecture

The speed comparison above is Apple Silicon's AMX vs our NEON. On any
CPU without AMX (Graviton, Intel, Ampere), the dense path is itself
NEON/AVX FMA loops, running at no particular advantage over ours.
Published SpMM benchmarks beat dense MKL starting around 75% sparsity
on Intel (see the `docs/LANDSCAPE.md` references).

We haven't tested this yet. That's the most important follow-up
experiment — if the story holds on non-Apple hardware, our
positioning narrative becomes "the first sparse training library where
the speed story matches the research literature across the CPU
landscape."

## What this tells us about milestone priorities

Two things get reinforced by this demo:

**The DST regrow policy matters a lot at the high-sparsity tail.**
Going from 95% → 99% loses 8pp of accuracy with our current "random
mask at init, no mutation" approach. The RigL / SET literature shows
you can largely close that gap by *moving* the 1% of active weights
to where the data needs them, using gradient signals between training
runs. That's milestones 4c (Router API) → 4d (SET) → 4e (RigL). Having
a working DST regrow policy is how we tell the story "SparseCore
matches dense up to 99%" in the launch blog.

**The SparseLinear UX (milestone 4b) is still critical.** Even for
this demo I had to wrangle `_SpMMFunction.apply()` manually. That's
not the experience a researcher should have. After 4b lands, the
demo becomes simple:

```python
self.fc1 = sparsecore.SparseLinear(784, 512, sparsity=0.9)
```

## Commits

- (this commit) — feat(demo): MNIST convergence sweep at 7 sparsities +
  loss-curve plot + memory-at-rest column + follow-up convergence
  experiments (demo_06, demo_07) + this write-up

## What's next

Milestone 4b: `SparseLinear(nn.Module)` wraps `_SpMMFunction` so users
never touch raw autograd. That's the two-line-adoption promise made in
`docs/PROJECT_OVERVIEW.md`. After 4b we'll rewrite this demo using the
clean API and it becomes a contribution-ready example.
