# Milestone 11 — Scaling run at ~40M parameters (1000 steps)

demo_15 showed convergence at 10M params over 10,000 steps. demo_16
answers a different question: **does the memory and wallclock story
hold when the model is 4× larger?**

Concretely: a ~40M-parameter decoder-only transformer trained on Tiny
Shakespeare for **1000 steps**, comparing:

1. **Dense baseline** — all projections as `nn.Linear`
2. **Sparse all** — attention at 70% sparsity + FFN at 90% sparsity

Same architecture, same seed, same data.

## What this demo claims

- ✅ **Memory ratio holds at 4× scale.** Sparse-all is **37.0% of
  dense inference memory at 40M**, exactly matching the 37% measured
  at 10M in demo_15. The ratio is stable as models get bigger.
- ✅ **Per-step slowdown narrowed slightly.** Sparse is **4.1× slower
  than dense per step at 40M** vs 4.6× at 10M. Kernel time is
  starting to dominate Python-side overhead, which is what we'd
  expect — and predicts the ratio keeps improving at larger scale.
- ✅ **Both paths train cleanly.** Loss descended monotonically on
  both paths across 1000 steps. No NaN, no OOM, no divergence. First
  confirmed training run in the project above 10M params.

## What this demo does *not* claim

- ❌ **Convergence parity at 40M.** 1000 steps on a 40M-param model
  is roughly 0.1 "epochs" of Tiny Shakespeare — both paths are still
  in the early descent. Whether sparse matches dense at convergence
  at this scale is open territory; it requires a real training run
  (days of compute).
- ❌ **Final text-sample quality.** Samples at step 1000 are
  character-soup on both paths.
- ❌ **Speed parity.** We're still slower than dense, and we say so.

## Architecture

```
MiniGPT-40M(
    tok_emb:  Embedding(65, 640)           # dense
    pos_emb:  Embedding(128, 640)          # dense
    blocks × 8:
        ln1:     LayerNorm(640)
        attn:    CausalSelfAttention
            qkv: (640 → 1920)              # dense | 70% sparse
            o:   (640 → 640)               # dense | 70% sparse
        ln2:     LayerNorm(640)
        ffn:
            fc_up:   (640 → 2560)          # dense | 90% sparse
            fc_down: (2560 → 640)          # dense | 90% sparse
    ln_f:     LayerNorm(640)
    head:     Linear(640, 65)              # dense
)
```

Reproduce:

```bash
python examples/demo_16_mini_gpt_60m.py --steps 1000 --path dense,all --tag 60m
```

Training: 1000 steps, batch=8, seq=128, lr=3e-3 (SGD), seed=42.
Hardware: M3 Pro MacBook. Timing via `time.perf_counter()` inside
the training loop (active compute only).

## Headline numbers

|                                        | Dense         | Sparse all (attn 70% + FFN 90%) |
|----------------------------------------|---------------|---------------------------------|
| **Total parameters**                   | 39,508,480    | 6,744,424                       |
| &nbsp;&nbsp;dense params (embeddings, LN, head) | 39,508,480 | 186,880                     |
| &nbsp;&nbsp;sparse live weights        | 0             | 6,557,544                       |
| &nbsp;&nbsp;sparse capacity (incl. padding) | 0        | 7,887,438                       |
| **Inference memory**                   | **150.7 MB**  | **55.8 MB (37.0% of dense)**    |
| **Training memory (weight + grad)**    | 301.4 MB      | 81.5 MB                         |
| **Training memory (+ CSR padding)**    | 301.4 MB      | 91.7 MB                         |
| **Per-step wallclock**                 | 320 ms        | 1326 ms                         |
| **Slowdown vs dense**                  | 1.0×          | **4.1× slower**                 |
| **Total wallclock (1000 steps)**       | ~5.3 min      | 22.1 min                        |
| **Val loss at step 100**               | 3.270         | 3.839                           |
| **Val loss at step 500**               | 2.854         | 3.068                           |
| **Val loss at step 1000**              | 2.727         | 2.836                           |

### The memory story plainly

At ~40M dense params:
- Dense inference footprint: **150.7 MB**
- Sparse inference footprint: **55.8 MB (37.0% of dense)**

Compare directly to the 10M demo_15 numbers: there we measured 37%
of dense inference memory with the same (70% attn, 90% FFN)
configuration. **The 37% ratio held exactly across a 4× parameter
scale-up**, which is the honest evidence that the memory story is
structural rather than a benchmarking coincidence.

Training memory (including CSR padding, which is the most pessimistic
and most accurate of what we actually allocate at training time):
**91.7 MB sparse vs 301.4 MB dense — 30.4% of dense**.

### The speed story plainly

Dense at 40M: **320 ms/step**.
Sparse all at 40M: **1326 ms/step → 4.1× slower**.

At 10M in demo_15, the slowdown was **4.6×**. The ratio **narrowed by
~11%** as we scaled up. That's a real signal: at smaller scales the
fixed per-layer Python overhead (numpy↔torch roundtrips, transpose
cache lookups, autograd boilerplate) dominates the sparse cost. At
larger scales, the `dW` kernel time grows linearly with live-weight
count, making the per-layer overhead a smaller fraction of each step.

Extrapolating: at 100M+ params, the ratio should continue narrowing
as kernel time dominates. This is a quantitative argument for why
the v0.2 `dW` NEON kernel matters — it's already the biggest cost
at 40M, and its share grows with scale.

### The convergence story, honestly

Both paths descend smoothly across 1000 steps:

| Step | Dense val | Sparse val | Gap (nats) |
|------|-----------|------------|------------|
| 100  | 3.270     | 3.839      | +0.569     |
| 500  | 2.854     | 3.068      | +0.214     |
| 1000 | 2.727     | 2.836      | +0.109     |

Gap narrows monotonically (0.569 → 0.214 → 0.109 nats). Both curves
are still moving — neither is close to plateau. **We make no claim
about convergence at 1000 steps.** What we observe is that the paths
are in the same regime, with sparse tracking dense within a fraction
of a nat, and the gap closing.

## Samples at step 1000

Both paths produce character-soup at this training budget. Committed
at `docs/demos/demo_16_60m_samples.txt` for completeness.

Prompt: `ROMEO:\n`, temperature 0.8, 200 new tokens.

**Dense (val 2.727):**

```
ROMEO:
TAero s ft t hendThasst tdGSur,

S me mIend thayoy pe wourenrourrihehosll hes and whe te hisowanerit'somine? b ss wely Etrntun he EutherseS'e inesousaes weanghilsthituNuayenme w, wiseel, wen ou
P, myo
```

**Sparse all (val 2.836):**

```
ROMEO:
Awiouke re gTupse ard sV&es gender Rz

VOr anMyy on athaNEolin y tonthe othethin o thr bJ?ithe lschids p NxsthesichacZCUfhallthan INIM the u hePHithang s tond y se our,es fhat be, lifoHe pshit I ware
```

Neither path has learned English-shaped distributions yet. That's
not a failure — it's what 1000 steps on a 40M model looks like on a
character-level task. The demo_15 10k-step run is where
structurally-correct samples start emerging. The question this demo
answers is memory and wallclock at scale, not sample quality.

## Why this scale and not 100M / GPT-2 small

Three reasons we stopped at ~40M for this run:

1. **Afternoon-scale compute.** 1000 steps at GPT-2 small (124M)
   would be ~4 hours dense + ~12+ hours sparse on an M3 Pro. ~40M
   finished in 27 minutes, keeping the feedback loop tight.
2. **Scaling signal, not absolute result.** The question is "does
   the ratio hold as you scale up?" — going 10M → 40M is enough to
   see the trend. Going to 124M adds compute time without adding a
   second qualitative data point.
3. **Clean two-path comparison.** At 124M the three-path (dense /
   ffn-only / all-sparse) run would have been a full day. The
   two-path scaled comparison (dense vs all-sparse) is the cleanest
   evidence we need.

**The 100M+ regime remains open territory.** If a contributor has
CPU cluster time to throw at it, a 124M scale-up with convergence
budget is a real community opportunity — it would be the first
independent reproduction of the memory claim above the author's
workstation.

## What we'll do next (v0.2, informed by this run)

1. **`dW` kernel vectorization.** At 40M the `dW` kernel was again
   the dominant cost, and the narrowing slowdown ratio (4.6× → 4.1×)
   shows it scales roughly linearly with live weights. A NEON `dW`
   is the single highest-ROI item on the v0.2 roadmap.
2. **Memory profiling at 100M+.** Memory ratio held at 4× scale-up.
   Circumstantial evidence it holds at 12× (the jump to GPT-2 small)
   is strong, but direct measurement from a volunteer's cluster is
   the v0.2 validation.
3. **Adam/AdamW path.** This run used SGD to match demo_15's
   methodology. Real training runs will use Adam, which doubles
   memory per *dense* parameter (m, v moment buffers) while scaling
   with *live count* for sparse. The memory advantage of sparse
   widens under Adam. Worth measuring.

## Reproducibility artifacts

- Runner: `examples/demo_16_mini_gpt_60m.py`
- Plot: `docs/demos/demo_16_60m_curves.png`
- Samples: `docs/demos/demo_16_60m_samples.txt`
- Log: `logs/demo16_60m_20260423_161456.log` (local-only; not
  committed due to size)
