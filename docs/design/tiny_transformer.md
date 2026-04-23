# Design: Tiny Transformer Launch Demo (Milestone 4g)

## What this demo is

The launch artifact. A tiny decoder-only transformer, trained on Tiny
Shakespeare, that actually converges and generates coherent text —
on a MacBook, in under 10 minutes, with sparse FFN weights via
SparseCore.

This is the launch demo: the "show, don't tell" proof that
sparse-from-scratch training is practical on CPU.

## What this demo is NOT

- **Not state-of-the-art.** We're training a ~200k-param model, not
  GPT-2. The bar is "generates recognizable English-like text," not
  "writes Shakespeare."
- **Not a fair compute comparison with GPUs.** CPU training is slower
  per GFLOP. The story is about *memory* and *accessibility*, not
  throughput.
- **Not fine-tuned for quality.** We train one seed, one
  configuration. Hyperparameter search is out of scope.

## The research claim we can make honestly

> "A 90%-sparse tiny transformer trained from scratch with SparseCore
> on an M3 Pro MacBook reaches comparable validation loss to its
> dense equivalent, in comparable wall-clock time, while using
> <20% of the dense memory footprint."

Things this claim requires us to actually measure:

1. **Validation loss parity** (or a documented small gap). Sparse @
   90% should be within ~10% of dense validation loss.
2. **Wall-clock comparable.** Not faster (we've been honest about
   autograd overhead). Comparable. Same order of magnitude.
3. **Memory reduction.** Computed exactly from tensor sizes, same
   method as demo_05.
4. **Generates text.** Qualitative check — samples at checkpoints
   showing the model has learned character transitions.

## Architecture

Matches the spike from the profile run, validated at that scale:

```
TinyTransformer(
    vocab_size   = 65,        # Tiny Shakespeare char set
    d_model      = 128,
    d_ff         = 512,       # 4x d_model, standard ratio
    n_heads      = 4,
    n_layers     = 2,
    seq_len      = 64,
)
```

**Parameter breakdown:**

| Component | Params | Sparsified? |
|-----------|--------|-------------|
| Token embeddings | 65 × 128 = 8,320 | no |
| Position embeddings | 64 × 128 = 8,192 | no |
| Attention QKV per layer | 3 × 128 × 128 = 49,152 | no |
| Attention O per layer | 128 × 128 = 16,384 | no |
| FFN up per layer | 128 × 512 = 65,536 | **yes** |
| FFN down per layer | 512 × 128 = 65,536 | **yes** |
| LayerNorms | ~1,000 | no |
| Output head | 128 × 65 = 8,320 | no |
| **Total (dense)** | **~285,000** | — |

At 90% FFN sparsity, live FFN params = 26,214. Total live =
~165k instead of ~285k — a 42% model-wide parameter reduction
(memory story). Note we're NOT sparsifying attention yet, so
the savings are FFN-only. In a real transformer at scale attention
is a much smaller fraction of total params, so this will look even
better at scale.

## Why not sparsify attention?

Three reasons to defer sparse attention to v0.2:

1. **QKV matrices are small** (128×128 = 16k params each) and
   under-benefit from sparsity at this scale.
2. **Published DST results on attention are mixed** — random masks
   on attention hurt more than on FFN because connectivity
   between tokens and heads matters a lot.
3. **Keeping attention dense lets us isolate the sparse-FFN effect.**
   Scientific clarity.

## Dataset

**Tiny Shakespeare** (`input.txt`, ~1 MB, ~1.1M characters). Standard
char-level language modeling benchmark used in countless tutorials
(karpathy/nanoGPT, etc.). It's small, ASCII, has clear structure
(speaker names, stage directions), and ~65 unique characters.

Download URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Split: 90% train, 10% validation. Validation = last 10% of the file
for reproducibility.

## Training config

```python
BATCH_SIZE   = 16
SEQ_LEN      = 64
LR           = 3e-3          # SGD works fine at this scale
N_STEPS      = 5000          # ~5 epochs equivalent
EVAL_EVERY   = 500           # log train+val loss
SAMPLE_EVERY = 1000          # generate text
```

Using SGD not Adam deliberately — we haven't tested Adam + SparseLinear
with DST exhaustively, and SGD is known to work. We can punt the
"does Adam play nice with DST" experiment to v0.2.

## Comparison paths

Three training runs, same seed, same config except for the sparsity
algorithm:

1. **Dense** — `nn.Linear` FFN, no sparsity. Gold standard.
2. **Sparse + Static** — `SparseLinear` at 90% with frozen random
   mask. This is the "naive sparse" baseline.
3. **Sparse + SET(0.1)** — best-performing DST from milestone 4f
   drop-fraction sweep.

We skip RigL here because:
- It lost to both Static and SET in our MNIST sweep
- Including a fourth path triples demo runtime
- The launch story is cleaner with 3 paths than 4

If RigL *would* have won at transformer scale (which it might),
we note this as a follow-up experiment. The infrastructure supports
swapping SET for RigL in one line.

## Text sampling

Every 1000 steps, for each path, generate a 200-character sample
starting from the prompt "ROMEO:\n". Temperature=1.0, no beam search.
We keep all samples in a file so the demo output includes:

- Step 0:    random garbage (baseline)
- Step 1000: "something that looks vaguely like character-level text"
- Step 5000: "recognizable English-like text with structure"

## What "success" looks like

A reasonable demo outcome:

```
Path             Val loss (final)   Wall-clock   Live params   Sample @ step 5000
Dense            ~1.6               ~8 min       285k          "ROMEO: I pray you, sir, be not..."
Sparse+Static    ~1.8               ~8 min       165k (58%)    "ROMEO: he wou be to the..."
Sparse+SET(0.1)  ~1.7               ~8 min       165k (58%)    "ROMEO: I pray thee, my lord..."
```

Numbers are illustrative — we'll measure the real ones. But the
qualitative story is: sparse training works, the samples look
similar in quality, and the memory footprint is genuinely smaller.

## What if it doesn't work?

Three failure modes to be prepared for:

1. **Sparse loss diverges.** Unlikely given MNIST worked, but if so,
   lower LR, smaller batch, or reduce sparsity to 80%. Debug before
   shipping.

2. **Sparse is meaningfully worse than dense (>20% val loss gap).**
   This would actually be interesting — we'd want to understand
   *why* before launching. Possibilities:
   - Transformer connectivity genuinely needs >10% density
   - Our mask at init is bad (RigL might help — try it as fallback)
   - Training budget too short

3. **Wall-clock is painful (>30 min).** Re-check autograd overhead,
   maybe reduce `n_steps` or shrink batch/seq.

If we hit any of these, we adjust and document the adjustment
honestly in the milestone writeup. No silent tuning.

## What lives in the final artifact

```
examples/demo_13_tiny_transformer.py       # the runnable demo
docs/demos/milestone_10.md                  # the launch writeup
docs/demos/demo_13_loss_curves.png          # train + val loss, 3 paths
docs/demos/demo_13_samples.txt              # text samples at checkpoints
```

## Prior art consulted

- **karpathy/nanoGPT** — architecture shape (decoder-only,
  pre-layer-norm, GELU FFN). Our model is essentially a shrunk
  nanoGPT with sparse FFN.
- **RigL paper § 5** — for language modeling results at 90%
  sparsity (they use wikitext/PTB, not Shakespeare, but the
  architectural observations carry over).
- **nothing from HuggingFace** — their transformer implementations
  are too complex for a 200k-param demo.
