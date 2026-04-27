"""
Demo 20 — Global-skip FFN transformer at 40M-param scale.

Research question
─────────────────
Does giving each transformer block's FFN direct sparse access to the
outputs of ALL previous blocks (not just the immediately prior one)
improve training dynamics vs the standard sequential FFN pattern?

This demo ONLY runs the global-skip configuration. We compare against
pre-recorded numbers from demo_16 (40M-scale scaling run):

    Config                   ms/step    val loss @ 1000 steps
    Dense (sequential FFN)      320              2.727
    Sparse (sequential FFN)    1326              2.836
    Global-skip FFN (this demo)  ?                  ?      ← what we measure

If global-skip FFN trains comparably to sparse-sequential at matched
live-param budget, the architectural flexibility is "free" — we should
explore it further. If it's substantially worse, the deeper
connectivity doesn't pay for itself at 1000 steps on this workload.

Architecture
────────────
Same transformer shape as demo_15 / demo_16:
  8 layers, d_model=640, d_ff=2560, 10 heads, seq=128, batch=8

In each block:
  h_in_N = LayerNorm(x_N)           # dense, pre-FFN normalization
  ffn_input_N = concat(h_in_N, h_out_0, h_out_1, ..., h_out_{N-1})
                # dim = d_model * (N+1)
  ffn_up(ffn_input_N) → (B, T, d_ff)         # SPARSE, global-skip
  gelu()
  ffn_down → (B, T, d_model)                 # SPARSE, standard
  x_{N+1} = x_N + ffn_down_output            # residual (local, not skipped)
  h_out_N = x_{N+1}                          # cached for later blocks

Attention is the same 70%-sparse pattern demo_16 uses. Only the FFN
architecture changes vs demo_16's all-sparse path.

Matched-param budget
────────────────────
Demo_16's all-sparse path has total live params ≈ 6.7M. We match
that here by choosing FFN sparsity such that the sum across all
global-skip FFN up/down weights equals ~5M live (attention adds ~1.5M).

Usage
─────
    python examples/demo_20_global_skip_transformer.py
    python examples/demo_20_global_skip_transformer.py --steps 500  # quicker

Runtime: ~30-45 min on M3 Pro for 1000 steps.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import demo_15_mini_gpt as d15        # reuse load_data, get_batch

import sparselab


# ─── Arch constants (match demo_16) ──────────────────────────────────
D_MODEL    = 640
D_FF       = 2560
N_HEADS    = 10
N_LAYERS   = 8
SEQ_LEN    = 128
BATCH_SIZE = 8
LR         = 3e-3
SEED       = 42

DEFAULT_N_STEPS = 1000
EVAL_EVERY      = 100
SAMPLE_EVERY    = 500

# Sparsity settings. FFN sparsity is chosen so the total live-param count
# across the 8 global-skip FFN layers matches demo_16's sparse-all FFN live
# count (~6.7M total).
#
# At FFN sparsity=0.965, total live = ~6.64M (within 1.5% of demo_16's 6.74M).
# The slightly higher sparsity vs demo_16's 0.90 FFN is because our
# fc_up weights are structurally larger: global-skip's layer-7 fc_up
# sees an 8*d_model concat input (5120 features), vs demo_16's 640.
FFN_SPARSITY  = 0.965
ATTN_SPARSITY = 0.70


# Rebind demo_15's module constants so the existing attention builder
# (CausalSelfAttention) sees the right shape.
d15.D_MODEL    = D_MODEL
d15.N_HEADS    = N_HEADS
d15.N_LAYERS   = N_LAYERS
d15.SEQ_LEN    = SEQ_LEN
d15.BATCH_SIZE = BATCH_SIZE
d15.LR         = LR
d15.SEED       = SEED
d15.EVAL_EVERY = EVAL_EVERY
d15.SAMPLE_EVERY = SAMPLE_EVERY


# ─────────────────────────────────────────────────────────────────────
#  Global-skip FFN block
# ─────────────────────────────────────────────────────────────────────
#
# Layer N's FFN-up reads from the concatenation of its LayerNormed
# input and the OUTPUTS of every previous block. Dimensions grow
# with N: layer 0 FFN-up sees just d_model, layer 7 sees 8*d_model.
# ─────────────────────────────────────────────────────────────────────

class GlobalSkipFFN(nn.Module):
    """FFN that reads from all previous block outputs (via sparse router).

    Args:
        layer_idx: position in the transformer stack (0-indexed). Block 0
                   sees just its own input; block N sees its own input +
                   outputs of all blocks 0..N-1.
        d_model:   per-block width. Individual block's contribution to
                   the concat.
        d_ff:      FFN hidden width (the middle of the gate).
        sparsity:  fraction of connections to drop in the sparse up/down
                   weights.
        near_bias: connectivity strategy for fc_up. Controls how live
                   connections are distributed across distance buckets:
                   - 0.0  (default): uniform random across all concat
                           features. This is the Demo 20 baseline.
                   - x > 0: near-biased stratified. x fraction of each
                           output row's live connections are drawn from
                           the "immediate-prior" bucket (the block
                           output that is 1 hop back — the most
                           recently computed features before this
                           block's input), and (1-x) are scattered
                           uniformly across all older buckets. Block 0
                           has no "prior" block, so it falls back to
                           uniform over its only bucket (its own LN
                           input).
    """
    def __init__(self, layer_idx: int, d_model: int, d_ff: int,
                 sparsity: float, near_bias: float = 0.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.near_bias = near_bias
        in_dim_up = d_model * (layer_idx + 1)          # +1 for own normed input

        # fc_up reads from the concat (layer_idx+1 block outputs worth of dim).
        # This is a SparseLinear whose col_indices range over the full
        # concat dim. Our col_indices are int32; at worst
        # d_model * N_LAYERS = 640 * 8 = 5120, well within int32.
        self.fc_up = sparselab.SparseLinear(
            in_dim_up, d_ff, sparsity=sparsity, bias=False
        )

        # If near-biased stratified is requested, replace fc_up's random
        # mask with a structured one. The layer's live-weight COUNT stays
        # the same (so sparsity and total-params match the baseline); only
        # the DISTRIBUTION across distance buckets changes.
        if near_bias > 0.0 and layer_idx > 0:
            _apply_near_biased_mask(self.fc_up, layer_idx, d_model,
                                    sparsity, near_bias)

        # fc_down is the standard sparse output projection. No need for
        # distance-biased mask here: fc_down's input is a single dense
        # d_ff-wide activation (the FFN's hidden layer), so there's no
        # "distance" concept to stratify over.
        self.fc_down = sparselab.SparseLinear(
            d_ff, d_model, sparsity=sparsity, bias=False
        )

    def forward(self, ffn_input_cat: torch.Tensor) -> torch.Tensor:
        # ffn_input_cat shape: (B, T, d_model * (layer_idx + 1))
        return self.fc_down(F.gelu(self.fc_up(ffn_input_cat)))


def _apply_near_biased_mask(
    sparse_linear: sparselab.SparseLinear,
    layer_idx: int,
    d_model: int,
    sparsity: float,
    near_bias: float,
) -> None:
    """Rebuild sparse_linear's internal PaddedCSR with a near-biased mask.

    The fc_up input for block N has shape (d_model * (N+1)) and is
    laid out (see GlobalSkipBlock.forward) as:
        [h_norm | block_0_out | block_1_out | ... | block_{N-1}_out]
    where h_norm (this block's LN-normalized input) occupies indices
    [0, d_model), and the remaining block outputs (the "history") fill
    indices [d_model, in_features).

    The "immediate prior" = h_norm. Why h_norm rather than
    block_{N-1}'s output? Because h_norm is what a vanilla transformer's
    FFN consumes, so at near_bias=1.0 we recover sparse-sequential's
    behavior exactly. This makes the near_bias knob a clean
    interpolation between sparse-sequential (1.0) and uniform
    global-skip (0.0).

    For each output row, we sample:
      - near_k = round(K * near_bias)  columns from h_norm's bucket
      - far_k  = K - near_k             columns scattered uniformly across
                                        the history bucket
    where K is the target live count for that row (matches what uniform
    random at the same sparsity would give on average, so total live
    count is preserved).

    Layer 0 has no history, so all K connections go to h_norm (which is
    also the entire input) — this is identical to a non-skip sparse FFN.

    Why this approach (rather than asking SparseLinear to accept a custom
    mask at construction): SparseLinear doesn't expose a mask-injection
    API yet, so we rebuild the internal PaddedCSR in place. This keeps
    the library API stable while letting us experiment with different
    connectivity priors in user code.
    """
    in_features = sparse_linear.in_features
    out_features = sparse_linear.out_features

    # Target live count per output row. We match what random-uniform at
    # this sparsity would give ON AVERAGE: each row gets round(F * (1-s))
    # live entries. This is deterministic (no binomial noise) because
    # we're constructing the mask directly.
    target_k = max(1, round(in_features * (1.0 - sparsity)))
    near_k = max(1, round(target_k * near_bias))
    far_k = max(0, target_k - near_k)

    # Bucket boundaries.
    #
    # Concat layout produced by GlobalSkipBlock.forward:
    #     [h_norm | block_0_out | block_1_out | ... | block_{N-1}_out]
    #      ^--- d_model ---^     ^--- d_model ---^   ^--- d_model ---^
    #
    # For near-biased stratified, "immediate prior" means the LayerNormed
    # input of THIS block (h_norm) — occupying indices [0, d_model). This
    # is the thing a vanilla transformer FFN consumes, so at near_bias=1.0
    # we recover the sparse-sequential behavior from demo_16 exactly.
    # All block-output history (indices [d_model, in_features)) is the
    # "far" bucket pool.
    near_start = 0
    near_end = d_model           # [0, d_model)
    far_start = d_model          # [d_model, in_features)
    far_end = in_features

    # Construct a dense float32 weight matrix and then a boolean mask.
    # The values come from the sparsity-aware Kaiming init we just
    # applied inside SparseLinear — we read them back, then re-sparsify
    # with our structured mask.
    current_W = np.asarray(sparse_linear._csr.values).copy()  # (total_cap,)
    # Re-draw values with the correct bound so near-k and far-k both
    # land in a statistically sensible range. We use the same
    # sparsity-aware Kaiming bound SparseLinear uses internally.
    effective_fan_in = max(1.0, in_features * (1.0 - sparsity))
    bound = 1.0 / math.sqrt(effective_fan_in)

    # Build the dense mask + values from scratch.
    rng = np.random.default_rng(seed=hash(("near_biased_mask", layer_idx)) & 0xFFFFFFFF)
    W_dense = np.zeros((out_features, in_features), dtype=np.float32)
    for i in range(out_features):
        # near-k columns from the "immediate prior" bucket (h_norm).
        near_cols = rng.choice(
            np.arange(near_start, near_end), size=near_k, replace=False
        )
        # far-k columns from block-output history (if any exists).
        # Layer 0 has no history yet, so far_end == far_start; fall
        # back to uniform over the entire input (which IS just h_norm).
        if far_end > far_start and far_k > 0:
            far_cols = rng.choice(
                np.arange(far_start, far_end), size=far_k, replace=False
            )
            cols = np.concatenate([near_cols, far_cols])
        else:
            # No history to sample from — the "near" allocation is all
            # we can do. If we didn't hit target_k, top up uniformly
            # from near bucket (avoids under-filling row).
            shortfall = target_k - len(near_cols)
            if shortfall > 0:
                extra = rng.choice(
                    np.setdiff1d(np.arange(near_start, near_end), near_cols),
                    size=min(shortfall, (near_end - near_start) - len(near_cols)),
                    replace=False,
                )
                cols = np.concatenate([near_cols, extra])
            else:
                cols = near_cols
        W_dense[i, cols] = rng.uniform(-bound, bound, size=len(cols)).astype(np.float32)

    # Replace the PaddedCSR and re-alias the Parameter.
    from sparselab import layout as _layout
    W_tensor = torch.from_numpy(W_dense)
    new_csr = _layout.from_dense(W_tensor, padding_ratio=sparse_linear.padding_ratio)
    sparse_linear._csr = new_csr
    values_np = np.asarray(new_csr.values)
    sparse_linear._values = nn.Parameter(torch.from_numpy(values_np))


# ─────────────────────────────────────────────────────────────────────
#  Global-skip block
# ─────────────────────────────────────────────────────────────────────

class GlobalSkipBlock(nn.Module):
    """Transformer block with global-skip FFN.

    Attention is the same sparse pattern demo_16 uses. Only the FFN
    changes: it reads from concat(its own LN input, all prior block
    outputs).

    Note: the block ALSO maintains a standard residual connection
    around the FFN. The global-skip is additive to the residual, not
    a replacement — the residual gives us the usual gradient
    superhighway, and the global-skip lets the FFN sample deeper
    into history for ITS input features.
    """
    def __init__(self, layer_idx: int, d_model: int, d_ff: int, n_heads: int,
                 attn_sparsity: float, ffn_sparsity: float,
                 near_bias: float = 0.0):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention LayerNorm
        self.ln1 = nn.LayerNorm(d_model)

        # Standard sparse attention (same as demo_15's CausalSelfAttention)
        self.attn = d15.CausalSelfAttention(d_model, n_heads, attn_sparsity)

        # Pre-FFN LayerNorm
        self.ln2 = nn.LayerNorm(d_model)

        # Global-skip FFN (with optional near-biased stratified mask)
        self.ffn = GlobalSkipFFN(layer_idx, d_model, d_ff, ffn_sparsity,
                                 near_bias=near_bias)

    def forward(self, x: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        """
        x:        current block's input, shape (B, T, d_model)
        history:  list of previous blocks' OUTPUTS, each shape (B, T, d_model)
                  (empty list for block 0)
        """
        # Attention with residual — identical to standard transformer.
        x = x + self.attn(self.ln1(x))

        # Build FFN input by concatenating this block's normed input
        # with all history block outputs (on the feature dim).
        h_norm = self.ln2(x)
        if len(history) > 0:
            ffn_input = torch.cat([h_norm] + history, dim=-1)
        else:
            ffn_input = h_norm

        # FFN with residual. FFN reads from the concat but only the
        # standard d_model width gets added back to x.
        x = x + self.ffn(ffn_input)
        return x


# ─────────────────────────────────────────────────────────────────────
#  Global-skip mini-GPT
# ─────────────────────────────────────────────────────────────────────

class GlobalSkipMiniGPT(nn.Module):
    def __init__(self, vocab_size: int,
                 ffn_sparsity: float = FFN_SPARSITY,
                 attn_sparsity: float = ATTN_SPARSITY,
                 near_bias: float = 0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            GlobalSkipBlock(
                layer_idx=i, d_model=D_MODEL, d_ff=D_FF, n_heads=N_HEADS,
                attn_sparsity=attn_sparsity, ffn_sparsity=ffn_sparsity,
                near_bias=near_bias,
            )
            for i in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(SEQ_LEN))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos_ids[:T])

        # History accumulates each block's OUTPUT for later blocks to
        # read. We append after the block runs.
        history: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, history)
            history.append(x)

        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n_new_tokens: int,
                 temperature: float = 0.8) -> torch.Tensor:
        self.eval()
        for _ in range(n_new_tokens):
            idx_cond = idx[:, -SEQ_LEN:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def count_live_params(model: nn.Module) -> tuple[int, int]:
    dense = 0
    sparse_live = 0
    for m in model.modules():
        if isinstance(m, sparselab.SparseLinear):
            sparse_live += m.nnz
            if m.bias is not None:
                dense += m.bias.numel()
        else:
            for p in m.parameters(recurse=False):
                dense += p.numel()
    return dense, sparse_live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--tag", type=str, default=None,
                        help="Filename suffix for samples/plot. "
                             "Default: auto-generated from near_bias.")
    parser.add_argument("--near_bias", type=float, default=0.0,
                        help="Fraction of each FFN's live connections to "
                             "route to h_norm (this block's LN input). "
                             "0.0 = uniform global-skip (demo_20 baseline). "
                             "1.0 = sparse-sequential equivalent. "
                             "0.5 = near-biased stratified. Default 0.0.")
    args = parser.parse_args()

    if args.tag is None:
        args.tag = f"nb{int(args.near_bias * 100):03d}"

    start_wallclock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("SparseLab demo 20 — Global-skip FFN transformer")
    print(f"  Started:   {start_wallclock}")
    print(f"  Arch:      {N_LAYERS}L × d_model={D_MODEL} × d_ff={D_FF}")
    print(f"  Sparsity:  FFN={FFN_SPARSITY}  ATTN={ATTN_SPARSITY}")
    print(f"  near_bias: {args.near_bias}  (fraction of live FFN-up "
          f"connections routed to h_norm bucket)")
    print(f"  Steps:     {args.steps}   Batch: {BATCH_SIZE}   Seq: {SEQ_LEN}")
    print(f"  Seed:      {SEED}")
    print(f"  Compare vs demo_16: dense=320ms/2.727, sparse-all=1326ms/2.836")
    print("=" * 72, flush=True)

    print("\nLoading Tiny Shakespeare...", flush=True)
    train_ids, val_ids, vocab_size, itos, stoi = d15.load_data()
    print(f"  {len(train_ids):,} train chars, vocab={vocab_size}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = GlobalSkipMiniGPT(vocab_size, near_bias=args.near_bias)
    dense_p, sparse_p = count_live_params(model)
    total_p = dense_p + sparse_p
    print(f"\nParameters:")
    print(f"  Dense:       {dense_p:>10,}")
    print(f"  Sparse live: {sparse_p:>10,}")
    print(f"  Total:       {total_p:>10,}   (target: match demo_16 ~6.7M)")
    print("=" * 72, flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=LR)

    samples_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        f"demo_20_{args.tag}_samples.txt",
    )
    if os.path.exists(samples_path):
        os.remove(samples_path)
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(f"Global-skip FFN transformer samples (demo_20, {args.tag})\n")
        f.write(f"Steps: {args.steps}  Seed: {SEED}  "
                f"Started: {start_wallclock}\n")
        f.write("=" * 60 + "\n")

    print("\nTraining...", flush=True)
    train_losses = []
    val_losses = []
    running_loss = 0.0
    n_running = 0
    t_start = time.perf_counter()

    for step in range(args.steps):
        x, y = d15.get_batch(train_ids, BATCH_SIZE, SEQ_LEN)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1))
        loss.backward()
        opt.step()

        running_loss += loss.item()
        n_running += 1

        if (step + 1) % EVAL_EVERY == 0:
            train_avg = running_loss / n_running
            running_loss, n_running = 0.0, 0
            val_avg = d15.compute_val_loss(model, val_ids)
            train_losses.append((step + 1, train_avg))
            val_losses.append((step + 1, val_avg))
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (step + 1) * (args.steps - step - 1)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] step {step+1:>5d}/{args.steps}  "
                  f"train={train_avg:.3f}  val={val_avg:.3f}  "
                  f"({elapsed:.0f}s, eta {eta/60:.1f}m)", flush=True)

        if (step + 1) % SAMPLE_EVERY == 0 or (step + 1) == args.steps:
            sample = d15.generate_sample(model, itos, stoi)
            with open(samples_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- step {step+1} ---\n{sample}\n")

    total_time = time.perf_counter() - t_start
    ms_per_step = (total_time * 1000.0) / args.steps
    final_val = val_losses[-1][1] if val_losses else float("nan")

    print("\n" + "=" * 72)
    print(f"Final — global-skip FFN transformer, {args.steps} steps")
    print("-" * 72)
    print(f"  Per-step wallclock:  {ms_per_step:>7.0f} ms")
    print(f"  Final val loss:      {final_val:>7.3f}")
    print(f"  Total params:        {total_p:>7,}")
    print(f"  Total time:          {total_time/60:>6.1f} min")
    print("-" * 72)
    print("\nComparison (demo_16, same arch family, 1000 steps):")
    print(f"  Dense baseline:       320 ms/step   val=2.727")
    print(f"  Sparse all (seq FFN):1326 ms/step   val=2.836")
    print(f"  Global-skip (now):   {ms_per_step:>4.0f} ms/step   val={final_val:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
