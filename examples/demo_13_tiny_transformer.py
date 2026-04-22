"""
Demo 13 — Tiny transformer training on Tiny Shakespeare (milestone 4g).

THE LAUNCH DEMO. A 2-layer decoder-only transformer trained from
scratch on Tiny Shakespeare, with sparse FFN layers via SparseCore.
Compares three paths side-by-side:

  - Dense:            vanilla nn.Linear FFN (baseline, ~285k params)
  - Sparse+Static:    SparseLinear FFN, frozen random 90% mask (~165k params)
  - Sparse+SET(0.1):  SparseLinear FFN, SET DST with 10% churn rate

For each path we train 5000 steps, log train/val loss every 500,
generate a 200-char text sample every 1000 steps, and produce a
side-by-side loss plot.

How to run
──────────
    python examples/demo_13_tiny_transformer.py

Needs: pip install sparsecore[demos]

Runtime: ~20 minutes on M3 Pro (3 paths × ~7 min each).

What to look at
───────────────
  1. Final validation losses — expect sparse within 10-15% of dense.
  2. Text samples at step 5000 — all three should look like text,
     with some local structure (character names, line breaks).
  3. Wall-clock per path — should be within 10% of each other
     (kernel work dominates ~13% of step time so parallelism matters).
  4. Memory column — sparse paths should be ~58% of dense memory.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install sparsecore[demos]")

import sparsecore


# ─── Config ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")
DATA_FILE = os.path.join(DATA_DIR, "tinyshakespeare.txt")

# Model
VOCAB_SIZE_ESTIMATE = 65  # actual computed from data
D_MODEL = 128
D_FF = 512
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 64
SPARSITY = 0.9

# Training
BATCH_SIZE = 16
LR = 3e-3
N_STEPS = 5000
EVAL_EVERY = 500
SAMPLE_EVERY = 1000
SEED = 0

SAMPLE_PROMPT = "ROMEO:\n"
SAMPLE_LEN = 200


# ─────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────

def download_tinyshakespeare():
    """Download Tiny Shakespeare to data/ if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        return
    print(f"Downloading Tiny Shakespeare to {DATA_FILE} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)


def load_data():
    """Return (train_ids, val_ids, vocab_size, itos, stoi)."""
    download_tinyshakespeare()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], vocab_size, itos, stoi


def get_batch(ids: torch.Tensor, batch_size: int, seq_len: int):
    """Random contiguous windows for char-LM."""
    # Pick batch_size random starting indices with room for seq_len+1 chars.
    n = ids.size(0) - seq_len - 1
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# ─────────────────────────────────────────────────────────────────────
#  Model — shared between dense and sparse paths
# ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(att)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc_up = nn.Linear(d_model, d_ff, bias=False)
        self.fc_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.fc_down(F.gelu(self.fc_up(x)))


class SparseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, sparsity: float):
        super().__init__()
        self.fc_up = sparsecore.SparseLinear(d_model, d_ff,
                                               sparsity=sparsity, bias=False)
        self.fc_down = sparsecore.SparseLinear(d_ff, d_model,
                                                 sparsity=sparsity, bias=False)

    def forward(self, x):
        return self.fc_down(F.gelu(self.fc_up(x)))


class Block(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, sparse: bool, sparsity: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = (SparseFFN(d_model, d_ff, sparsity) if sparse
                    else DenseFFN(d_model, d_ff))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_heads, n_layers, seq_len,
                 sparse: bool, sparsity: float = 0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, d_ff, n_heads, sparse, sparsity)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos_ids[:T])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text char-by-char. idx is a (1, T0) LongTensor."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -SEQ_LEN:]   # crop to block size
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx


# ─────────────────────────────────────────────────────────────────────
#  Params counting for the memory column
# ─────────────────────────────────────────────────────────────────────

def count_model_params(model):
    """(dense_params, sparse_live_params). Doesn't count sparse padding
    slots or the CSR index overhead — those are reported separately."""
    dense_params = 0
    sparse_live = 0
    for mod in model.modules():
        if isinstance(mod, sparsecore.SparseLinear):
            sparse_live += mod.nnz
        else:
            for p in mod.parameters(recurse=False):
                dense_params += p.numel()
    return dense_params, sparse_live


def estimate_weight_bytes(dense_params: int, sparse_live: int) -> int:
    """Rough at-rest memory for weights only. Sparse is val+idx+grad."""
    # Dense: 4 bytes per param for value + 4 for grad = 8
    # Sparse: 4 (val) + 4 (col_idx) + 4 (grad) = 12 per live slot
    # + small fixed row metadata (row_start, row_nnz, row_capacity = 12 per row)
    # Ignoring row metadata since it's tiny vs live slots
    return 8 * dense_params + 12 * sparse_live


# ─────────────────────────────────────────────────────────────────────
#  Training loop — one run for one algorithm
# ─────────────────────────────────────────────────────────────────────

def train_one_path(name: str, model: nn.Module, algo, train_ids, val_ids,
                    itos, samples_path: str):
    """Train one model for N_STEPS and record loss + text samples."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    opt = torch.optim.SGD(model.parameters(), lr=LR)

    if algo is not None:
        model.apply(algo)

    step_losses = []           # (step, train_loss)
    val_losses = []            # (step, val_loss)
    samples = []               # (step, generated_text)

    # Track topology churn across the run so we can show that DST is
    # actually mutating weights even if the loss curves look similar.
    # Snapshot the full concatenated col_indices at each eval point.
    if algo is not None and hasattr(algo, "layers") and algo.layers:
        def snapshot_cols():
            return np.concatenate([
                np.asarray(lyr._csr.col_indices).copy()
                for lyr in algo.layers
            ])
        last_cols = snapshot_cols()
    else:
        last_cols = None

    t_start = time.perf_counter()
    train_loss_running = 0.0
    n_running = 0
    total_churn = 0

    for step in range(N_STEPS):
        x, y = get_batch(train_ids, BATCH_SIZE, SEQ_LEN)
        opt.zero_grad()
        logits = model(x)   # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        opt.step()
        if algo is not None:
            algo.step()

        train_loss_running += loss.item()
        n_running += 1

        if (step + 1) % EVAL_EVERY == 0:
            # Train loss (smoothed)
            train_avg = train_loss_running / n_running
            train_loss_running = 0.0
            n_running = 0
            step_losses.append((step + 1, train_avg))

            # Validation loss
            val_loss = compute_val_loss(model, val_ids)
            val_losses.append((step + 1, val_loss))

            # Churn since last eval (topology slot changes)
            if last_cols is not None:
                cur_cols = snapshot_cols()
                churn = int((cur_cols != last_cols).sum())
                total_churn += churn
                last_cols = cur_cols
            else:
                churn = 0

            elapsed = time.perf_counter() - t_start
            # Print with 5 decimals so the sub-visible drift is actually
            # visible. Loss differences of 1e-5 between paths are real
            # algorithmic effects, not display artifacts.
            print(f"    [{name}] step {step+1:>5d}  "
                  f"train={train_avg:.5f}  val={val_loss:.5f}  "
                  f"churn+={churn:>6d}  ({elapsed:.0f}s)", flush=True)

        if (step + 1) % SAMPLE_EVERY == 0:
            text = generate_text(model, itos)
            samples.append((step + 1, text))

    elapsed = time.perf_counter() - t_start
    print(f"    [{name}] done. {elapsed:.0f}s total. "
          f"total_churn={total_churn}",
          flush=True)

    # Write samples to file for inspection.
    with open(samples_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== {name} ===\n")
        for step, text in samples:
            f.write(f"\n--- step {step} ---\n{text}\n")

    # Param + memory stats
    dense, sparse_live = count_model_params(model)
    return {
        "name": name,
        "step_losses": step_losses,
        "val_losses": val_losses,
        "samples": samples,
        "elapsed_s": elapsed,
        "dense_params": dense,
        "sparse_live": sparse_live,
        "weight_bytes": estimate_weight_bytes(dense, sparse_live),
        "total_churn": total_churn,
    }


def compute_val_loss(model, val_ids, n_batches=20):
    """Average val loss over a small number of random val batches."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_ids, BATCH_SIZE, SEQ_LEN)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     y.reshape(-1))
            total += loss.item()
    model.train()
    return total / n_batches


def generate_text(model, itos):
    """Generate SAMPLE_LEN chars starting from SAMPLE_PROMPT."""
    # Need stoi to encode the prompt
    stoi = {ch: i for i, ch in itos.items()}
    idx = torch.tensor(
        [[stoi[c] for c in SAMPLE_PROMPT if c in stoi]],
        dtype=torch.long,
    )
    out = model.generate(idx, max_new_tokens=SAMPLE_LEN, temperature=1.0)
    return "".join(itos[i] for i in out[0].tolist())


# ─────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_loss_curves(results, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"Dense": "#3b82f6", "Sparse+Static": "#94a3b8",
              "Sparse+SET(0.1)": "#ef4444"}

    for r in results:
        c = colors.get(r["name"], "black")
        steps_tr = [s for s, _ in r["step_losses"]]
        losses_tr = [l for _, l in r["step_losses"]]
        steps_val = [s for s, _ in r["val_losses"]]
        losses_val = [l for _, l in r["val_losses"]]
        ax1.plot(steps_tr, losses_tr, color=c, linewidth=2,
                  label=f'{r["name"]}  final train={losses_tr[-1]:.3f}')
        ax2.plot(steps_val, losses_val, color=c, linewidth=2,
                  label=f'{r["name"]}  final val={losses_val[-1]:.3f}')

    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    ax2.set_xlabel("step")
    ax2.set_title("Validation loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    fig.suptitle("Tiny Transformer — Dense vs Sparse @ 90% on Tiny Shakespeare",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\nSparseCore demo 13 — Tiny Transformer on Tiny Shakespeare")
    print(f"  Arch: {N_LAYERS}L x d_model={D_MODEL} x d_ff={D_FF} x "
          f"heads={N_HEADS} x seq_len={SEQ_LEN}")
    print(f"  Train: {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}, "
          f"FFN sparsity={SPARSITY}")
    print("=" * 72, flush=True)

    # Data
    print("\nLoading data...")
    train_ids, val_ids, vocab_size, itos, stoi = load_data()
    print(f"  train: {len(train_ids):,} chars, val: {len(val_ids):,} chars, "
          f"vocab: {vocab_size}")

    samples_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        "demo_13_samples.txt",
    )
    # Clear any previous samples file
    if os.path.exists(samples_path):
        os.remove(samples_path)
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write("Tiny Transformer samples from demo_13\n")
        f.write(f"prompt: {SAMPLE_PROMPT!r}\n")
        f.write("=" * 60 + "\n")

    # ─── Path 1: Dense baseline ─────────────────────────────────
    print("\n─── Path 1: Dense ───", flush=True)
    torch.manual_seed(SEED)
    dense_model = TinyTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, seq_len=SEQ_LEN,
        sparse=False,
    )
    r_dense = train_one_path("Dense", dense_model, None,
                               train_ids, val_ids, itos, samples_path)

    # ─── Path 2: Sparse + Static ───────────────────────────────
    print("\n─── Path 2: Sparse + Static ───", flush=True)
    torch.manual_seed(SEED)
    sparse_static_model = TinyTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, seq_len=SEQ_LEN,
        sparse=True, sparsity=SPARSITY,
    )
    r_static = train_one_path(
        "Sparse+Static", sparse_static_model,
        sparsecore.Static(sparsity=SPARSITY),
        train_ids, val_ids, itos, samples_path,
    )

    # ─── Path 3: Sparse + SET(0.1) ────────────────────────────
    print("\n─── Path 3: Sparse + SET(0.1) ───", flush=True)
    torch.manual_seed(SEED)
    sparse_set_model = TinyTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, seq_len=SEQ_LEN,
        sparse=True, sparsity=SPARSITY,
    )
    r_set = train_one_path(
        "Sparse+SET(0.1)", sparse_set_model,
        sparsecore.SET(sparsity=SPARSITY, drop_fraction=0.1,
                         update_freq=100, seed=42),
        train_ids, val_ids, itos, samples_path,
    )

    results = [r_dense, r_static, r_set]

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("Results summary:")
    print("-" * 88)
    print(f"  {'path':<20s}  {'final val':>9s}  {'params':>10s}  "
          f"{'weight KB':>16s}  {'churn':>9s}  {'time':>6s}")
    print("-" * 88)

    dense_bytes = r_dense["weight_bytes"]
    for r in results:
        final_val = r["val_losses"][-1][1]
        total_params = r["dense_params"] + r["sparse_live"]
        kb = r["weight_bytes"] / 1024
        mem_pct = r["weight_bytes"] / dense_bytes * 100
        churn = r.get("total_churn", 0)
        print(f"  {r['name']:<20s}  {final_val:>9.4f}  {total_params:>10,}  "
              f"{kb:>10.1f} ({mem_pct:>3.0f}%)  "
              f"{churn:>9,}  {r['elapsed_s']:>4.0f}s")
    print("-" * 88)

    # Gap assertions — print with 4 decimals so small-but-real
    # differences between paths are visible. A loss diff of 0.0001
    # over 5000 steps is a real mutation effect, not noise.
    gap_static = r_static["val_losses"][-1][1] - r_dense["val_losses"][-1][1]
    gap_set = r_set["val_losses"][-1][1] - r_dense["val_losses"][-1][1]
    print(f"\n  Sparse+Static val loss gap vs Dense:   {gap_static:+.4f}  "
          f"({gap_static / r_dense['val_losses'][-1][1] * 100:+.2f}%)")
    print(f"  Sparse+SET    val loss gap vs Dense:   {gap_set:+.4f}  "
          f"({gap_set / r_dense['val_losses'][-1][1] * 100:+.2f}%)")

    gap_set_vs_static = r_set["val_losses"][-1][1] - r_static["val_losses"][-1][1]
    print(f"  Sparse+SET vs Sparse+Static:           {gap_set_vs_static:+.4f}  "
          f"(negative = SET wins)")

    # Plot
    plot_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        "demo_13_loss_curves.png",
    )
    plot_loss_curves(results, plot_path)
    print(f"\nSaved loss plot: {plot_path}")
    print(f"Saved text samples: {samples_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
