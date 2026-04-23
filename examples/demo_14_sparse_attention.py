"""
Demo 14 — Sparse attention experiment (milestone 4g follow-up).

Demo 13 showed FFN-only sparsity at 90% matches dense loss within 2%.
Open question: can we sparsify ATTENTION too, and how much accuracy
do we trade for how much memory?

This demo trains four paths on Tiny Shakespeare, same seed, same
training budget, varying only what's sparsified and at what level:

  1. Dense                         — all nn.Linear, baseline
  2. Sparse FFN 90%               — only FFN sparse (demo_13 setup)
  3. Sparse mixed (attn 70%, FFN 90%) — gentler on attention
  4. Sparse uniform 90% everywhere  — stress test

The hypothesis being tested: attention is MORE connectivity-sensitive
than FFN because random Q/K/V sparsity severs specific token-to-head
paths. We expect path 3 to land between paths 2 and 4. Path 4 might
fail to converge.

How to run
──────────
    python examples/demo_14_sparse_attention.py

Needs: pip install sparselab[demos]

Runtime: ~10 min on M3 Pro (4 paths × ~2 min each).

What to look at
───────────────
  1. Final val losses — how much do we lose by going more aggressive?
  2. Memory column — what's the upside of sparsifying attention?
  3. The plot shows all four trajectories on one axis.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install sparselab[demos]")

import sparselab


warnings.filterwarnings("ignore", category=UserWarning)


# ─── Config (matches demo_13 so results are directly comparable) ─────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "tinyshakespeare.txt")

D_MODEL = 128
D_FF = 512
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 64

BATCH_SIZE = 16
LR = 3e-3
N_STEPS = 5000
EVAL_EVERY = 500
SEED = 0


# ─────────────────────────────────────────────────────────────────────
#  Data (copied from demo_13 — light enough to duplicate)
# ─────────────────────────────────────────────────────────────────────

def load_data():
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")
        urllib.request.urlretrieve(url, DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def get_batch(ids: torch.Tensor, batch_size: int, seq_len: int):
    n = ids.size(0) - seq_len - 1
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# ─────────────────────────────────────────────────────────────────────
#  Model with configurable per-component sparsity
# ─────────────────────────────────────────────────────────────────────
#
# The key new piece: we accept (attn_sparsity, ffn_sparsity) and
# switch the Linear construction accordingly. A sparsity of 0.0 means
# "use dense nn.Linear"; anything else means "use SparseLinear at
# that sparsity." This lets us drop in any component-level mix from
# a single model class.

def make_linear(in_f: int, out_f: int, sparsity: float, bias: bool = False):
    """Factory: dense nn.Linear if sparsity==0, else SparseLinear."""
    if sparsity <= 0.0:
        return nn.Linear(in_f, out_f, bias=bias)
    return sparselab.SparseLinear(in_f, out_f, sparsity=sparsity, bias=bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, sparsity: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Both QKV (3× wide) and output projection respect the same
        # attention sparsity level.
        self.qkv = make_linear(d_model, 3 * d_model, sparsity)
        self.o = make_linear(d_model, d_model, sparsity)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(att)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, sparsity: float):
        super().__init__()
        self.fc_up = make_linear(d_model, d_ff, sparsity)
        self.fc_down = make_linear(d_ff, d_model, sparsity)

    def forward(self, x):
        return self.fc_down(F.gelu(self.fc_up(x)))


class Block(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, attn_sparsity, ffn_sparsity):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, attn_sparsity)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, ffn_sparsity)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ConfigurableTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_heads, n_layers,
                 seq_len, attn_sparsity: float, ffn_sparsity: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, d_ff, n_heads, attn_sparsity, ffn_sparsity)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Output head stays dense — it's the prediction layer, sparsity
        # here hurts quality disproportionately and it's only ~8k params.
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos_ids[:T])
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ─────────────────────────────────────────────────────────────────────
#  Param counting
# ─────────────────────────────────────────────────────────────────────

def count_params(model):
    dense = 0
    sparse_live = 0
    for mod in model.modules():
        if isinstance(mod, sparselab.SparseLinear):
            sparse_live += mod.nnz
        else:
            for p in mod.parameters(recurse=False):
                dense += p.numel()
    return dense, sparse_live


def estimate_weight_bytes(dense: int, sparse_live: int) -> int:
    # 4 bytes value + 4 bytes grad for dense
    # 4 value + 4 col_idx + 4 grad for sparse (plus tiny row metadata)
    return 8 * dense + 12 * sparse_live


# ─────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────

def compute_val_loss(model, val_ids, n_batches=20):
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


def train_one_path(name: str, attn_sparsity: float, ffn_sparsity: float,
                    vocab_size: int, train_ids, val_ids):
    print(f"\n─── {name} (attn={attn_sparsity}, ffn={ffn_sparsity}) ───",
          flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = ConfigurableTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, seq_len=SEQ_LEN,
        attn_sparsity=attn_sparsity, ffn_sparsity=ffn_sparsity,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    step_losses = []
    val_losses = []
    train_loss_running = 0.0
    n_running = 0

    t_start = time.perf_counter()
    for step in range(N_STEPS):
        x, y = get_batch(train_ids, BATCH_SIZE, SEQ_LEN)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1))
        loss.backward()
        opt.step()

        train_loss_running += loss.item()
        n_running += 1

        if (step + 1) % EVAL_EVERY == 0:
            train_avg = train_loss_running / n_running
            train_loss_running = 0.0
            n_running = 0
            step_losses.append((step + 1, train_avg))
            val_loss = compute_val_loss(model, val_ids)
            val_losses.append((step + 1, val_loss))
            elapsed = time.perf_counter() - t_start
            print(f"    step {step+1:>5d}  train={train_avg:.4f}  "
                  f"val={val_loss:.4f}  ({elapsed:.0f}s)", flush=True)

    elapsed = time.perf_counter() - t_start
    dense, sparse_live = count_params(model)
    return {
        "name": name,
        "attn_sparsity": attn_sparsity,
        "ffn_sparsity": ffn_sparsity,
        "step_losses": step_losses,
        "val_losses": val_losses,
        "elapsed_s": elapsed,
        "dense_params": dense,
        "sparse_live": sparse_live,
        "weight_bytes": estimate_weight_bytes(dense, sparse_live),
    }


def plot_all(results, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = [
        "#3b82f6",   # Dense — blue
        "#10b981",   # Sparse FFN 90% — green
        "#f59e0b",   # Mixed — amber
        "#ef4444",   # Uniform 90% — red
    ]

    for r, c in zip(results, colors):
        steps_tr = [s for s, _ in r["step_losses"]]
        losses_tr = [l for _, l in r["step_losses"]]
        steps_val = [s for s, _ in r["val_losses"]]
        losses_val = [l for _, l in r["val_losses"]]
        ax1.plot(steps_tr, losses_tr, color=c, linewidth=2,
                  label=f'{r["name"]}  final={losses_tr[-1]:.3f}')
        ax2.plot(steps_val, losses_val, color=c, linewidth=2,
                  label=f'{r["name"]}  final={losses_val[-1]:.3f}')

    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.set_xlabel("step")
    ax2.set_title("Validation loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Sparse attention experiment on Tiny Shakespeare",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    print("\nSparseLab demo 14 — sparse attention experiment")
    print(f"  Arch: {N_LAYERS}L x d_model={D_MODEL} x d_ff={D_FF} x "
          f"heads={N_HEADS} x seq_len={SEQ_LEN}")
    print(f"  Training: {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
    print("=" * 72, flush=True)

    train_ids, val_ids, vocab_size = load_data()
    print(f"  Data: {len(train_ids):,} train chars, vocab={vocab_size}")

    configs = [
        ("Dense",                 0.0,  0.0),
        ("FFN 90%",               0.0,  0.9),
        ("Attn 70% + FFN 90%",    0.7,  0.9),
        ("Uniform 90%",           0.9,  0.9),
    ]

    results = []
    for name, attn_s, ffn_s in configs:
        r = train_one_path(name, attn_s, ffn_s, vocab_size,
                             train_ids, val_ids)
        results.append(r)

    # ─── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 88)
    print("Summary — what you trade for how much:")
    print("-" * 88)
    print(f"  {'path':<22s}  {'final val':>10s}  "
          f"{'total params':>14s}  {'weight KB':>14s}  {'vs dense':>10s}")
    print("-" * 88)

    dense_bytes = results[0]["weight_bytes"]
    dense_val = results[0]["val_losses"][-1][1]
    for r in results:
        final_val = r["val_losses"][-1][1]
        total_params = r["dense_params"] + r["sparse_live"]
        kb = r["weight_bytes"] / 1024
        mem_pct = r["weight_bytes"] / dense_bytes * 100
        loss_gap = (final_val - dense_val) / dense_val * 100
        print(f"  {r['name']:<22s}  {final_val:>10.4f}  "
              f"{total_params:>14,}  {kb:>8.1f} ({mem_pct:>3.0f}%)  "
              f"{loss_gap:>+9.2f}%")
    print("-" * 88)

    # Final plot
    plot_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        "demo_14_sparse_attention.png",
    )
    plot_all(results, plot_path)
    print(f"\nSaved plot: {plot_path}\n")


if __name__ == "__main__":
    main()
