"""
Demo 15 — Mini GPT on Tiny Shakespeare (launch-grade demo).

A 10M-parameter transformer trained from scratch on Tiny Shakespeare
on your laptop. Compares dense vs sparse-FFN-at-90% side-by-side:
model size, training time, convergence quality, and text samples.

Architecture (scaled up from demo_13):
    d_model = 384
    d_ff    = 1536    (4x d_model)
    n_heads = 6
    n_layers = 6
    seq_len = 128
    batch   = 16

Dense total: ~10.7M params
Sparse total: ~4.3M params (FFN at 90%, attention+embeddings dense)
  — 40% of dense memory

This is the "ship-worthy demo" for v0.1 launch. It runs in ~60-90
minutes on an M3 Pro MacBook at the default 5000 steps, producing
loss curves and text samples showing both paths converging (sparse
at 40% of dense memory).

How to run
──────────
    pip install sparsecore[demos]
    python examples/demo_15_mini_gpt.py              # 5000 steps, ~75 min
    python examples/demo_15_mini_gpt.py --steps 10000  # 10000 steps, ~2.5 hr

What you'll see
───────────────
    - Live loss curves (updated every 250 steps) for both paths
    - Generated text samples every 1000 steps
    - Final comparison: params, memory, time, final val loss
    - Side-by-side samples at convergence
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from datetime import datetime

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


# ─── Configuration ───────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "tinyshakespeare.txt")
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")

# Model: aim for ~10M params dense, ~4M live sparse
D_MODEL = 384
D_FF = 1536
N_HEADS = 6
N_LAYERS = 6
SEQ_LEN = 128
BATCH_SIZE = 16

# Training
LR = 3e-3
DEFAULT_N_STEPS = 5000
EVAL_EVERY = 250
SAMPLE_EVERY = 1000
SEED = 42

SPARSITY = 0.9            # FFN sparsity
ATTN_SPARSITY = 0.7       # attention sparsity (used only when path=all)
SAMPLE_PROMPT = "ROMEO:\n"
SAMPLE_LEN = 200


# ─────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────

def load_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    # 90/10 train/val split
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars), itos, stoi


def get_batch(ids: torch.Tensor, batch_size: int, seq_len: int):
    n = ids.size(0) - seq_len - 1
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# ─────────────────────────────────────────────────────────────────────
#  Model — switches between dense nn.Linear and sparsecore.SparseLinear
# ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    ``attn_sparsity`` controls whether the qkv and o projections use
    dense ``nn.Linear`` (attn_sparsity=0) or ``sparsecore.SparseLinear``
    at the given sparsity level (attn_sparsity>0). The attention math
    itself (the softmax and the scaled-dot-product) is always dense —
    we only sparsify the projection weights, not the attention pattern.
    """
    def __init__(self, d_model: int, n_heads: int, attn_sparsity: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if attn_sparsity > 0:
            self.qkv = sparsecore.SparseLinear(
                d_model, 3 * d_model, sparsity=attn_sparsity, bias=False)
            self.o = sparsecore.SparseLinear(
                d_model, d_model, sparsity=attn_sparsity, bias=False)
        else:
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
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc_up = nn.Linear(d_model, d_ff, bias=False)
        self.fc_down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.fc_down(F.gelu(self.fc_up(x)))


class SparseFFN(nn.Module):
    """The only line that changes — SparseLinear instead of nn.Linear."""
    def __init__(self, d_model, d_ff, sparsity):
        super().__init__()
        self.fc_up = sparsecore.SparseLinear(d_model, d_ff,
                                               sparsity=sparsity, bias=False)
        self.fc_down = sparsecore.SparseLinear(d_ff, d_model,
                                                 sparsity=sparsity, bias=False)
    def forward(self, x):
        return self.fc_down(F.gelu(self.fc_up(x)))


class Block(nn.Module):
    """One transformer block. ``ffn_sparsity`` and ``attn_sparsity``
    control whether the FFN / attention projections are sparse. Any
    value > 0 swaps in ``sparsecore.SparseLinear`` at that sparsity."""
    def __init__(self, d_model, d_ff, n_heads,
                 ffn_sparsity: float, attn_sparsity: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, attn_sparsity)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = (SparseFFN(d_model, d_ff, ffn_sparsity) if ffn_sparsity > 0
                    else DenseFFN(d_model, d_ff))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, ffn_sparsity: float, attn_sparsity: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([
            Block(D_MODEL, D_FF, N_HEADS, ffn_sparsity, attn_sparsity)
            for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(SEQ_LEN))

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos_ids[:T])
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, n_new_tokens, temperature=0.8):
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
#  Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────

def count_params(model):
    """Returns (dense_params, sparse_live_count, sparse_capacity_count).

    dense_params: plain nn.Linear / nn.Embedding / LayerNorm params.
    sparse_live_count: number of LIVE (nonzero) entries in the
      PaddedCSRs (ignores padding slots).
    sparse_capacity_count: total allocated slots in the PaddedCSRs
      (live + padding). Used for an honest "what's the real memory
      footprint" row that includes the padding overhead.
    """
    dense_p = 0
    sparse_live = 0
    sparse_capacity = 0
    for m in model.modules():
        if isinstance(m, sparsecore.SparseLinear):
            sparse_live += m.nnz
            # total_capacity = nnz + padding slots. Reflects the actual
            # allocated backing array size.
            sparse_capacity += int(m._csr.total_capacity)
        else:
            for p in m.parameters(recurse=False):
                dense_p += p.numel()
    return dense_p, sparse_live, sparse_capacity


def memory_breakdown(
    dense_p: int, sparse_live: int, sparse_capacity: int
) -> dict[str, float]:
    """Three honest memory numbers in MB, labeled clearly.

    - inference_mb: what you'd load to do forward passes only. No
      gradient buffers. Sparse columns cost 4 B (int32 index) + 4 B
      (float32 value) = 8 B per live entry. Padding slots are 4 B
      (just the index array has to carry them; values at padding are
      implicitly 0).
    - training_mb: add a 4 B gradient per trainable scalar. For
      sparse, that's 4 B per live entry (we don't store gradient at
      padding slots).
    - training_with_padding_mb: most honest training figure —
      includes the PaddedCSR row padding in the storage cost.

    Returns a dict with all three so callers can pick which number
    they want to headline.
    """
    # Dense: 4 B value + 4 B grad = 8 B per training-time param.
    dense_inference = 4 * dense_p
    dense_training = 8 * dense_p

    # Sparse live entries: 4 B value + 4 B col_idx + (optionally 4 B
    # grad). Padding slots: 4 B col_idx only (padding values are
    # implicitly 0 and we don't compute gradients there).
    sparse_padding = max(0, sparse_capacity - sparse_live)
    sparse_inference = 8 * sparse_live + 4 * sparse_padding
    sparse_training_nopad = 12 * sparse_live + 4 * sparse_padding
    sparse_training_withpad = 12 * sparse_capacity

    return {
        "inference_mb": (dense_inference + sparse_inference) / (1024 * 1024),
        "training_mb": (dense_training + sparse_training_nopad) / (1024 * 1024),
        "training_with_padding_mb":
            (dense_training + sparse_training_withpad) / (1024 * 1024),
        # Also return dense-only decomposition so the table can split it.
        "dense_inference_mb": dense_inference / (1024 * 1024),
        "dense_training_mb": dense_training / (1024 * 1024),
        "sparse_inference_mb": sparse_inference / (1024 * 1024),
        "sparse_training_mb": sparse_training_nopad / (1024 * 1024),
    }


def compute_val_loss(model, val_ids, n_batches=30):
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


def generate_sample(model, itos, stoi, prompt=SAMPLE_PROMPT, n_tokens=SAMPLE_LEN):
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]],
                         dtype=torch.long)
    out = model.generate(idx, n_new_tokens=n_tokens, temperature=0.8)
    return "".join(itos[i] for i in out[0].tolist())


def train_one_path(name: str, ffn_sparsity: float, attn_sparsity: float,
                    train_ids, val_ids,
                    vocab_size: int, itos: dict, stoi: dict,
                    samples_path: str, n_steps: int):
    """Train one model configuration and record progress.

    ffn_sparsity: 0.0 for dense FFN, or the target sparsity (e.g. 0.9).
    attn_sparsity: 0.0 for dense attention, or the target sparsity.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n{'='*72}", flush=True)
    print(f" Training path: {name}", flush=True)
    print(f"  ffn_sparsity={ffn_sparsity}  attn_sparsity={attn_sparsity}",
          flush=True)
    print(f"{'='*72}", flush=True)

    model = MiniGPT(vocab_size,
                    ffn_sparsity=ffn_sparsity,
                    attn_sparsity=attn_sparsity)
    dense_p, sparse_live, sparse_capacity = count_params(model)
    total_p = dense_p + sparse_live
    mem = memory_breakdown(dense_p, sparse_live, sparse_capacity)

    print(f"  Parameters: total={total_p:,}  "
          f"(dense={dense_p:,}, sparse_live={sparse_live:,}, "
          f"sparse_capacity={sparse_capacity:,})", flush=True)
    print(f"  Weight memory:", flush=True)
    print(f"    inference only              : {mem['inference_mb']:.1f} MB",
          flush=True)
    print(f"    training (weight+grad)      : {mem['training_mb']:.1f} MB",
          flush=True)
    print(f"    training (incl. CSR padding): "
          f"{mem['training_with_padding_mb']:.1f} MB", flush=True)
    print(f"", flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=LR)

    train_losses, val_losses, samples = [], [], []
    running_loss = 0.0
    running_n = 0
    t_start = time.perf_counter()

    for step in range(n_steps):
        x, y = get_batch(train_ids, BATCH_SIZE, SEQ_LEN)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1))
        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_n += 1

        if (step + 1) % EVAL_EVERY == 0:
            train_avg = running_loss / running_n
            running_loss, running_n = 0.0, 0
            val_avg = compute_val_loss(model, val_ids)
            train_losses.append((step + 1, train_avg))
            val_losses.append((step + 1, val_avg))
            elapsed = time.perf_counter() - t_start
            eta_sec = elapsed / (step + 1) * (n_steps - step - 1)
            eta_min = eta_sec / 60
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}][{name}] step {step+1:>5d}/{n_steps}  "
                  f"train={train_avg:.3f}  val={val_avg:.3f}  "
                  f"({elapsed:.0f}s, eta {eta_min:.1f}m)", flush=True)

        if (step + 1) % SAMPLE_EVERY == 0 or (step + 1) == n_steps:
            sample = generate_sample(model, itos, stoi)
            samples.append((step + 1, sample))

    total_s = time.perf_counter() - t_start
    print(f"  [{name}] DONE: {total_s:.0f}s total "
          f"({total_s/60:.1f} min)", flush=True)

    # Persist samples
    with open(samples_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*72}\n {name}\n{'='*72}\n")
        for step, text in samples:
            f.write(f"\n--- step {step} ---\n{text}\n")

    return {
        "name":                    name,
        "train_losses":            train_losses,
        "val_losses":              val_losses,
        "samples":                 samples,
        "final_val":               val_losses[-1][1],
        "total_s":                 total_s,
        "total_params":            total_p,
        "dense_params":            dense_p,
        "sparse_live":             sparse_live,
        "sparse_capacity":         sparse_capacity,
        "inference_mb":            mem["inference_mb"],
        "training_mb":             mem["training_mb"],
        "training_with_padding_mb": mem["training_with_padding_mb"],
        # Back-compat with the old field name (some downstream code
        # checked r["weight_mb"]). Kept until all callers updated.
        "weight_mb":               mem["training_mb"],
    }


# ─────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_results(results, out_path):
    """Plot training + validation curves for any number of paths.

    Colors are assigned deterministically by path name so repeated
    runs look the same in screenshots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {
        "Dense":              "#3b82f6",  # blue
        "Sparse (FFN 90%)":   "#ef4444",  # red
        "Sparse (all: attn 70% + FFN 90%)": "#10b981",  # green
    }
    for r in results:
        c = colors.get(r["name"], "black")
        steps_tr = [s for s, _ in r["train_losses"]]
        losses_tr = [l for _, l in r["train_losses"]]
        steps_val = [s for s, _ in r["val_losses"]]
        losses_val = [l for _, l in r["val_losses"]]
        ax1.plot(steps_tr, losses_tr, color=c, linewidth=2,
                 label=f'{r["name"]}  final={losses_tr[-1]:.3f}')
        ax2.plot(steps_val, losses_val, color=c, linewidth=2,
                 label=f'{r["name"]}  final={losses_val[-1]:.3f}')
    ax1.set_xlabel("step"); ax1.set_ylabel("loss")
    ax1.set_title("Training loss"); ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)
    ax2.set_xlabel("step")
    ax2.set_title("Validation loss"); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    path_names = ", ".join(r["name"] for r in results)
    fig.suptitle(
        f"Mini-GPT on Tiny Shakespeare — {path_names}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  Path configurations
# ─────────────────────────────────────────────────────────────────────
#
#  Each entry defines one training configuration. The --path CLI flag
#  selects one, or "all" (default) runs every entry in order.
#  Keys match the pretty-printed name used in logs and plots.
# ─────────────────────────────────────────────────────────────────────

PATH_CONFIGS = {
    "dense": {
        "name": "Dense",
        "ffn_sparsity": 0.0,
        "attn_sparsity": 0.0,
    },
    "ffn": {
        "name": "Sparse (FFN 90%)",
        "ffn_sparsity": SPARSITY,
        "attn_sparsity": 0.0,
    },
    "all": {
        "name": "Sparse (all: attn 70% + FFN 90%)",
        "ffn_sparsity": SPARSITY,
        "attn_sparsity": ATTN_SPARSITY,
    },
}


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    # ─── CLI ──────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Mini-GPT on Tiny Shakespeare — SparseCore launch demo."
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_N_STEPS,
        help=f"Number of training steps (default: {DEFAULT_N_STEPS}). "
             f"Use 10000 for the full-convergence artifact run.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Optional tag appended to output filenames "
             "(demo_15_{tag}_samples.txt, demo_15_{tag}_plot.png).",
    )
    parser.add_argument(
        "--path", type=str, default="dense,ffn",
        help="Which configurations to train. Comma-separated subset of "
             f"{sorted(PATH_CONFIGS.keys())} or 'all-three' for all. "
             "Default: 'dense,ffn' (matches prior demo_15 behavior). "
             "Use 'all' for attention+FFN sparse only, 'all-three' for "
             "all three paths in one run.",
    )
    args = parser.parse_args()
    n_steps = args.steps
    file_tag = f"_{args.tag}" if args.tag else ""

    # ─── Resolve which paths to run ───────────────────────────────────
    if args.path == "all-three":
        paths_to_run = ["dense", "ffn", "all"]
    else:
        paths_to_run = [p.strip() for p in args.path.split(",")]
    for p in paths_to_run:
        if p not in PATH_CONFIGS:
            raise SystemExit(
                f"--path got unknown value: {p!r}. "
                f"Valid: {sorted(PATH_CONFIGS.keys())} or 'all-three'."
            )

    start_wallclock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("SparseCore demo 15 — Mini GPT on Tiny Shakespeare")
    print(f"  Started: {start_wallclock}")
    print(f"  Arch: {N_LAYERS}L × d_model={D_MODEL} × d_ff={D_FF} × "
          f"heads={N_HEADS} × seq={SEQ_LEN}")
    print(f"  Training: {n_steps} steps, batch={BATCH_SIZE}, lr={LR}")
    print(f"  Configs: {paths_to_run}")
    print(f"  Seed: {SEED}")
    print("="*72)

    print("\nLoading Tiny Shakespeare...", flush=True)
    train_ids, val_ids, vocab_size, itos, stoi = load_data()
    print(f"  {len(train_ids):,} train chars, {len(val_ids):,} val chars, "
          f"vocab={vocab_size}")

    samples_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        f"demo_15{file_tag}_samples.txt",
    )
    if os.path.exists(samples_path):
        os.remove(samples_path)
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(f"Mini-GPT samples from demo_15 (Tiny Shakespeare)\n")
        f.write(f"Prompt: {SAMPLE_PROMPT!r}\n")
        f.write(f"Steps: {n_steps}   Seed: {SEED}   "
                f"Started: {start_wallclock}\n")
        f.write(f"Paths: {paths_to_run}\n")
        f.write("="*60 + "\n")

    # Run each requested path. Same seed, same inputs, sequential.
    results = []
    for p in paths_to_run:
        cfg = PATH_CONFIGS[p]
        r = train_one_path(
            name=cfg["name"],
            ffn_sparsity=cfg["ffn_sparsity"],
            attn_sparsity=cfg["attn_sparsity"],
            train_ids=train_ids,
            val_ids=val_ids,
            vocab_size=vocab_size,
            itos=itos,
            stoi=stoi,
            samples_path=samples_path,
            n_steps=n_steps,
        )
        results.append(r)

    # ─── Final comparison table ───────────────────────────────────────
    #
    # Three memory figures, so the reader can pick the honest one for
    # their use case:
    #   inference  — forward-only (no grad buffers). Most favorable.
    #   training   — weight + grad, no CSR padding. Middle figure.
    #   +padding   — training + PaddedCSR padding slots (capacity).
    #                The most pessimistic and most accurate of what we
    #                actually allocate in memory at training time.
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("Final comparison:")
    print("-"*80)
    print(f"  {'path':<36}  {'params':>10}  {'final val':>10}  {'time':>8}")
    print("-"*80)
    for r in results:
        print(f"  {r['name']:<36}  {r['total_params']:>10,}  "
              f"{r['final_val']:>10.3f}  {r['total_s']/60:>7.1f}m")
    print()
    print(f"  {'path':<36}  {'infer MB':>10}  {'train MB':>10}  "
          f"{'+padding':>10}")
    print("-"*80)
    # Dense-baseline numbers used for the percentage comparison column.
    r_dense = next((r for r in results if r["name"] == "Dense"), None)
    for r in results:
        infer_mb = r["inference_mb"]
        train_mb = r["training_mb"]
        padded_mb = r["training_with_padding_mb"]
        if r_dense is not None and r is not r_dense:
            infer_pct = f"{infer_mb/r_dense['inference_mb']*100:.0f}%"
            train_pct = f"{train_mb/r_dense['training_mb']*100:.0f}%"
        else:
            infer_pct = ""
            train_pct = ""
        print(f"  {r['name']:<36}  "
              f"{infer_mb:>7.1f} {infer_pct:>3s}  "
              f"{train_mb:>7.1f} {train_pct:>3s}  "
              f"{padded_mb:>9.1f}")
    print("="*80)
    print("  infer MB = 4B per dense param, 8B per sparse live "
          "(idx + value, no grad).")
    print("  train MB = adds 4B grad per trainable scalar.")
    print("  +padding = train MB but including PaddedCSR row padding "
          "slots (capacity).")
    print("="*80)

    # ─── Plot + summary output files ──────────────────────────────────
    plot_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "demos",
        f"demo_15{file_tag}_mini_gpt.png",
    )
    plot_results(results, plot_path)
    print(f"\n  Saved plot:    {plot_path}")
    print(f"  Saved samples: {samples_path}")

    # Show the last sample from each configuration.
    for r in results:
        print("\n" + "="*72)
        print(f"Final sample — {r['name']}:")
        print("-"*72)
        print(r["samples"][-1][1])
    end_wallclock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("="*72)
    print(f"  Started: {start_wallclock}")
    print(f"  Ended:   {end_wallclock}\n")


if __name__ == "__main__":
    main()
