"""
Spike — profile where time actually goes at transformer scale.

Not a real training demo. This is pre-milestone-4g reconnaissance:
a tiny transformer whose only job is to tell us how much of a training
step is kernel work vs autograd overhead vs optimizer vs everything
else. Decides whether we need to do autograd-overhead optimization
(Fix B from earlier notes) before shipping the 4g launch demo.

Architecture (matches 4g plan):
  - 2-layer decoder-only transformer
  - d_model=128, d_ff=512, n_heads=4, seq_len=64
  - Vocab=65 (Shakespeare-scale)
  - FFN uses sparselab.SparseLinear at 90% sparsity
  - Attention (Q, K, V, O projections) stays dense for now

How to run
──────────
    python examples/spike_transformer_profile.py

What to look at
───────────────
  Breakdown table at the end. If sparse kernel is <25% of total step
  time, we need Fix B before 4g ships. If sparse kernel is >50%, we
  can ship 4g as-is and revisit autograd overhead post-launch.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sparselab


# ─── Config ──────────────────────────────────────────────────────────
VOCAB_SIZE = 65
D_MODEL = 128
D_FF = 512
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 64
BATCH_SIZE = 16
SPARSITY = 0.9   # FFN sparsity
WARMUP_STEPS = 5
MEASURE_STEPS = 30


# ─────────────────────────────────────────────────────────────────────
#  Model: tiny decoder-only transformer with sparse FFN
# ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention. All projections dense
    — we're only sparsifying FFN in v0.1."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                             # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (B, n_heads, T, d_head) for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Scaled dot-product attention with causal mask. Use torch's
        # fused impl; we're not profiling attention.
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(att)


class SparseFFN(nn.Module):
    """Two-layer FFN with both layers as sparselab.SparseLinear.
    This is the part we actually sparsify."""
    def __init__(self, d_model: int, d_ff: int, sparsity: float):
        super().__init__()
        self.fc_up = sparselab.SparseLinear(d_model, d_ff,
                                               sparsity=sparsity, bias=False)
        self.fc_down = sparselab.SparseLinear(d_ff, d_model,
                                                 sparsity=sparsity, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SparseLinear handles arbitrary leading dims (B, T, D) naturally.
        return self.fc_down(F.gelu(self.fc_up(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, sparsity: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SparseFFN(d_model, d_ff, sparsity)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int,
                 n_heads: int, n_layers: int, seq_len: int,
                 sparsity: float):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, d_ff, n_heads, sparsity) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("pos_ids", torch.arange(seq_len))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(self.pos_ids[:T])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab)


# ─────────────────────────────────────────────────────────────────────
#  Profiling helpers
# ─────────────────────────────────────────────────────────────────────

class Timer:
    """Accumulate time spent in a named bucket. Use as a context manager."""
    def __init__(self):
        self.totals: dict[str, float] = {}

    def __call__(self, name: str):
        return _Scope(self, name)


class _Scope:
    def __init__(self, timer: Timer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *args):
        dt = time.perf_counter() - self.t0
        self.timer.totals[self.name] = self.timer.totals.get(self.name, 0.0) + dt


def count_params(model: nn.Module) -> tuple[int, int]:
    """Count (dense_params, sparse_live_params)."""
    dense = 0
    sparse_live = 0
    for name, mod in model.named_modules():
        if isinstance(mod, sparselab.SparseLinear):
            sparse_live += mod.nnz
        else:
            for p in mod.parameters(recurse=False):
                dense += p.numel()
    return dense, sparse_live


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\nSparseLab spike — transformer-scale timing breakdown")
    print(f"  Architecture: {N_LAYERS}L x d_model={D_MODEL} x d_ff={D_FF} x "
          f"heads={N_HEADS} x seq_len={SEQ_LEN}")
    print(f"  Batch: {BATCH_SIZE}  Vocab: {VOCAB_SIZE}  FFN sparsity: {SPARSITY}")
    print(f"  Steps: warmup={WARMUP_STEPS}, measure={MEASURE_STEPS}")
    print("=" * 72)

    torch.manual_seed(0)
    model = TinyTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, seq_len=SEQ_LEN,
        sparsity=SPARSITY,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    dense_params, sparse_live = count_params(model)
    total = dense_params + sparse_live
    print(f"\nModel size:")
    print(f"  Dense params (attention, embeddings, norms, head):     {dense_params:>10,}")
    print(f"  Sparse live params (FFN fc_up + fc_down, 90% sparsity): {sparse_live:>10,}")
    print(f"  Total:                                                  {total:>10,}")
    equiv_dense_ffn = 2 * N_LAYERS * D_MODEL * D_FF   # what FFN would be at 0% sparsity
    print(f"  (FFN would be {equiv_dense_ffn:,} dense; we're using "
          f"{sparse_live:,} live = {sparse_live / equiv_dense_ffn * 100:.1f}%)")

    # ── Warmup
    print(f"\nWarmup ({WARMUP_STEPS} steps)...", flush=True)
    for _ in range(WARMUP_STEPS):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        opt.zero_grad()
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        opt.step()

    # ── Measurement pass with breakdown
    print(f"Measuring ({MEASURE_STEPS} steps)...", flush=True)
    timer = Timer()

    t_overall = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        with timer("data_prep"):
            idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
            targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        with timer("zero_grad"):
            opt.zero_grad()

        with timer("forward"):
            logits = model(idx)

        with timer("loss"):
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                     targets.reshape(-1))

        with timer("backward"):
            loss.backward()

        with timer("opt_step"):
            opt.step()

    total_wall = time.perf_counter() - t_overall
    ms_per_step = total_wall / MEASURE_STEPS * 1000

    # ── Additional measurement: time raw sparse kernel calls in isolation
    # so we can estimate how much of forward/backward is the kernels.
    # We'll do a forward+backward with JUST the FFN's fc_up layer to get
    # the floor: kernel + minimal autograd overhead.
    print("\nMicro-benchmark: single SparseLinear fwd+bwd time...", flush=True)
    fc_up = model.blocks[0].ffn.fc_up
    # fc_up is shape (D_FF, D_MODEL) = (512, 128) after 90% sparsity
    # Input expected shape: (*, 128)
    x_dummy = torch.randn(BATCH_SIZE * SEQ_LEN, D_MODEL, requires_grad=True)
    fwd_bwd_times = []
    for _ in range(30):
        if fc_up._values.grad is not None:
            fc_up._values.grad.zero_()
        if x_dummy.grad is not None:
            x_dummy.grad.zero_()
        t0 = time.perf_counter()
        y = fc_up(x_dummy)
        y.sum().backward()
        fwd_bwd_times.append(time.perf_counter() - t0)
    one_sparse_fwd_bwd = np.median(fwd_bwd_times) * 1000

    # Raw SpMM forward-only kernel
    from sparselab import _core
    W_raw = fc_up._csr
    # Kernel wants (D_MODEL, batch) = (128, 1024). Pre-build numpy input.
    X_raw = np.random.randn(D_MODEL, BATCH_SIZE * SEQ_LEN).astype(np.float32)
    # Warmup
    for _ in range(5):
        _core.spmm_simd(W_raw, X_raw)
    kernel_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _core.spmm_simd(W_raw, X_raw)
        kernel_times.append(time.perf_counter() - t0)
    one_kernel_call = np.median(kernel_times) * 1000

    # ── Print breakdown
    print("\n" + "=" * 72)
    print(f"Full training step breakdown (avg over {MEASURE_STEPS} steps):")
    print("-" * 72)
    print(f"  {'bucket':<16s} {'ms/step':>10s} {'% of step':>12s}")
    print("-" * 72)
    for name in ["data_prep", "zero_grad", "forward", "loss", "backward", "opt_step"]:
        avg_ms = timer.totals[name] / MEASURE_STEPS * 1000
        pct = avg_ms / ms_per_step * 100
        print(f"  {name:<16s} {avg_ms:>10.2f} {pct:>11.1f}%")
    print("-" * 72)
    print(f"  {'TOTAL':<16s} {ms_per_step:>10.2f}")

    print("\n" + "=" * 72)
    print("Sparse kernel vs autograd overhead estimate:")
    print("-" * 72)
    print(f"  One SparseLinear fwd+bwd (with autograd):  {one_sparse_fwd_bwd:.3f} ms")
    print(f"  One raw spmm_simd kernel call (no autograd): {one_kernel_call:.3f} ms")
    print(f"  Autograd overhead per sparse fwd+bwd:      "
          f"{one_sparse_fwd_bwd - one_kernel_call:.3f} ms")

    # Estimate total kernel-only vs total-autograd over the full step:
    # 4 SparseLinear calls per block (2 FFN layers, each has 1 forward + 1 backward
    # that internally does 2 kernel calls — forward kernel + grad_w kernel +
    # transpose-and-spmm for dX). N_LAYERS * 2 FFN layers = 4 total SparseLinear
    # modules, each doing fwd+bwd = 4 * one_sparse_fwd_bwd total.
    n_sparse_fwd_bwd = N_LAYERS * 2
    est_kernel_time = n_sparse_fwd_bwd * one_kernel_call * 3  # fwd + grad_w + WT@dY ≈ 3x kernel
    est_overhead = n_sparse_fwd_bwd * (one_sparse_fwd_bwd - one_kernel_call)
    print(f"\n  Estimated kernel-only work across full step:  {est_kernel_time:.2f} ms")
    print(f"  Estimated autograd overhead across full step: {est_overhead:.2f} ms")
    pct_kernel = est_kernel_time / ms_per_step * 100
    pct_overhead = est_overhead / ms_per_step * 100
    print(f"  Kernel: {pct_kernel:.1f}% of step | "
          f"Sparse-autograd overhead: {pct_overhead:.1f}% of step")

    # ── Verdict
    print("\n" + "=" * 72)
    print("Verdict:")
    if pct_kernel < 25:
        print(f"  ⚠️  Kernel is only {pct_kernel:.0f}% of training step time.")
        print("      Autograd overhead dominates. Fix B (zero-copy pybind11)")
        print("      would help before launching 4g.")
    elif pct_kernel < 50:
        print(f"  →  Kernel is {pct_kernel:.0f}% of training step time.")
        print("      Autograd fix would help but isn't blocking. 4g can ship,")
        print("      Fix B is a good post-launch cleanup.")
    else:
        print(f"  ✓  Kernel is {pct_kernel:.0f}% of training step time.")
        print("      Kernel dominates. No autograd fix needed before 4g.")
    print("=" * 72)


if __name__ == "__main__":
    main()
