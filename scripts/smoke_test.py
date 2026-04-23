#!/usr/bin/env python3
"""
SparseLab smoke test — verifies a fresh install works end-to-end.

Used by:
  - cibuildwheel to validate each built wheel in its own environment
  - Fresh-machine manual verification (Docker, SageMaker, Colab)
  - Anyone who just installed sparselab and wants to sanity check

What this checks:
  1. sparselab imports without error
  2. Version string is non-empty
  3. PaddedCSR round-trips correctly (dense → sparse → dense identity)
  4. spmm produces numerically correct output vs a dense reference
  5. Autograd backward pass flows gradients correctly
  6. SparseLinear works inside nn.Module / loss.backward() / optimizer
  7. A SparsityAlgorithm (SET) can attach and step without crashing

What this does NOT check:
  - Performance (wheel might be slow but still correct)
  - Every kernel variant (that's what pytest is for)
  - Every DST algorithm (we only exercise SET)

Exit codes:
  0 — everything works
  non-zero — some assertion failed, look at stderr for details

Run time: ~2-5 seconds on any modern machine.
"""

from __future__ import annotations

import sys
import traceback


def section(name: str) -> None:
    print(f"\n── {name}")


def check(label: str, cond: bool) -> None:
    mark = "✓" if cond else "✗"
    print(f"  {mark} {label}")
    if not cond:
        raise SystemExit(f"smoke test failed at: {label}")


def main() -> int:
    section("1. Import")
    import sparselab
    import torch
    check("sparselab imported", hasattr(sparselab, "__version__"))
    check(f"version string present ({sparselab.__version__})",
          bool(sparselab.__version__))
    check("torch imported", hasattr(torch, "__version__"))

    section("2. PaddedCSR round-trip")
    # Dense → sparse → dense must be identity for threshold=0.
    W_dense = torch.tensor(
        [[0.5, 0.0, 0.0, 0.3],
         [0.0, 0.0, 1.1, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.2, 0.7, 0.0, 0.4]],
        dtype=torch.float32,
    )
    W_csr = sparselab.PaddedCSR.from_dense(W_dense)
    W_rt = W_csr.to_dense()
    check("shape preserved", W_rt.shape == W_dense.shape)
    check("values preserved exactly", torch.equal(W_rt, W_dense))
    check("nnz matches live entries", W_csr.nnz == 6)  # 6 non-zero entries

    section("3. SpMM vs dense reference")
    torch.manual_seed(0)
    W = sparselab.PaddedCSR.random(64, 32, sparsity=0.8, seed=42)
    X = torch.randn(32, 16)
    Y_sparse = sparselab.spmm(W, X)
    Y_ref = W.to_dense() @ X
    max_diff = (Y_sparse - Y_ref).abs().max().item()
    check(f"forward matches dense (max |diff| = {max_diff:.2e})",
          max_diff < 1e-5)

    section("4. Autograd backward")
    X_grad = torch.randn(32, 16, requires_grad=True)
    Y = sparselab.spmm(W, X_grad)
    loss = Y.sum()
    loss.backward()
    check("X received a gradient", X_grad.grad is not None)
    check("X gradient is finite", torch.isfinite(X_grad.grad).all().item())

    section("5. SparseLinear inside nn.Module")
    model = torch.nn.Sequential(
        sparselab.SparseLinear(32, 64, sparsity=0.9),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 8),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 32)
    y = model(x)
    check("forward shape", tuple(y.shape) == (4, 8))
    # One training step.
    loss = y.pow(2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    check("training step completed without error", True)

    section("6. SparsityAlgorithm attachment")
    algo = sparselab.SET(sparsity=0.9, drop_fraction=0.3, update_freq=5)
    model.apply(algo)
    check("algorithm attached", len(algo.layers) == 1)
    # Run a few steps so SET actually fires at least once.
    for _ in range(12):
        x = torch.randn(4, 32)
        model(x).pow(2).sum().backward()
        opt.step()
        algo.step()
        opt.zero_grad()
    check(f"ran 12 training steps with SET (current _step_idx={algo._step_idx})",
          algo._step_idx == 12)

    print("\n✓ All smoke tests passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print("\n✗ Smoke test hit an unexpected exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
