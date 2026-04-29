"""
Validate end-to-end dW SIMD speedup at 40M-param transformer scale.

What this measures
──────────────────
The dW SIMD kernel (NEON on ARM, scalar-fallback on x86 pre-AVX2,
future AVX2 on x86) claims a measurable end-to-end speedup on the
40M-param sparse transformer (demo_16). Per-layer microbenches can
mislead: real backward passes include weight decay, optimizer steps,
and attention work that dilute any kernel win. This script measures
the honest wallclock ratio end-to-end.

Methodology
───────────
- Run demo_16's all-sparse path at --steps 200 with kernel forced
  to "scalar", record ms/step and final val loss.
- Run again with kernel forced to "simd", same seed, same data.
- Compare both: wallclock ratio AND final val-loss gap.

We force the kernel by monkey-patching _SpMMFunction.forward to
override the kernel string on ctx, which drives backward's dispatch.

What a "pass" looks like
────────────────────────
- simd step time < scalar step time by >= 1.5x on platforms that
  have a real SIMD dW kernel (currently ARM NEON).
- On platforms without SIMD yet (x86_64 pre-AVX2), scalar ≈ simd;
  this script then just serves as a "before" baseline capture.
- |simd final val − scalar final val| < 0.01 nats (identical
  training dynamics — no silent regression from accumulator
  reordering at 40M scale).

Platform notes
──────────────
- Apple Silicon macOS / Linux aarch64: simd dispatches to the NEON
  kernel in csrc/kernels/spmm_grad_neon.cpp. Expect ~2x ratio.
- Linux x86_64 (current main): simd dispatches to the AVX2 stub in
  csrc/kernels/spmm_grad_avx2.cpp, which is scalar-identical. Ratio
  will be ~1.0x; final val losses should match exactly. This is the
  pre-AVX2 baseline for milestone 14's "before" anchor.
- Linux x86_64 (post-Phase-B): simd will dispatch to the real
  AVX2 kernel. Expected ratio >= 1.5x.

Runtime
───────
200 steps at 40M-param scale:
  - scalar all-sparse: ~4-5 min on warm M3 Pro, ~8-10 min on CI x86
  - simd   all-sparse: ~2-3 min on warm M3 Pro, ~8-10 min on CI x86
    (pre-AVX2), ~4-5 min (post-AVX2, projected)

Not a test — lives in examples/ as a runnable validation script and
as the driver for the milestone-14 "before/after" CI artifact.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Make examples/ importable so we can pull in demo_15/demo_16 setup.
# This script lives in examples/ so that path is just __file__'s dir.
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXAMPLES_DIR)

import demo_15_mini_gpt as d15
import demo_16_mini_gpt_60m as d16  # noqa: F401 — rebinds demo_15 constants

import sparselab  # noqa: F401 — ensure package loads before patching
from sparselab.ops import _SpMMFunction


# ─── Kernel override ─────────────────────────────────────────────────
#
# demo_15.train_one_path uses SparseLinear, which calls
# _SpMMFunction.apply with kernel="simd" (see sparselab/nn.py).
# To make the scalar/simd comparison fair we force the kernel on
# every forward via a monkey-patch on _SpMMFunction.forward.
# ─────────────────────────────────────────────────────────────────────

_original_forward = _SpMMFunction.forward


def make_forced_forward(forced_kernel: str):
    """Return a forward that ignores the caller's kernel and forces our own."""
    def forced_forward(ctx, W_values, W, X, kernel):
        # Call the original with our forced kernel instead of whatever
        # SparseLinear passed. The backward path reads ctx.kernel, so
        # it dispatches to the matching dW path.
        return _original_forward(ctx, W_values, W, X, forced_kernel)
    return forced_forward


def patch_kernel(kernel: str):
    """Install a monkey-patch forcing this kernel on every _SpMMFunction call."""
    _SpMMFunction.forward = staticmethod(make_forced_forward(kernel))


def unpatch_kernel():
    """Restore the original forward."""
    _SpMMFunction.forward = staticmethod(_original_forward)


# ─── Single-path training helper ─────────────────────────────────────

def run_one(
    label: str,
    kernel: str,
    n_steps: int,
    train_ids, val_ids, vocab_size, itos, stoi,
):
    """Run demo_16's all-sparse path with the forced kernel. Return dict."""
    print(f"\n{'=' * 72}")
    print(f"  Run: {label}  (kernel={kernel}, steps={n_steps})")
    print(f"{'=' * 72}", flush=True)

    # Pin all RNG state before building the model — scalar and simd
    # runs see identical weights at init, identical batches, identical
    # random masks.
    torch.manual_seed(d15.SEED)
    np.random.seed(d15.SEED)

    # Force the kernel for the whole training loop.
    patch_kernel(kernel)
    try:
        cfg = d15.PATH_CONFIGS["all"]
        # Write samples to .scratch/ so committed demo_16 artifacts
        # under docs/demos aren't clobbered. .scratch/ is gitignored.
        scratch_dir = os.path.join(
            os.path.dirname(__file__), "..", ".scratch",
        )
        os.makedirs(scratch_dir, exist_ok=True)
        scratch_samples = os.path.join(
            scratch_dir, f"validate_40m_samples_{kernel}.txt",
        )
        if os.path.exists(scratch_samples):
            os.remove(scratch_samples)
        with open(scratch_samples, "w", encoding="utf-8") as f:
            f.write(f"Validation run ({kernel}) — {n_steps} steps\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

        t0 = time.perf_counter()
        result = d15.train_one_path(
            name=cfg["name"],
            ffn_sparsity=cfg["ffn_sparsity"],
            attn_sparsity=cfg["attn_sparsity"],
            train_ids=train_ids,
            val_ids=val_ids,
            vocab_size=vocab_size,
            itos=itos,
            stoi=stoi,
            samples_path=scratch_samples,
            n_steps=n_steps,
        )
        elapsed = time.perf_counter() - t0
    finally:
        unpatch_kernel()

    result["wallclock_s"] = elapsed
    result["ms_per_step"] = (elapsed * 1000.0) / n_steps
    result["kernel"] = kernel
    return result


# ─── Main ────────────────────────────────────────────────────────────

def main():
    N_STEPS = 200  # cheap validation; enough to see step-time ratio clearly

    print("=" * 72)
    print("  dW SIMD end-to-end validation at 40M-param transformer scale")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Arch: 8L × d_model=640 × d_ff=2560 (all-sparse)")
    print(f"  Steps: {N_STEPS} (per kernel)  Seed: {d15.SEED}")
    print("=" * 72, flush=True)

    print("\nLoading Tiny Shakespeare...", flush=True)
    train_ids, val_ids, vocab_size, itos, stoi = d15.load_data()

    # Run scalar first so its cache state is cold (simd runs second
    # and gets any warmup benefit — slightly favors simd but fair
    # because real users hit the warm path too).
    scalar_result = run_one(
        "scalar backward", "scalar", N_STEPS,
        train_ids, val_ids, vocab_size, itos, stoi,
    )
    simd_result = run_one(
        "simd backward", "simd", N_STEPS,
        train_ids, val_ids, vocab_size, itos, stoi,
    )

    # ─── Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    step_ratio = scalar_result["ms_per_step"] / simd_result["ms_per_step"]
    loss_gap = abs(scalar_result["final_val"] - simd_result["final_val"])

    print()
    print(f"  {'Kernel':<12} {'ms/step':>10} {'final_val':>12} {'wallclock':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    print(f"  {'scalar':<12} {scalar_result['ms_per_step']:>10.1f} "
          f"{scalar_result['final_val']:>12.4f} "
          f"{scalar_result['wallclock_s']:>10.1f}s")
    print(f"  {'simd':<12} {simd_result['ms_per_step']:>10.1f} "
          f"{simd_result['final_val']:>12.4f} "
          f"{simd_result['wallclock_s']:>10.1f}s")
    print()
    print(f"  Speedup (scalar/simd):     {step_ratio:.2f}x")
    print(f"  Val-loss gap |s − n|:      {loss_gap:.4f} nats")

    # ─── Pass/fail ──────────────────────────────────────────────────
    print()
    print("  Acceptance criteria:")
    # Platforms without a real SIMD kernel (x86 pre-AVX2) should
    # produce ratio ≈ 1.0 and loss gap ≈ 0. The >=1.5x gate is for
    # platforms that ship a real SIMD dW kernel.
    speed_ok = step_ratio >= 1.5
    loss_ok = loss_gap < 0.01
    print(f"    end-to-end speedup ≥ 1.5x: "
          f"{'✓' if speed_ok else '✗'} ({step_ratio:.2f}x)")
    print(f"    val-loss gap < 0.01 nats:  "
          f"{'✓' if loss_ok else '✗'} ({loss_gap:.4f})")
    print()

    if speed_ok and loss_ok:
        print("  OVERALL: ✓ validated — SIMD dW kernel delivers real end-to-end win.")
    elif loss_ok and not speed_ok:
        print("  OVERALL: — loss matches; no speed win on this platform (expected")
        print("            on x86 pre-AVX2 or if simd falls back to scalar).")
    else:
        print("  OVERALL: ✗ validation failed — diagnose before shipping.")

    print("=" * 72)


if __name__ == "__main__":
    main()
