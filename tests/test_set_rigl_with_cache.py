"""
Integration tests: SET and RigL training through the transpose cache.

Why this file exists
────────────────────
The transpose cache in `sparsecore.ops._cached_transpose` keys on
`W.topology_version`. Every SET/RigL update calls `csr.rewrite_row`,
which bumps the version, which should invalidate the cache. This is
tested at the unit level in `test_transpose_cache.py`.

What's NOT covered by the unit tests is a LONG training loop where
SET/RigL fires repeatedly through the backward path. If there were
any subtle bug in the invalidation — off-by-one on the version,
wrong cache eviction, value-refresh missing a slot — it would show
up as drifting loss or mismatched gradients at step 200+ but not at
step 1.

Our running Mini-GPT demo (demo_15) uses STATIC sparsity, so it
doesn't exercise this interaction either. This file fills the gap.

Strategy
────────
For both SET and RigL:
  1. Run N steps of training with the cache enabled (production path).
  2. Run the SAME N steps with the cache force-cleared after every
     step (pure fresh-transpose baseline).
  3. Assert the loss trajectories match to 1e-5 across all N steps.

If (1) and (2) diverge, the cache has a correctness bug that an
isolated unit test couldn't catch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import sparsecore
from sparsecore.ops import _clear_transpose_cache


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _run_training(
    *,
    algo_ctor,
    n_steps: int,
    batch_size: int,
    in_features: int,
    out_features: int,
    sparsity: float,
    seed: int,
    clear_cache_every_step: bool,
) -> list[float]:
    """
    Run a tiny fixed training loop with the given sparsity algorithm.

    Returns the loss trajectory as a list of floats (one per step).

    clear_cache_every_step=True forces the pure fresh-transpose path
    on every backward; =False uses the production cache.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    layer = sparsecore.SparseLinear(
        in_features, out_features, sparsity=sparsity, bias=False
    )
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)

    algo = algo_ctor()
    layer.apply(algo)

    # Fixed dataset — same across both runs so the comparison is apples-to-apples
    torch.manual_seed(seed + 1000)
    data_x = torch.randn(n_steps * batch_size, in_features)
    data_y = torch.randn(n_steps * batch_size, out_features)

    losses: list[float] = []
    for step in range(n_steps):
        if clear_cache_every_step:
            # Force every backward to rebuild the transpose from scratch.
            # If production (cached) loss matches this trajectory, the
            # cache is correct.
            _clear_transpose_cache()

        x = data_x[step * batch_size : (step + 1) * batch_size]
        y = data_y[step * batch_size : (step + 1) * batch_size]

        opt.zero_grad()
        pred = layer(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        algo.step()   # may or may not fire .update() depending on schedule

        losses.append(loss.item())

    return losses


# ─────────────────────────────────────────────────────────────────────
#  SET × cache
# ─────────────────────────────────────────────────────────────────────

def test_set_loss_trajectory_matches_fresh_transpose():
    """
    250-step training run with SET firing every 25 steps. Loss at each
    step must match between the cached and fresh-transpose paths.

    A stale-cache bug would manifest as divergence starting at step 26
    (first SET update) and growing.
    """
    def make_algo():
        return sparsecore.SET(
            sparsity=0.8,
            drop_fraction=0.3,
            update_freq=25,
            seed=123,
        )

    kwargs = dict(
        algo_ctor=make_algo,
        n_steps=250,
        batch_size=16,
        in_features=32,
        out_features=16,
        sparsity=0.8,
        seed=42,
    )

    # Start both runs from a fully clean cache state.
    _clear_transpose_cache()
    losses_cached = _run_training(**kwargs, clear_cache_every_step=False)

    _clear_transpose_cache()
    losses_fresh = _run_training(**kwargs, clear_cache_every_step=True)

    # Both trajectories should be identical to float32 numerical precision.
    # A per-step abs-diff tolerance of 1e-5 is tight for a 250-step loop —
    # accumulated float32 error over that many steps sits well inside it
    # when the cache is correct.
    for step, (lc, lf) in enumerate(zip(losses_cached, losses_fresh)):
        assert abs(lc - lf) < 1e-5, (
            f"SET: cache and fresh-transpose diverged at step {step}: "
            f"cached={lc:.6f}, fresh={lf:.6f}, "
            f"diff={abs(lc-lf):.2e}. A bug in transpose-cache "
            f"invalidation after SET.update() is the most likely cause."
        )

    # And loss should actually decrease — sanity that training happened.
    # First 10 vs last 10 — the SET perturbation means we can't assume
    # monotone decrease, so compare rolling averages.
    assert sum(losses_cached[-10:]) / 10 < sum(losses_cached[:10]) / 10, (
        "SET training didn't reduce loss — something upstream broken"
    )


def test_set_many_updates_cache_stays_correct():
    """
    Aggressive: SET fires every 5 steps for 100 steps = 20 topology
    changes. Each change must correctly invalidate the cache. If the
    cache ever returns a stale WT after a SET update, gradients go
    wrong and loss diverges within 2-3 steps.
    """
    def make_algo():
        return sparsecore.SET(
            sparsity=0.7, drop_fraction=0.4, update_freq=5, seed=7
        )

    kwargs = dict(
        algo_ctor=make_algo,
        n_steps=100, batch_size=8,
        in_features=24, out_features=12,
        sparsity=0.7, seed=0,
    )

    _clear_transpose_cache()
    losses_cached = _run_training(**kwargs, clear_cache_every_step=False)

    _clear_transpose_cache()
    losses_fresh = _run_training(**kwargs, clear_cache_every_step=True)

    max_diff = max(abs(a - b) for a, b in zip(losses_cached, losses_fresh))
    assert max_diff < 1e-5, (
        f"SET (aggressive schedule): max loss divergence {max_diff:.2e} "
        f"across 100 steps. Cache is not invalidating correctly."
    )


# ─────────────────────────────────────────────────────────────────────
#  RigL × cache
# ─────────────────────────────────────────────────────────────────────

def test_rigl_loss_trajectory_matches_fresh_transpose():
    """
    Same test shape as the SET version but for RigL. RigL takes a
    different code path through the gradient-based regrow logic, so
    it's worth testing independently.
    """
    def make_algo():
        return sparsecore.RigL(
            sparsity=0.8,
            drop_fraction=0.3,
            update_freq=25,
            seed=123,
        )

    kwargs = dict(
        algo_ctor=make_algo,
        n_steps=250,
        batch_size=16,
        in_features=32,
        out_features=16,
        sparsity=0.8,
        seed=42,
    )

    _clear_transpose_cache()
    losses_cached = _run_training(**kwargs, clear_cache_every_step=False)

    _clear_transpose_cache()
    losses_fresh = _run_training(**kwargs, clear_cache_every_step=True)

    for step, (lc, lf) in enumerate(zip(losses_cached, losses_fresh)):
        assert abs(lc - lf) < 1e-5, (
            f"RigL: cache and fresh-transpose diverged at step {step}: "
            f"cached={lc:.6f}, fresh={lf:.6f}, diff={abs(lc-lf):.2e}"
        )

    assert sum(losses_cached[-10:]) / 10 < sum(losses_cached[:10]) / 10, (
        "RigL training didn't reduce loss — something upstream broken"
    )
