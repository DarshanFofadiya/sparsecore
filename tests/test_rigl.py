"""
Tests for sparselab.RigL (milestone 4f).

RigL is similar to SET, with the same DynamicSparsityAlgorithm base,
but a smarter grow criterion: grow at positions where the dense
gradient magnitude is highest.

Test strategy:
  - Construction + schedule gating (same contract as SET)
  - topology mutation preserves nnz, invariants
  - grown positions correspond to TOP-|G|, not random
  - end-to-end training step after RigL update works
  - reproducibility under seed
"""

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import sparselab
from sparselab import RigL, SparseLinear, Static


# Silence the benign "Full backward hook is firing..." warning that
# appears in tests where input tensors don't have requires_grad=True.
# In real training the upstream layer's output has requires_grad=True
# so this warning doesn't fire.
warnings.filterwarnings("ignore", category=UserWarning,
                         message=".*backward hook.*")


# ─────────────────────────────────────────────────────────────────────
#  Construction
# ─────────────────────────────────────────────────────────────────────

def test_rigl_constructs_with_defaults():
    algo = RigL(sparsity=0.9)
    assert algo.sparsity == 0.9
    assert algo.drop_fraction == 0.3
    assert algo.update_freq == 100
    assert algo.layers == []


def test_rigl_accepts_custom_args():
    algo = RigL(sparsity=0.8, drop_fraction=0.5, update_freq=50, seed=42)
    assert algo.drop_fraction == 0.5
    assert algo.update_freq == 50


def test_rigl_inherits_base_class_validation():
    with pytest.raises(ValueError, match="drop_fraction"):
        RigL(sparsity=0.9, drop_fraction=0.0)
    with pytest.raises(ValueError, match="update_freq"):
        RigL(sparsity=0.9, update_freq=0)


# ─────────────────────────────────────────────────────────────────────
#  Attach + hook installation
# ─────────────────────────────────────────────────────────────────────

def test_rigl_attach_registers_hooks():
    """Attaching RigL to a layer should install forward + backward hooks.
    Running a forward+backward then inspecting captured state should show
    X and dY stashed."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5, bias=False)
    algo = RigL(sparsity=0.5, drop_fraction=0.3, update_freq=100, seed=42)
    layer.apply(algo)

    # Run forward+backward to trigger the hooks.
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    captured = algo._captured[id(layer)]
    assert "X" in captured
    assert "dY" in captured
    assert captured["X"].shape == (4, 16)
    assert captured["dY"].shape == (4, 8)


# ─────────────────────────────────────────────────────────────────────
#  Update semantics
# ─────────────────────────────────────────────────────────────────────

def test_rigl_update_preserves_nnz():
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(32, 16, sparsity=0.7, bias=False)
    algo = RigL(sparsity=0.7, drop_fraction=0.3, update_freq=1, seed=42)
    layer.apply(algo)

    # Run one forward+backward step so hooks capture state.
    x = torch.randn(8, 32, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    nnz_before = layer.nnz
    algo.update()
    assert layer.nnz == nnz_before


def test_rigl_update_preserves_invariants():
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(16, 32, sparsity=0.8, bias=False)
    algo = RigL(sparsity=0.8, drop_fraction=0.4, update_freq=1, seed=42)
    layer.apply(algo)

    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    # Run 10 training steps, each triggering an update.
    for _ in range(10):
        opt.zero_grad()
        x = torch.randn(4, 16, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        opt.step()
        algo.step()
        layer._csr.assert_invariants()


def test_rigl_grows_at_high_gradient_positions():
    """The core RigL property: grown positions should be where |G|
    is high, not random. We test this by constructing a scenario
    where the gradient at certain columns is known to be large, then
    verify RigL grows there."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Layer at high sparsity so we have lots of empty columns.
    # 1 row × 20 columns, only 2 live.
    layer = SparseLinear(20, 1, sparsity=0.9, bias=False)
    algo = RigL(sparsity=0.9, drop_fraction=0.5, update_freq=1, seed=0)
    layer.apply(algo)

    # Force the layer to have live weights at known columns.
    # We can't easily do this through public API — use internals.
    # Start fresh: overwrite row 0 to have exactly cols [0, 1] live.
    layer._csr.rewrite_row(
        0,
        np.array([0, 1], dtype=np.int32),
        np.array([0.01, 0.02], dtype=np.float32),
    )

    # Now construct an input and dY such that the dense gradient G[0, :]
    # has a known shape. G[0, k] = sum_j dY[0, j] * X[k, j].
    # Pick dY = [[1.0]] (shape 1x1), X = shape (20, 1).
    # Then G[0, k] = 1.0 * X[k, 0] = X[k, 0].
    # So to make column 15 the winner, set X[15, 0] very large.
    #
    # We do this by faking the captured state directly.
    X_fake = torch.zeros(1, 20)
    X_fake[0, 15] = 100.0      # column 15 should dominate
    X_fake[0, 12] = 50.0       # column 12 second
    dY_fake = torch.ones(1, 1)
    algo._captured[id(layer)] = {"X": X_fake, "dY": dY_fake}

    # Now update. Should drop one of the (small-magnitude) live slots
    # and grow at column 15 (the highest-|G| empty position).
    algo.update()

    # After update: we expect column 15 to be live (possibly also 12
    # depending on drop fraction, but at minimum 15).
    cols_after = set(
        int(c) for c in np.asarray(layer._csr.col_indices)[
            int(np.asarray(layer._csr.row_start)[0]) :
            int(np.asarray(layer._csr.row_start)[0]) +
            int(np.asarray(layer._csr.row_nnz)[0])
        ]
    )
    assert 15 in cols_after, (
        f"Expected column 15 (highest |G|) to be grown; got cols {cols_after}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Integration: full training loop
# ─────────────────────────────────────────────────────────────────────

def test_rigl_in_training_loop_doesnt_crash():
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(16, 8, sparsity=0.7, bias=False)
    algo = RigL(sparsity=0.7, drop_fraction=0.3, update_freq=2, seed=42)
    layer.apply(algo)

    opt = torch.optim.SGD(layer.parameters(), lr=0.01)
    labels = torch.randint(0, 8, (6,))

    for step in range(10):
        opt.zero_grad()
        x = torch.randn(6, 16, requires_grad=True)
        logits = layer(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        algo.step()

    # Still invariant-valid after 10 steps and ~5 updates
    layer._csr.assert_invariants()


def test_rigl_without_any_forward_is_noop_update():
    """If update() is called before any forward+backward, there's no
    captured gradient info — should just skip gracefully, not crash."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5, bias=False)
    algo = RigL(sparsity=0.5, drop_fraction=0.3, update_freq=1, seed=42)
    layer.apply(algo)

    nnz_before = layer.nnz
    algo.update()  # no prior forward/backward → no captures
    assert layer.nnz == nnz_before  # unchanged


def test_rigl_reproducible_under_seed():
    """Same seed + same input sequence → same topology after update."""

    def run_once():
        torch.manual_seed(0)
        np.random.seed(0)
        layer = SparseLinear(16, 16, sparsity=0.7, bias=False)
        algo = RigL(sparsity=0.7, drop_fraction=0.3, update_freq=1, seed=42)
        layer.apply(algo)

        x = torch.randn(4, 16, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        algo.update()
        return np.array(layer._csr.col_indices, copy=True)

    cols_a = run_once()
    cols_b = run_once()
    np.testing.assert_array_equal(cols_a, cols_b)


def test_model_apply_attaches_rigl_to_multiple_layers():
    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = SparseLinear(32, 16, sparsity=0.7, bias=False)
            self.fc2 = SparseLinear(16, 8, sparsity=0.7, bias=False)

    model = MLP()
    algo = RigL(sparsity=0.7, drop_fraction=0.3, update_freq=1, seed=42)
    model.apply(algo)

    assert len(algo.layers) == 2
    assert model.fc1 in algo.layers
    assert model.fc2 in algo.layers
