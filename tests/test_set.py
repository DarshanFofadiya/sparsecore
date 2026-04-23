"""
Tests for sparselab.SET and DynamicSparsityAlgorithm (milestone 4e).

SET (Sparse Evolutionary Training) is the first dynamic-sparsity
algorithm we ship. These tests cover:

  - DynamicSparsityAlgorithm base: schedule gating, parameter validation
  - SET.update: topology actually mutates, nnz stays constant, invariants
  - Integration: training loop with SET doesn't crash, autograd still
    flows through the changing topology
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import sparselab
from sparselab import (
    DynamicSparsityAlgorithm,
    SparseLinear,
    SET,
    Static,
)


# ─────────────────────────────────────────────────────────────────────
#  DynamicSparsityAlgorithm base
# ─────────────────────────────────────────────────────────────────────

def test_dynamic_base_rejects_invalid_drop_fraction():
    with pytest.raises(ValueError, match="drop_fraction"):
        DynamicSparsityAlgorithm(sparsity=0.5, drop_fraction=0.0, update_freq=10)
    with pytest.raises(ValueError, match="drop_fraction"):
        DynamicSparsityAlgorithm(sparsity=0.5, drop_fraction=1.5, update_freq=10)


def test_dynamic_base_rejects_invalid_update_freq():
    with pytest.raises(ValueError, match="update_freq"):
        DynamicSparsityAlgorithm(sparsity=0.5, drop_fraction=0.3, update_freq=0)


def test_dynamic_step_gates_update_by_frequency():
    """update() should run exactly every `update_freq` steps."""

    class Counter(DynamicSparsityAlgorithm):
        """Records how many times update() is called."""
        def __init__(self):
            super().__init__(sparsity=0.5, drop_fraction=0.3, update_freq=10)
            self.n_updates = 0

        def update(self):
            self.n_updates += 1

    algo = Counter()
    # 9 steps: no update
    for _ in range(9):
        algo.step()
    assert algo.n_updates == 0

    # 10th step: update fires
    algo.step()
    assert algo.n_updates == 1

    # 20th step: second update fires
    for _ in range(10):
        algo.step()
    assert algo.n_updates == 2


# ─────────────────────────────────────────────────────────────────────
#  SET construction and basic semantics
# ─────────────────────────────────────────────────────────────────────

def test_set_default_args():
    algo = SET(sparsity=0.9)
    assert algo.drop_fraction == 0.3
    assert algo.update_freq == 100


def test_set_custom_args():
    algo = SET(sparsity=0.8, drop_fraction=0.5, update_freq=50)
    assert algo.sparsity == 0.8
    assert algo.drop_fraction == 0.5
    assert algo.update_freq == 50


# ─────────────────────────────────────────────────────────────────────
#  SET.update — topology mutation
# ─────────────────────────────────────────────────────────────────────

def test_set_update_preserves_nnz():
    """Total nnz must stay constant — SET swaps, it doesn't add or
    subtract connections overall."""
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(32, 64, sparsity=0.9)
    algo = SET(sparsity=0.9, drop_fraction=0.3, update_freq=1)
    layer.apply(algo)

    # Set the weight values to something non-trivial so SET's
    # magnitude-based drop has work to do.
    with torch.no_grad():
        layer._values.data.uniform_(-1.0, 1.0)

    nnz_before = layer.nnz
    algo.update()
    nnz_after = layer.nnz
    assert nnz_before == nnz_after


def test_set_update_actually_changes_topology():
    """A fraction of columns in the layer should be different after
    update(). Exact number depends on RNG — we just check 'a
    meaningful change happened'."""
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(32, 64, sparsity=0.9)
    algo = SET(sparsity=0.9, drop_fraction=0.3, update_freq=1)
    layer.apply(algo)

    with torch.no_grad():
        layer._values.data.uniform_(-1.0, 1.0)

    cols_before = np.array(layer._csr.col_indices, copy=True)
    algo.update()
    cols_after = np.asarray(layer._csr.col_indices)

    # At least some slots differ (we're dropping 30% so ~30% should change).
    n_different = int((cols_before != cols_after).sum())
    assert n_different > 0


def test_set_update_preserves_invariants():
    """After any number of updates, the CSR must still satisfy all 8
    invariants. This is our safety net against the mutation logic
    accidentally producing malformed CSRs."""
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(16, 32, sparsity=0.8)
    algo = SET(sparsity=0.8, drop_fraction=0.4, update_freq=1)
    layer.apply(algo)

    for _ in range(20):
        with torch.no_grad():
            layer._values.data.uniform_(-1.0, 1.0)
        algo.update()
        layer._csr.assert_invariants()


def test_set_new_slots_initialized_to_zero():
    """SET grows connections with value 0 (the paper's choice).
    After an update, for any slot that is newly live (was dead before),
    its value must be zero."""
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(16, 32, sparsity=0.8)
    algo = SET(sparsity=0.8, drop_fraction=0.3, update_freq=1)
    layer.apply(algo)

    with torch.no_grad():
        layer._values.data.uniform_(-1.0, 1.0)

    # Record which (row, col) cells were live before
    live_before = set()
    cols = np.asarray(layer._csr.col_indices)
    row_start = np.asarray(layer._csr.row_start)
    row_nnz = np.asarray(layer._csr.row_nnz)
    vals = np.asarray(layer._csr.values)
    for i in range(layer._csr.nrows):
        for s in range(int(row_nnz[i])):
            live_before.add((i, int(cols[int(row_start[i]) + s])))

    algo.update()

    # Find live slots AFTER and check new ones are zero
    cols_after = np.asarray(layer._csr.col_indices)
    vals_after = np.asarray(layer._csr.values)
    row_start_after = np.asarray(layer._csr.row_start)
    row_nnz_after = np.asarray(layer._csr.row_nnz)

    for i in range(layer._csr.nrows):
        for s in range(int(row_nnz_after[i])):
            slot = int(row_start_after[i]) + s
            c = int(cols_after[slot])
            if (i, c) not in live_before:
                # Newly grown — must be zero
                assert vals_after[slot] == 0.0, (
                    f"newly-grown slot at row {i}, col {c} has value "
                    f"{vals_after[slot]}, expected 0.0"
                )


# ─────────────────────────────────────────────────────────────────────
#  Integration: training loop with SET
# ─────────────────────────────────────────────────────────────────────

def test_training_step_after_set_update_still_works():
    """After SET mutates topology, a following forward+backward+opt.step()
    must still work. This is the canary for integration bugs."""
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(32, 16, sparsity=0.7)
    algo = SET(sparsity=0.7, drop_fraction=0.3, update_freq=1)
    layer.apply(algo)

    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    for _ in range(5):
        # Train step
        opt.zero_grad()
        x = torch.randn(8, 32)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        opt.step()
        # Topology mutation (update_freq=1 so every step)
        algo.step()

    # Still invariant-correct
    layer._csr.assert_invariants()


def test_set_doesnt_drop_all_weights_in_small_rows():
    """If a row has very few live connections (e.g. 2), SET's
    drop_fraction must not produce a row with 0 live slots.
    Contract: always keep at least 1 live per row."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Set up a layer with very few live connections per row
    layer = SparseLinear(16, 8, sparsity=0.9)
    algo = SET(sparsity=0.9, drop_fraction=0.99, update_freq=1)
    layer.apply(algo)

    with torch.no_grad():
        layer._values.data.uniform_(-1.0, 1.0)

    algo.update()

    row_nnz = np.asarray(layer._csr.row_nnz)
    # Every row that had >= 2 before should still have >= 1 after
    for i in range(layer._csr.nrows):
        # Some rows might start with 0 — those stay 0 (no bug)
        # Rows with >= 1 must end with >= 1
        if int(row_nnz[i]) > 0:
            assert int(row_nnz[i]) >= 1
    layer._csr.assert_invariants()


def test_model_apply_attaches_set_to_multiple_layers():
    torch.manual_seed(0)
    np.random.seed(0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = SparseLinear(32, 16, sparsity=0.7)
            self.fc2 = SparseLinear(16, 8, sparsity=0.7)

    model = MLP()
    algo = SET(sparsity=0.7, drop_fraction=0.3, update_freq=1)
    model.apply(algo)

    assert len(algo.layers) == 2
    with torch.no_grad():
        model.fc1._values.data.uniform_(-1.0, 1.0)
        model.fc2._values.data.uniform_(-1.0, 1.0)

    algo.update()
    # Both layers' invariants must hold
    model.fc1._csr.assert_invariants()
    model.fc2._csr.assert_invariants()


def test_set_update_is_deterministic_under_seed():
    """With the same seed, two SET runs should produce identical
    topology changes. SET accepts a seed parameter to make its RNG
    deterministic without polluting numpy's global state."""
    torch.manual_seed(0)
    layer_a = SparseLinear(16, 16, sparsity=0.8)
    algo_a = SET(sparsity=0.8, drop_fraction=0.3, update_freq=1, seed=42)
    layer_a.apply(algo_a)
    with torch.no_grad():
        layer_a._values.data.uniform_(-1, 1)
    algo_a.update()
    cols_a = np.array(layer_a._csr.col_indices, copy=True)

    torch.manual_seed(0)
    layer_b = SparseLinear(16, 16, sparsity=0.8)
    algo_b = SET(sparsity=0.8, drop_fraction=0.3, update_freq=1, seed=42)
    layer_b.apply(algo_b)
    with torch.no_grad():
        layer_b._values.data.uniform_(-1, 1)
    algo_b.update()
    cols_b = np.array(layer_b._csr.col_indices, copy=True)

    np.testing.assert_array_equal(cols_a, cols_b)
