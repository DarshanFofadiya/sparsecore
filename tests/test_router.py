"""
Tests for sparsecore.router (Milestone 4d).

The Router API is pure Python — it doesn't touch kernels — so the
tests focus on the API contract:

  - SparsityAlgorithm base class: construction, attach, step counter
  - Static reference: no-op on update, doesn't mutate topology
  - Module-walk semantics: model.apply filters to SparseLinear only
  - Back-pointer: layer._sparsity_algorithm stays in sync
  - Base class: update() raises if a subclass doesn't override it
  - Training still works after attachment (sanity regression)
"""

import pytest
import torch
import torch.nn as nn

import sparsecore
from sparsecore import SparseLinear, SparsityAlgorithm, Static


# ─────────────────────────────────────────────────────────────────────
#  SparsityAlgorithm — construction / validation
# ─────────────────────────────────────────────────────────────────────

def test_static_constructs_with_valid_sparsity():
    algo = Static(sparsity=0.9)
    assert algo.sparsity == 0.9
    assert algo.layers == []
    assert algo._step_idx == 0


def test_static_rejects_invalid_sparsity():
    with pytest.raises(ValueError, match="sparsity"):
        Static(sparsity=1.0)
    with pytest.raises(ValueError, match="sparsity"):
        Static(sparsity=-0.1)


def test_base_update_raises_not_implemented():
    """Directly constructing and updating the abstract base class
    should fail loudly, not silently do nothing."""
    algo = SparsityAlgorithm(sparsity=0.5)
    with pytest.raises(NotImplementedError):
        algo.update()


# ─────────────────────────────────────────────────────────────────────
#  attach / layer.apply
# ─────────────────────────────────────────────────────────────────────

def test_attach_sets_back_pointer():
    torch.manual_seed(0)
    layer = SparseLinear(32, 16, sparsity=0.5)
    algo = Static(sparsity=0.5)
    algo.attach(layer)
    assert algo.layers == [layer]
    assert layer._sparsity_algorithm is algo


def test_attach_rejects_non_sparse_linear():
    """attach() should refuse a plain nn.Linear with a clear error."""
    dense = nn.Linear(16, 8)
    algo = Static(sparsity=0.5)
    with pytest.raises(TypeError, match="SparseLinear"):
        algo.attach(dense)


def test_attach_is_idempotent():
    """Attaching the same layer twice should not duplicate it in .layers."""
    torch.manual_seed(0)
    layer = SparseLinear(32, 16, sparsity=0.5)
    algo = Static(sparsity=0.5)
    algo.attach(layer)
    algo.attach(layer)
    assert len(algo.layers) == 1


def test_layer_apply_routes_to_algorithm():
    """layer.apply(algo) is PyTorch's module-walk. For a single layer,
    the algo should receive exactly one __call__ and attach to it."""
    torch.manual_seed(0)
    layer = SparseLinear(32, 16, sparsity=0.5)
    algo = Static(sparsity=0.5)
    layer.apply(algo)
    assert algo.layers == [layer]


# ─────────────────────────────────────────────────────────────────────
#  model.apply recursion — filters to SparseLinear
# ─────────────────────────────────────────────────────────────────────

class _MixedMLP(nn.Module):
    """Model with both sparse and dense layers, used to verify
    that model.apply(algo) only attaches to the sparse ones."""
    def __init__(self):
        super().__init__()
        self.fc1 = SparseLinear(64, 128, sparsity=0.9)   # sparse
        self.bn = nn.BatchNorm1d(128)                    # dense
        self.fc2 = nn.Linear(128, 64)                    # dense
        self.fc3 = SparseLinear(64, 32, sparsity=0.9)    # sparse


def test_model_apply_attaches_only_to_sparse_linear():
    torch.manual_seed(0)
    model = _MixedMLP()
    algo = Static(sparsity=0.9)
    model.apply(algo)
    # Should attach to fc1 and fc3, skip bn, fc2, and the top-level module.
    assert len(algo.layers) == 2
    assert model.fc1 in algo.layers
    assert model.fc3 in algo.layers


def test_model_apply_back_pointers_on_all_sparse_layers():
    torch.manual_seed(0)
    model = _MixedMLP()
    algo = Static(sparsity=0.9)
    model.apply(algo)
    assert model.fc1._sparsity_algorithm is algo
    assert model.fc3._sparsity_algorithm is algo


def test_different_algos_can_govern_different_layers():
    """Nothing in our API forbids using two algorithms on a model:
    one on some layers, one on others. This is the precursor to the
    Group abstraction (4d-ii)."""
    torch.manual_seed(0)
    model = _MixedMLP()
    algo1 = Static(sparsity=0.9)
    algo2 = Static(sparsity=0.5)
    model.fc1.apply(algo1)
    model.fc3.apply(algo2)
    assert algo1.layers == [model.fc1]
    assert algo2.layers == [model.fc3]
    assert model.fc1._sparsity_algorithm is algo1
    assert model.fc3._sparsity_algorithm is algo2


# ─────────────────────────────────────────────────────────────────────
#  step counter and Static update()
# ─────────────────────────────────────────────────────────────────────

def test_step_increments_counter():
    algo = Static(sparsity=0.5)
    assert algo._step_idx == 0
    algo.step()
    assert algo._step_idx == 1
    for _ in range(9):
        algo.step()
    assert algo._step_idx == 10


def test_static_update_is_noop_on_topology():
    """Static.update() must not change nnz, col_indices, or values
    (values can change because the user trains them — but the TOPOLOGY
    shouldn't). We check nnz and col_indices identity here."""
    torch.manual_seed(0)
    layer = SparseLinear(32, 16, sparsity=0.7)
    algo = Static(sparsity=0.7)
    layer.apply(algo)

    nnz_before = layer.nnz
    # Snapshot col_indices — they're a numpy read-only view.
    import numpy as np
    cols_before = np.array(layer._csr.col_indices, copy=True)

    # Multiple step() + explicit update() should all be no-ops.
    for _ in range(5):
        algo.step()
    algo.update()

    assert layer.nnz == nnz_before
    cols_after = np.asarray(layer._csr.col_indices)
    np.testing.assert_array_equal(cols_before, cols_after)


def test_repr_includes_sparsity_and_layer_count():
    algo = Static(sparsity=0.9)
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.9)
    layer.apply(algo)
    algo.step()
    text = repr(algo)
    assert "Static" in text
    assert "sparsity=0.9" in text
    assert "n_layers=1" in text
    assert "step=1" in text


# ─────────────────────────────────────────────────────────────────────
#  Sanity regression — a training loop with Static works identically
#  to one without. This is the canary that says we didn't break
#  anything by introducing the algorithm plumbing.
# ─────────────────────────────────────────────────────────────────────

def test_training_loop_with_static_matches_no_attach():
    """Two identical models trained with identical inputs. One has
    a Static algorithm attached; the other doesn't. Final weights
    must be bit-identical: Static.step() is a no-op.
    """
    def make_model(seed):
        torch.manual_seed(seed)
        fc1 = SparseLinear(16, 8, sparsity=0.5, bias=False)
        fc2 = nn.Linear(8, 4, bias=False)
        return fc1, fc2

    fc1_a, fc2_a = make_model(0)
    fc1_b, fc2_b = make_model(0)

    algo = Static(sparsity=0.5)
    fc1_b.apply(algo)

    opt_a = torch.optim.SGD(
        list(fc1_a.parameters()) + list(fc2_a.parameters()), lr=0.01
    )
    opt_b = torch.optim.SGD(
        list(fc1_b.parameters()) + list(fc2_b.parameters()), lr=0.01
    )

    # Same sequence of inputs to both.
    torch.manual_seed(123)
    xs = [torch.randn(6, 16) for _ in range(10)]
    torch.manual_seed(123)  # same seed — same inputs
    xs_b = [torch.randn(6, 16) for _ in range(10)]

    for xa, xb in zip(xs, xs_b):
        # Path A — no algorithm
        opt_a.zero_grad()
        loss_a = fc2_a(torch.relu(fc1_a(xa))).sum()
        loss_a.backward()
        opt_a.step()

        # Path B — with Static algo
        opt_b.zero_grad()
        loss_b = fc2_b(torch.relu(fc1_b(xb))).sum()
        loss_b.backward()
        opt_b.step()
        algo.step()

    torch.testing.assert_close(fc1_a._values.data, fc1_b._values.data,
                                rtol=0, atol=0)
    torch.testing.assert_close(fc2_a.weight.data, fc2_b.weight.data,
                                rtol=0, atol=0)
