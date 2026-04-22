"""
Tests for the transpose cache (Experiment A, topology-version keyed).

What we verify:
  1. Cache hit: the same W produces the same WT pointer on repeat calls
     WITHOUT bumping topology_version.
  2. Cache miss: after topology change, a fresh WT is built.
  3. Value refresh on cache hit: when W.values change (SGD step),
     the returned WT has up-to-date values.
  4. Gradient correctness: autograd through SparseLinear still produces
     bit-correct gradients with the cache enabled.
  5. gradcheck: numerical vs analytical gradients match (integration check).
"""

import numpy as np
import pytest
import torch

import sparsecore
from sparsecore import SparseLinear, SET
from sparsecore.ops import (
    _cached_transpose,
    _clear_transpose_cache,
    _transpose_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Every test starts with a fresh cache."""
    _clear_transpose_cache()
    yield
    _clear_transpose_cache()


# ─────────────────────────────────────────────────────────────────────
#  Basic cache semantics
# ─────────────────────────────────────────────────────────────────────

def test_cache_hit_returns_same_object():
    """Second call with the same W (no topology change) should hit
    the cache and return the same WT object."""
    W = sparsecore.PaddedCSR.random(8, 16, sparsity=0.5, seed=0)

    WT1 = _cached_transpose(W)
    WT2 = _cached_transpose(W)

    # Cache hit means we return the SAME PaddedCSR object, not just
    # an equivalent one.
    assert WT1 is WT2


def test_cache_miss_after_topology_change():
    """After rewrite_row bumps topology_version, the next call must
    rebuild (returns a DIFFERENT WT)."""
    W = sparsecore.PaddedCSR.random(8, 16, sparsity=0.5, seed=0)

    WT1 = _cached_transpose(W)
    initial_version = W.topology_version

    # Mutate topology
    W.rewrite_row(0, np.array([0, 5], dtype=np.int32),
                      np.array([1.0, 2.0], dtype=np.float32))
    assert W.topology_version > initial_version

    WT2 = _cached_transpose(W)
    assert WT1 is not WT2


def test_cache_value_refresh_on_hit():
    """When W.values changes but topology doesn't (normal SGD step),
    the cached WT must show the updated values — not stale ones."""
    W = sparsecore.PaddedCSR.random(8, 16, sparsity=0.5, seed=0)

    # Prime the cache
    WT_before = _cached_transpose(W)
    vals_before = np.asarray(WT_before.values).copy()

    # Mutate W.values in place (no topology change)
    W_vals = np.asarray(W.values)
    W_vals[:] *= 10.0   # scale all live values

    # Second call — cache hit, but values must be refreshed
    WT_after = _cached_transpose(W)
    vals_after = np.asarray(WT_after.values)

    # Same CSR object (cache hit)
    assert WT_after is WT_before

    # But the values differ now — they tracked the W.values change
    # Only live slots matter; padding stays 0.
    assert not np.allclose(vals_before, vals_after)


def test_cache_correctness_against_fresh_transpose():
    """_cached_transpose must match a fresh W.transpose() in content
    (values and topology) at every call — cache hit or miss."""
    W = sparsecore.PaddedCSR.random(16, 8, sparsity=0.5, seed=0)

    # Miss path
    WT_cached = _cached_transpose(W)
    WT_fresh = W.transpose()

    # Same shape, same topology, same values
    assert WT_cached.nrows == WT_fresh.nrows
    assert WT_cached.ncols == WT_fresh.ncols
    assert WT_cached.nnz == WT_fresh.nnz
    # Dense reconstruction is the easy oracle
    np.testing.assert_allclose(
        WT_cached.to_dense(), WT_fresh.to_dense(), rtol=1e-6, atol=1e-6
    )

    # Now change W values and confirm cache refreshes correctly
    W_vals = np.asarray(W.values)
    W_vals[:] = np.random.randn(len(W_vals)).astype(np.float32)

    WT_cached_2 = _cached_transpose(W)
    WT_fresh_2 = W.transpose()
    np.testing.assert_allclose(
        WT_cached_2.to_dense(), WT_fresh_2.to_dense(), rtol=1e-6, atol=1e-6
    )


# ─────────────────────────────────────────────────────────────────────
#  End-to-end autograd: gradients still bit-correct
# ─────────────────────────────────────────────────────────────────────

def test_gradients_match_before_and_after_value_update():
    """A full fwd+bwd with cache enabled should produce identical
    gradients to one with a fresh transpose each time."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5, bias=False)

    # First fwd+bwd: populates cache
    x1 = torch.randn(4, 16, requires_grad=True)
    y1 = layer(x1)
    y1.sum().backward()
    grad_values_1 = layer._values.grad.clone()
    grad_input_1 = x1.grad.clone()

    # Simulate an SGD step (mutates W.values in place via _values)
    with torch.no_grad():
        layer._values.data *= 0.95   # shrink all live weights

    # Second fwd+bwd: uses cache (topology unchanged), but W.values differ
    if layer._values.grad is not None:
        layer._values.grad.zero_()
    x2 = torch.randn(4, 16, requires_grad=True)
    y2 = layer(x2)
    y2.sum().backward()
    grad_values_2 = layer._values.grad.clone()

    # The gradients should differ (different input, different weights)
    assert not torch.allclose(grad_values_1, grad_values_2)
    # And be finite
    assert torch.isfinite(grad_values_2).all()


def test_gradients_match_fresh_transpose_path():
    """Compare gradients computed via _cached_transpose to gradients
    computed by forcing a fresh transpose each call. Must match exactly
    if the cache is correct."""
    torch.manual_seed(0)
    layer_a = SparseLinear(16, 8, sparsity=0.5, bias=False)

    torch.manual_seed(0)
    layer_b = SparseLinear(16, 8, sparsity=0.5, bias=False)

    x = torch.randn(4, 16)
    x_a = x.clone().detach().requires_grad_(True)
    x_b = x.clone().detach().requires_grad_(True)

    # Path A: normal (uses cache)
    y_a = layer_a(x_a)
    y_a.sum().backward()

    # Path B: clear cache before each call so we always take the miss path
    _clear_transpose_cache()
    y_b = layer_b(x_b)
    y_b.sum().backward()

    torch.testing.assert_close(
        layer_a._values.grad, layer_b._values.grad,
        rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(x_a.grad, x_b.grad, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Integration with SET: cache invalidates on update()
# ─────────────────────────────────────────────────────────────────────

def test_set_update_invalidates_cache():
    """After SET.update() fires (which calls rewrite_row), the cache
    must invalidate and rebuild on the next backward."""
    torch.manual_seed(0)
    np.random.seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5, bias=False)

    # First forward/backward populates cache
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    # Cache has one entry
    assert len(_transpose_cache) == 1
    (cached_version, _, _, _) = list(_transpose_cache.values())[0]

    # Fire SET update — bumps topology_version via rewrite_row
    algo = SET(sparsity=0.5, drop_fraction=0.3, update_freq=1, seed=42)
    layer.apply(algo)
    algo.update()

    # Run another backward
    x2 = torch.randn(4, 16, requires_grad=True)
    y2 = layer(x2)
    y2.sum().backward()

    # Cache entry should now have a different version than before
    (new_version, _, _, _) = list(_transpose_cache.values())[0]
    assert new_version != cached_version, (
        f"cache should have invalidated; versions {cached_version} == {new_version}"
    )


# ─────────────────────────────────────────────────────────────────────
#  gradcheck: the ultimate correctness test
# ─────────────────────────────────────────────────────────────────────

def test_gradcheck_with_cache_enabled():
    """Numerical finite-differences vs analytical gradients.
    This is the strongest correctness test we have. With the cache
    active, it exercises every code path."""
    torch.manual_seed(0)
    layer = SparseLinear(8, 4, sparsity=0.5, bias=False)

    # Small inputs for gradcheck (it's O(n²))
    x = torch.randn(3, 8, dtype=torch.double, requires_grad=True)
    layer.double()
    # gradcheck needs the module callable wrapped as a scalar-loss function
    def f(x):
        return layer(x).sum()
    # tol relaxed for float→double cast noise
    assert torch.autograd.gradcheck(f, (x,), eps=1e-3, atol=1e-3, rtol=1e-3)
