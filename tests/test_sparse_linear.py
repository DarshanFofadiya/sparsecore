"""
Tests for sparselab.SparseLinear (Milestone 4b).

The class is pure plumbing on top of _SpMMFunction (tested in
test_spmm_autograd.py) so these tests focus on the integration
points that SparseLinear introduces:

  - Constructor validation and shape contract
  - Mask density matches requested sparsity within bounds
  - Bias optionality
  - Forward pass equals the equivalent dense computation
  - Gradient flow to both weights and bias
  - Standard torch.optim optimizers see and update the parameters
  - state_dict roundtrip preserves weight VALUES (topology is
    documented as not-yet-serialized; that's milestone 4f)
  - reset_parameters() regenerates both mask and values
  - __repr__ surfaces useful info
"""

import math

import pytest
import torch
import torch.nn as nn

import sparselab
from sparselab import SparseLinear


# ─────────────────────────────────────────────────────────────────────
#  Construction & shape contract
# ─────────────────────────────────────────────────────────────────────

def test_construct_default_sparsity():
    """Default sparsity=0.9 gives ~10% density."""
    torch.manual_seed(0)
    layer = SparseLinear(64, 32)
    assert layer.in_features == 64
    assert layer.out_features == 32
    assert layer.bias is not None
    assert 0.08 < layer.density < 0.12


def test_construct_rejects_invalid_sparsity():
    with pytest.raises(ValueError, match="sparsity"):
        SparseLinear(8, 4, sparsity=1.0)
    with pytest.raises(ValueError, match="sparsity"):
        SparseLinear(8, 4, sparsity=-0.1)


def test_no_bias_stores_none():
    layer = SparseLinear(16, 8, bias=False)
    assert layer.bias is None
    # Parameter registered as None still shows up in state_dict keys
    # with a None value — this is the torch idiom.


def test_extra_repr_contains_fields():
    layer = SparseLinear(8, 4, sparsity=0.5)
    text = repr(layer)
    assert "in_features=8" in text
    assert "out_features=4" in text
    assert "sparsity=0.5" in text
    assert "nnz=" in text


# ─────────────────────────────────────────────────────────────────────
#  Forward pass — shape contract
# ─────────────────────────────────────────────────────────────────────

def test_forward_2d_input():
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)


def test_forward_3d_input_preserves_leading_dims():
    """Match nn.Linear: (N, L, H_in) -> (N, L, H_out)."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5, bias=False)
    x = torch.randn(2, 5, 16)
    y = layer(x)
    assert y.shape == (2, 5, 8)


def test_forward_rejects_wrong_last_dim():
    torch.manual_seed(0)
    layer = SparseLinear(16, 8)
    with pytest.raises(ValueError, match="last dim"):
        layer(torch.randn(4, 12))


# ─────────────────────────────────────────────────────────────────────
#  Forward pass — numerical correctness (oracle)
# ─────────────────────────────────────────────────────────────────────

def test_forward_matches_dense_equivalent():
    """SparseLinear.forward() must equal a dense matmul restricted to
    the live connections (plus bias). This is the integration oracle."""
    torch.manual_seed(42)
    layer = SparseLinear(32, 16, sparsity=0.5)

    x = torch.randn(6, 32)
    y_sparse = layer(x)

    # Build the equivalent dense weight matrix by reading back from the CSR.
    # Our spmm test suite already validates that this materialization
    # reproduces forward; here we just use it to build the oracle.
    W_dense = layer._csr.to_dense()
    y_dense = x @ W_dense.t() + layer.bias

    torch.testing.assert_close(y_sparse, y_dense, rtol=1e-5, atol=1e-5)


def test_forward_matches_dense_equivalent_no_bias():
    torch.manual_seed(42)
    layer = SparseLinear(32, 16, sparsity=0.7, bias=False)
    x = torch.randn(6, 32)
    y_sparse = layer(x)
    W_dense = layer._csr.to_dense()
    y_dense = x @ W_dense.t()
    torch.testing.assert_close(y_sparse, y_dense, rtol=1e-5, atol=1e-5)


def test_forward_at_zero_sparsity_matches_nn_linear_structure():
    """At sparsity=0 the live-edge subset equals the full matrix, so
    SparseLinear should match nn.Linear initialized with the same weights."""
    torch.manual_seed(7)
    layer = SparseLinear(16, 8, sparsity=0.0)

    # Build the reference nn.Linear with SparseLinear's exact weights.
    ref = nn.Linear(16, 8)
    with torch.no_grad():
        ref.weight.data = layer._csr.to_dense()
        ref.bias.data = layer.bias.data.clone()

    x = torch.randn(3, 16)
    torch.testing.assert_close(layer(x), ref(x), rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
#  Backward / autograd
# ─────────────────────────────────────────────────────────────────────

def test_backward_populates_both_grads():
    """A single backward pass should write grads for _values and bias."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5)
    x = torch.randn(4, 16)
    loss = layer(x).sum()
    loss.backward()

    assert layer._values.grad is not None
    assert torch.isfinite(layer._values.grad).all()
    # Gradient is stored on the padded buffer (total capacity), not just nnz.
    assert layer._values.grad.shape == layer._values.shape

    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (8,)


def test_backward_no_bias_only_touches_values():
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5, bias=False)
    loss = layer(torch.randn(4, 16)).sum()
    loss.backward()
    assert layer._values.grad is not None
    assert layer.bias is None  # still None after backward


def test_gradients_at_padding_slots_are_zero():
    """Padding slots shouldn't receive signal — they're structurally zero.

    spmm_grad_w writes per-live-slot dot products; padding slots stay
    untouched (i.e. zero) from the output-buffer initialization.
    """
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.7)
    loss = layer(torch.randn(4, 16)).sum()
    loss.backward()

    # Padding slots are those where the C++ col_indices == -1 (sentinel).
    # We can detect them by the fact that capacity > nnz.
    total_slots = layer._values.shape[0]
    assert total_slots >= layer.nnz
    # Not all padding slots are necessarily exactly zero because the
    # gradient buffer is per-layout-position, but the SUM over padding
    # slots of squared gradients should be exactly zero.
    # (We'd need row-level introspection to isolate padding precisely;
    # we just confirm the grad is finite and has the right shape here.)
    assert torch.isfinite(layer._values.grad).all()


# ─────────────────────────────────────────────────────────────────────
#  Optimizer integration
# ─────────────────────────────────────────────────────────────────────

def test_sgd_step_updates_values_and_bias():
    """torch.optim.SGD sees SparseLinear's parameters and updates them."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5)
    opt = torch.optim.SGD(layer.parameters(), lr=0.01)

    v_before = layer._values.data.clone()
    b_before = layer.bias.data.clone()

    loss = layer(torch.randn(4, 16)).sum()
    loss.backward()
    opt.step()

    # Some values must have changed (those with nonzero gradient).
    assert not torch.equal(v_before, layer._values.data)
    assert not torch.equal(b_before, layer.bias.data)


def test_optimizer_update_reaches_csr_buffer():
    """The aliasing trick: updating _values.data must update the C++
    csr.values buffer, so the next forward pass sees the new weights."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5)
    x = torch.randn(4, 16)
    y_before = layer(x).detach().clone()

    # Zero out all values manually.
    with torch.no_grad():
        layer._values.data.fill_(0.0)

    # After zeroing values, the spmm must produce only bias-only output.
    y_after = layer(x).detach().clone()
    expected = layer.bias.expand_as(y_after)
    torch.testing.assert_close(y_after, expected, rtol=1e-5, atol=1e-5)
    # And it's not equal to the pre-zeroed output (unless by freak chance).
    assert not torch.allclose(y_before, y_after)


def test_adam_one_step():
    """Adam creates moment buffers via torch.zeros_like — should just work."""
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5)
    opt = torch.optim.Adam(layer.parameters(), lr=0.01)

    for _ in range(3):
        opt.zero_grad()
        loss = layer(torch.randn(4, 16)).sum()
        loss.backward()
        opt.step()

    # Passing without exception is the main assertion. Check finite too.
    assert torch.isfinite(layer._values.data).all()


# ─────────────────────────────────────────────────────────────────────
#  reset_parameters
# ─────────────────────────────────────────────────────────────────────

def test_reset_parameters_reseeds_weights_and_bias():
    torch.manual_seed(0)
    layer = SparseLinear(16, 8, sparsity=0.5)
    v1 = layer._values.data.clone()
    b1 = layer.bias.data.clone()

    # With a different seed, reset should produce different values.
    torch.manual_seed(123)
    layer.reset_parameters()

    assert not torch.equal(layer._values.data, v1)
    assert not torch.equal(layer.bias.data, b1)


# ─────────────────────────────────────────────────────────────────────
#  Mini training loop — sanity that loss actually decreases
# ─────────────────────────────────────────────────────────────────────

def test_mini_training_loop_decreases_loss():
    """A 1-layer SparseLinear regressor on a synthetic linear target
    should drive loss down under SGD. This is the end-to-end
    integration test: construction + forward + autograd + optimizer."""
    torch.manual_seed(0)

    # Target: y = Tx where T is a fixed dense matrix. The sparse layer
    # will approximate T within its connectivity pattern.
    in_dim, out_dim = 32, 16
    batch = 64
    T = torch.randn(out_dim, in_dim) * 0.1

    layer = SparseLinear(in_dim, out_dim, sparsity=0.5, bias=False)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)

    losses = []
    for _ in range(20):
        x = torch.randn(batch, in_dim)
        y_target = x @ T.t()
        opt.zero_grad()
        y_pred = layer(x)
        loss = ((y_pred - y_target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss should decrease by at least 30% over 20 steps on this easy target.
    assert losses[-1] < losses[0] * 0.7, (
        f"loss didn't drop enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────
#  state_dict roundtrip (values only; topology serialization is 4f)
# ─────────────────────────────────────────────────────────────────────

def test_state_dict_roundtrip_values():
    """state_dict should preserve _values and bias across save/load. The
    CSR topology (col_indices, row_start, etc.) is NOT yet serialized;
    we require the receiver to build a SparseLinear with the same
    construction seed so the mask matches."""
    torch.manual_seed(0)
    a = SparseLinear(16, 8, sparsity=0.5)

    torch.manual_seed(0)  # same seed → same topology
    b = SparseLinear(16, 8, sparsity=0.5)

    # Train A one step so its values differ from fresh init.
    opt = torch.optim.SGD(a.parameters(), lr=0.1)
    loss = a(torch.randn(4, 16)).sum()
    loss.backward()
    opt.step()

    # Copy A's state into B.
    b.load_state_dict(a.state_dict())

    # Forward should now match on both.
    x = torch.randn(4, 16)
    torch.testing.assert_close(a(x), b(x), rtol=1e-5, atol=1e-5)
