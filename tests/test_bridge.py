"""
Milestone 1 Oracle tests — the PyTorch/C++ bridge.

These tests verify the C++/Python plumbing works end-to-end. They don't
test sparsity, SIMD, or any SparseCore-specific logic — just that bytes
can cross the FFI boundary and math-through-C++ matches math-through-PyTorch
to floating-point tolerance.

The Oracle for every test is a PyTorch dense operation (`x * 2.0`). If our
C++ output disagrees with the Oracle at 1e-5 tolerance, the C++ is wrong
(see docs/SYSTEM_PROMPT.md §3 Rule 2 — never modify the Oracle).

Run with:
    pytest tests/test_bridge.py -v
"""

import numpy as np
import pytest
import torch


# Oracle tolerance for all floating-point comparisons.
# 1e-5 is our project-wide standard (see SYSTEM_PROMPT.md).
ORACLE_ATOL = 1e-5


# ─────────────────────────────────────────────────────────────────────
#  Group 1 — Import sanity
#
#  These tests catch "the build succeeded but the module is broken"
#  failure modes. Surprisingly common with C++ extensions.
# ─────────────────────────────────────────────────────────────────────


def test_sparsecore_package_imports():
    """The `sparsecore` Python package is on the path and imports cleanly."""
    import sparsecore  # noqa: F401 — we're testing that import itself works


def test_core_module_imports():
    """The compiled C++ module `sparsecore._core` loads."""
    from sparsecore import _core  # noqa: F401


def test_double_tensor_is_registered():
    """`double_tensor` is exposed on the `_core` module."""
    from sparsecore import _core
    assert hasattr(_core, "double_tensor"), (
        "C++ function `double_tensor` is missing from sparsecore._core. "
        "Check that PYBIND11_MODULE(_core, m) in csrc/bindings.cpp calls "
        "m.def(\"double_tensor\", ...)."
    )


def test_double_tensor_is_callable():
    """The registered symbol is actually a callable, not e.g. a type."""
    from sparsecore import _core
    assert callable(_core.double_tensor)



# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Oracle correctness
#
#  Every test in this group has the same structure:
#    1. Build an input tensor.
#    2. Compute C++ output via our double_tensor.
#    3. Compute PyTorch Oracle output as `x * 2.0`.
#    4. Assert they match within ORACLE_ATOL.
#
#  The @pytest.mark.parametrize decorator runs the same function across
#  many sizes — pytest reports each as a separate test case.
# ─────────────────────────────────────────────────────────────────────


def _double_via_cpp(x_torch: torch.Tensor) -> torch.Tensor:
    """
    Helper: route a PyTorch tensor through our C++ double_tensor and
    return the result as a PyTorch tensor.

    The dance is:  torch.Tensor -> numpy -> C++ -> numpy -> torch.Tensor
    This crosses the FFI boundary twice (in and out), which is exactly
    the round-trip we want to verify.

    Forces float32 and contiguous memory because that's what our C++
    binding accepts; non-contiguous tensors are handled by np.ascontiguousarray.
    """
    x_np = np.ascontiguousarray(x_torch.detach().numpy(), dtype=np.float32)
    from sparsecore import _core
    y_np = _core.double_tensor(x_np)
    return torch.from_numpy(y_np)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 15, 16, 17, 100, 1000, 10_000])
def test_oracle_various_sizes(size: int):
    """
    For a range of sizes, C++ output must match torch.Tensor * 2.0 exactly.

    Sizes include:
      - 1 (minimum): the one-element edge case
      - 2, 3, 5 (small): general small sizes
      - 8, 16 (SIMD-lane-aligned): prepares us for NEON in Milestone 2
      - 15, 17 (off-by-one around lane boundary): catches remainder-loop bugs
      - 100, 1000, 10000 (scale): catches bugs that only appear with many elements
    """
    # Use deterministic inputs; generating with a fixed seed keeps
    # test failures reproducible across runs.
    torch.manual_seed(42 + size)
    x = torch.randn(size, dtype=torch.float32)

    y_cpp = _double_via_cpp(x)
    y_oracle = x * 2.0

    assert torch.allclose(y_cpp, y_oracle, atol=ORACLE_ATOL), (
        f"C++ output disagrees with Oracle at size={size}. "
        f"Max abs diff: {(y_cpp - y_oracle).abs().max().item():.2e}"
    )


def test_oracle_zeros():
    """Doubling zero is still zero. Canary for math going haywire."""
    x = torch.zeros(10, dtype=torch.float32)
    y_cpp = _double_via_cpp(x)
    assert torch.all(y_cpp == 0.0)


def test_oracle_negative_values():
    """Negative inputs flow through correctly — no sign bugs."""
    x = torch.tensor([-1.0, -2.5, -0.0, 3.14], dtype=torch.float32)
    y_cpp = _double_via_cpp(x)
    y_oracle = x * 2.0
    assert torch.allclose(y_cpp, y_oracle, atol=ORACLE_ATOL)


def test_oracle_large_values():
    """Large floats double without overflow to inf (2 * ~1e38 is within float32)."""
    x = torch.tensor([1e37, -1e37, 1.5e38], dtype=torch.float32)
    y_cpp = _double_via_cpp(x)
    y_oracle = x * 2.0
    # Note: 2 * 1.5e38 ≈ 3e38, past float32 max (~3.4e38) — but 2 * 1e37 is fine.
    # The Oracle and our C++ should both produce the same answer whatever it is,
    # including inf if overflow occurs.
    assert torch.equal(torch.isinf(y_cpp), torch.isinf(y_oracle))
    finite_mask = torch.isfinite(y_cpp)
    assert torch.allclose(y_cpp[finite_mask], y_oracle[finite_mask], atol=ORACLE_ATOL)



# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Immutability guarantees
#
#  PyTorch's `x * 2.0` returns a NEW tensor; `x` is unchanged. Our C++
#  function makes the same promise. These tests verify the promise.
#  If broken, users would see mysterious bugs where their original
#  tensors mutate under them.
# ─────────────────────────────────────────────────────────────────────


def test_input_not_mutated():
    """Calling double_tensor does not modify the input array."""
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    x_before = x.clone()
    _ = _double_via_cpp(x)
    assert torch.equal(x, x_before), "double_tensor mutated its input!"


def test_output_is_independent_memory():
    """Output array has its own memory — mutating the output doesn't affect input."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    from sparsecore import _core
    y = _core.double_tensor(x)
    # Modify the output array. If output shares memory with input, this breaks x.
    y[0] = 999.0
    assert x[0] == 1.0, "Output and input share memory — they should not."


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — Smoke test for determinism
#
#  Multiply-by-2 is a deterministic operation; running it twice on the
#  same input must produce bit-identical results. This catches
#  uninitialized-memory bugs in the output allocation.
# ─────────────────────────────────────────────────────────────────────


def test_determinism():
    """Two calls on the same input produce identical results (no uninit memory)."""
    x = np.arange(100, dtype=np.float32)
    from sparsecore import _core
    y1 = _core.double_tensor(x)
    y2 = _core.double_tensor(x)
    assert np.array_equal(y1, y2), (
        "double_tensor returned different results on identical inputs — "
        "likely uninitialized memory in the output allocation."
    )
