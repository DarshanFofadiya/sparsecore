"""
Milestone 2a Oracle tests — the scalar vector_dot kernel.

Every test compares our C++ dot product against torch.dot. Because dot
product is a reduction over elementwise products, floating-point associativity
matters — we use rtol=1e-5 AND atol=1e-5, as per our tolerance policy
(see docs/SYSTEM_PROMPT.md §3 Rule 2).

Scalar implementation: csrc/kernels/vector_dot.cpp
This test file will also run unchanged against the NEON version in 2b.

Run with:
    pytest tests/test_vector_dot.py -v
"""

import numpy as np
import pytest
import torch

from sparsecore import _core


# Our standard tolerance for float32 reductions.
RTOL = 1e-5
ATOL = 1e-5


# ─────────────────────────────────────────────────────────────────────
#  Group 1 — Registration
# ─────────────────────────────────────────────────────────────────────

def test_vector_dot_is_registered():
    """vector_dot is exposed on the _core module."""
    assert hasattr(_core, "vector_dot")


def test_vector_dot_is_callable():
    """The registered symbol is actually callable."""
    assert callable(_core.vector_dot)


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Oracle correctness across sizes.
#
#  Sizes are chosen to exercise SIMD boundary conditions even though
#  2a is scalar-only — when 2b lands, these same tests will catch
#  remainder-loop bugs immediately:
#    1, 2, 3      : sub-lane-width (NEON lane=4, so these are all "remainder")
#    4            : exactly one lane
#    5, 7         : one lane + remainder
#    8, 16        : perfect multiples of lane width (main loop only)
#    9, 15, 17    : main loop + 1-to-3 remainder
#    100, 10_000  : many lanes + remainder, general scale
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 10_000])
def test_oracle_various_sizes(size: int):
    """For each size, our C++ dot matches torch.dot within tolerance."""
    torch.manual_seed(42 + size)
    a_torch = torch.randn(size, dtype=torch.float32)
    b_torch = torch.randn(size, dtype=torch.float32)

    a_np = a_torch.numpy()
    b_np = b_torch.numpy()

    result_cpp = _core.vector_dot(a_np, b_np)
    result_oracle = torch.dot(a_torch, b_torch).item()

    # We compare as 0-D tensors so torch.allclose handles both tol modes
    # identically to how we'll use it everywhere else.
    assert torch.allclose(
        torch.tensor(result_cpp),
        torch.tensor(result_oracle),
        rtol=RTOL, atol=ATOL,
    ), (
        f"Dot product disagreement at size={size}: "
        f"C++={result_cpp:.6f}, Oracle={result_oracle:.6f}, "
        f"diff={abs(result_cpp - result_oracle):.2e}"
    )



# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Known-analytic cases.
#
#  These don't need an Oracle — we know the answer from math class.
#  Useful as canaries: if even these break, something is very wrong.
# ─────────────────────────────────────────────────────────────────────


def test_orthogonal_vectors_dot_to_zero():
    """[1, 0] · [0, 1] = 0 (canonical orthogonality)."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert _core.vector_dot(a, b) == 0.0


def test_self_dot_equals_sum_of_squares():
    """x · x = Σ x_i²  — a sanity check that ties dot to a familiar quantity."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected = float((x * x).sum())  # 1 + 4 + 9 + 16 = 30
    assert _core.vector_dot(x, x) == expected


def test_empty_vectors_dot_to_zero():
    """Empty inputs return 0.0 — the additive identity."""
    a = np.array([], dtype=np.float32)
    b = np.array([], dtype=np.float32)
    assert _core.vector_dot(a, b) == 0.0


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — Large-input accumulation.
#
#  Float32 accumulation error scales roughly with sqrt(n) for random
#  inputs. At n=10_000 the absolute error can be large, which is
#  exactly why we have rtol. If rtol is missing from our tolerance
#  comparison, this test is what would fail.
# ─────────────────────────────────────────────────────────────────────


def test_large_input_accumulation():
    """10K elements: rtol (not atol alone) is what keeps this test passing."""
    torch.manual_seed(0)
    a_torch = torch.randn(10_000, dtype=torch.float32)
    b_torch = torch.randn(10_000, dtype=torch.float32)

    result_cpp = _core.vector_dot(a_torch.numpy(), b_torch.numpy())
    result_oracle = torch.dot(a_torch, b_torch).item()

    assert torch.allclose(
        torch.tensor(result_cpp),
        torch.tensor(result_oracle),
        rtol=RTOL, atol=ATOL,
    ), (
        f"Large-input dot disagreement: C++={result_cpp}, Oracle={result_oracle}. "
        f"If this fails only here, reduce tolerance is likely too strict for "
        f"float32 accumulation at n=10_000."
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 5 — Error paths.
#
#  Bad input must raise ValueError (C++ std::invalid_argument auto-
#  translates to Python ValueError via pybind11). The messages should
#  be informative enough that a user knows what they did wrong.
# ─────────────────────────────────────────────────────────────────────


def test_mismatched_lengths_raises():
    """Different lengths → ValueError with helpful message."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="input lengths must match"):
        _core.vector_dot(a, b)


def test_2d_input_raises():
    """2-D input → ValueError, not silent flatten."""
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D"):
        _core.vector_dot(a, b)


def test_scalar_input_raises():
    """0-D input (a scalar) → ValueError."""
    a = np.float32(3.0)
    b = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D"):
        _core.vector_dot(a, b)


# ─────────────────────────────────────────────────────────────────────
#  Group 6 — Dtype coercion.
#
#  Our binding uses py::array::forcecast, which silently converts
#  wrong-dtype inputs to float32. This is researcher-friendly: no
#  need to remember .astype(float32) on every call. We verify the
#  coercion actually happens.
# ─────────────────────────────────────────────────────────────────────


def test_float64_input_is_coerced():
    """float64 arrays are accepted and silently converted to float32."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # note: double
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result = _core.vector_dot(a, b)
    # 1*4 + 2*5 + 3*6 = 32
    assert result == 32.0
