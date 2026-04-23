"""
Milestone 2a/2b Oracle tests — vector_dot kernels (scalar + NEON SIMD).

Every correctness and error-path test runs against BOTH kernels via
pytest parametrization. If either kernel misbehaves on any size, the
failure message tells us exactly which kernel and which size.

The two kernels:
  - vector_dot        (scalar, csrc/kernels/vector_dot.cpp)
  - vector_dot_simd   (NEON,   csrc/kernels/vector_dot_neon.cpp)

Because dot product is a reduction, we use rtol=atol=1e-5 per our
project-wide tolerance policy (never modify the Oracle to make a failing
test pass; if the kernel disagrees with PyTorch's reference, the kernel
is wrong).

Run with:
    pytest tests/test_vector_dot.py -v
"""

import numpy as np
import pytest
import torch

from sparselab import _core


# Our standard tolerance for float32 reductions.
RTOL = 1e-5
ATOL = 1e-5


# Both kernels we test. Each parametrized test will run twice — once
# per kernel. The second element of each tuple is a short label that
# pytest prints in the test case name.
KERNELS = [
    pytest.param(_core.vector_dot,      id="scalar"),
    pytest.param(_core.vector_dot_simd, id="neon"),
]


# ─────────────────────────────────────────────────────────────────────
#  Group 1 — Registration (per-kernel, not parametrized).
# ─────────────────────────────────────────────────────────────────────

def test_vector_dot_is_registered():
    """The scalar vector_dot is exposed on _core."""
    assert hasattr(_core, "vector_dot") and callable(_core.vector_dot)


def test_vector_dot_simd_is_registered():
    """The NEON vector_dot_simd is exposed on _core."""
    assert hasattr(_core, "vector_dot_simd") and callable(_core.vector_dot_simd)


# ─────────────────────────────────────────────────────────────────────
#  Group 2 — Oracle correctness across sizes, both kernels.
#
#  Sizes are chosen to exercise NEON's 4-wide lane boundary conditions:
#    1, 2, 3      : sub-lane-width (all in the remainder loop)
#    4            : exactly one SIMD iteration, zero remainder
#    5, 7         : one SIMD iter + 1 or 3 remainder
#    8, 16        : perfect multiples of lane width (main loop only)
#    9, 15, 17    : main loop + 1 to 3 remainder elements
#    100, 10_000  : many SIMD iters + remainder, general scale
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kernel_fn", KERNELS)
@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 10_000])
def test_oracle_various_sizes(kernel_fn, size: int):
    """For each (kernel, size) pair, output matches torch.dot within tolerance."""
    torch.manual_seed(42 + size)
    a_torch = torch.randn(size, dtype=torch.float32)
    b_torch = torch.randn(size, dtype=torch.float32)

    result = kernel_fn(a_torch.numpy(), b_torch.numpy())
    oracle = torch.dot(a_torch, b_torch).item()

    assert torch.allclose(
        torch.tensor(result),
        torch.tensor(oracle),
        rtol=RTOL, atol=ATOL,
    ), (
        f"Dot-product disagreement at size={size}: "
        f"kernel={result:.6f}, oracle={oracle:.6f}, "
        f"abs diff={abs(result - oracle):.2e}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 3 — Known-analytic cases.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_orthogonal_vectors_dot_to_zero(kernel_fn):
    """[1, 0] · [0, 1] = 0 — canonical orthogonality."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert kernel_fn(a, b) == 0.0


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_self_dot_equals_sum_of_squares(kernel_fn):
    """x · x = Σ x_i² — ties dot to a familiar quantity."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected = float((x * x).sum())  # 1 + 4 + 9 + 16 = 30
    assert kernel_fn(x, x) == expected


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_empty_vectors_dot_to_zero(kernel_fn):
    """Empty inputs return 0.0 — the additive identity."""
    a = np.array([], dtype=np.float32)
    b = np.array([], dtype=np.float32)
    assert kernel_fn(a, b) == 0.0


# ─────────────────────────────────────────────────────────────────────
#  Group 4 — Large-input accumulation.
#
#  Float32 reduction error scales with sqrt(n) for random inputs.
#  At n=10_000 the absolute error can be meaningful, which is why
#  we rely on rtol. If rtol were missing from our tolerance, this
#  test would be the one to fail.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_large_input_accumulation(kernel_fn):
    """10K elements: rtol (not atol alone) keeps this passing."""
    torch.manual_seed(0)
    a_torch = torch.randn(10_000, dtype=torch.float32)
    b_torch = torch.randn(10_000, dtype=torch.float32)

    result = kernel_fn(a_torch.numpy(), b_torch.numpy())
    oracle = torch.dot(a_torch, b_torch).item()

    assert torch.allclose(
        torch.tensor(result),
        torch.tensor(oracle),
        rtol=RTOL, atol=ATOL,
    )


# ─────────────────────────────────────────────────────────────────────
#  Group 5 — Error paths.
#
#  Both kernels share the validation helper in bindings.cpp, so both
#  must reject bad input identically.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_mismatched_lengths_raises(kernel_fn):
    """Different lengths → ValueError."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="input lengths must match"):
        kernel_fn(a, b)


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_2d_input_raises(kernel_fn):
    """2-D input → ValueError; no silent flatten."""
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D"):
        kernel_fn(a, b)


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_scalar_input_raises(kernel_fn):
    """0-D input → ValueError."""
    a = np.float32(3.0)
    b = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D"):
        kernel_fn(a, b)


# ─────────────────────────────────────────────────────────────────────
#  Group 6 — Dtype coercion.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kernel_fn", KERNELS)
def test_float64_input_is_coerced(kernel_fn):
    """float64 arrays are accepted and silently converted to float32."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    # 1*4 + 2*5 + 3*6 = 32
    assert kernel_fn(a, b) == 32.0


# ─────────────────────────────────────────────────────────────────────
#  Group 7 — Scalar vs NEON cross-check.
#
#  The scalar and NEON implementations should agree closely on the
#  same input. This is a DIFFERENT check from the Oracle tests above:
#  here we compare our two kernels directly, not against torch.dot.
#  Catches bugs where one kernel has drifted away from the other.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [1, 4, 15, 17, 100, 10_000])
def test_scalar_and_neon_agree(size: int):
    """vector_dot and vector_dot_simd agree to within rtol=atol=1e-5."""
    torch.manual_seed(size)
    a = torch.randn(size, dtype=torch.float32).numpy()
    b = torch.randn(size, dtype=torch.float32).numpy()

    scalar = _core.vector_dot(a, b)
    simd = _core.vector_dot_simd(a, b)

    assert torch.allclose(
        torch.tensor(scalar),
        torch.tensor(simd),
        rtol=RTOL, atol=ATOL,
    ), f"scalar-vs-NEON mismatch at size={size}: {scalar} vs {simd}"
