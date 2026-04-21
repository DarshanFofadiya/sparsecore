"""SparseCore build script."""
import sys
import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


# ─────────────────────────────────────────────────────────────────────
# Platform-specific compiler flags.
#
# -O3              : maximum optimization; non-negotiable for SIMD code.
# -std=c++17       : C++17 standard (matches docs/SYSTEM_PROMPT.md).
# -Wall -Wextra    : enable warnings. Warnings are future bugs.
# -fvisibility=    : hide C++ symbols; pybind11 handles exported ones.
#                    Keeps the .so small and avoids symbol collisions.
# -mcpu=apple-m1   : target Apple Silicon. Works on M1 and forward
#                    (M2/M3/M4). Unlocks NEON + Apple-specific tuning.
#                    We use -mcpu (CPU-specific) instead of -march
#                    (arch-generic) because Apple Clang treats them
#                    differently on arm64-darwin.
# ─────────────────────────────────────────────────────────────────────

IS_APPLE_SILICON = (
    sys.platform == "darwin" and platform.machine() == "arm64"
)

if IS_APPLE_SILICON:
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-fvisibility=hidden",
        "-mcpu=apple-m1",
    ]
else:
    # Conservative defaults for non-Apple-Silicon builds.
    # Full x86/AVX support is a post-v0.1 contribution target.
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-fvisibility=hidden",
    ]


# ─────────────────────────────────────────────────────────────────────
# The C++ extension module.
#
# Name: "sparsecore._core"
#   The dotted name means: produce a .so file importable as
#   `sparsecore._core`. It physically lives at sparsecore/_core.so after
#   install. The leading underscore marks it as private — users import
#   from `sparsecore`, not from `sparsecore._core`.
#
# Sources: list the .cpp files to compile. Milestone 1c declares the
#   build machinery without any sources yet; Milestone 1d will add
#   csrc/bindings.cpp as the first source.
# ─────────────────────────────────────────────────────────────────────

ext_modules = [
    Pybind11Extension(
        name="sparsecore._core",
        # All C++ sources that need to be compiled and linked together.
        # Kernels go in csrc/kernels/*; bindings.cpp is the pybind11 entry point.
        sources=[
            "csrc/bindings.cpp",
            "csrc/kernels/double_tensor.cpp",
            "csrc/kernels/vector_dot.cpp",
            "csrc/kernels/vector_dot_neon.cpp",
            "csrc/kernels/padded_csr.cpp",
            "csrc/kernels/spmm.cpp",
            "csrc/kernels/spmm_neon.cpp",
            "csrc/kernels/spmm_grad.cpp",
        ],
        # Include paths used for `#include "kernels/foo.hpp"` etc.
        include_dirs=["csrc"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
