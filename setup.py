"""SparseCore build script."""
import os
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
IS_MACOS = sys.platform == "darwin"

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

extra_link_args: list[str] = []


# ─────────────────────────────────────────────────────────────────────
# OpenMP setup — optional but recommended.
#
# On macOS, Apple Clang does NOT ship OpenMP support by default. Users
# install libomp via Homebrew (`brew install libomp`). When it's
# present at the standard Homebrew paths we wire it in; if it's absent
# we build without OpenMP and the kernels fall back to their
# sequential path via the #ifdef _OPENMP guard in C++.
#
# On Linux, gcc/clang typically support `-fopenmp` directly. We try
# that unconditionally; if the user's compiler doesn't know the flag
# the build fails loudly (they can override via SPARSECORE_NO_OPENMP=1).
#
# Environment overrides:
#   SPARSECORE_NO_OPENMP=1      → force-disable (useful for CI or
#                                 debugging a non-OpenMP build)
#   SPARSECORE_LIBOMP_PREFIX=/…  → point at a custom libomp install
# ─────────────────────────────────────────────────────────────────────

def configure_openmp() -> tuple[list[str], list[str], list[str]]:
    """Return (compile_args, link_args, include_dirs) additions for OpenMP.

    Returns three empty lists if OpenMP is disabled or unavailable.

    Macos note: PyTorch ships its OWN libomp.dylib inside its wheel. If
    we link a different libomp (e.g. Homebrew's) and both get loaded
    into the same Python process, the two OpenMP runtimes abort each
    other on startup. Our strategy:

      1. If PyTorch is importable, prefer its bundled libomp headers
         (from Homebrew) for compile, and link a weak SONAME so the
         loader resolves to whichever libomp is already in the process
         — which, when torch imports first, will be torch's.
      2. If PyTorch isn't importable at build time, fall back to
         Homebrew's libomp directly.
    """
    if os.environ.get("SPARSECORE_NO_OPENMP") == "1":
        return [], [], []

    if IS_MACOS:
        # Headers only come from Homebrew (PyTorch's wheel doesn't ship
        # the omp.h development header, only the runtime .dylib).
        include_candidates = [
            os.environ.get("SPARSECORE_LIBOMP_PREFIX"),
            "/opt/homebrew/opt/libomp",
            "/usr/local/opt/libomp",
        ]
        include_path = None
        for prefix in include_candidates:
            if prefix and os.path.isfile(os.path.join(prefix, "include", "omp.h")):
                include_path = os.path.join(prefix, "include")
                break

        if include_path is None:
            msg = (
                "\n"
                "══════════════════════════════════════════════════════════════════\n"
                "  sparsecore: libomp NOT FOUND — building WITHOUT OpenMP.\n"
                "  The kernels will run SEQUENTIALLY (roughly 4-6x slower\n"
                "  on an Apple Silicon Mac with >=4 cores).\n"
                "\n"
                "  To get parallel kernels, install libomp:\n"
                "    macOS:  brew install libomp\n"
                "    Linux:  (already bundled with gcc/clang)\n"
                "\n"
                "  Then rebuild:\n"
                "    pip install -e . --no-build-isolation --no-deps --force-reinstall\n"
                "\n"
                "  To silence this warning intentionally, set:\n"
                "    SPARSECORE_NO_OPENMP=1\n"
                "══════════════════════════════════════════════════════════════════\n"
            )
            print(msg, file=sys.stderr)
            return [], [], []

        # Link strategy: prefer PyTorch's bundled libomp if we can find
        # it, otherwise Homebrew's. Using `-rpath` tells the macOS
        # dynamic loader where to search at runtime.
        #
        # We always add Homebrew's libomp prefix as a FALLBACK rpath
        # too. That way, if a user somehow imports sparsecore without
        # torch first (unusual — sparsecore always imports torch in
        # its __init__), the dynamic loader still finds a libomp.
        #
        # include_path here is "<homebrew-prefix>/libomp/include"
        # (e.g. /opt/homebrew/opt/libomp/include). Strip ONE level to
        # get the libomp prefix (/opt/homebrew/opt/libomp), then
        # append "lib" for the actual library directory. Previous
        # implementation stripped two levels by mistake and landed on
        # /opt/homebrew/opt/lib, which doesn't exist — broke every
        # non-editable wheel build.
        hb_prefix = os.path.dirname(include_path)
        hb_lib = os.path.join(hb_prefix, "lib")

        link_args: list[str] = []
        try:
            import torch  # type: ignore
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isfile(os.path.join(torch_lib, "libomp.dylib")):
                # Search torch/lib FIRST so we resolve to the same libomp
                # torch itself loaded. Homebrew's libomp is the fallback.
                link_args = [
                    "-L" + torch_lib,
                    "-Wl,-rpath," + torch_lib,
                    "-Wl,-rpath," + hb_lib,
                    "-lomp",
                ]
        except ImportError:
            pass

        if not link_args:
            # No torch at build time: just use Homebrew's libomp.
            link_args = [
                "-L" + hb_lib,
                "-Wl,-rpath," + hb_lib,
                "-lomp",
            ]

        return (
            ["-Xpreprocessor", "-fopenmp"],
            link_args,
            [include_path],
        )

    # Linux (and other POSIX): assume the compiler handles -fopenmp.
    return ["-fopenmp"], ["-fopenmp"], []


omp_compile, omp_link, omp_include = configure_openmp()
extra_compile_args.extend(omp_compile)
extra_link_args.extend(omp_link)


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
            "csrc/kernels/dense_grad.cpp",
        ],
        # Include paths used for `#include "kernels/foo.hpp"` etc.
        # OpenMP includes are appended by configure_openmp() above.
        include_dirs=["csrc", *omp_include],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
