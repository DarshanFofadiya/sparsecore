"""SparseLab build script."""
import os
import sys
import platform
import subprocess
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


# ─────────────────────────────────────────────────────────────────────
# Platform-specific compiler flags.
#
# -O3              : maximum optimization; non-negotiable for SIMD code.
# -std=c++17       : C++17 standard (our project-wide baseline).
# -Wall -Wextra    : enable warnings. Warnings are future bugs.
# -fvisibility=    : hide C++ symbols; pybind11 handles exported ones.
#                    Keeps the .so small and avoids symbol collisions.
# -mcpu=apple-m1   : target Apple Silicon. Works on M1 and forward
#                    (M2/M3/M4). Unlocks NEON + Apple-specific tuning.
#                    We use -mcpu (CPU-specific) instead of -march
#                    (arch-generic) because Apple Clang treats them
#                    differently on arm64-darwin.
# -march=x86-64-v3 : Linux x86_64 baseline. Targets AVX + AVX2 + FMA +
#                    BMI1/2 + LZCNT + F16C — every x86 CPU from 2013+
#                    (Haswell / Zen). Required for the AVX2 dW kernel
#                    to compile (_mm256_fmadd_ps is gated on FMA
#                    being available at build time), and lets Clang
#                    emit AVX2 FMAs in auto-vectorizable sibling
#                    kernels too. Pre-2013 x86 CPUs are not supported.
#                    See docs/design/spmm_backward_avx2.md for the
#                    measurement data motivating this change.
# ─────────────────────────────────────────────────────────────────────

IS_APPLE_SILICON = (
    sys.platform == "darwin" and platform.machine() == "arm64"
)
IS_MACOS = sys.platform == "darwin"

# ARM64 covers Apple Silicon, Linux aarch64 (Graviton, RPi 5, Ampere),
# and any other arm64 target. NEON is available on all ARM64 hardware,
# which is our gating condition for compiling the NEON kernels.
# On x86 / x86_64 we must NOT try to compile the NEON sources at all —
# their arm_neon.h include fails and arm_neon intrinsics don't exist.
IS_ARM64 = platform.machine() in ("arm64", "aarch64")

# x86_64 covers Linux x86_64 (the platform we actively support with
# AVX2 kernels) and technically Intel macOS + Windows x86_64, which
# are out of scope for v0.2 wheels (see CHANGELOG v0.1.1 for the
# Intel Mac carve-out, issue #8 for Windows). The -march flag below
# is emitted on any x86_64 build that reaches this file, but our CI
# / wheel matrix only exercises the Linux x86_64 path.
IS_X86_64 = platform.machine() in ("x86_64", "AMD64")

if IS_APPLE_SILICON:
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-fvisibility=hidden",
        "-mcpu=apple-m1",
    ]
elif IS_X86_64:
    # x86_64 (Linux, or a source build on Intel Mac / Windows). The
    # -march=x86-64-v3 target is supported by Clang 12+ and GCC 11+;
    # manylinux_2_28 (our wheel build image) ships compilers that
    # support it natively. Minimum CPU requirement for the resulting
    # wheel: Haswell (Intel 2013+) or Zen 1 (AMD 2017+).
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-fvisibility=hidden",
        "-march=x86-64-v3",
    ]
else:
    # Fallback for any platform we haven't explicitly accounted for
    # (e.g. Linux aarch64 using the non-Apple-Silicon branch). This
    # build will produce a working .so but without architecture-
    # specific SIMD tuning.
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
# the build fails loudly (they can override via SPARSELAB_NO_OPENMP=1).
#
# Environment overrides:
#   SPARSELAB_NO_OPENMP=1      → force-disable (useful for CI or
#                                 debugging a non-OpenMP build)
#   SPARSELAB_LIBOMP_PREFIX=/…  → point at a custom libomp install
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
    if os.environ.get("SPARSELAB_NO_OPENMP") == "1":
        return [], [], []

    if IS_MACOS:
        # Headers only come from Homebrew (PyTorch's wheel doesn't ship
        # the omp.h development header, only the runtime .dylib).
        include_candidates = [
            os.environ.get("SPARSELAB_LIBOMP_PREFIX"),
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
                "  sparselab: libomp NOT FOUND — building WITHOUT OpenMP.\n"
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
                "    SPARSELAB_NO_OPENMP=1\n"
                "══════════════════════════════════════════════════════════════════\n"
            )
            print(msg, file=sys.stderr)
            return [], [], []

        # Link strategy: prefer PyTorch's bundled libomp if we can find
        # it, otherwise Homebrew's. Using `-rpath` tells the macOS
        # dynamic loader where to search at runtime.
        #
        # We always add Homebrew's libomp prefix as a FALLBACK rpath
        # too. That way, if a user somehow imports sparselab without
        # torch first (unusual — sparselab always imports torch in
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
# macOS libomp post-build repair.
#
# Our C++ extension links against Homebrew's libomp at an absolute
# path: /opt/homebrew/opt/libomp/lib/libomp.dylib (arm64) or
# /usr/local/opt/libomp/lib/libomp.dylib (x86_64 Intel Mac).
#
# At import time, torch has already loaded its OWN bundled libomp
# (from torch/lib/libomp.dylib inside the torch wheel). If our .so
# then loads a different libomp, OpenMP's runtime detects two
# copies in the process and calls abort() with the infamous
# "OMP: Error #15" message.
#
# The wheel build path solves this via scripts/repair_wheel_macos.sh
# (post-build, invoked by cibuildwheel). Editable installs
# (pip install -e .) skip that repair script and ship a .so with
# the absolute libomp install name — which reliably aborts on
# import as soon as torch is in the same process.
#
# BuildExtWithRepair runs the same two install_name_tool commands
# the wheel repair script uses, but inline after each build_extension()
# call. That way editable installs produce a correctly-linked .so on
# the first try. Non-macOS platforms (where libgomp ships with gcc
# and doesn't double-init) no-op this step.
# ─────────────────────────────────────────────────────────────────────

class BuildExtWithRepair(build_ext):
    """pybind11 build_ext + post-build libomp repair on macOS."""

    def build_extension(self, ext):
        super().build_extension(ext)
        if IS_MACOS:
            self._repair_libomp_install_name(ext)

    def _repair_libomp_install_name(self, ext):
        """
        Rewrite the built .so so its libomp reference uses @rpath and
        a relative search path that points at torch's bundled libomp
        at import time. See module-level comment above for the
        motivation. No-op if the .so doesn't reference a Homebrew
        libomp (e.g., SPARSELAB_NO_OPENMP builds).
        """
        so_path = self.get_ext_fullpath(ext.name)
        if not os.path.isfile(so_path):
            return

        # Find the current libomp install name (if any). Absolute
        # Homebrew paths will be rewritten; already-relative @rpath
        # references are left alone.
        otool_out = subprocess.run(
            ["otool", "-L", so_path],
            check=False, capture_output=True, text=True,
        ).stdout
        homebrew_libomp = None
        for line in otool_out.splitlines():
            line = line.strip()
            # Match /opt/homebrew/opt/libomp/lib/libomp.dylib
            # or    /usr/local/opt/libomp/lib/libomp.dylib
            if line.startswith(("/opt/homebrew/opt/libomp/",
                                "/usr/local/opt/libomp/")):
                homebrew_libomp = line.split(" ")[0]
                break

        if homebrew_libomp is None:
            return  # already @rpath-style, or no libomp linked

        print(
            f"[sparselab] rewriting libomp install name: "
            f"{homebrew_libomp} -> @rpath/libomp.dylib",
            file=sys.stderr,
        )
        subprocess.run(
            ["install_name_tool", "-change",
             homebrew_libomp, "@rpath/libomp.dylib", so_path],
            check=True,
        )

        # Add an rpath pointing at torch's bundled libomp, the one
        # torch will have already loaded by the time we import.
        #
        # For WHEEL installs, sparselab/_core.so and torch/ live
        # next to each other inside site-packages, so the relative
        # path @loader_path/../torch/lib resolves correctly. The
        # wheel repair script (scripts/repair_wheel_macos.sh) uses
        # that relative form and it's how CI-built wheels work in
        # production.
        #
        # For EDITABLE installs (pip install -e .), the .so lives in
        # the source tree (e.g. /path/to/repo/sparselab/_core.so)
        # while torch is in site-packages — they are NOT siblings,
        # so @loader_path/../torch/lib resolves to a nonexistent
        # directory. To keep editable installs working we also add
        # an ABSOLUTE rpath pointing at torch/lib in the current
        # Python environment. This absolute path is a
        # development-only artifact: it's baked into your local .so
        # and isn't a problem because the .so itself lives in your
        # repo, not in a distributed wheel. Wheel builds still use
        # scripts/repair_wheel_macos.sh, which actively strips
        # absolute rpaths before publishing.
        #
        # We check each rpath before adding to avoid
        # "file already has LC_RPATH for" errors on incremental builds.
        otool_rpaths = subprocess.run(
            ["otool", "-l", so_path],
            check=False, capture_output=True, text=True,
        ).stdout

        rpaths_to_add = ["@loader_path/../torch/lib"]
        try:
            import torch  # type: ignore
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isfile(os.path.join(torch_lib, "libomp.dylib")):
                rpaths_to_add.append(torch_lib)
        except ImportError:
            pass

        for want_rpath in rpaths_to_add:
            if want_rpath not in otool_rpaths:
                print(
                    f"[sparselab] adding rpath: {want_rpath}",
                    file=sys.stderr,
                )
                subprocess.run(
                    ["install_name_tool", "-add_rpath",
                     want_rpath, so_path],
                    check=True,
                )


# ─────────────────────────────────────────────────────────────────────
# The C++ extension module.
#
# Name: "sparselab._core"
#   The dotted name means: produce a .so file importable as
#   `sparselab._core`. It physically lives at sparselab/_core.so after
#   install. The leading underscore marks it as private — users import
#   from `sparselab`, not from `sparselab._core`.
#
# Sources: list the .cpp files to compile. Milestone 1c declares the
#   build machinery without any sources yet; Milestone 1d will add
#   csrc/bindings.cpp as the first source.
# ─────────────────────────────────────────────────────────────────────

ext_modules = [
    Pybind11Extension(
        name="sparselab._core",
        # All C++ sources that need to be compiled and linked together.
        # Kernels go in csrc/kernels/*; bindings.cpp is the pybind11 entry point.
        #
        # NEON sources (spmm_neon.cpp, vector_dot_neon.cpp,
        # spmm_grad_neon.cpp) are gated on IS_ARM64. On x86 they can't
        # be compiled — their #include <arm_neon.h> fails and they
        # self-guard with an #error.
        #
        # AVX2 sources (spmm_grad_avx2.cpp) are gated on IS_X86_64. On
        # non-x86 they can't be compiled — their #include <immintrin.h>
        # is missing AVX2 intrinsic declarations without -march=x86-64-v3,
        # and they self-guard with an #error.
        #
        # The two branches are MUTUALLY EXCLUSIVE: at no point does a
        # single build compile both NEON and AVX2 sources. Both SIMD
        # kernel files define the same C++ symbol name
        # (sparselab::spmm_grad_w_simd) so the bindings layer in
        # bindings.cpp calls the right function automatically — which
        # file defines that symbol depends on which branch the setup
        # took.
        sources=[
            "csrc/bindings.cpp",
            "csrc/kernels/double_tensor.cpp",
            "csrc/kernels/vector_dot.cpp",
            "csrc/kernels/padded_csr.cpp",
            "csrc/kernels/spmm.cpp",
            "csrc/kernels/spmm_grad.cpp",
            "csrc/kernels/dense_grad.cpp",
        ] + (
            [
                "csrc/kernels/vector_dot_neon.cpp",
                "csrc/kernels/spmm_neon.cpp",
                "csrc/kernels/spmm_grad_neon.cpp",
            ] if IS_ARM64 else []
        ) + (
            [
                "csrc/kernels/spmm_grad_avx2.cpp",
            ] if IS_X86_64 else []
        ),
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
    cmdclass={"build_ext": BuildExtWithRepair},
)
