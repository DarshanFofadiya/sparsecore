// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_avx2.hpp
//
//  AVX2 + FMA SIMD declaration for the forward SpMM path
//  (Y = W @ X, W in PaddedCSR) on Linux x86_64. Sibling of
//  spmm_neon.hpp on the ARM side.
//
//  Companion to the dW AVX2 kernel shipped in milestone 14
//  (spmm_grad_avx2.hpp / .cpp). Together these two files close
//  the x86 parity story: backward (milestone 14) + forward
//  (this spec). See docs/design/spmm_forward_avx2.md for the
//  full rationale, projected numbers, and risk register.
//
//  Same contract as spmm_scalar (see spmm.hpp for full
//  signature documentation). The only difference is this
//  version vectorizes the inner N-length loop with 256-bit
//  AVX2 + FMA intrinsics, processing 16 float32 elements per
//  Phase-A iteration using two independent 8-wide write-through
//  streams (load → fmadd → store into Y).
//
//  Correctness: this version is verified against spmm_scalar
//  within rtol=atol=1e-5 across all shapes in the test suite
//  (tests/test_spmm.py parametrized + tests/test_spmm_avx2.py).
//  Per-slot reordering is minimal — unlike the dW kernel, there
//  is no accumulator reduction; each lane writes straight back
//  to Y. Scalar and AVX2 outputs are often bit-identical on
//  FFN-sized problems and within 1 ULP otherwise.
//
//  Performance: memory-bandwidth-bound on Zen 4 (~100 GB/s L1
//  per core). Inner loop is 96 B/iter (32 B load X + 32 B load
//  Y + 32 B store Y) against 8 FMAs/iter — target ~15-20 GF/s
//  sustained in the kernel (4-5x over scalar's ~3.8 GF/s on
//  the GitHub ubuntu-24.04 runner). See design §6.2 for the
//  full bandwidth-ceiling argument.
//
//  Symbol naming
//  ─────────────
//  This kernel declares `sparselab::spmm_simd_avx2` — distinct
//  from the NEON symbol `sparselab::spmm_simd_neon`. Follows the
//  existing forward-kernel naming convention (arch-specific
//  symbols) rather than the shared-symbol convention used by
//  the dW kernel. Rationale in docs/design/spmm_forward_avx2.md
//  §4.3. The Python-facing binding `_core.spmm_simd` stays
//  unchanged across platforms; the `#if/#elif/#else` dispatch
//  in bindings.cpp picks the right symbol per build.
//
//  Status (Phase A stub)
//  ─────────────────────
//  This file ships as part of Phase A plumbing: the .cpp
//  currently implements a scalar-identical stub body so we can
//  land the build + binding + dispatch plumbing in a commit
//  that is arithmetic-risk-free. Phase B replaces the inner
//  j-loop body with the real dual-stream AVX2 code (design
//  §3.4) in a follow-up commit.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include "padded_csr.hpp"


namespace sparselab {

// See spmm.hpp for full parameter documentation. This is a
// drop-in replacement for spmm_scalar — same signature, same
// memory contracts, same self-zeroing behavior. The declared
// symbol name is `spmm_simd_avx2` (arch-specific per forward
// kernel convention); bindings.cpp dispatches to it via the
// x86-gated `#elif defined(__AVX2__) && defined(__FMA__)` branch
// in py_spmm_simd.
void spmm_simd_avx2(
    const PaddedCSR& W,
    const float* X, int64_t K, int64_t N,
    float* Y
);

}  // namespace sparselab
