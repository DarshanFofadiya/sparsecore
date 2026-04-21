// ═══════════════════════════════════════════════════════════════════════════
//  kernels/spmm_grad.hpp
//
//  Backward-pass kernels for Y = W @ X (milestone 4a).
//
//  Two kernels live here:
//    spmm_grad_w — gradient of the loss with respect to W, computed ONLY
//                  at live slots. Produces a 1-D array of length nnz.
//                  This is the DST moat: we never materialize the dense
//                  M×K gradient that dense-simulated libraries allocate.
//
//  The gradient of the loss with respect to X (dL/dX = Wᵀ @ dY) is NOT
//  a separate kernel — it's just a forward SpMM on Wᵀ, so we reuse the
//  existing spmm_simd and don't ship a dedicated C++ function.
//
//  See docs/design/spmm_backward.md for the full derivation.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include "padded_csr.hpp"


namespace sparsecore {

// ─────────────────────────────────────────────────────────────────────────
//  spmm_grad_w — dL/dW at live slots only.
//
//  Given the upstream gradient dY and the original forward input X, compute:
//
//      dW_values[s] = dot(dY[i, :], X[k, :])
//
//  for each live slot s in W, where (i, k) is the (row, col) position of
//  that slot. Output is a contiguous 1-D array aligned with W.values —
//  dW_values[s] is the gradient for the value at W.values[s]. Padding
//  slots are left at 0.0 (from the kernel's internal memset), so
//  optimizers can do a single vectorized update: W.values -= lr * dW.
//
//  Parameters:
//    W          — the sparse weight. col_indices, row_start, row_nnz are
//                 read to enumerate live slots. W.values is NOT read
//                 (values don't appear in dL/dW's formula).
//    dY         — upstream gradient, dense (M, N) row-major. M*N floats.
//    N          — columns of dY and X.
//    X          — original forward input, dense (K, N) row-major. K*N floats.
//    K          — rows of X (must equal W.ncols).
//    dW_values  — output buffer, length W.total_capacity() (same as
//                 W.values). The kernel zeros it internally.
//
//  Algorithmic note:
//    For each live slot s = (i, k), we do a single N-length dot product.
//    Total FMAs: nnz * N. Matches the forward pass complexity exactly.
//    Memory access pattern:
//      - Sequential reads of dY[i, :]  (good cache behavior)
//      - Sequential reads of X[k, :]   (good cache behavior — different k
//                                       per slot, but within each slot the
//                                       N floats are contiguous)
//      - Single scalar write of dW_values[s]
//
//  Complexity: O(nnz * N) floating-point ops. Same as forward.
// ─────────────────────────────────────────────────────────────────────────
void spmm_grad_w(
    const PaddedCSR& W,
    const float* dY, int64_t N,
    const float* X,  int64_t K,
    float* dW_values
);

}  // namespace sparsecore
