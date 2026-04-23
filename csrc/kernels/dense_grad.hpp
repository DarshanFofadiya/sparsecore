// ═══════════════════════════════════════════════════════════════════════════
//  kernels/dense_grad.hpp
//
//  Full-dense gradient kernel: G = dY @ X^T.
//
//  This is the kernel that RigL (and any gradient-regrow DST algorithm)
//  uses to find the positions where the task *wants* new connections.
//  Unlike spmm_grad_w, which only computes gradients at currently-live
//  slots, dense_grad computes the full (M, K) matrix so RigL can see
//  what the gradient would be at currently-dead positions too.
//
//  See docs/design/rigl.md for context and why we materialize the full
//  matrix rather than compute top-K directly.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>


namespace sparselab {

// ─────────────────────────────────────────────────────────────────────────
//  dense_grad: compute the FULL dense gradient of the loss w.r.t. W.
//
//  Inputs:
//    M, K, N: shape parameters.
//    dY: (M, N) row-major, upstream gradient from the layer's output.
//    X:  (K, N) row-major, original forward-pass input.
//
//  Output:
//    G:  (M, K) row-major, where G[i, k] = sum_j dY[i, j] * X[k, j].
//        Equivalently G = dY @ X^T.
//
//  The kernel makes no reference to a PaddedCSR — it doesn't care which
//  positions of W are currently live. The Python side uses live-mask info
//  afterward to pick the top-K among the currently-dead positions.
//
//  Complexity: O(M * K * N) FMAs. Parallelized across the outer M loop
//  via OpenMP (same pattern as spmm_grad_w).
// ─────────────────────────────────────────────────────────────────────────
void dense_grad(
    int64_t M, int64_t K, int64_t N,
    const float* dY,    // (M, N)
    const float* X,     // (K, N)
    float* G            // (M, K) — output
);

}  // namespace sparselab
