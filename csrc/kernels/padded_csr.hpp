// ═══════════════════════════════════════════════════════════════════════════
//  kernels/padded_csr.hpp
//
//  The PaddedCSR storage struct.
//
//  PaddedCSR is CSR with per-row padding slots reserved for future
//  insertions. Standard CSR requires shifting every entry past the
//  insertion point on every grow operation — O(nnz) per insertion,
//  unusable for dynamic sparse training where topology changes every
//  few hundred steps. Padded-CSR trades 10-20% extra memory for O(1)
//  insertions within a row's preallocated capacity.
//
//  See docs/design/padded_csr.md for the full specification — this
//  file implements §2.1 (fields) and §4 (C++ side of the API).
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstddef>   // std::size_t
#include <cstdint>   // int32_t, int64_t
#include <string>    // std::string
#include <vector>


namespace sparselab {

// ─────────────────────────────────────────────────────────────────────────
//  PaddedCSR: sparse matrix with padded rows for fast insertion.
//
//  Layout (see docs/design/padded_csr.md §2.1):
//
//    values[]       — length total_capacity(), per-slot float32 values.
//                     Live slots hold real values; padding slots hold 0.0f.
//    col_indices[]  — length total_capacity(), per-slot int32 column index.
//                     Live slots: 0 <= col < ncols. Padding slots: -1.
//    row_start[]    — length nrows, index into values[]/col_indices[] where
//                     each row begins.
//    row_nnz[]      — length nrows, count of LIVE entries per row (<= capacity).
//    row_capacity[] — length nrows, total allocated slots per row.
//
//  Within each row, the first row_nnz[i] slots are live and column-sorted;
//  the trailing row_capacity[i] - row_nnz[i] slots are padding.
//
//  See the 8 invariants in docs/design/padded_csr.md §2.2 for the hard
//  guarantees this struct must uphold.
// ─────────────────────────────────────────────────────────────────────────
struct PaddedCSR {
    // Shape — int64_t matches PyTorch convention and allows future large
    // matrices without re-typing.
    int64_t nrows;
    int64_t ncols;

    // Per-slot arrays. Length must equal total_capacity() (sum of row_capacity).
    std::vector<float>   values;
    std::vector<int32_t> col_indices;

    // Per-row arrays. Length must equal nrows.
    // int32_t is sufficient: our v0.1 scope caps nnz below 2 billion
    // (see design doc §2.3 "Why int32").
    std::vector<int32_t> row_start;
    std::vector<int32_t> row_nnz;
    std::vector<int32_t> row_capacity;

    // ─── Topology version counter ─────────────────────────────────────
    //
    // Monotonically increasing integer that bumps every time rewrite_row
    // modifies the CSR's live-set. Used by callers to cache derived
    // structures (like the transposed CSR, which is expensive to
    // materialize) and invalidate them when topology changes.
    //
    // Starts at 0. Any code that wants to cache a transpose should
    // record the version it was computed for, and recompute only when
    // the current version differs.
    int64_t topology_version = 0;

    // ─── Constructors ──────────────────────────────────────────────────

    // Empty constructor: builds a valid zero-nnz, zero-capacity matrix
    // of the given shape. Useful as a starting point for manual tests
    // and as a default-construction fallback.
    PaddedCSR(int64_t nrows, int64_t ncols);

    // Full constructor: takes ownership of all six data arrays via move.
    // Move semantics matter — a 1M-element vector<float> is 4 MB, and
    // copying it on construction would waste memory bandwidth for no reason.
    // Python factories use this path: they build the arrays in NumPy, then
    // move them across the FFI boundary.
    //
    // Caller's responsibility to provide arrays that satisfy all 8 invariants
    // (see design doc §2.2). Use check_invariants() post-construction during
    // development to verify.
    PaddedCSR(
        int64_t nrows, int64_t ncols,
        std::vector<float>&&   values,
        std::vector<int32_t>&& col_indices,
        std::vector<int32_t>&& row_start,
        std::vector<int32_t>&& row_nnz,
        std::vector<int32_t>&& row_capacity
    );

    // ─── Read-only accessors ───────────────────────────────────────────

    // Total live non-zero entries (sum of row_nnz).
    // O(nrows) — not cached because nrows is typically small
    // (<= 4096 for realistic transformer layers).
    int64_t nnz() const;

    // Total allocated slots including padding (sum of row_capacity).
    // O(nrows).
    int64_t total_capacity() const;

    // Total padding slots (total_capacity - nnz). O(nrows).
    int64_t padding_slots() const;

    // ─── Mutation: rewrite a single row atomically ─────────────────────
    //
    // Replaces row `i`'s live content with the (new_cols, new_values) pair.
    // Remaining slots in the row's capacity become padding (col=-1, val=0).
    //
    // Preconditions (checked at runtime, throws std::invalid_argument):
    //   - 0 <= row_idx < nrows
    //   - new_cols.size() == new_values.size()
    //   - new_cols.size() <= row_capacity[row_idx]
    //   - new_cols sorted strictly ascending
    //   - all new_cols in [0, ncols)
    //
    // This is the single mutation primitive DST algorithms use. They
    // compute the desired new live set for a row in Python (SET: drop
    // smallest K, grow K random empties; RigL: drop smallest K, grow K
    // positions with highest dense gradient), then hand the complete
    // row to this function. All invariant maintenance (padding sentinel,
    // row_nnz update, sort order) is done here.
    //
    // Complexity: O(row_capacity[i]) — one pass over the row's slots.
    // For realistic training workloads (row_capacity ~100-1000 per row)
    // this is a few microseconds per call, invoked at most every N
    // training steps.
    void rewrite_row(
        int64_t row_idx,
        const std::vector<int32_t>& new_cols,
        const std::vector<float>&   new_values
    );
};

// ─────────────────────────────────────────────────────────────────────────
//  Invariant checking.
//
//  Two entry points:
//
//    check_invariants_str(p)
//        Returns an empty string if p satisfies all 8 invariants from
//        design doc §2.2. Otherwise returns a human-readable error
//        message describing the first violation found.
//        Useful in tests that want to assert invalidity without catching
//        an exception.
//
//    assert_invariants(p)
//        Calls check_invariants_str; if non-empty, throws
//        std::invalid_argument with the message. Pybind11 auto-translates
//        this to a Python ValueError.
//
//  Complexity: O(total_capacity). Dominated by the "columns are sorted
//  within each row" check. Not cheap — intended for construction-time
//  validation and tests, not for per-operation hot paths.
// ─────────────────────────────────────────────────────────────────────────
std::string check_invariants_str(const PaddedCSR& p);
void assert_invariants(const PaddedCSR& p);

}  // namespace sparselab
