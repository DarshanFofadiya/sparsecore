// ═══════════════════════════════════════════════════════════════════════════
//  kernels/padded_csr.cpp
//
//  Implementation of the PaddedCSR struct's constructors and accessors.
//  Invariant checking will be added in sub-milestone 3b-ii.
// ═══════════════════════════════════════════════════════════════════════════

#include "padded_csr.hpp"

#include <numeric>    // std::accumulate
#include <stdexcept>  // std::invalid_argument


namespace sparsecore {

// ─────────────────────────────────────────────────────────────────────────
//  Empty constructor: zero-capacity matrix of given shape.
//  All six arrays are default-constructed (empty). Valid by invariants
//  (§2.2): nrows >= 0, row_start/row_nnz/row_capacity have length nrows = 0,
//  total_capacity = 0, etc. A matrix with nrows > 0 but total_capacity = 0
//  is also valid — it has empty row_capacity vectors of correct length 0.
// ─────────────────────────────────────────────────────────────────────────
PaddedCSR::PaddedCSR(int64_t nrows_, int64_t ncols_)
    : nrows(nrows_),
      ncols(ncols_),
      values(),
      col_indices(),
      row_start(nrows_, 0),            // nrows zeros
      row_nnz(nrows_, 0),              // nrows zeros
      row_capacity(nrows_, 0) {        // nrows zeros
    // Note: values and col_indices are zero-length; that's correct for
    // an empty matrix because total_capacity = sum(row_capacity) = 0.
}

// ─────────────────────────────────────────────────────────────────────────
//  Full constructor: takes ownership of all six arrays via move.
// ─────────────────────────────────────────────────────────────────────────
PaddedCSR::PaddedCSR(
    int64_t nrows_, int64_t ncols_,
    std::vector<float>&&   values_,
    std::vector<int32_t>&& col_indices_,
    std::vector<int32_t>&& row_start_,
    std::vector<int32_t>&& row_nnz_,
    std::vector<int32_t>&& row_capacity_
)
    : nrows(nrows_),
      ncols(ncols_),
      values(std::move(values_)),
      col_indices(std::move(col_indices_)),
      row_start(std::move(row_start_)),
      row_nnz(std::move(row_nnz_)),
      row_capacity(std::move(row_capacity_)) {
    // Invariant checking comes in 3b-ii. For now, trust the caller.
}

// ─────────────────────────────────────────────────────────────────────────
//  Accessors — trivial sums over small arrays.
// ─────────────────────────────────────────────────────────────────────────

int64_t PaddedCSR::nnz() const {
    // std::accumulate with int64_t init forces the sum into int64 space
    // to avoid overflow at very large nrows. Not a concern at v0.1
    // scales (nrows <= 4096), but cheap to future-proof.
    return std::accumulate(row_nnz.begin(), row_nnz.end(), int64_t{0});
}

int64_t PaddedCSR::total_capacity() const {
    return std::accumulate(row_capacity.begin(), row_capacity.end(), int64_t{0});
}

int64_t PaddedCSR::padding_slots() const {
    return total_capacity() - nnz();
}

// ═════════════════════════════════════════════════════════════════════════
//  Invariant checking — §2.2 of docs/design/padded_csr.md
// ═════════════════════════════════════════════════════════════════════════

std::string check_invariants_str(const PaddedCSR& p) {
    // ─── Invariant 1: shape is non-negative ──────────────────────────────
    if (p.nrows < 0 || p.ncols < 0) {
        return "Invariant 1 violated: shape must be non-negative, got (" +
               std::to_string(p.nrows) + ", " + std::to_string(p.ncols) + ").";
    }

    // ─── Invariant (metadata arrays correct length) ──────────────────────
    // Implicit consequence of storage layout — if row_start etc. aren't
    // length nrows, nothing else works. Check it explicitly.
    const std::size_t nrows_sz = static_cast<std::size_t>(p.nrows);
    if (p.row_start.size()    != nrows_sz ||
        p.row_nnz.size()      != nrows_sz ||
        p.row_capacity.size() != nrows_sz) {
        return "Metadata array length mismatch: row_start=" +
               std::to_string(p.row_start.size()) +
               ", row_nnz=" + std::to_string(p.row_nnz.size()) +
               ", row_capacity=" + std::to_string(p.row_capacity.size()) +
               ", expected nrows=" + std::to_string(p.nrows) + ".";
    }

    // ─── Invariant 2: values and col_indices have the same length ────────
    // and that length equals total_capacity.
    if (p.values.size() != p.col_indices.size()) {
        return "Invariant 2 violated: values (" + std::to_string(p.values.size()) +
               ") and col_indices (" + std::to_string(p.col_indices.size()) +
               ") must have the same length.";
    }
    const int64_t total_cap = p.total_capacity();
    if (static_cast<int64_t>(p.values.size()) != total_cap) {
        return "Invariant 2 violated: values.size()=" +
               std::to_string(p.values.size()) +
               " does not match total_capacity=" +
               std::to_string(total_cap) + " (sum of row_capacity).";
    }

    // ─── Invariant 3: row_start is the running sum of row_capacity ───────
    // row_start[0] must be 0; row_start[i+1] must be row_start[i] + row_capacity[i].
    if (p.nrows > 0 && p.row_start[0] != 0) {
        return "Invariant 3 violated: row_start[0] must be 0, got " +
               std::to_string(p.row_start[0]) + ".";
    }
    for (int64_t i = 1; i < p.nrows; ++i) {
        const int32_t expected = p.row_start[i - 1] + p.row_capacity[i - 1];
        if (p.row_start[i] != expected) {
            return "Invariant 3 violated: row_start[" + std::to_string(i) +
                   "]=" + std::to_string(p.row_start[i]) +
                   " should equal row_start[" + std::to_string(i - 1) +
                   "] + row_capacity[" + std::to_string(i - 1) +
                   "] = " + std::to_string(expected) + ".";
        }
    }

    // ─── Invariant 4: 0 <= row_nnz[i] <= row_capacity[i] ─────────────────
    for (int64_t i = 0; i < p.nrows; ++i) {
        if (p.row_nnz[i] < 0 || p.row_nnz[i] > p.row_capacity[i]) {
            return "Invariant 4 violated at row " + std::to_string(i) +
                   ": row_nnz=" + std::to_string(p.row_nnz[i]) +
                   " must be in [0, " +
                   std::to_string(p.row_capacity[i]) + "].";
        }
    }

    // ─── Per-row slot-level checks (invariants 5, 6, 7) ──────────────────
    // For each row, walk its slots:
    //   [row_start, row_start + row_nnz)            → live slots
    //   [row_start + row_nnz, row_start + row_cap)  → padding slots
    for (int64_t i = 0; i < p.nrows; ++i) {
        const int32_t start = p.row_start[i];
        const int32_t n_live = p.row_nnz[i];
        const int32_t n_cap  = p.row_capacity[i];

        // Invariant 6: live column indices are in [0, ncols) and strictly
        // sorted ascending (PyTorch requires sorted + unique within a row).
        int32_t prev_col = -1;
        for (int32_t k = 0; k < n_live; ++k) {
            const int32_t c = p.col_indices[start + k];
            if (c < 0 || c >= p.ncols) {
                return "Invariant 6 violated at row " + std::to_string(i) +
                       ", slot " + std::to_string(k) +
                       ": col_indices=" + std::to_string(c) +
                       " must be in [0, " + std::to_string(p.ncols) + ").";
            }
            if (c <= prev_col) {
                return "Invariant 6 violated at row " + std::to_string(i) +
                       ", slot " + std::to_string(k) +
                       ": col_indices=" + std::to_string(c) +
                       " must be strictly greater than previous=" +
                       std::to_string(prev_col) +
                       " (live columns must be sorted ascending and distinct).";
            }
            prev_col = c;
        }

        // Invariants 5 and 7: padding slots come after live slots.
        // We require col_indices[padding slot] == -1 as a sentinel that
        // mutation code uses to detect "is this slot currently live?"
        // We do NOT require padding values to be zero — user code (the
        // optimizer, weight decay, etc.) might legitimately write into
        // the raw _values buffer, and the SpMM kernels never read past
        // row_nnz so there's no correctness requirement. rewrite_row
        // still resets padding values to 0 as a defensive-consistency
        // measure, but the invariant check only enforces the col_idx
        // sentinel.
        for (int32_t k = n_live; k < n_cap; ++k) {
            if (p.col_indices[start + k] != -1) {
                return "Invariants 5+7 violated at row " + std::to_string(i) +
                       ", slot " + std::to_string(k) +
                       ": padding col_indices=" +
                       std::to_string(p.col_indices[start + k]) +
                       " must be -1.";
            }
        }
    }

    // All 8 invariants hold.
    return "";
}

void assert_invariants(const PaddedCSR& p) {
    const std::string err = check_invariants_str(p);
    if (!err.empty()) {
        throw std::invalid_argument("PaddedCSR invariant check failed: " + err);
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  Mutation: rewrite_row
// ═════════════════════════════════════════════════════════════════════════
//
//  This is the single mutation primitive used by DST algorithms (SET,
//  RigL). The Python side decides which columns a row should contain
//  after drop+grow; this function writes them in.
//
//  Under the hood: overwrites the row's slot range, fills trailing slots
//  with the padding sentinel, and updates row_nnz. All invariant
//  maintenance happens here, so Python code can't accidentally break
//  the CSR by passing malformed data.
// ═════════════════════════════════════════════════════════════════════════

void PaddedCSR::rewrite_row(
    int64_t row_idx,
    const std::vector<int32_t>& new_cols,
    const std::vector<float>&   new_values
) {
    // ─── Bounds check ──────────────────────────────────────────────────
    if (row_idx < 0 || row_idx >= nrows) {
        throw std::invalid_argument(
            "rewrite_row: row_idx=" + std::to_string(row_idx) +
            " out of range [0, " + std::to_string(nrows) + ").");
    }

    // ─── Parallel-array length check ───────────────────────────────────
    if (new_cols.size() != new_values.size()) {
        throw std::invalid_argument(
            "rewrite_row: new_cols.size()=" +
            std::to_string(new_cols.size()) +
            " must equal new_values.size()=" +
            std::to_string(new_values.size()));
    }

    // ─── Capacity check ────────────────────────────────────────────────
    const int32_t cap = row_capacity[row_idx];
    const int32_t new_n = static_cast<int32_t>(new_cols.size());
    if (new_n > cap) {
        throw std::invalid_argument(
            "rewrite_row: row " + std::to_string(row_idx) +
            " has capacity " + std::to_string(cap) +
            " but got " + std::to_string(new_n) + " new entries.");
    }

    // ─── Validate new_cols: strictly ascending, in bounds, distinct ────
    int32_t prev = -1;
    for (int32_t k = 0; k < new_n; ++k) {
        const int32_t c = new_cols[k];
        if (c < 0 || c >= ncols) {
            throw std::invalid_argument(
                "rewrite_row: new_cols[" + std::to_string(k) +
                "]=" + std::to_string(c) +
                " out of range [0, " + std::to_string(ncols) + ").");
        }
        if (c <= prev) {
            // We require strictly increasing — catches both unsorted
            // input AND duplicate columns within a row (PyTorch CSR
            // invariant).
            throw std::invalid_argument(
                "rewrite_row: new_cols must be strictly ascending, got " +
                std::to_string(prev) + " followed by " + std::to_string(c) +
                " at position " + std::to_string(k) + ".");
        }
        prev = c;
    }

    // ─── Do the write ─────────────────────────────────────────────────
    const int32_t base = row_start[row_idx];

    // Write the live slots.
    for (int32_t k = 0; k < new_n; ++k) {
        col_indices[base + k] = new_cols[k];
        values[base + k]      = new_values[k];
    }

    // Fill trailing slots with padding sentinel (col=-1, val=0).
    // This is what the SpMM kernel relies on to skip dead slots without
    // a second dereference — if col_indices is -1 we know to stop early,
    // and value 0 means even if we accidentally multiply, nothing happens.
    for (int32_t k = new_n; k < cap; ++k) {
        col_indices[base + k] = -1;
        values[base + k]      = 0.0f;
    }

    // Update the live count.
    row_nnz[row_idx] = new_n;

    // Bump the topology version so anyone caching derived structures
    // (like the transpose) can detect this change and invalidate.
    topology_version += 1;
}

}  // namespace sparsecore
