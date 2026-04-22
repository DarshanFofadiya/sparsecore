// ═══════════════════════════════════════════════════════════════════════════
//  csrc/bindings.cpp
//
//  Sole file that registers SparseCore's C++ kernels as Python callables.
//
//  This file contains NO math. Its only jobs are:
//    1. Unwrap py::array_t<float> objects into raw (float*, size_t) pairs
//    2. Call the kernel implementations in csrc/kernels/
//    3. Wrap results back into Python objects
//
//  Every new kernel added to SparseCore gets:
//    - Its own file in csrc/kernels/<name>.cpp + .hpp (pure C++)
//    - A thin Python wrapper function here
//    - A single m.def() line in PYBIND11_MODULE
//
//  Pattern borrowed from pytorch/ao and xformers (Borrow-Don't-Reinvent).
// ═══════════════════════════════════════════════════════════════════════════

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // enables auto-conversion Python list <-> std::vector

#include "kernels/double_tensor.hpp"
#include "kernels/vector_dot.hpp"
#include "kernels/vector_dot_neon.hpp"
#include "kernels/padded_csr.hpp"
#include "kernels/spmm.hpp"
#include "kernels/spmm_neon.hpp"
#include "kernels/spmm_grad.hpp"

namespace py = pybind11;


// ─────────────────────────────────────────────────────────────────────────
//  Python wrapper: double_tensor
//
//  Accepts a 1-D NumPy-compatible float32 array, returns a new array of the
//  same shape with every element doubled. Input is not mutated.
// ─────────────────────────────────────────────────────────────────────────
py::array_t<float> py_double_tensor(py::array_t<float> input) {
    py::buffer_info in_info = input.request();
    const float* in_ptr = static_cast<const float*>(in_info.ptr);
    const std::size_t n = static_cast<std::size_t>(in_info.size);

    auto output = py::array_t<float>(in_info.shape);
    py::buffer_info out_info = output.request();
    float* out_ptr = static_cast<float*>(out_info.ptr);

    // Delegate the actual math to the kernel. Bindings do no computation.
    sparsecore::double_tensor_scalar(in_ptr, out_ptr, n);

    return output;
}


// ─────────────────────────────────────────────────────────────────────────
//  Internal helper: validate_and_extract_dot_inputs
//
//  Shared validation for any kernel with the vector_dot signature:
//    - Both arrays 1-D
//    - Equal length
//  On success, populates out-params with raw pointers and length.
//  On failure, throws std::invalid_argument (pybind11 → ValueError).
//
//  Not exposed to Python — file-local (anonymous namespace).
// ─────────────────────────────────────────────────────────────────────────
namespace {

void validate_and_extract_dot_inputs(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
    const char* kernel_name,
    const float*& a_ptr,
    const float*& b_ptr,
    std::size_t& n
) {
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.ndim != 1 || b_info.ndim != 1) {
        throw std::invalid_argument(
            std::string(kernel_name) + ": both inputs must be 1-D arrays. "
            "Got shapes with ndim=" + std::to_string(a_info.ndim) +
            " and ndim=" + std::to_string(b_info.ndim) + "."
        );
    }
    if (a_info.size != b_info.size) {
        throw std::invalid_argument(
            std::string(kernel_name) + ": input lengths must match. "
            "Got sizes " + std::to_string(a_info.size) +
            " and " + std::to_string(b_info.size) + "."
        );
    }

    a_ptr = static_cast<const float*>(a_info.ptr);
    b_ptr = static_cast<const float*>(b_info.ptr);
    n = static_cast<std::size_t>(a_info.size);
}

}  // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────
//  Python wrapper: vector_dot (scalar)
// ─────────────────────────────────────────────────────────────────────────
float py_vector_dot(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    py::array_t<float, py::array::c_style | py::array::forcecast> b
) {
    const float* a_ptr;
    const float* b_ptr;
    std::size_t n;
    validate_and_extract_dot_inputs(a, b, "vector_dot", a_ptr, b_ptr, n);
    return sparsecore::vector_dot_scalar(a_ptr, b_ptr, n);
}


// ─────────────────────────────────────────────────────────────────────────
//  Python wrapper: vector_dot_simd (NEON)
//
//  Same contract as vector_dot — 1-D float32 arrays in, single float
//  out. The only difference is it calls the NEON SIMD kernel.
//
//  Numerical note: the two kernels may differ by ~1 ULP due to
//  different accumulation orders. Both are verified against torch.dot
//  with rtol=atol=1e-5 in the test suite.
// ─────────────────────────────────────────────────────────────────────────
float py_vector_dot_simd(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    py::array_t<float, py::array::c_style | py::array::forcecast> b
) {
    const float* a_ptr;
    const float* b_ptr;
    std::size_t n;
    validate_and_extract_dot_inputs(a, b, "vector_dot_simd", a_ptr, b_ptr, n);
    return sparsecore::vector_dot_simd_neon(a_ptr, b_ptr, n);
}


// ─────────────────────────────────────────────────────────────────────────
//  Python wrapper: spmm_scalar / spmm_simd
//
//  Both scalar and NEON SpMM have identical input/output shapes, so we
//  factor validation + output-allocation into a shared helper and have
//  two thin wrappers that just pick which kernel to call.
//
//  Compute Y = W @ X where W is a PaddedCSR and X is a dense float32 tensor.
//  Allocates a fresh NumPy array for Y. The kernel self-zeros it internally,
//  so we don't need to pre-zero on the Python side.
//
//  Validation:
//    - X must be 2-D. A 1-D X is ambiguous (column vector? row vector?);
//      reject and let the caller reshape explicitly.
//    - X.shape[0] must equal W.ncols (the inner matmul dimension).
//    - X is accepted as any contiguous array with float-castable dtype;
//      forcecast silently converts float64 / float16 → float32 to match
//      our v0.1 dtype scope.
//
//  Returns a new NumPy array of shape (W.nrows, X.shape[1]), dtype float32.
//  The caller can wrap it in a torch.Tensor via torch.from_numpy(...).
// ─────────────────────────────────────────────────────────────────────────
namespace {

// Shared setup: validates inputs and allocates Y. Returns the three raw
// pointers / shapes the kernel needs; the caller picks the kernel.
struct SpmmPlan {
    py::array_t<float> Y;
    const float* x_ptr;
    float* y_ptr;
    int64_t K;
    int64_t N;
};

SpmmPlan prepare_spmm(
    const sparsecore::PaddedCSR& W,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& X,
    const char* kernel_name
) {
    py::buffer_info x_info = X.request();
    if (x_info.ndim != 2) {
        throw std::invalid_argument(
            std::string(kernel_name) + ": X must be 2-D, got ndim=" +
            std::to_string(x_info.ndim) + "."
        );
    }

    const int64_t K = x_info.shape[0];
    const int64_t N = x_info.shape[1];

    if (W.ncols != K) {
        throw std::invalid_argument(
            std::string(kernel_name) + ": shape mismatch. W has ncols=" +
            std::to_string(W.ncols) +
            " but X has shape[0]=" + std::to_string(K) +
            ". Inner dimensions must match for matmul."
        );
    }

    const int64_t M = W.nrows;
    auto Y = py::array_t<float>({M, N});
    py::buffer_info y_info = Y.request();

    SpmmPlan plan;
    plan.Y = Y;
    plan.x_ptr = static_cast<const float*>(x_info.ptr);
    plan.y_ptr = static_cast<float*>(y_info.ptr);
    plan.K = K;
    plan.N = N;
    return plan;
}

}  // anonymous namespace


py::array_t<float> py_spmm_scalar(
    const sparsecore::PaddedCSR& W,
    py::array_t<float, py::array::c_style | py::array::forcecast> X
) {
    auto plan = prepare_spmm(W, X, "spmm_scalar");
    sparsecore::spmm_scalar(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
    return plan.Y;
}


py::array_t<float> py_spmm_simd(
    const sparsecore::PaddedCSR& W,
    py::array_t<float, py::array::c_style | py::array::forcecast> X
) {
    auto plan = prepare_spmm(W, X, "spmm_simd");
    sparsecore::spmm_simd_neon(W, plan.x_ptr, plan.K, plan.N, plan.y_ptr);
    return plan.Y;
}


// ─────────────────────────────────────────────────────────────────────────
//  Python wrapper: spmm_grad_w
//
//  Compute dL/dW at live slots of W (milestone 4a-iv).
//
//  Takes three inputs:
//    W   — sparse weight (we only read its index arrays, not values)
//    dY  — upstream gradient, dense (M, N) float32
//    X   — original forward input, dense (K, N) float32
//
//  Returns a 1-D NumPy float32 array of length W.total_capacity(),
//  indexed the same way as W.values. Live slots get computed
//  gradients; padding slots stay at 0.0 (from the kernel's internal
//  memset). This layout aligns exactly with W.values, so an optimizer
//  can do W.values -= lr * dW as a single vectorized subtraction
//  without needing any slot-index mapping.
//
//  Shape checks:
//    - dY and X must both be 2-D
//    - dY.shape[0] must equal W.nrows
//    - X.shape[0] must equal W.ncols
//    - dY.shape[1] must equal X.shape[1] (the shared "N" dim)
// ─────────────────────────────────────────────────────────────────────────
py::array_t<float> py_spmm_grad_w(
    const sparsecore::PaddedCSR& W,
    py::array_t<float, py::array::c_style | py::array::forcecast> dY,
    py::array_t<float, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info dy_info = dY.request();
    py::buffer_info x_info = X.request();

    if (dy_info.ndim != 2) {
        throw std::invalid_argument(
            "spmm_grad_w: dY must be 2-D, got ndim=" +
            std::to_string(dy_info.ndim) + "."
        );
    }
    if (x_info.ndim != 2) {
        throw std::invalid_argument(
            "spmm_grad_w: X must be 2-D, got ndim=" +
            std::to_string(x_info.ndim) + "."
        );
    }

    const int64_t M_dy = dy_info.shape[0];
    const int64_t N_dy = dy_info.shape[1];
    const int64_t K_x = x_info.shape[0];
    const int64_t N_x = x_info.shape[1];

    if (M_dy != W.nrows) {
        throw std::invalid_argument(
            "spmm_grad_w: dY.shape[0]=" + std::to_string(M_dy) +
            " but W.nrows=" + std::to_string(W.nrows) +
            ". dY must have the same row count as W."
        );
    }
    if (K_x != W.ncols) {
        throw std::invalid_argument(
            "spmm_grad_w: X.shape[0]=" + std::to_string(K_x) +
            " but W.ncols=" + std::to_string(W.ncols) +
            ". X must have the same row count as W has columns."
        );
    }
    if (N_dy != N_x) {
        throw std::invalid_argument(
            "spmm_grad_w: dY.shape[1]=" + std::to_string(N_dy) +
            " must equal X.shape[1]=" + std::to_string(N_x) +
            " (shared inner dim N)."
        );
    }

    // Size the output to total_capacity to align with W.values. Padding
    // slots stay 0.0 (from the kernel's internal memset), which is the
    // correct neutral for gradient-descent updates.
    const int32_t cap = W.total_capacity();
    auto dW = py::array_t<float>({static_cast<py::ssize_t>(cap)});
    py::buffer_info dw_info = dW.request();

    const float* dY_ptr = static_cast<const float*>(dy_info.ptr);
    const float* X_ptr = static_cast<const float*>(x_info.ptr);
    float* dW_ptr = static_cast<float*>(dw_info.ptr);

    sparsecore::spmm_grad_w(W, dY_ptr, N_dy, X_ptr, K_x, dW_ptr);

    return dW;
}


// ─────────────────────────────────────────────────────────────────────────
//  Module registration.
//  The first argument (_core) must match the name in setup.py's
//  Pybind11Extension ("sparsecore._core" → everything after the dot).
// ─────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    m.doc() = "SparseCore C++ core — compiled kernels for sparse training.";

    m.def("double_tensor", &py_double_tensor,
          "Multiply every float in a 1-D array by 2.0. "
          "Returns a new array; the input is not mutated.");

    m.def("vector_dot", &py_vector_dot,
          "Compute the dot product of two 1-D float32 arrays. "
          "Returns a single float (sum of elementwise products).");

    m.def("vector_dot_simd", &py_vector_dot_simd,
          "NEON SIMD version of vector_dot. Same contract, ~3-4x faster "
          "on Apple Silicon. Numerically agrees with vector_dot within "
          "rtol=atol=1e-5.");

    m.def("spmm_scalar", &py_spmm_scalar,
          "Scalar sparse-dense matmul Y = W @ X. "
          "W is a PaddedCSR (M, K), X is a dense float32 array (K, N). "
          "Returns a new (M, N) float32 NumPy array. "
          "Reference implementation — no SIMD. See also: spmm_simd (NEON).");

    m.def("spmm_simd", &py_spmm_simd,
          "NEON SIMD sparse-dense matmul Y = W @ X. Same contract as "
          "spmm_scalar; the inner loop is vectorized 4-wide with NEON "
          "FMA. Numerically agrees with spmm_scalar within rtol=atol=1e-5.");

    m.def("spmm_grad_w", &py_spmm_grad_w,
          "Compute dL/dW at live slots of W. Given upstream gradient dY "
          "(M, N) and forward input X (K, N), returns a 1-D float32 array "
          "of length W.nnz() aligned with W.values. The dense-simulated "
          "anti-pattern materializes a full (M, K) gradient; we do not.");

    // ═════════════════════════════════════════════════════════════════════
    //  PaddedCSR — sparse matrix storage with padded rows for O(1) insert.
    //  See docs/design/padded_csr.md for the full specification.
    // ═════════════════════════════════════════════════════════════════════
    //
    //  Exposes the C++ sparsecore::PaddedCSR struct as a Python class.
    //  The `values`, `col_indices`, `row_start`, `row_nnz`, `row_capacity`
    //  properties return zero-copy NumPy views over the C++ vectors.
    //  Python code must treat them as read-only (enforced by numpy's
    //  writable=False flag below).
    //
    //  The Python-facing user API (from_dense, from_torch_sparse_csr,
    //  random) is defined in sparsecore/layout.py — that module constructs
    //  the underlying C++ object by calling this class's full constructor.
    // ─────────────────────────────────────────────────────────────────────

    // Helper lambda: build a zero-copy read-only numpy view over a vector.
    // Captures the PaddedCSR-owning lifetime via the `parent` handle so
    // the underlying memory stays alive as long as the array does.
    auto make_readonly_view = [](auto& vec, py::handle parent) {
        using T = typename std::remove_reference_t<decltype(vec)>::value_type;
        auto arr = py::array_t<T>(
            {static_cast<py::ssize_t>(vec.size())},  // shape
            {static_cast<py::ssize_t>(sizeof(T))},   // strides
            vec.data(),                              // data pointer
            parent                                   // keeps owner alive
        );
        // Mark non-writable so Python users can't mutate C++ state behind
        // its back. Mutation goes through explicit methods in Phase 4.
        py::detail::array_proxy(arr.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return arr;
    };

    // Helper lambda: zero-copy WRITABLE numpy view.
    //
    // Used only for the `values` array, which optimizers need to
    // update in-place during training (W.values -= lr * dW_values).
    // All structural arrays (col_indices, row_start, row_nnz,
    // row_capacity) stay read-only via make_readonly_view — writing
    // to those would break PaddedCSR invariants.
    //
    // Safety boundary:
    //   - Writable: `values` (just floats, any value is valid)
    //   - Read-only: everything that defines the sparsity structure
    //
    // Topology mutation (growing/pruning live slots) will happen in
    // milestone 4c through explicit methods that enforce invariants,
    // never through direct array mutation.
    auto make_writable_view = [](auto& vec, py::handle parent) {
        using T = typename std::remove_reference_t<decltype(vec)>::value_type;
        return py::array_t<T>(
            {static_cast<py::ssize_t>(vec.size())},
            {static_cast<py::ssize_t>(sizeof(T))},
            vec.data(),
            parent
        );
    };

    py::class_<sparsecore::PaddedCSR>(m, "PaddedCSR",
        "Sparse matrix with padded-row CSR storage for O(1) insertion. "
        "See docs/design/padded_csr.md for full specification.")

        // ─── Constructors ────────────────────────────────────────────────
        .def(py::init<int64_t, int64_t>(),
             py::arg("nrows"), py::arg("ncols"),
             "Create an empty (zero-capacity) PaddedCSR of the given shape.")

        .def(py::init([](int64_t nrows, int64_t ncols,
                         std::vector<float> values,
                         std::vector<int32_t> col_indices,
                         std::vector<int32_t> row_start,
                         std::vector<int32_t> row_nnz,
                         std::vector<int32_t> row_capacity) {
                auto p = std::make_unique<sparsecore::PaddedCSR>(
                    nrows, ncols,
                    std::move(values), std::move(col_indices),
                    std::move(row_start), std::move(row_nnz),
                    std::move(row_capacity)
                );
                sparsecore::assert_invariants(*p);  // validate on construction
                return p;
             }),
             py::arg("nrows"), py::arg("ncols"),
             py::arg("values"), py::arg("col_indices"),
             py::arg("row_start"), py::arg("row_nnz"), py::arg("row_capacity"),
             "Full constructor. Validates all 8 invariants from design "
             "doc §2.2; raises ValueError on any violation.")

        // ─── Shape ───────────────────────────────────────────────────────
        .def_property_readonly("shape", [](const sparsecore::PaddedCSR& self) {
            return py::make_tuple(self.nrows, self.ncols);
        })
        .def_property_readonly("nrows", [](const sparsecore::PaddedCSR& self) {
            return self.nrows;
        })
        .def_property_readonly("ncols", [](const sparsecore::PaddedCSR& self) {
            return self.ncols;
        })

        // ─── Aggregate accessors ─────────────────────────────────────────
        .def_property_readonly("nnz", &sparsecore::PaddedCSR::nnz,
            "Total live non-zero entries (sum of row_nnz).")
        .def_property_readonly("total_capacity", &sparsecore::PaddedCSR::total_capacity,
            "Total allocated slots including padding (sum of row_capacity).")
        .def_property_readonly("padding_slots", &sparsecore::PaddedCSR::padding_slots,
            "Number of slots allocated but not yet used (total_capacity - nnz).")
        .def_property_readonly("sparsity", [](const sparsecore::PaddedCSR& self) {
            // Fraction of logical cells that are zero = 1 - nnz / (nrows*ncols).
            if (self.nrows == 0 || self.ncols == 0) return 1.0;
            return 1.0 - static_cast<double>(self.nnz()) /
                         static_cast<double>(self.nrows * self.ncols);
        })

        // ─── Zero-copy array views ───────────────────────────────────────
        // values[] is WRITABLE — optimizers update it during training via
        // W.values -= lr * dW_values. The structural arrays below are
        // read-only because mutating them would break PaddedCSR invariants.
        .def_property_readonly("values", [make_writable_view](py::object self_obj) {
            auto& self = self_obj.cast<sparsecore::PaddedCSR&>();
            return make_writable_view(self.values, self_obj);
        })
        .def_property_readonly("col_indices", [make_readonly_view](py::object self_obj) {
            auto& self = self_obj.cast<sparsecore::PaddedCSR&>();
            return make_readonly_view(self.col_indices, self_obj);
        })
        .def_property_readonly("row_start", [make_readonly_view](py::object self_obj) {
            auto& self = self_obj.cast<sparsecore::PaddedCSR&>();
            return make_readonly_view(self.row_start, self_obj);
        })
        .def_property_readonly("row_nnz", [make_readonly_view](py::object self_obj) {
            auto& self = self_obj.cast<sparsecore::PaddedCSR&>();
            return make_readonly_view(self.row_nnz, self_obj);
        })
        .def_property_readonly("row_capacity", [make_readonly_view](py::object self_obj) {
            auto& self = self_obj.cast<sparsecore::PaddedCSR&>();
            return make_readonly_view(self.row_capacity, self_obj);
        })

        // ─── Invariants ──────────────────────────────────────────────────
        .def("assert_invariants", &sparsecore::assert_invariants,
             "Verify all 8 invariants from design doc §2.2; raise ValueError "
             "with a descriptive message on any violation.")

        // ─── Mutation: rewrite a single row atomically ───────────────────
        //
        // Python-facing signature: `csr.rewrite_row(row_idx, new_cols, new_values)`
        //
        // new_cols and new_values are accepted as Python sequences
        // (lists, numpy arrays, tuples) and copied into std::vector
        // inside the binding. This ~30 μs copy is amortized over
        // whole-row mutations that happen at most every N training
        // steps, so the overhead is irrelevant.
        //
        // DST algorithms (SET, RigL) compute the desired row content
        // in Python and call this to materialize it. All invariant
        // maintenance (column sort, padding sentinel, row_nnz) lives
        // in C++ so Python code cannot corrupt the CSR.
        .def("rewrite_row",
             [](sparsecore::PaddedCSR& self,
                int64_t row_idx,
                py::array_t<int32_t, py::array::c_style | py::array::forcecast> new_cols,
                py::array_t<float,   py::array::c_style | py::array::forcecast> new_values) {
                 // py::array_t::forcecast means pybind11 converts any
                 // numeric input (int lists, float32/64 arrays) to the
                 // declared dtype automatically. This matches how our
                 // other kernel bindings accept numpy arrays.

                 auto cols_buf = new_cols.request();
                 auto vals_buf = new_values.request();

                 if (cols_buf.ndim != 1 || vals_buf.ndim != 1) {
                     throw std::invalid_argument(
                         "rewrite_row: new_cols and new_values must be 1-D arrays");
                 }

                 const int32_t* cols_ptr = static_cast<const int32_t*>(cols_buf.ptr);
                 const float*   vals_ptr = static_cast<const float*>(vals_buf.ptr);
                 std::vector<int32_t> cols(cols_ptr, cols_ptr + cols_buf.shape[0]);
                 std::vector<float>   vals(vals_ptr, vals_ptr + vals_buf.shape[0]);

                 self.rewrite_row(row_idx, cols, vals);
             },
             py::arg("row_idx"),
             py::arg("new_cols"),
             py::arg("new_values"),
             "Replace row row_idx's live content with the given columns and values.\n"
             "new_cols must be strictly ascending, distinct, and within [0, ncols).\n"
             "Trailing slots in the row's capacity become padding (col=-1, value=0).\n"
             "Used by DST algorithms (SET, RigL) for topology mutation.")

        // ─── Repr ────────────────────────────────────────────────────────
        .def("__repr__", [](const sparsecore::PaddedCSR& self) {
            return "PaddedCSR(nrows=" + std::to_string(self.nrows) +
                   ", ncols=" + std::to_string(self.ncols) +
                   ", nnz=" + std::to_string(self.nnz()) +
                   ", capacity=" + std::to_string(self.total_capacity()) +
                   ")";
        })
    ;
}
