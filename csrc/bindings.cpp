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
//  Python wrapper: spmm_scalar
//
//  Compute Y = W @ X where W is a PaddedCSR and X is a dense float32 tensor.
//  Allocates a fresh NumPy array for Y (zero-initialized by NumPy's default
//  allocator — matching the kernel's "Y must be pre-zeroed" contract).
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
py::array_t<float> py_spmm_scalar(
    const sparsecore::PaddedCSR& W,
    py::array_t<float, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info x_info = X.request();

    // Shape validation: 2-D input required.
    if (x_info.ndim != 2) {
        throw std::invalid_argument(
            "spmm_scalar: X must be 2-D, got ndim=" +
            std::to_string(x_info.ndim) + "."
        );
    }

    const int64_t K = x_info.shape[0];
    const int64_t N = x_info.shape[1];

    // Inner-dimension match: W's column count must equal X's row count.
    if (W.ncols != K) {
        throw std::invalid_argument(
            "spmm_scalar: shape mismatch. W has ncols=" +
            std::to_string(W.ncols) +
            " but X has shape[0]=" + std::to_string(K) +
            ". Inner dimensions must match for matmul."
        );
    }

    // Allocate the output array. py::array_t's shape constructor calls
    // numpy.zeros-equivalent underneath — memory is zero-initialized, which
    // is exactly what spmm_scalar's "pre-zeroed Y" contract requires.
    const int64_t M = W.nrows;
    auto Y = py::array_t<float>({M, N});
    py::buffer_info y_info = Y.request();

    const float* x_ptr = static_cast<const float*>(x_info.ptr);
    float* y_ptr = static_cast<float*>(y_info.ptr);

    sparsecore::spmm_scalar(W, x_ptr, K, N, y_ptr);

    return Y;
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
          "Reference implementation — no SIMD. See also: spmm_simd_neon (3d).");

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
        .def_property_readonly("values", [make_readonly_view](py::object self_obj) {
            auto& self = self_obj.cast<sparsecore::PaddedCSR&>();
            return make_readonly_view(self.values, self_obj);
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
