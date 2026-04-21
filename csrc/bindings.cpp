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

#include "kernels/double_tensor.hpp"
#include "kernels/vector_dot.hpp"
#include "kernels/vector_dot_neon.hpp"

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
}
