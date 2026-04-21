// ═══════════════════════════════════════════════════════════════════════════
//  SparseCore C++ bindings — Milestone 1: The PyTorch Bridge
// ═══════════════════════════════════════════════════════════════════════════
//
//  This is the first C++ file in SparseCore. Its job is deliberately tiny:
//  accept a float array from Python, multiply every element by 2.0, return
//  a new float array. No sparsity, no SIMD, no PyTorch C++ API — just the
//  bridge that proves Python and C++ can talk to each other over the FFI.
//
//  Everything sophisticated (Padded-CSR, NEON kernels, autograd) will build
//  on top of this same pattern: Python passes raw float buffers across the
//  boundary, C++ reads/writes them, returns new buffers. We deliberately do
//  NOT depend on libtorch headers — that keeps our .so binary compatible
//  with any future PyTorch release (see docs/SYSTEM_PROMPT.md §2a).
//
// ═══════════════════════════════════════════════════════════════════════════

// pybind11.h: the core C++/Python interop machinery.
// Gives us the PYBIND11_MODULE macro and the py::* types.
#include <pybind11/pybind11.h>

// pybind11/numpy.h: adds py::array_t<T> — a NumPy-compatible array view.
// This is how we accept float buffers from Python without linking libtorch.
// PyTorch tensors auto-convert to NumPy arrays (same memory, zero-copy
// when the dtype and device match), so this handles our use case.
#include <pybind11/numpy.h>

namespace py = pybind11;  // shorthand — every pybind11 project does this


// ─────────────────────────────────────────────────────────────────────────
//  double_tensor
//
//  Takes a 1-D NumPy-compatible float32 array, returns a new array of the
//  same shape where every element is doubled.
//
//  Parameters:
//    input - a py::array_t<float>. When called from Python, pybind11
//            automatically accepts anything that satisfies the NumPy array
//            protocol: numpy.ndarray, torch.Tensor (via .numpy()), or
//            Python lists/tuples of floats (auto-converted).
//
//  Returns:
//    A newly allocated py::array_t<float> of the same shape. The input is
//    NOT mutated — this follows PyTorch's convention that `x * 2` produces
//    a new tensor rather than modifying `x` in place.
//
//  Complexity: O(n), single-threaded, no SIMD. Milestone 2 rewrites this
//  pattern with NEON intrinsics; Milestone 1 just proves the pipe works.
// ─────────────────────────────────────────────────────────────────────────
py::array_t<float> double_tensor(py::array_t<float> input) {
    // Request a buffer_info view of the input. This is NOT a copy — it's a
    // descriptor struct with:
    //   .ptr    : raw void* into the underlying float memory
    //   .shape  : vector of dimension sizes (e.g. {5} for a 1-D array of 5)
    //   .size   : total element count (product of shape)
    //   .ndim   : number of dimensions
    //   .itemsize, .format, .strides : not needed for this milestone
    py::buffer_info input_info = input.request();

    // Cast the void* to float* so we can do pointer arithmetic on it.
    // `const` because we promise not to mutate the input (see docstring).
    const float* input_ptr = static_cast<const float*>(input_info.ptr);
    const size_t n = static_cast<size_t>(input_info.size);

    // Allocate a fresh output array with the SAME shape as the input.
    // py::array_t's constructor takes the shape vector and allocates
    // contiguous memory in one step. Memory is uninitialized at this
    // point — we'll overwrite every byte in the loop below.
    auto output = py::array_t<float>(input_info.shape);
    py::buffer_info output_info = output.request();
    float* output_ptr = static_cast<float*>(output_info.ptr);

    // The math itself. One scalar multiply per element.
    // Note: we use 2.0f (float literal), not 2.0 (double literal). The `f`
    // suffix matters — without it, the compiler promotes each element to
    // double, does the multiply in double precision, then demotes back to
    // float. Correct answer, but slower and harder to reason about.
    for (size_t i = 0; i < n; ++i) {
        output_ptr[i] = input_ptr[i] * 2.0f;
    }

    return output;
}


// ─────────────────────────────────────────────────────────────────────────
//  PYBIND11_MODULE — registers our C++ functions as a Python module.
//
//  The first argument (_core) MUST match the module name declared in
//  setup.py's Pybind11Extension (i.e. "sparsecore._core", everything after
//  the dot). If they mismatch, Python imports the module and finds nothing
//  inside — a silent, confusing failure mode.
//
//  The second argument (m) is the local variable name for the module
//  object inside this macro's body. By convention it's always `m`.
// ─────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_core, m) {
    // Module-level docstring. Shows up as `sparsecore._core.__doc__` in
    // Python, and in help(sparsecore._core).
    m.doc() = "SparseCore C++ core — Milestone 1: PyTorch bridge sanity check.";

    // Register `double_tensor` as a callable named `double_tensor` in
    // Python. The third argument is the Python-visible docstring; it
    // appears in help(sparsecore._core.double_tensor).
    m.def("double_tensor", &double_tensor,
          "Multiply every float in a 1-D array by 2.0. "
          "Returns a new array; the input is not mutated.");
}
