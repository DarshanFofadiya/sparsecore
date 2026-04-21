"""
SparseCore — dynamic sparse training for PyTorch, CPU-native, Apple Silicon first.

Public API (v0.1):
    PaddedCSR          — sparse matrix storage with padded rows for O(1) insert.
    spmm(W, X)         — sparse-dense matmul Y = W @ X, returns torch.Tensor.

Factory helpers on PaddedCSR:
    PaddedCSR.from_dense(W, *, threshold=0.0, padding_ratio=0.2)
    PaddedCSR.from_torch_sparse_csr(csr, *, padding_ratio=0.2)
    PaddedCSR.random(nrows, ncols, *, sparsity, padding_ratio=0.2, seed=None)
    PaddedCSR.to_dense() -> torch.Tensor

This __init__ attaches the factory helpers (defined in sparsecore.layout) as
classmethod-style staticmethods on the C++-backed PaddedCSR class. The result
is that users see a single coherent PaddedCSR class, even though its methods
span two implementation languages.

See docs/PROJECT_OVERVIEW.md for the full mission and roadmap.
"""

from sparsecore._core import PaddedCSR as _PaddedCSRCpp
from sparsecore import layout as _layout
from sparsecore.ops import spmm


# ─────────────────────────────────────────────────────────────────────
#  Attach Python factories to the C++-backed class as static methods.
#
#  Rationale: rather than subclass PaddedCSR (which would shadow the
#  C++ type and confuse pybind11), we monkey-patch the factories onto
#  the class. Users write `PaddedCSR.from_dense(W)` and it just works.
# ─────────────────────────────────────────────────────────────────────

_PaddedCSRCpp.from_dense = staticmethod(_layout.from_dense)
_PaddedCSRCpp.from_torch_sparse_csr = staticmethod(_layout.from_torch_sparse_csr)
_PaddedCSRCpp.random = staticmethod(_layout.random)
_PaddedCSRCpp.to_dense = _layout.to_dense
_PaddedCSRCpp.transpose = _layout.transpose


# Public re-export with the canonical name.
PaddedCSR = _PaddedCSRCpp


__all__ = ["PaddedCSR", "spmm"]
__version__ = "0.0.1"
