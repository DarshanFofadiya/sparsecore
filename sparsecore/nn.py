"""
sparsecore.nn — PyTorch-native ``nn.Module`` layers backed by sparse storage.

This is where the "two-line adoption" promise is delivered: a researcher
swaps ``nn.Linear`` for ``sparsecore.SparseLinear`` and adds a
``sparsity=0.9`` keyword, and the rest of their training script
(optimizers, schedulers, loss functions, ``.to(device)``, ``state_dict``)
keeps working. Under the hood the weight matrix lives as a ``PaddedCSR``,
the forward pass dispatches to our NEON ``_SpMMFunction``, and the
backward pass participates in ``loss.backward()`` automatically.

Design doc: ``docs/design/sparse_linear.md``.

Inspired by:
  - ``torch.nn.Linear`` (parameter name and init conventions)
  - ``cerebras.pytorch.sparse`` (separation of layer vs. sparsity algorithm)
  - ``hyeon95y/SparseLinear`` (constructor signature)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from sparsecore._core import PaddedCSR as _PaddedCSR
from sparsecore import layout as _layout
from sparsecore.ops import _SpMMFunction


class SparseLinear(nn.Module):
    """
    A drop-in replacement for ``nn.Linear`` whose weight is stored as a
    ``PaddedCSR`` at the requested sparsity. Forward and backward go
    through our C++ kernels; autograd hands the gradient for the live
    (non-zero) weight slots back to the standard torch optimizer stack.

    Args:
        in_features:  input feature dimension (``H_in``).
        out_features: output feature dimension (``H_out``).
        bias:         if ``True``, learn an additive bias of shape
                      ``(out_features,)``. Default ``True``.
        sparsity:     fraction of weights that are structurally zero.
                      Must be in ``[0.0, 1.0)``. Default ``0.9``.
        padding_ratio: passed through to ``PaddedCSR.from_dense``; extra
                       capacity per row reserved for future topology
                       mutation (SET/RigL). Default ``0.2``.

    Shape:
        - Input:  ``(*, in_features)`` — leading batch dims are arbitrary.
        - Output: ``(*, out_features)``.

    Example:
        >>> import torch, sparsecore
        >>> layer = sparsecore.SparseLinear(784, 512, sparsity=0.9)
        >>> x = torch.randn(128, 784)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([128, 512])
    """

    # ─────────────────────────────────────────────────────────────
    #  Construction
    # ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.9,
        padding_ratio: float = 0.2,
    ):
        super().__init__()
        if not (0.0 <= sparsity < 1.0):
            raise ValueError(
                f"sparsity must be in [0.0, 1.0), got {sparsity}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.padding_ratio = padding_ratio

        # Build the layer weights. Doing this in a separate method means
        # ``reset_parameters`` can regenerate them in-place later.
        self._csr, values_param = self._build_csr_and_parameter(sparsity)

        # ``_values`` holds the trainable non-zero weight values. It is
        # aliased (shares storage) with the C++-backed ``self._csr.values``
        # buffer so that an optimizer step that updates ``_values.data``
        # is immediately visible to the next forward pass's C++ kernel.
        self._values = values_param

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            # ``register_parameter(..., None)`` is the idiomatic way to
            # mark a parameter as absent. It shows up in ``state_dict``
            # but as ``None`` rather than silently vanishing.
            self.register_parameter("bias", None)

        # Draw bias values (weights were already sampled during the
        # PaddedCSR construction to use the same RNG stream).
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _build_csr_and_parameter(
        self, sparsity: float
    ) -> tuple[_PaddedCSR, nn.Parameter]:
        """
        Draw a fresh random mask + values and materialize them as a
        ``PaddedCSR`` plus a torch ``nn.Parameter`` aliased to the CSR
        value buffer.

        This is the one place we do the numpy↔torch aliasing trick.
        ``torch.from_numpy`` shares memory with the source numpy array,
        so updates to the returned Parameter's ``.data`` propagate
        straight into the C++ buffer. See ``docs/design/sparse_linear.md``.
        """
        bound = 1.0 / math.sqrt(self.in_features)

        # Kaiming-uniform init on a dense scratch matrix: same
        # distribution ``nn.Linear`` uses, so a user swapping
        # ``nn.Linear -> SparseLinear`` sees statistically identical
        # starting weights (on the live edges).
        W_dense = torch.empty(self.out_features, self.in_features)
        nn.init.uniform_(W_dense, -bound, bound)

        # Draw a random binary mask at the requested sparsity. We use
        # torch's RNG (not numpy's) so ``torch.manual_seed(...)`` seeds
        # both the mask and the bias reproducibly.
        mask = (torch.rand(self.out_features, self.in_features) >= sparsity)
        W_dense = W_dense * mask.float()

        # Convert to PaddedCSR. This allocates padded rows with extra
        # capacity for future topology mutation.
        csr = _layout.from_dense(W_dense, padding_ratio=self.padding_ratio)

        # Alias the C++ value buffer as a torch Parameter. The numpy
        # array from ``csr.values`` is a zero-copy view of the C++
        # ``std::vector<float>``; ``torch.from_numpy`` keeps that view.
        values_np = np.asarray(csr.values)
        values_param = nn.Parameter(torch.from_numpy(values_np))
        return csr, values_param

    def reset_parameters(self) -> None:
        """
        Reset weights, mask, and bias to freshly sampled values. This
        mirrors ``nn.Linear.reset_parameters()``.

        Note: because the mask is re-drawn, the set of live connections
        changes. For most use cases this is what you want — if you need
        to keep the topology and only re-init values, do that manually
        via ``self._values.data.uniform_(-bound, bound)``.
        """
        self._csr, new_values = self._build_csr_and_parameter(self.sparsity)
        # Keep the Parameter object; just overwrite its data in-place
        # so any external references (e.g. an optimizer state dict) stay valid.
        with torch.no_grad():
            self._values.data = new_values.data
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    # ─────────────────────────────────────────────────────────────
    #  Forward pass
    # ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute ``y = x @ Wᵀ + b`` with W stored as PaddedCSR.

        We accept any input shape ending in ``in_features``. Extra
        leading dims (batch, sequence, etc.) are flattened, the SpMM
        runs on the flattened batch, then the output is reshaped. This
        matches the behaviour of ``nn.Linear``.
        """
        # Preserve the original leading shape for the final reshape.
        orig_shape = x.shape
        if orig_shape[-1] != self.in_features:
            raise ValueError(
                f"SparseLinear expects input last dim {self.in_features}, "
                f"got {orig_shape[-1]}"
            )

        # Flatten all batch-like dims into one: (batch, in_features).
        # contiguous() because .view() requires it and our kernel does too.
        x_flat = x.reshape(-1, self.in_features).contiguous()

        # Our SpMM computes Y = W @ X where X has shape (in_features, batch).
        # So we transpose to match that layout. The autograd function
        # then hands us Y of shape (out_features, batch), which we
        # transpose back to (batch, out_features).
        X_col = x_flat.t().contiguous()
        Y_col = _SpMMFunction.apply(self._values, self._csr, X_col, "simd")
        y_flat = Y_col.t()  # (batch, out_features)

        if self.bias is not None:
            # Broadcast: (batch, out_features) + (out_features,) → (batch, out_features).
            y_flat = y_flat + self.bias

        # Restore leading batch dims.
        out_shape = orig_shape[:-1] + (self.out_features,)
        return y_flat.reshape(out_shape)

    # ─────────────────────────────────────────────────────────────
    #  Inspection helpers
    # ─────────────────────────────────────────────────────────────

    @property
    def nnz(self) -> int:
        """Number of structurally non-zero weights (live connections)."""
        return self._csr.nnz

    @property
    def density(self) -> float:
        """Live connections as a fraction of the dense matrix size."""
        return self.nnz / (self.out_features * self.in_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"sparsity={self.sparsity}, "
            f"nnz={self.nnz}"
        )
