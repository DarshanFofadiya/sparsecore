"""
sparselab.nn — PyTorch-native ``nn.Module`` layers backed by sparse storage.

This is where the "two-line adoption" promise is delivered: a researcher
swaps ``nn.Linear`` for ``sparselab.SparseLinear`` and adds a
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

from sparselab._core import PaddedCSR as _PaddedCSR
from sparselab import layout as _layout
from sparselab.ops import _SpMMFunction


__all__ = ["SparseLinear"]


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
        >>> import torch, sparselab
        >>> layer = sparselab.SparseLinear(784, 512, sparsity=0.9)
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
        # PaddedCSR construction to use the same RNG stream). Use the
        # same effective-fan-in as the weight init above — keeps the
        # bias's natural scale compatible with how much signal the
        # live weights actually carry.
        if self.bias is not None:
            effective_fan_in = max(1.0, in_features * (1.0 - sparsity))
            bound = 1.0 / math.sqrt(effective_fan_in)
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
        # ─── Sparsity-aware Kaiming-uniform init ──────────────────────
        # A dense Linear with fan-in F uses bound = 1/sqrt(F) so that the
        # variance of its output matches the variance of its input
        # (the PyTorch convention that inspired this init). For a sparse
        # layer, only (1 - sparsity) * F of the in_features are live —
        # the rest are masked to zero. If we keep the dense bound, we
        # under-scale the live weights by sqrt(1 - sparsity) and the
        # signal shrinks by that factor at every layer.
        #
        # For a single sparse layer surrounded by dense math (demo_05,
        # demo_12) or sandwiched between LayerNorm + residuals
        # (demo_15's transformer), the training loop recovers from this
        # under-scaling within a few hundred steps. But a stack of 5+
        # SparseLinear layers with ReLU between them (demo_18's
        # sparse_sequential MLP) sees the undershoot compound
        # exponentially — signal collapse stalls training at chance
        # accuracy for many epochs.
        #
        # Fix: scale the bound by sqrt(F / F_effective). That's the
        # correct Kaiming bound for the actual number of live inputs
        # feeding each output unit. This matches how Cerebras scales
        # weights in cerebras.pytorch.sparse (they call it "sparsity-
        # compensated init"), and is consistent with the standard
        # derivation of Kaiming init applied to a sparse mask.
        #
        # Safe for existing callers: at sparsity=0 this reduces to the
        # dense bound unchanged. At sparsity=0.9 it scales up by
        # sqrt(10) ≈ 3.16x, which makes single-layer demo_05 slightly
        # warmer at init but converges to the same final accuracy (we
        # verify this in tests).
        effective_fan_in = max(1.0, self.in_features * (1.0 - sparsity))
        bound = 1.0 / math.sqrt(effective_fan_in)

        # Kaiming-uniform init on a dense scratch matrix. The sparsity
        # compensation above ensures the live weights (after masking)
        # have unit-variance-compatible magnitudes.
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
