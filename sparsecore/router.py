"""
sparsecore.router — the pluggable sparsity algorithm API.

This module defines the contract that every DST (Dynamic Sparse
Training) algorithm — RigL, SET, a new one you're inventing — must
fit into. The contract is deliberately small so that expressing a
new algorithm is ~50-100 lines of Python.

Design doc: ``docs/design/router.md``

Inspired by `cerebras.pytorch.sparse.SparsityAlgorithm`. We adopt
their `model.apply(sparsity)` user surface, their
`DynamicSparsityAlgorithm` convenience base, and their "single
abstract `update()` method" pattern. We diverge on the mask
representation — Cerebras stores dense weight + separate mask; we
store PaddedCSR, so "update the mask" becomes "mutate the CSR
topology."

User story (intended workflow):

    layer = sparsecore.SparseLinear(784, 512, sparsity=0.9)

    # 4d: Static — no-op reference implementation
    sparsity = sparsecore.Static(sparsity=0.9)
    layer.apply(sparsity)                     # or model.apply(sparsity)

    # Training loop (unchanged except for the final line):
    for x, y in loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        sparsity.step()                       # <-- NEW: triggers update()
                                              #     if schedule says so
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type hints to avoid a circular import at runtime
    # (nn.py → router.py → nn.py). The string quotes on forward refs
    # below handle the actual typing.
    from sparsecore.nn import SparseLinear


# ─────────────────────────────────────────────────────────────────────
#  Base class: SparsityAlgorithm
# ─────────────────────────────────────────────────────────────────────

class SparsityAlgorithm:
    """
    Base class for every sparsity policy attached to SparseLinear layers.

    Subclasses override :meth:`update` to mutate topology. The base
    class handles:

      - Tracking which layers this algorithm is attached to
          (``self.layers``)
      - Advancing an internal step counter on ``self.step()``
      - Being callable via ``__call__(module)`` so ``model.apply(algo)``
        (PyTorch's recursive module walk) routes attachment correctly.

    Concrete subclasses in this module: :class:`Static` (no-op).
    DST subclasses (SET, RigL) land in milestones 4e and 4f.

    Args:
        sparsity: target sparsity level in ``[0, 1)``. Informational
                  only at the base class level — subclasses can use it
                  to decide how many connections to drop/grow.
    """

    def __init__(self, sparsity: float):
        if not (0.0 <= sparsity < 1.0):
            raise ValueError(
                f"sparsity must be in [0.0, 1.0), got {sparsity}"
            )
        self.sparsity = sparsity
        # List of layers this algorithm governs. Populated by attach().
        self.layers: list["SparseLinear"] = []
        # Step counter advanced by self.step(). Used by subclasses to
        # gate periodic updates.
        self._step_idx: int = 0

    # ─── Attachment ──────────────────────────────────────────────

    def attach(self, layer: "SparseLinear") -> None:
        """
        Register ``layer`` as governed by this algorithm.

        Called once per layer, typically via ``model.apply(sparsity)``.
        Idempotent: attaching the same layer twice is a no-op.

        Most users never call this directly — they use the ``__call__``
        path below.
        """
        # Late import to avoid circular import at module load time.
        from sparsecore.nn import SparseLinear

        if not isinstance(layer, SparseLinear):
            raise TypeError(
                f"SparsityAlgorithm.attach expects a SparseLinear, "
                f"got {type(layer).__name__}"
            )
        if layer not in self.layers:
            self.layers.append(layer)
            # The layer also keeps a back-pointer so it can report its
            # current algorithm in __repr__ and in future hooks.
            layer._sparsity_algorithm = self

    def __call__(self, module) -> None:
        """
        Module-walk entry point for ``model.apply(algorithm)``.

        PyTorch's ``nn.Module.apply(fn)`` recursively invokes
        ``fn(submodule)`` on every submodule. We accept any module
        here and only attach when it's a :class:`SparseLinear`. Dense
        submodules are silently skipped — exactly the behaviour we
        want, since a single algorithm instance can govern every
        SparseLinear in a multi-layer network.
        """
        from sparsecore.nn import SparseLinear

        if isinstance(module, SparseLinear):
            self.attach(module)

    # ─── Stepping ────────────────────────────────────────────────

    def step(self) -> None:
        """
        Advance the algorithm's internal step counter and optionally
        mutate topology.

        Intended call site: end of a training step, after
        ``optimizer.step()``. The base class just increments the
        counter; subclasses that inherit from
        :class:`DynamicSparsityAlgorithm` use the counter to decide
        when to call ``update()``.

        :class:`Static` uses the bare base-class step (no update).
        """
        self._step_idx += 1

    # ─── Abstract hook ───────────────────────────────────────────

    def update(self) -> None:  # pragma: no cover — override in subclasses
        """
        Mutate topology on each governed layer.

        Subclasses must override this. The base raises so that a user
        accidentally calling ``sparsity.update()`` on a policy that
        doesn't define it gets a clear error rather than silently
        doing nothing.
        """
        raise NotImplementedError(
            "SparsityAlgorithm.update() must be overridden by subclasses"
        )

    # ─── Introspection ──────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"sparsity={self.sparsity}, "
            f"n_layers={len(self.layers)}, "
            f"step={self._step_idx})"
        )


# ─────────────────────────────────────────────────────────────────────
#  Static — the no-op reference implementation
# ─────────────────────────────────────────────────────────────────────

class Static(SparsityAlgorithm):
    """
    Holds the init-time random mask unchanged for the entire training
    run. Exists as the reference implementation and the baseline every
    DST algorithm is compared against.

    Why have a no-op algorithm at all?

      1. It concretizes the API. Users can call ``model.apply(algo)``
         with ``Static`` first to verify the plumbing works before
         switching to SET/RigL.
      2. It gives the test suite a minimal but real ``SparsityAlgorithm``
         subclass to exercise.
      3. DST research papers consistently compare against "static
         random masks at the same sparsity" as the control. Ours
         ships as a first-class citizen.

    Args:
        sparsity: same as :class:`SparsityAlgorithm`.

    Example:
        >>> import sparsecore
        >>> layer = sparsecore.SparseLinear(128, 64, sparsity=0.9)
        >>> algo = sparsecore.Static(sparsity=0.9)
        >>> layer.apply(algo)                    # attaches
        >>> algo.step()                          # no topology change
        >>> layer.nnz                            # unchanged from init
    """

    def update(self) -> None:
        """No-op. Static sparsity never mutates topology."""
        pass
