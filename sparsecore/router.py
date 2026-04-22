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


__all__ = [
    "SparsityAlgorithm",
    "DynamicSparsityAlgorithm",
    "Static",
    "SET",
    "RigL",
]


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



# ─────────────────────────────────────────────────────────────────────
#  DynamicSparsityAlgorithm — base for algorithms that mutate topology
# ─────────────────────────────────────────────────────────────────────

class DynamicSparsityAlgorithm(SparsityAlgorithm):
    """
    Base for algorithms that mutate topology on a schedule (SET, RigL).

    Handles the common bookkeeping — step counter, 'is this an update
    step?' decision — so the subclass only has to say *what* mutations
    to perform when an update fires.

    Args:
        sparsity:      target sparsity in ``[0, 1)``.
        drop_fraction: fraction of live connections to churn per update.
                       0.3 means 30% of a layer's live slots get dropped
                       and replaced each time ``update()`` fires.
        update_freq:   interval, in calls to ``step()``, between updates.
                       100 means ``update()`` runs every 100 steps.
                       Smaller = more exploration, bigger overhead.

    Subclass contract:

        def update(self):
            for layer in self.layers:
                self._update_layer(layer)

        def _update_layer(self, layer: SparseLinear) -> None:
            # Compute the new (cols, values) for each row of the layer
            # and call layer._csr.rewrite_row(row_idx, new_cols, new_vals).
            ...

    The base's ``step()`` is overridden here to gate on ``update_freq``.
    """

    def __init__(self, sparsity: float, drop_fraction: float, update_freq: int):
        super().__init__(sparsity=sparsity)
        if not (0.0 < drop_fraction <= 1.0):
            raise ValueError(
                f"drop_fraction must be in (0.0, 1.0], got {drop_fraction}"
            )
        if update_freq < 1:
            raise ValueError(
                f"update_freq must be >= 1, got {update_freq}"
            )
        self.drop_fraction = drop_fraction
        self.update_freq = update_freq

    def step(self) -> None:
        """
        Advance the step counter. If ``_step_idx`` is an integer
        multiple of ``update_freq``, call ``update()`` to mutate
        topology.

        The user calls ``algo.step()`` after ``optimizer.step()`` in
        their training loop. For SET/RigL with ``update_freq=100`` this
        means ~1% of training steps trigger a topology mutation.
        """
        self._step_idx += 1
        if self._step_idx % self.update_freq == 0:
            self.update()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"sparsity={self.sparsity}, "
            f"drop_fraction={self.drop_fraction}, "
            f"update_freq={self.update_freq}, "
            f"n_layers={len(self.layers)}, "
            f"step={self._step_idx})"
        )


# ─────────────────────────────────────────────────────────────────────
#  SET — Sparse Evolutionary Training
# ─────────────────────────────────────────────────────────────────────

class SET(DynamicSparsityAlgorithm):
    """
    Sparse Evolutionary Training (Mocanu et al., 2018).

    The simplest DST algorithm. Every ``update_freq`` steps:

      1. Drop the ``drop_fraction × row_nnz`` smallest-magnitude weights
         in each row.
      2. Grow the same number of new connections at random empty
         positions in each row. New weights start at zero.

    Total ``nnz`` stays constant. Over training, the topology evolves
    toward positions where the data actually needs connections.

    Why SET is the right v0.1 DST algorithm:
      - Uses only information the sparse kernel already exposes
        (weight magnitudes). No need for dense gradients → cheap.
      - The simplest possible control for "does topology mutation help
        at all?" — every DST paper compares to SET.
      - Tiny implementation: ~30 lines of real logic here.

    For the more sophisticated RigL (gradient-based regrow) see
    milestone 4f.

    Args:
        sparsity:      same as base. Sets the overall density of layers
                       this algorithm attaches to.
        drop_fraction: fraction of live connections per layer to churn
                       on each update. Paper default: 0.3.
        update_freq:   steps between updates. Paper uses 100 for CIFAR;
                       we default to 100 too.

    Example:
        >>> layer = sparsecore.SparseLinear(784, 512, sparsity=0.9)
        >>> algo = sparsecore.SET(sparsity=0.9, drop_fraction=0.3,
        ...                        update_freq=100)
        >>> layer.apply(algo)
        >>> # training loop
        >>> # ...
        >>> # opt.step(); algo.step()   # last line per step
    """

    def __init__(
        self,
        sparsity: float,
        drop_fraction: float = 0.3,
        update_freq: int = 100,
        seed: int | None = None,
    ):
        super().__init__(
            sparsity=sparsity,
            drop_fraction=drop_fraction,
            update_freq=update_freq,
        )
        # Lazy import to keep module-load cheap.
        import numpy as np
        # Dedicated RNG per-algorithm-instance so SET's stochasticity
        # doesn't mess with or depend on torch/numpy's global state.
        # If seed is None, we derive from numpy's default entropy source
        # (so two algorithm instances created back-to-back don't
        # accidentally produce the same mutations).
        self._rng = np.random.default_rng(seed)

    def update(self) -> None:
        """
        Drop smallest-magnitude weights and regrow random new ones.

        Implementation: *global* magnitude threshold, not per-row.

        Why global: if row 0 has 50 live connections and row 5 has 200,
        per-row dropping gives them equal weight in the mutation, even
        though row 5's "smallest 30%" might still be more important
        than row 0's "largest 70%". Global dropping lets the algorithm
        concentrate kept connections where the data needs them most.

        Each layer is processed independently (global within a layer,
        not across layers). Per-row dropping was the v0.1 prototype
        and is preserved in git history for comparison.
        """
        for layer in self.layers:
            self._update_layer(layer, self._rng)

    def _update_layer(self, layer, rng) -> None:
        """Mutate a single layer's topology using GLOBAL magnitude
        threshold. Separated for testability."""
        import numpy as np

        csr = layer._csr
        col_indices = np.asarray(csr.col_indices)
        values_np   = np.asarray(csr.values)
        row_start   = np.asarray(csr.row_start)
        row_nnz     = np.asarray(csr.row_nnz)
        ncols       = csr.ncols
        total_nnz   = int(csr.nnz)

        if total_nnz == 0:
            return

        # ── PASS 1: find global k-th smallest magnitude across all live slots.
        # This is the threshold below which we drop.
        # We iterate rows to gather only LIVE slot values (skipping padding).
        all_live_vals = np.empty(total_nnz, dtype=np.float32)
        cursor = 0
        for i in range(csr.nrows):
            n = int(row_nnz[i])
            if n == 0:
                continue
            start = int(row_start[i])
            all_live_vals[cursor:cursor + n] = values_np[start:start + n]
            cursor += n
        assert cursor == total_nnz

        abs_vals = np.abs(all_live_vals)
        k_drop = max(1, int(self.drop_fraction * total_nnz))
        # argpartition finds the threshold where the k-th smallest lies.
        # We'll use this magnitude as our drop cutoff.
        threshold = np.partition(abs_vals, k_drop - 1)[k_drop - 1]

        # ── PASS 2: for each row, drop slots with |value| <= threshold,
        # grow new connections at random empty columns, and rewrite the
        # row atomically.
        for i in range(csr.nrows):
            start = int(row_start[i])
            n_live = int(row_nnz[i])
            if n_live == 0:
                continue

            live_cols = col_indices[start : start + n_live]
            live_vals = values_np[start : start + n_live]

            # Which slots survive: |value| strictly greater than threshold.
            # Ties at the threshold get to survive — we slightly under-drop
            # which is safer than over-dropping (never produces empty rows).
            keep_mask = np.abs(live_vals) > threshold
            n_keep = int(keep_mask.sum())

            # Corner case: if threshold is small and the row has tiny
            # magnitudes across the board, keep_mask could be all False.
            # Force at least 1 survivor to avoid empty rows.
            if n_keep == 0 and n_live > 0:
                # Keep the largest-magnitude slot in this row.
                best_idx = int(np.argmax(np.abs(live_vals)))
                keep_mask[best_idx] = True
                n_keep = 1

            survivor_cols = live_cols[keep_mask]
            survivor_vals = live_vals[keep_mask]
            n_drop = n_live - n_keep  # how many slots we freed

            # Grow n_drop new connections so total nnz stays constant.
            # Pick from columns NOT currently in this row's live set.
            live_set = set(int(c) for c in survivor_cols)
            # Building the empty column list via list comprehension is
            # O(ncols). At ncols = 784 (MNIST) this is fine; at transformer
            # FFN scale (say 8192) it's still microseconds.
            empty_cols = np.array(
                [c for c in range(ncols) if c not in live_set],
                dtype=np.int32,
            )

            if empty_cols.size == 0 or n_drop == 0:
                # No growth possible (row is now dense) or no drops happened.
                # Write back survivors in sorted order.
                order = np.argsort(survivor_cols, kind="stable")
                csr.rewrite_row(
                    i,
                    survivor_cols[order].astype(np.int32),
                    survivor_vals[order].astype(np.float32),
                )
                continue

            n_grow = min(n_drop, empty_cols.size)
            grow_cols = rng.choice(empty_cols, size=n_grow, replace=False)
            grow_vals = np.zeros(n_grow, dtype=np.float32)  # SET: zero-init

            # Merge survivors + new, sort by column, write back.
            merged_cols_unsorted = np.concatenate([survivor_cols, grow_cols])
            merged_vals_unsorted = np.concatenate([survivor_vals, grow_vals])
            order = np.argsort(merged_cols_unsorted, kind="stable")
            merged_cols = merged_cols_unsorted[order].astype(np.int32)
            merged_vals = merged_vals_unsorted[order].astype(np.float32)

            csr.rewrite_row(i, merged_cols, merged_vals)

        # After topology change, the _values torch Parameter is still
        # aliased to csr.values numpy view, so in-place writes above
        # are automatically visible to the optimizer at the next step.



# ─────────────────────────────────────────────────────────────────────
#  RigL — Rigging the Lottery
# ─────────────────────────────────────────────────────────────────────

class RigL(DynamicSparsityAlgorithm):
    """
    Rigging the Lottery (Evci et al., 2020,
    https://arxiv.org/abs/1911.11134).

    The smarter cousin of SET. Same drop criterion (smallest-magnitude
    live weights), but a smarter grow criterion: instead of random empty
    positions, grow at positions where the *dense gradient* |dL/dW| is
    largest — i.e., positions the loss would start training toward
    immediately if the connection existed.

    Why it works: `dL/dW[i, k]` at a currently-dead position `(i, k)`
    says "if you added this connection, this is the force that would
    act on it." Large magnitude → the task wants this connection.
    So we grow there. The random-regrow of SET is like buying lottery
    tickets at random; RigL reads the answer key first.

    On short training budgets (where SET often ties or loses to
    Static), RigL's information advantage shows up immediately — the
    paper shows a measurable accuracy improvement within the first
    ~10 epochs on CIFAR.

    Args:
        sparsity:      target sparsity in ``[0, 1)``.
        drop_fraction: fraction of live connections to churn per update.
                       Paper default: 0.3.
        update_freq:   steps between updates. Paper default: 100.
        seed:          optional RNG seed for reproducibility (in case
                       of ties in the top-K selection).

    Implementation detail: RigL needs the forward-pass input `X` and
    the backward-pass upstream gradient `dY` for each layer, at the
    most recent training step before `update()` fires. We capture
    these via hooks registered when the layer is attached — a
    forward hook to stash `X`, and a full backward hook to stash
    `dY`. These are consumed and cleared at every `update()`.

    Example:
        >>> layer = sparsecore.SparseLinear(784, 512, sparsity=0.9)
        >>> algo = sparsecore.RigL(sparsity=0.9, drop_fraction=0.3,
        ...                          update_freq=100, seed=42)
        >>> layer.apply(algo)
        >>> # training loop
        >>> #   opt.step(); algo.step()
    """

    def __init__(
        self,
        sparsity: float,
        drop_fraction: float = 0.3,
        update_freq: int = 100,
        seed: int | None = None,
    ):
        super().__init__(
            sparsity=sparsity,
            drop_fraction=drop_fraction,
            update_freq=update_freq,
        )
        import numpy as np
        self._rng = np.random.default_rng(seed)
        # Per-layer captured state: maps layer id → captured tensors.
        # Populated by the forward/backward hooks on each training step;
        # read by update() and then cleared.
        self._captured: dict[int, dict] = {}
        # Handles to the hooks so we can remove them if needed.
        self._hook_handles: list = []

    # ─── Hook installation ─────────────────────────────────────────

    def attach(self, layer) -> None:
        """Override attach to additionally install forward/backward hooks
        that capture X (input) and dY (upstream gradient) at each
        training step."""
        super().attach(layer)

        # Forward hook: captures the input X.
        #
        # PyTorch passes (module, input_tuple, output) to forward hooks.
        # Our SparseLinear.forward takes a single positional arg.
        def _fwd_hook(mod, inputs, output):
            # inputs is a tuple; inputs[0] is the raw input tensor (N, H_in).
            # We stash a detached copy so autograd doesn't hold a ref.
            # The final forward of the layer operates on (H_in, N) after
            # a transpose inside forward(); we stash the USER-FACING (N, H_in)
            # shape and transpose inside update().
            self._captured.setdefault(id(mod), {})["X"] = inputs[0].detach().clone()

        # Full backward hook: captures the gradient w.r.t. the OUTPUT of the layer.
        #
        # PyTorch's signature: (module, grad_input, grad_output).
        # grad_output is a tuple with one entry for each output tensor;
        # for our layer it's a single (N, H_out) gradient. We stash it.
        def _bwd_hook(mod, grad_input, grad_output):
            dY = grad_output[0]
            if dY is not None:
                self._captured.setdefault(id(mod), {})["dY"] = dY.detach().clone()

        self._hook_handles.append(layer.register_forward_hook(_fwd_hook))
        self._hook_handles.append(layer.register_full_backward_hook(_bwd_hook))

    # ─── Update ────────────────────────────────────────────────────

    def update(self) -> None:
        """Drop smallest-magnitude weights, grow at highest-gradient
        empty positions."""
        for layer in self.layers:
            self._update_layer(layer, self._rng)

    def _update_layer(self, layer, rng) -> None:
        """Mutate one layer's topology using the captured gradient info."""
        import numpy as np
        import torch
        from sparsecore import _core

        # ── Retrieve captured X and dY from the hooks.
        captured = self._captured.get(id(layer))
        if captured is None or "X" not in captured or "dY" not in captured:
            # No forward/backward has run since attach — can't update.
            # This can happen if update_freq is reached before any training
            # step. Just skip this cycle.
            return

        # Shapes:
        #   X:  (N, H_in)  — user-facing forward input shape
        #   dY: (N, H_out) — gradient w.r.t. layer output
        # For our dense_grad kernel we need:
        #   dY': (H_out=M, N) — transpose
        #   X':  (H_in =K, N) — transpose
        X_user = captured["X"]      # (N, H_in)
        dY_user = captured["dY"]    # (N, H_out)

        # Flatten leading batch dims the same way forward() does.
        X_flat = X_user.reshape(-1, layer.in_features).contiguous()
        dY_flat = dY_user.reshape(-1, layer.out_features).contiguous()

        # Transpose to (H_out, N) and (H_in, N).
        dY_np = dY_flat.t().contiguous().numpy()
        X_np = X_flat.t().contiguous().numpy()

        # ── Compute the full dense gradient G = dY @ X^T, shape (M, K).
        # M = layer.out_features (= csr.nrows)
        # K = layer.in_features  (= csr.ncols)
        G = _core.dense_grad(dY_np, X_np)   # shape (M, K)
        abs_G = np.abs(G)

        # ── Drop: same as SET, global magnitude threshold.
        csr = layer._csr
        col_indices = np.asarray(csr.col_indices)
        values_np   = np.asarray(csr.values)
        row_start   = np.asarray(csr.row_start)
        row_nnz     = np.asarray(csr.row_nnz)
        ncols       = csr.ncols
        total_nnz   = int(csr.nnz)

        if total_nnz == 0:
            self._captured.pop(id(layer), None)
            return

        # Collect all live |values| to find the drop threshold.
        all_live_vals = np.empty(total_nnz, dtype=np.float32)
        cursor = 0
        for i in range(csr.nrows):
            n = int(row_nnz[i])
            if n == 0:
                continue
            start = int(row_start[i])
            all_live_vals[cursor:cursor + n] = values_np[start:start + n]
            cursor += n

        abs_vals = np.abs(all_live_vals)
        k_drop = max(1, int(self.drop_fraction * total_nnz))
        threshold = np.partition(abs_vals, k_drop - 1)[k_drop - 1]

        # ── Per-row: drop low-magnitude, grow at top-|G| empty positions.
        for i in range(csr.nrows):
            start = int(row_start[i])
            n_live = int(row_nnz[i])
            if n_live == 0:
                continue

            live_cols = col_indices[start : start + n_live]
            live_vals = values_np[start : start + n_live]

            # Keep slots strictly above the drop threshold.
            keep_mask = np.abs(live_vals) > threshold
            n_keep = int(keep_mask.sum())

            # Don't produce empty rows.
            if n_keep == 0 and n_live > 0:
                best_idx = int(np.argmax(np.abs(live_vals)))
                keep_mask[best_idx] = True
                n_keep = 1

            survivor_cols = live_cols[keep_mask]
            survivor_vals = live_vals[keep_mask]
            n_drop = n_live - n_keep

            if n_drop == 0:
                # No drops in this row → no grows needed.
                order = np.argsort(survivor_cols, kind="stable")
                csr.rewrite_row(
                    i,
                    survivor_cols[order].astype(np.int32),
                    survivor_vals[order].astype(np.float32),
                )
                continue

            # ── NEW vs SET: grow at top-|G| empty positions.
            # Mask out currently-live cols so we only consider empties.
            row_grad_abs = abs_G[i, :].copy()   # length ncols
            row_grad_abs[survivor_cols] = -np.inf  # exclude live

            # Pick the top n_drop grow candidates.
            # argpartition gets us the top-n without full sort.
            available = int((row_grad_abs > -np.inf).sum())
            n_grow = min(n_drop, available)
            if n_grow == 0:
                # Row is fully dense after drops — can't grow.
                order = np.argsort(survivor_cols, kind="stable")
                csr.rewrite_row(
                    i,
                    survivor_cols[order].astype(np.int32),
                    survivor_vals[order].astype(np.float32),
                )
                continue

            # argpartition returns an unsorted partition; the first n_grow
            # elements contain the top n_grow values (not sorted among
            # themselves, but each >= all elements after position n_grow).
            # Negate abs_G so "top K largest" becomes "top K smallest of
            # the negation", which is what argpartition[:n_grow] gives us.
            top_indices = np.argpartition(-row_grad_abs, n_grow - 1)[:n_grow]
            grow_cols = top_indices.astype(np.int32)
            grow_vals = np.zeros(n_grow, dtype=np.float32)  # RigL: zero-init

            # Merge survivors + grown, sort by column.
            merged_cols_unsorted = np.concatenate([survivor_cols, grow_cols])
            merged_vals_unsorted = np.concatenate([survivor_vals, grow_vals])
            order = np.argsort(merged_cols_unsorted, kind="stable")
            merged_cols = merged_cols_unsorted[order].astype(np.int32)
            merged_vals = merged_vals_unsorted[order].astype(np.float32)

            csr.rewrite_row(i, merged_cols, merged_vals)

        # Free captured state so we don't hold a reference to stale tensors.
        self._captured.pop(id(layer), None)
