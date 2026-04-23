# Design: `SparseLinear` (Milestone 4b)

## What it is

`SparseLinear` is a `torch.nn.Module` drop-in replacement for
`torch.nn.Linear` whose weight matrix is stored as a PaddedCSR rather
than a dense tensor. This is the "two-line adoption" promise of
SparseCore: a researcher should be able to take a dense MLP and make
it sparse by renaming `nn.Linear` to `sparsecore.SparseLinear` and
adding a `sparsity=0.9` keyword argument.

That's it. That's the whole milestone. It's pure plumbing on top of
`_SpMMFunction`, which already exists and is tested.

## The user story this serves

Before Milestone 4b (what demo_05 had to do):

```python
torch.manual_seed(0)
bound = 1.0 / (784 ** 0.5)
W1_init = (torch.rand(HIDDEN, 784) * 2 - 1) * bound
mask = (torch.rand(HIDDEN, 784) >= sparsity).float()
W1_init = W1_init * mask
W_csr = PaddedCSR.from_dense(W1_init)
W_vals = torch.from_numpy(np.asarray(W_csr.values)).requires_grad_(True)
fc2 = nn.Linear(HIDDEN, 10, bias=False)
# ... custom training step with _SpMMFunction.apply and manual weight update ...
```

After Milestone 4b:

```python
fc1 = sparsecore.SparseLinear(784, 512, sparsity=0.9, bias=False)
fc2 = nn.Linear(512, 10, bias=False)
# ... standard nn.Module training step ...
```

## Prior art (what we're adopting and from whom)

Survey of the projects whose APIs and storage layouts influenced ours:

| Library | What we borrow | What we diverge on |
| ------- | -------------- | ------------------ |
| **hyeon95y/SparseLinear** (PyPI `sparselinear`) | Constructor signature (`in_features, out_features, bias, sparsity, connectivity`); Kaiming-uniform init with `bound = 1/sqrt(in_features)`; `register_buffer` for topology, `nn.Parameter` for values; `extra_repr` format. | They store COO `(2, nnz)` indices and implement SpMM as `to_dense().mm()` — defeats the purpose. We store PaddedCSR and use our `_SpMMFunction`. |
| **Cerebras `cerebras.pytorch.sparse`** | Split of concerns: layer (SparseLinear) owns **storage + forward**; sparsity algorithm (next milestone: Router) owns **mutation policy**. Layer does NOT know about SET/RigL/etc. | — |
| **rigl-torch** | Minimalist user footprint — aim for their "2 lines of code" vibe. | They attach RigL as an optimizer hook on top of dense `nn.Linear`; we have real sparse storage, so our shape is a layer replacement. |
| **PyTorch core `nn.Linear`** | Parameter name conventions (`weight`, `bias`), shape conventions `(out_features, in_features)`, `reset_parameters()` method. | We can't register `weight` as the full dense tensor; it lives in `self._values` as the PaddedCSR value buffer. We expose a read-only `.weight` property that materializes a `torch.sparse_csr_tensor` view — matches PyTorch's serialization expectations. |

## Public API

```python
class SparseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.9,
        padding_ratio: float = 0.2,
    ): ...

    # Standard nn.Module methods
    def forward(self, input: Tensor) -> Tensor: ...
    def reset_parameters(self) -> None: ...
    def extra_repr(self) -> str: ...

    # Inspection
    @property
    def nnz(self) -> int: ...
    @property
    def density(self) -> float: ...
```

### Constructor behaviour

1. Generate a random boolean mask of shape `(out_features, in_features)`
   with density `1 - sparsity`. Seed from PyTorch's default RNG so
   `torch.manual_seed(...)` works as expected.
2. Kaiming-uniform init on the whole dense matrix, zero-out where mask
   is False. (Matches the init distribution of `nn.Linear`.)
3. Convert to PaddedCSR via `PaddedCSR.from_dense(...)`.
4. Register:
   - `self._values`: `nn.Parameter(torch.from_numpy(csr.values))` — trainable.
   - `self._csr`: the `PaddedCSR` object itself (NOT a buffer — it's
     opaque C++ state; we store it as a plain attribute).
   - `self.bias`: `nn.Parameter` of shape `(out_features,)` if `bias=True`,
     else `None`.

### `forward(input)` contract

- Input shape: `(*, in_features)` — leading batch dims allowed.
- Output shape: `(*, out_features)`.
- Internally: flatten to `(batch, in_features)`, transpose to
  `(in_features, batch)` (what our SpMM expects), call
  `_SpMMFunction.apply(self._values, self._csr, X, "simd")`,
  transpose back, add bias, reshape to output shape.

### Key detail: `self._values` ↔ `self._csr.values` aliasing

This is the one subtle part. The `_SpMMFunction` expects:
- `W_values` (the autograd-tracked Tensor) — what the optimizer updates.
- `W` (the `PaddedCSR`) — whose C++ kernel reads through `W.values`.

These two must point at the *same underlying memory*. We achieve
this by:

1. `csr_values_np = np.asarray(csr.values)` — numpy view of C++ buffer.
2. `self._values = nn.Parameter(torch.from_numpy(csr_values_np))` —
   torch view of the same memory.

Then when the optimizer does `self._values.data -= lr * self._values.grad`,
the C++ `csr.values` buffer is automatically updated — no manual sync
needed. This is the trick that makes `SparseLinear` work with any
standard `torch.optim` optimizer (SGD, Adam, etc.) out of the box.

Caveat: some optimizers (notably Adam, RMSProp) create an internal
state tensor (momentum, second moment) by calling `torch.zeros_like()`
on the parameter. Because our parameter is a torch view of numpy-backed
memory, `zeros_like` produces a normal torch tensor — no aliasing issue.
The momentum lives in regular torch memory; at step time Adam does
`param.data.addcdiv_(m, v.sqrt(), value=-lr)`, which writes in-place
into the numpy-aliased buffer. We verified this already for SGD in
milestone 4a-vi; this just re-confirms it for arbitrary optimizers.

## What `SparseLinear` does NOT do (deferred to later milestones)

- **No topology mutation.** The mask is frozen at construction. Adding
  a drop/regrow step lives in Milestone 4d (Router API) and 4e (SET).
- **No parallelization.** Forward is still single-threaded NEON from
  `_SpMMFunction`. Milestone 4c adds OpenMP.
- **No 2:4 or block-structured sparsity.** Unstructured only.

## Tests to write

1. **Shape contract.** Input shapes `(N, H_in)` and `(N, L, H_in)` both
   return correct output shapes.
2. **Bias optional.** `bias=False` path omits addition; `bias=True`
   adds a learnable vector.
3. **`nnz` matches the requested sparsity** within rounding.
4. **`reset_parameters()` reproduces the initializer.**
5. **Forward equals the equivalent dense computation** on the
   live-edges-only weight matrix (oracle test).
6. **Gradient check** via `torch.autograd.gradcheck` on a small layer
   (composition of our already-gradchecked `_SpMMFunction` with bias
   add, so this is an integration test).
7. **SGD convergence**: train a 2-layer SparseLinear MLP on a tiny
   synthetic task, verify loss decreases.
8. **`state_dict` roundtrip**: save and restore; verify values and
   topology survive.
9. **Optimizer interop**: one full step with `torch.optim.SGD`
   matches a manual update; same with `torch.optim.Adam`.

The `demo_05` rewrite will be the end-to-end integration test:
same accuracy, simpler code.

## Out-of-scope questions this doc deliberately doesn't answer

- *How do we serialize the topology across train→inference?* PyTorch
  `state_dict` handles `_values` and `bias` automatically. The CSR
  index arrays (`col_indices`, `row_start`, etc.) are currently not
  in `state_dict`. This is fine for v0.1 because the default behaviour
  is "rebuild the layer at load time with the same random seed", but
  for true checkpoint portability we will need to serialize the full
  CSR. Scoped for milestone 4f or later.

- *Why doesn't `SparseLinear` inherit from `nn.Linear`?* Because
  `nn.Linear` has `self.weight: nn.Parameter` of shape
  `(out_features, in_features)` and we cannot honour that contract
  with padded-CSR storage. Duck-typing (matching the interface
  without the inheritance) is cleaner.

## Next milestone after this

Milestone 4c: OpenMP parallelization of `spmm_simd` and
`spmm_grad_w`. Independent of `SparseLinear`, kernel-level only,
~20–30% of 4b's LOC. Target: measurable per-epoch speedup on M3
Pro's 11 P-cores.
