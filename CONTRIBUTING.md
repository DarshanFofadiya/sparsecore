# Contributing to SparseLab

Thanks for your interest. SparseLab is built to be **the canonical
scaffolding for dynamic sparse training research**, which only works
if the community can contribute. This doc tells you how.

If you have 30 seconds, read the "How we review PRs" section below.
That's the part that will save you the most time.

---

## Ways to contribute

### 1. Pick an existing issue

We maintain curated starter issues:

- [`good first issue`](https://github.com/DarshanFofadiya/sparselab/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) —
  small, well-scoped tasks good for getting familiar with the codebase
- [`help wanted`](https://github.com/DarshanFofadiya/sparselab/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) —
  larger contributions we'd love help with (new DST algorithms, CPU
  perf work, CUDA port, Windows wheels, etc.)
- [`v0.2`](https://github.com/DarshanFofadiya/sparselab/issues?q=is%3Aopen+is%3Aissue+label%3Av0.2) —
  planned for the next release

You do **not** need to ask permission to start work on an issue. Just
open a PR when you have something to show. We avoid assigning
issues so multiple people can attempt a fix in parallel —
whoever ships working code first wins. If you want to coordinate,
leave a comment saying "I'm working on this" but don't expect an
exclusivity lock.

### 2. Propose a new feature or bug fix

Open an issue first, especially for anything that changes public API
or adds a new top-level module. One-sentence description of the
change plus one paragraph of rationale is enough to start the
discussion.

### 3. File a bug report

Use the "Bug report" issue template. Include:

- Your OS, Python version, `sparselab.__version__`, and `torch.__version__`
- A **minimum reproduction** (5-10 lines of code ideally)
- What you expected vs what happened
- The full traceback if applicable

### 4. Ask a question

If something's unclear, open an issue with the `question` label or
ask in a GitHub Discussion. Questions are first-class contributions —
every question answered publicly helps the next person.

---

## How we review PRs

We review against a handful of explicit principles. Reading these
before you open a PR saves cycles for everyone:

1. **Comments that teach, not just describe.** Every public function
   gets a header comment explaining what it does, what the inputs
   mean, what it returns, and any non-obvious assumptions. SIMD blocks
   get 2-3 sentences on memory movement and vectorization choices.
   Aim for the reader who is learning, not just reading. See the
   existing code in `csrc/kernels/` for the standard.

2. **Small, focused PRs.** Target: <50 lines of real logic per PR
   where possible. A 500-line PR is harder to review than five
   100-line PRs. For pure bulk work (renames, docstring updates),
   larger is fine. For behavior changes, keep it small.

3. **Oracle tests for math kernels.** Every new compute kernel must
   have a test that compares its output against a reference (PyTorch,
   numpy, or a documented scalar implementation) at 1e-5 tolerance.
   See `tests/test_spmm.py` for the pattern.

4. **Edge cases.** Empty inputs, size-1, sizes not divisible by SIMD
   lane width, maximum realistic sizes. The suite already has oracle
   fixtures for these sizes — if you're touching a kernel, your test
   needs to hit the same grid.

5. **Don't break the test suite.** `pytest` must still pass after
   your change. Current suite: 442 tests (run `pytest tests/ -q` to
   see the full count on your machine).

6. **Borrow, don't reinvent.** If there's prior art (another
   library's API, an existing paper's algorithm), use it and credit
   it. Before designing a new public API, check the Cerebras,
   torchao, and rigl-torch projects for what's already converged on.
   We prioritize adopting hardened community patterns over inventing
   new ones — it's how a stable scaffolding library gets built.

7. **Honest numbers.** If your PR claims a speedup, include the
   before/after numbers from a reproducible benchmark in the PR
   description. "It should be faster" is not a measurement.
   Estimated and hand-waved numbers will be sent back.

### How PR reviews flow

- Small fixes (docs, typos, single-line bugfixes): same-day review typical.
- Regular PRs: review within a few days.
- Large PRs (new kernel, new algorithm, new module): may take a week
  or more if the design needs discussion. Feel free to ping if a
  week has passed with no response.

We will sometimes request changes. That's not a rejection — it's how
PRs land cleanly. If a review feels confusing, ask for specifics
rather than guessing what we meant.

---

## Development setup

```bash
git clone https://github.com/DarshanFofadiya/sparselab.git
cd sparselab

# macOS only (for OpenMP):
brew install libomp

# Install in editable mode with dev deps
pip install -e '.[dev]'

# Run the test suite
pytest
# Expected: 372 passed in ~3s
```

The editable install rebuilds the C++ kernels whenever you touch a
file in `csrc/`. First build takes ~45 seconds.

### Running demos

```bash
pip install -e '.[demos]'   # adds matplotlib, torchvision
python examples/demo_05_mnist.py    # or any other demo_XX file
```

---

## What kinds of contributions we're actively looking for

- **New DST algorithms as plugins.** Sparse Momentum, Top-KAST, GraNet,
  and any newer paper you want to reproduce. `sparselab.SparsityAlgorithm`
  is ~100 lines; a new algorithm is another ~100. See `router.py`.
- **CPU kernel performance.** The `dW` kernel is the current
  bottleneck and is not yet NEON-vectorized. AVX-512 for x86_64 is
  also open. See issues [#1](https://github.com/DarshanFofadiya/sparselab/issues/1)
  and [#2](https://github.com/DarshanFofadiya/sparselab/issues/2).
- **CUDA port.** v0.1 is CPU-only on principle. The layout is
  GPU-friendly; a CUDA port is [issue #3](https://github.com/DarshanFofadiya/sparselab/issues/3).
- **Scale validation.** We validated the memory ratio at 40M params
  in [`milestone_11.md`](docs/demos/milestone_11.md). A 100M+
  independent reproduction would be the strongest corroboration.
  See [issue #10](https://github.com/DarshanFofadiya/sparselab/issues/10).
- **Good-first-issue work.** Documentation, new demos, CI cleanups.

## What kinds of PRs we're likely to push back on

- Large refactors without a clear motivation ("cleaner" is not enough).
- New top-level APIs without an issue first discussing the design.
- Benchmark claims without reproducible before/after numbers.
- AI-generated PRs that don't compile or don't pass tests. We can
  tell. The most obvious sign is the PR description not engaging
  with any specific file or number from the repo.
- Changes to existing public API names. Those names are community
  commitments — renaming `SparseLinear` to `SparseLinearLayer`
  breaks every user's import statement.

---

## Code style

- Python: follow PEP 8 where reasonable. We use 4-space indent,
  double quotes for strings in library code, f-strings over
  `.format()`. The codebase doesn't have a strict formatter; match
  the surrounding file's style.
- C++: C++17, clang-format default, 4-space indent, snake_case for
  functions and variables. See `csrc/kernels/*.cpp` for examples.
- Naming: class names are `PascalCase`, module-level functions are
  `snake_case`, constants are `UPPER_SNAKE_CASE`. Private helpers
  start with `_`.
- Docstrings: Google style. Every public function gets one.

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you're agreeing to it. In short: be direct, be kind,
cite your work, credit the community.

## Licensing

By contributing to SparseLab, you agree that your contributions will
be licensed under the [MIT License](LICENSE), same as the rest of
the project.

---

Questions that don't fit in an issue? Email the maintainer:
[darshanfofadiya@gmail.com](mailto:darshanfofadiya@gmail.com).

Thanks for being here.
