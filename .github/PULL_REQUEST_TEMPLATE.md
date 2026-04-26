<!--
Thanks for opening a PR. A few things to know before you submit:

1. Small PRs land fastest. <50 lines of real logic is ideal; larger is fine for bulk work.
2. Every math kernel needs an oracle test at 1e-5 tolerance.
3. If you claim a speedup, include reproducible before/after numbers.
4. The full 372-test suite must pass. Run `pytest` locally.
5. See CONTRIBUTING.md for the full review rubric.
-->

## What this PR does

<!-- One-paragraph summary of the change. -->

## Why

<!--
The motivation. What problem does this solve? Link the issue it
closes if there is one.
-->

Closes #

## How it works

<!--
A short technical description. Point at the key files changed and
explain the non-obvious design choices. Two or three paragraphs
is enough for most PRs.
-->

## Testing

<!--
What did you verify? Paste the key test output.
-->

```
<paste `pytest` output or a relevant subset>
```

- [ ] Full test suite passes locally (`pytest`)
- [ ] New tests added for new functionality (if applicable)
- [ ] Oracle tests added for new math kernels (if applicable)

## Benchmarks

<!--
If this PR affects performance, paste the measured before/after
numbers from a reproducible benchmark. "It should be faster" is
not a measurement.

Delete this section if the PR is not performance-related.
-->

```
Before: <numbers>
After:  <numbers>
Benchmark: <command to reproduce>
```

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] Public API changes (if any) are documented with updated docstrings
- [ ] Comments explain the why, not just the what, for non-obvious code
- [ ] If this adds a new DST algorithm, the paper is cited in the class docstring
- [ ] If this borrows API shape from another project, the inspiration is credited

## Additional notes

<!-- Anything else reviewers should know — caveats, open questions, follow-up PRs planned. -->
