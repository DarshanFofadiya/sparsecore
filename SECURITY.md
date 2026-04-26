# Security Policy

SparseLab is a machine learning library. The most realistic security
concerns for a library like this are supply-chain issues (compromised
dependencies, typosquatted packages, malicious PRs) and bugs that
cause incorrect gradient computations, which matter for correctness
rather than confidentiality but are still worth reporting
responsibly.

## Supported versions

Only the latest published version on PyPI receives security updates.

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Email the maintainer directly:
[darshanfofadiya@gmail.com](mailto:darshanfofadiya@gmail.com)

Include:

- A description of the issue
- Steps to reproduce (a minimum code sample is ideal)
- The version of SparseLab, Python, and PyTorch
- Your assessment of impact and severity

### Response time

- Acknowledgement of receipt: within 3 business days
- Initial triage and severity assessment: within a week
- Fix or public disclosure timeline: depends on severity

For critical issues (supply-chain compromise, remote code execution
via malicious model loading, etc.) expect a faster turnaround. For
lower-severity issues we may coordinate a public disclosure date
that gives dependent projects time to update.

## What we consider in scope

- Supply-chain issues in the SparseLab wheels (tampered binaries,
  malicious code committed to the repo)
- Memory-safety bugs in the C++ kernels (out-of-bounds reads/writes
  triggered by untrusted input tensor shapes)
- Bugs that cause `SparseLinear.forward` or backward to produce
  mathematically incorrect results with valid inputs (the kernels
  are not yet audited for correctness under adversarial tensor
  contents)
- Issues with the PyPI publishing pipeline (compromised tokens,
  etc.)

## What we consider out of scope

- Denial of service via oversized tensors (you can always crash a
  process by allocating too much memory; that's a resource issue,
  not a vulnerability)
- Issues in dependencies (report those to the upstream project —
  PyTorch, OpenMP, etc.)
- Social engineering that doesn't involve a SparseLab-specific
  vector (generic phishing against maintainer accounts, etc.)

## Hall of thanks

Contributors who have reported valid security issues will be
credited here (with their permission) after the fix lands.

_None yet._

---

Thanks for helping keep SparseLab and its users safe.
