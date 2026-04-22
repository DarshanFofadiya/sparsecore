# Testing SparseCore on SageMaker (Linux x86_64 and Graviton aarch64)

This is the heavy-duty fresh-machine test — run before tagging a
release if we want to be paranoid about the Linux wheels working on
real AWS infrastructure (not just Docker / GitHub Actions runners).

**Most of the time this is overkill.** Prefer the Docker-based test
in `scripts/test_fresh_install.sh` and the Colab notebook for
day-to-day verification. Reach for SageMaker when:

- You're about to tag a release and want a final "yes, it works on
  real AWS" sanity check
- You want to verify the Linux aarch64 (Graviton) wheel on a native
  Graviton instance, not QEMU emulation
- A user has reported a bug that only reproduces on SageMaker

## Quick-start: SageMaker Studio Lab (free)

[Studio Lab](https://studiolab.sagemaker.aws/) gives a free CPU
runtime that behaves like SageMaker Studio. Best for a quick sanity
check without an AWS bill.

1. Sign in at https://studiolab.sagemaker.aws/ (one-time account
   approval required, usually fast).
2. Start a new CPU runtime.
3. Open a terminal.
4. Run:
   ```bash
   pip install sparsecore
   curl -sSL https://raw.githubusercontent.com/DarshanFofadiya/sparsecore/main/scripts/smoke_test.py \
     -o smoke_test.py
   python smoke_test.py
   ```
5. Expected output: the 13-check smoke test all green, final line
   `✓ All smoke tests passed.`

Studio Lab runtimes are x86_64 Linux with Python 3.9/3.10 typically —
exercise our Linux x86_64 wheel path.

## SageMaker Studio notebook (paid, for real testing)

If you have an AWS account and want to test on a specific instance
type (e.g., Graviton for the aarch64 wheel):

### For Linux x86_64 wheel

1. Create a SageMaker domain if you don't have one:
   - AWS console → SageMaker → Domains → Create domain → quick
     setup
2. Launch Studio and create a new Python 3 notebook.
3. Instance type: `ml.t3.medium` ($0.05/hr) is plenty for a smoke
   test. Any x86_64 CPU instance works.
4. Kernel: `conda_python3` (Python 3.11-ish, x86_64).
5. In a notebook cell:
   ```python
   !pip install sparsecore
   import urllib.request
   urllib.request.urlretrieve(
       "https://raw.githubusercontent.com/DarshanFofadiya/sparsecore/main/scripts/smoke_test.py",
       "smoke_test.py",
   )
   !python smoke_test.py
   ```
6. When done, stop the instance in the Studio UI (billing is hourly).

### For Linux aarch64 wheel (Graviton)

1. Same as above, but choose `ml.m7g.medium` or `ml.c7g.medium` as
   the instance type. These are Graviton3 / aarch64 Linux.
2. The environment uses the same JupyterLab interface; kernel
   detection should pick up Python 3.11 on aarch64.
3. Same smoke-test commands as x86_64.
4. This exercises the NEON kernel path on a genuine Graviton chip
   (vs QEMU emulation locally).

**Why bother?** NEON on Apple Silicon and NEON on Graviton are the
same ISA in theory but different silicon. A Graviton test catches
any Apple-specific compile flag we accidentally depended on (such
as `-mcpu=apple-m1` — we gate that behind `IS_APPLE_SILICON` in
`setup.py`, but a real Graviton test confirms it).

## Cost estimate

Studio Lab: free. Full SageMaker:

| Instance | Arch | Hourly | Test duration |
|-|-|-|-|
| `ml.t3.medium` | x86_64 | $0.05 | ~5 min → ~$0.005 |
| `ml.m7g.medium` | aarch64 | $0.05 | ~5 min → ~$0.005 |

Single-digit cents per run. The dominant cost is your time, not
AWS's.

## Troubleshooting

**"No matching distribution found for sparsecore"**
- Check you're on a supported platform (Linux x86_64 or aarch64).
- If you're on something else, the install falls back to source and
  needs a compiler: `pip install build-essential` (Debian) or
  equivalent.

**"Import error: libomp.so.5 not found"**
- Shouldn't happen with our wheels since we bundle libomp, but if it
  does: `apt-get install libgomp1` (which provides GNU OpenMP
  runtime) will unblock.

**Smoke test passes but MNIST demo is very slow**
- Studio Lab CPUs are shared and throttled. Don't use them for
  benchmarking — use them for correctness testing only.

## When to prefer this over Docker

Prefer the Docker test script (`scripts/test_fresh_install.sh`) for:
- Fast iteration during release prep (seconds to spin up).
- Reproducibility — contributors don't need AWS accounts.

Prefer SageMaker for:
- Graviton testing without QEMU emulation overhead.
- "Does this work on the same infrastructure our customers actually
  run on?" sanity checks.
- Final pre-tag verification on a v0.1.0 release candidate.
