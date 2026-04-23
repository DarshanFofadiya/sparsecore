#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# test_fresh_install.sh — verify sparselab installs + works on a
# genuinely clean machine via Docker.
#
# This is the primary gate we hold ourselves to before tagging v0.1:
# if this script fails, the wheels aren't ready for PyPI.
#
# What it does:
#   1. Pulls a clean python:3.11-slim image (no torch, no compilers).
#   2. Installs sparselab from the wheel we built locally (or from
#      TestPyPI once we've pushed there).
#   3. Runs the smoke test inside the container.
#
# Usage:
#   # Test a locally-built wheel
#   ./scripts/test_fresh_install.sh wheelhouse/sparselab-0.1.0-*.whl
#
#   # Test from TestPyPI
#   ./scripts/test_fresh_install.sh --testpypi
#
#   # Test from the real PyPI (post-launch sanity check)
#   ./scripts/test_fresh_install.sh --pypi
#
#   # Test aarch64 specifically (slower, uses QEMU)
#   ./scripts/test_fresh_install.sh --arch aarch64 wheelhouse/...
#
# Platforms:
#   Default: linux/amd64 (native on Intel, emulated on Apple Silicon via Docker Desktop)
#   --arch aarch64: linux/arm64 (native on Apple Silicon, emulated on Intel)
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON_VERSION="3.11"
PLATFORM="linux/amd64"
SOURCE=""
WHEEL_FILE=""

# ─── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            case "$2" in
                aarch64|arm64) PLATFORM="linux/arm64" ;;
                x86_64|amd64)  PLATFORM="linux/amd64" ;;
                *) echo "Unknown arch: $2"; exit 1 ;;
            esac
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --testpypi)
            SOURCE="testpypi"
            shift
            ;;
        --pypi)
            SOURCE="pypi"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--arch aarch64] [--python 3.12] (WHEEL_FILE | --testpypi | --pypi)"
            exit 0
            ;;
        *)
            WHEEL_FILE="$1"
            SOURCE="wheel"
            shift
            ;;
    esac
done

if [[ -z "$SOURCE" ]]; then
    echo "ERROR: provide either a wheel file, --testpypi, or --pypi"
    echo "Run with --help for usage."
    exit 1
fi

# ─── Construct the pip install command that will run inside Docker ──
case "$SOURCE" in
    wheel)
        if [[ ! -f "$WHEEL_FILE" ]]; then
            echo "ERROR: wheel file not found: $WHEEL_FILE"
            exit 1
        fi
        WHEEL_BASENAME=$(basename "$WHEEL_FILE")
        PIP_CMD="pip install /wheels/${WHEEL_BASENAME}"
        MOUNT_ARGS=(-v "$(cd "$(dirname "$WHEEL_FILE")" && pwd):/wheels:ro")
        ;;
    testpypi)
        PIP_CMD="pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sparselab"
        MOUNT_ARGS=()
        ;;
    pypi)
        PIP_CMD="pip install sparselab"
        MOUNT_ARGS=()
        ;;
esac

# ─── Mount the smoke test so we can run it inside ──────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOUNT_ARGS+=(-v "${REPO_ROOT}/scripts/smoke_test.py:/smoke_test.py:ro")

# ─── Run the test ─────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo " Fresh-install test"
echo "   Platform:   $PLATFORM"
echo "   Python:     $PYTHON_VERSION"
echo "   Source:     $SOURCE"
if [[ -n "$WHEEL_FILE" ]]; then
    echo "   Wheel:      $WHEEL_FILE"
fi
echo "════════════════════════════════════════════════════════════════"

docker run --rm --platform "$PLATFORM" "${MOUNT_ARGS[@]}" \
    "python:${PYTHON_VERSION}-slim" \
    bash -c "
        set -e
        echo '── Container Python version ──'
        python --version
        echo
        echo '── Installing sparselab ──'
        $PIP_CMD
        echo
        echo '── Installed versions ──'
        python -c 'import sparselab, torch; print(f\"sparselab {sparselab.__version__}\"); print(f\"torch {torch.__version__}\")'
        echo
        echo '── Running smoke test ──'
        python /smoke_test.py
    "

echo
echo "✓ Fresh-install test PASSED on $PLATFORM / Python $PYTHON_VERSION"
