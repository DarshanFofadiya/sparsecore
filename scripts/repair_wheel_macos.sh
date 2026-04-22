#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# repair_wheel_macos.sh — delocate-wheel replacement for SparseCore macOS
#
# Called by cibuildwheel's repair-wheel-command on macOS runners.
#
# Why this script instead of a one-liner delocate call:
#
#  We deliberately do NOT bundle libomp.dylib into our wheel. See the
#  [tool.cibuildwheel.macos] section in pyproject.toml for the full
#  rationale. Short version: bundling would create a second libomp in
#  the process alongside torch's libomp, and OpenMP's runtime aborts
#  when it detects duplicates.
#
#  Instead, we rely on PyTorch's libomp (torch ships its own inside its
#  wheel) which is always loaded first because sparsecore/__init__.py
#  does `import torch` before loading our _core.so. For this to work
#  at runtime on an arbitrary user's machine:
#
#   1. Our _core.so's libomp load path must be a name the dynamic
#      loader will match against the already-loaded torch libomp —
#      specifically, @rpath/libomp.dylib.
#
#   2. Our _core.so's rpath list must include a path relative to
#      @loader_path that resolves to the user's torch/lib directory.
#      sparsecore/_core.so lives at
#        <site-packages>/sparsecore/_core.so
#      and torch's libomp lives at
#        <site-packages>/torch/lib/libomp.dylib
#      So @loader_path/../torch/lib is the right rpath.
#
# Usage (invoked by cibuildwheel):
#   repair_wheel_macos.sh <wheel> <dest_dir> <archs>
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

WHEEL="$1"
DEST_DIR="$2"
ARCHS="$3"

echo "── repair_wheel_macos.sh"
echo "   wheel:    $WHEEL"
echo "   dest_dir: $DEST_DIR"
echo "   archs:    $ARCHS"

# Step 1: run delocate with --exclude libomp.dylib. This means delocate
# will sanity-check the wheel and copy any OTHER delocatable deps
# (if we ever add another lib dependency), but leave libomp alone.
# It also preserves the original (non-rewritten) install name for
# libomp in our _core.so.
echo "── Step 1: delocate-wheel with --exclude libomp.dylib"
delocate-wheel \
    --exclude libomp.dylib \
    --require-archs "$ARCHS" \
    -w "$DEST_DIR" \
    -v \
    "$WHEEL"

# Find the repaired wheel in the dest dir (there may be more than one
# iteration of the same name if we're debugging; take the latest).
REPAIRED_WHEEL="$(ls -t "$DEST_DIR"/*.whl | head -n 1)"
echo "── Step 2: repaired wheel is $REPAIRED_WHEEL"

# Step 2: unpack, rewrite, repack. We need to change the libomp
# install name in _core.so from an absolute path (the build-time
# /opt/homebrew/opt/libomp/lib/libomp.dylib) to @rpath/libomp.dylib.
# We also add the rpath entry that lets the dynamic loader find
# torch's libomp at runtime.
WORKDIR="$(mktemp -d)"
echo "── Step 3: unpack to $WORKDIR"
cd "$WORKDIR"
unzip -q "$REPAIRED_WHEEL"

# Find the .so file. Only one per wheel, but glob in case naming changes.
SO_FILE=$(find sparsecore -name "_core*.so" | head -n 1)
if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: could not find sparsecore/_core*.so in the wheel"
    exit 1
fi
echo "── Step 4: rewriting install names in $SO_FILE"

# Print current state for debugging.
echo "   BEFORE:"
otool -L "$SO_FILE" | sed 's/^/     /'
otool -l "$SO_FILE" | grep -A2 LC_RPATH | sed 's/^/     /' || true

# Find any load command referencing libomp and rewrite it. The original
# path is whatever setup.py's linker step produced — typically
# /opt/homebrew/opt/libomp/lib/libomp.dylib but could vary. We use
# otool -L to discover it rather than hardcoding.
OLD_LIBOMP_PATH="$(otool -L "$SO_FILE" | grep -E 'libomp\.dylib' | awk '{print $1}' | head -n 1)"
if [[ -n "$OLD_LIBOMP_PATH" && "$OLD_LIBOMP_PATH" != "@rpath/libomp.dylib" ]]; then
    echo "   rewriting libomp install name from $OLD_LIBOMP_PATH"
    install_name_tool -change "$OLD_LIBOMP_PATH" "@rpath/libomp.dylib" "$SO_FILE"
else
    echo "   libomp install name already correct or absent"
fi

# Add rpath entries the loader can use to find libomp at runtime:
#   @loader_path/../torch/lib  — the path to torch/lib/ from our .so,
#                                when both are installed as siblings
#                                under <site-packages>.
# Deduplication: install_name_tool -add_rpath fails if the rpath is
# already present, so we check first.
add_rpath_if_missing() {
    local rpath="$1"
    if otool -l "$SO_FILE" | grep -q "path $rpath "; then
        echo "   rpath $rpath already present"
    else
        echo "   adding rpath $rpath"
        install_name_tool -add_rpath "$rpath" "$SO_FILE"
    fi
}
add_rpath_if_missing "@loader_path/../torch/lib"

# Sanitize: remove any absolute rpaths that point at build-time paths
# (e.g. /opt/homebrew/opt/libomp/lib or /Users/runner/...). These
# won't exist on user machines and produce dyld warnings.
for rpath_candidate in \
    "/opt/homebrew/opt/libomp/lib" \
    "/usr/local/opt/libomp/lib" \
; do
    if otool -l "$SO_FILE" | grep -q "path $rpath_candidate "; then
        echo "   removing build-time rpath $rpath_candidate"
        install_name_tool -delete_rpath "$rpath_candidate" "$SO_FILE" || true
    fi
done

echo "   AFTER:"
otool -L "$SO_FILE" | sed 's/^/     /'
otool -l "$SO_FILE" | grep -A2 LC_RPATH | sed 's/^/     /' || true

# Step 5: re-zip back into a wheel.
WHEEL_NAME="$(basename "$REPAIRED_WHEEL")"
echo "── Step 5: repacking into $DEST_DIR/$WHEEL_NAME"
zip -r -q "$DEST_DIR/$WHEEL_NAME.tmp" .
mv -f "$DEST_DIR/$WHEEL_NAME.tmp" "$DEST_DIR/$WHEEL_NAME"

cd - > /dev/null
rm -rf "$WORKDIR"

echo "── repair_wheel_macos.sh: done"
