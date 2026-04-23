#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# repair_wheel_macos.sh — minimal wheel repair for SparseCore macOS
#
# Called by cibuildwheel's repair-wheel-command on macOS runners.
#
# Why this script instead of `delocate-wheel`:
#   1. We don't want to bundle libomp.dylib into the wheel (see
#      [tool.cibuildwheel.macos] in pyproject.toml for full rationale).
#      In short: torch ships its own libomp, our wheel bundling a
#      second copy would cause OpenMP's runtime to abort the process.
#   2. delocate won't even analyze a wheel whose dependencies can't be
#      resolved — so we can't just call `delocate-wheel --exclude
#      libomp.dylib` because the build-time libomp path doesn't exist
#      on the repair machine (it was torch's install path from a
#      different environment). Observed locally as
#         delocate.libsana.DelocationError:
#           /opt/llvm-openmp/lib/libomp.dylib not found
#   3. We only have one delocatable dep (libomp) and we know exactly
#      what to do with it — rewrite its load path to @rpath/libomp.dylib
#      and add a rpath entry pointing at torch's libomp on the user's
#      machine. That's ~20 lines of otool + install_name_tool, simpler
#      than working around delocate.
#
# At runtime: sparsecore/__init__.py imports torch first, which loads
# torch's libomp.dylib. When _core.so loads, its @rpath/libomp.dylib
# reference resolves to the already-loaded torch libomp. One copy in
# the process. No collision.
#
# Usage (invoked by cibuildwheel):
#   repair_wheel_macos.sh <wheel> <dest_dir> <archs>
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# Resolve to absolute paths BEFORE we cd anywhere, otherwise relative
# paths passed by the caller (or by cibuildwheel) break once we move
# into the temp workdir.
WHEEL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
DEST_DIR="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
ARCHS="$3"  # currently unused; kept in signature for parity with delocate-wheel

echo "── repair_wheel_macos.sh"
echo "   wheel:    $WHEEL"
echo "   dest_dir: $DEST_DIR"
echo "   archs:    $ARCHS"

mkdir -p "$DEST_DIR"
WHEEL_BASENAME="$(basename "$WHEEL")"

# Unpack wheel into a temp dir. Wheels are zip archives.
WORKDIR="$(mktemp -d)"
echo "── Step 1: unpack $WHEEL_BASENAME to $WORKDIR"
cd "$WORKDIR"
unzip -q "$WHEEL"

# Find the .so file. There should be exactly one per wheel, but glob in
# case future builds produce more.
SO_FILE=$(find sparsecore -name "_core*.so" | head -n 1)
if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: could not find sparsecore/_core*.so in the wheel"
    exit 1
fi
echo "── Step 2: rewriting install names in $SO_FILE"

# Print current state for debugging CI failures.
echo "   BEFORE:"
otool -L "$SO_FILE" | sed 's/^/     /'
otool -l "$SO_FILE" | awk '/cmd LC_RPATH/{getline; getline; print}' | sed 's/^/     rpath:/' || true

# Step 2a: find any load command referencing libomp.
# On the CI runner, the build-time path is
#   /Users/runner/.../site-packages/torch/lib/libomp.dylib  (torch's)
# or
#   /opt/homebrew/opt/libomp/lib/libomp.dylib  (Homebrew's)
# We don't hardcode either — instead we grep otool output and rewrite
# whatever we find to the portable @rpath form.
OLD_LIBOMP_PATH="$(otool -L "$SO_FILE" | grep -E 'libomp\.dylib' | awk '{print $1}' | head -n 1)"
if [[ -n "$OLD_LIBOMP_PATH" && "$OLD_LIBOMP_PATH" != "@rpath/libomp.dylib" ]]; then
    echo "   rewriting libomp install name: $OLD_LIBOMP_PATH → @rpath/libomp.dylib"
    install_name_tool -change "$OLD_LIBOMP_PATH" "@rpath/libomp.dylib" "$SO_FILE"
else
    echo "   libomp install name already @rpath form or absent"
fi

# Step 2b: add the rpath that lets the dynamic loader find torch's
# libomp at runtime.
#   sparsecore/_core.so lives at <site-packages>/sparsecore/_core.so
#   torch's libomp lives at <site-packages>/torch/lib/libomp.dylib
#   So @loader_path/../torch/lib resolves correctly.
add_rpath_if_missing() {
    local rpath="$1"
    if otool -l "$SO_FILE" | grep -q "path $rpath "; then
        echo "   rpath already present: $rpath"
    else
        echo "   adding rpath: $rpath"
        install_name_tool -add_rpath "$rpath" "$SO_FILE"
    fi
}
add_rpath_if_missing "@loader_path/../torch/lib"

# Step 2c: remove any absolute rpaths pointing at build-time paths
# that won't exist on the user's machine. The loader silently ignores
# missing rpaths, but cleaning them up keeps `otool -l` output
# diagnostic-friendly for users who inspect our wheel.
for rpath_candidate in \
    "/opt/homebrew/opt/libomp/lib" \
    "/usr/local/opt/libomp/lib" \
    "/opt/llvm-openmp/lib" \
; do
    if otool -l "$SO_FILE" | grep -q "path $rpath_candidate "; then
        echo "   removing build-time rpath: $rpath_candidate"
        install_name_tool -delete_rpath "$rpath_candidate" "$SO_FILE" || true
    fi
done

# Also remove any rpath that points into a specific torch install path
# (those baked-in absolute /Users/runner/... or /private/var/folders/...
# paths won't exist on user machines either). We match any rpath
# containing "/torch/lib" that isn't the portable @loader_path form.
while read -r absolute_torch_rpath; do
    if [[ -n "$absolute_torch_rpath" && "$absolute_torch_rpath" != "@loader_path/../torch/lib" ]]; then
        echo "   removing absolute torch rpath: $absolute_torch_rpath"
        install_name_tool -delete_rpath "$absolute_torch_rpath" "$SO_FILE" || true
    fi
done < <(
    otool -l "$SO_FILE" \
        | awk '/cmd LC_RPATH/{getline; getline; sub(/^[ \t]*path /,""); sub(/ \(offset.*$/,""); print}' \
        | grep "/torch/lib" || true
)

echo "   AFTER:"
otool -L "$SO_FILE" | sed 's/^/     /'
otool -l "$SO_FILE" | awk '/cmd LC_RPATH/{getline; getline; print}' | sed 's/^/     rpath:/' || true

# Step 3: re-zip back into a wheel. The wheel format is just a zip
# archive with a specific file layout (we preserve the layout here).
echo "── Step 3: repacking into $DEST_DIR/$WHEEL_BASENAME"
zip -r -q "$DEST_DIR/$WHEEL_BASENAME.tmp" .
mv -f "$DEST_DIR/$WHEEL_BASENAME.tmp" "$DEST_DIR/$WHEEL_BASENAME"

cd - > /dev/null
rm -rf "$WORKDIR"

echo "── repair_wheel_macos.sh: done"
