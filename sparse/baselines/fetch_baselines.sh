#!/bin/bash
# Clone the external baseline sources at the commits behind the paper results and apply this repo's
# patches (patches/*.patch: int64 nnz counters for >2^32-nnz inputs, a GCC 14 OpenBLAS flag).
#
#   GrassRMA -> sparse/GrassRMA   (C++ grassRMA_demo + pybind11 sparse_hnswlib for the Python driver)
#   vsag     -> sparse/vsag       (SINDI; built separately with GCC, see README)
#   pyanns   -> sparse/pyanns     (pybind11, built into the venv by common/build_bench_venv_perlmutter.sh)
#
# SEISMIC and kANNolo are built from their PyPI source releases plus patches/<pkg>.patch by
# common/build_bench_venv_perlmutter.sh.
# An existing folder is never modified: the script only reports whether it matches the pinned
# commit and patch. Needs network (run on a login node).
#
#   bash sparse/baselines/fetch_baselines.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SPARSE="$(cd "$HERE/.." && pwd)"

# name | url | commit | folder under sparse/ | patch
SOURCES=(
    "GrassRMA|https://github.com/Leslie-Chung/GrassRMA.git|43ab48bd71239754b856ad05a648010b09e50b2d|GrassRMA|grassrma.patch"
    "vsag|https://github.com/antgroup/vsag.git|efdaf17a10e96cdb5222baf558d50dfacbdc672e|vsag|vsag.patch"
    "pyanns|https://github.com/hhy3/pyanns.git|cbb5c8189d280419def6b96bc1f781e0d24b77ff|pyanns|pyanns.patch"
)

status=0
for entry in "${SOURCES[@]}"; do
    IFS='|' read -r name url commit dir patch <<<"$entry"
    dst="$SPARSE/$dir"
    if [ ! -e "$dst" ]; then
        echo "== $name: cloning $url @ ${commit:0:7} -> $dst"
        git clone --quiet "$url" "$dst"
        git -C "$dst" checkout --quiet "$commit"
        git -C "$dst" apply "$HERE/patches/$patch"
        echo "   applied patches/$patch"
        continue
    fi
    head=$(git -C "$dst" rev-parse HEAD 2>/dev/null || echo none)
    if [ "$head" != "$commit" ]; then
        echo "== $name: WARNING $dst is at ${head:0:7}, expected ${commit:0:7}" >&2; status=1
    elif git -C "$dst" apply --reverse --check "$HERE/patches/$patch" 2>/dev/null; then
        echo "== $name: ok ($dst @ ${commit:0:7}, patches/$patch applied)"
    elif git -C "$dst" apply --check "$HERE/patches/$patch" 2>/dev/null; then
        echo "== $name: WARNING patches/$patch is not applied in $dst" >&2
        echo "   apply it with: git -C $dst apply $HERE/patches/$patch" >&2; status=1
    else
        echo "== $name: WARNING $dst has local changes that differ from patches/$patch" >&2; status=1
    fi
done
exit $status
