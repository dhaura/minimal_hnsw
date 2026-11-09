#!/bin/bash
# Build the baseline venv on NERSC Perlmutter (AMD Zen3): SEISMIC + kANNolo (Rust, PyPI), GrassRMA and
# PyANNS (pybind11, from the sources fetch_baselines.sh pins). Run on a login node (needs network).
# SPKNN_VENV picks the target (default in bench_env_perlmutter.sh); an existing venv there is replaced.

set -u
REPO=${SPKNN_HNSW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "$REPO"
source "$REPO/sparse/scripts/bench_env_perlmutter.sh"

ARCH="$SPKNN_ARCH"

module load python/3.11-24.1.0 2>/dev/null
module load rust/stable 2>/dev/null

export RUSTUP_HOME=$SCRATCH/.rustup
export CARGO_HOME=$SCRATCH/.cargo
export PATH="$CARGO_HOME/bin:$PATH"

echo "=================== BUILD PROVENANCE ==================="
echo "date   : $(date -Is)"
echo "host   : $(hostname)"
echo "cpu    : $(lscpu | sed -n 's/^Model name: *//p')"
echo "avx    : $(lscpu | grep -o -E 'avx[0-9a-z_]*' | sort -u | tr '\n' ' ')"
echo "python : $(python3 -V 2>&1)  ($(which python3))"
echo "gcc    : $(gcc --version | head -1)"
echo "rustc  : $(rustc --version 2>/dev/null)"
echo "arch   : $ARCH"
echo "venv   : $SPKNN_VENV"
echo "========================================================"

if ! timeout 30 curl -sSI https://pypi.org/simple/ >/dev/null 2>&1; then
    echo "FATAL: no outbound network from $(hostname)." >&2
    echo "Run this on a Perlmutter login node, not a compute node." >&2
    exit 1
fi

rm -rf "$SPKNN_VENV"
mkdir -p "$(dirname "$SPKNN_VENV")"
python3 -m venv "$SPKNN_VENV"
source "$SPKNN_VENV/bin/activate"
pip install -q pip==26.2 setuptools==83.0.0 wheel==0.47.0   # results venv versions

declare -A STATUS

step() {  # step <name> <command...>
    local name=$1; shift
    echo; echo "=== $name ==="
    if "$@"; then STATUS[$name]=OK; else STATUS[$name]=FAILED; fi
}

# Versions of the venv that produced the paper results (python 3.11.7).
step "base-stack" pip install -q numpy==2.4.6 scipy==1.17.1 pandas==3.0.5 matplotlib==3.11.1 pybind11==3.0.4

# --- kANNolo + SEISMIC (Rust) ---------------------------------------------
# The recipe behind the paper results (wheels built 2026-08-05): the PyPI source releases plus one
# patch each -- kANNolo 0.7.1 with lto = "fat"; SEISMIC 0.5.1 with a regenerated Cargo.lock, because
# its locked git dependency (fixed @ 02d24b04) no longer exists upstream -- built by maturin 1.14.1
# with nightly-2026-08-04 (rustc 1.99.0-nightly 504869653) and -C target-cpu=$ARCH.
build_rust_pkgs() {
    rustup toolchain install nightly-2026-08-04 --profile minimal || return 1
    export RUSTUP_TOOLCHAIN=nightly-2026-08-04     # overrides the packages' rust-toolchain.toml
    export RUSTFLAGS="-C target-cpu=$ARCH"
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"
    local work=${SPKNN_RUST_BUILD:-$SCRATCH/build/minimal_hnsw_rustpkgs}
    rm -rf "${work:?}"; mkdir -p "$work/wheels"
    echo "maturin==1.14.1" > "$work/build-constraints.txt"
    local pkg name ver url
    for pkg in kannolo-0.7.1 pyseismic_lsr-0.5.1; do
        name=${pkg%-*}; ver=${pkg##*-}
        url=$(curl -sSf "https://pypi.org/pypi/${name//_/-}/$ver/json" | python3 -c \
            "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])") || return 1
        curl -sSfL "$url" | tar -xz -C "$work" || return 1
        (cd "$work/$pkg" && patch -p1 < "$REPO/sparse/baselines/patches/$pkg.patch") || return 1
        pip wheel --no-deps --no-cache-dir --build-constraint "$work/build-constraints.txt" \
            -w "$work/wheels" "$work/$pkg" || return 1
    done
    pip install --no-deps --force-reinstall "$work"/wheels/*.whl
}
step "kannolo+seismic" build_rust_pkgs

# --- GrassRMA sparse_hnswlib (pybind11) ------------------------------------
build_grassrma() {
    cd "$SPKNN_GRASSRMA_SRC" || return 1
    # HNSWLIB_NO_NATIVE stops setup.py forcing -march=native; the flags apply to this step only.
    HNSWLIB_NO_NATIVE=1 CFLAGS="-march=$ARCH -mtune=$ARCH" CXXFLAGS="-march=$ARCH -mtune=$ARCH" \
        pip install --no-deps --no-cache-dir .
}
step "grassRMA" build_grassrma

# --- PyANNS (pybind11) -----------------------------------------------------
build_pyanns() {
    cd "$SPKNN_PYANNS_SRC/python" || return 1
    rm -rf build dist
    # setup.py's own flags (-Ofast -march=native), no CFLAGS: the build behind the paper results.
    env -u CFLAGS -u CXXFLAGS python setup.py bdist_wheel >/dev/null || return 1
    pip install --no-deps --force-reinstall --no-cache-dir dist/pyanns-*.whl
}
step "pyanns" build_pyanns

# --- report ----------------------------------------------------------------
echo
echo "=================== BUILD SUMMARY ======================"
for k in base-stack kannolo+seismic grassRMA pyanns; do
    printf '%-20s %s\n' "$k" "${STATUS[$k]:-SKIPPED}"
done
echo "--------------------------------------------------------"
echo "compiled artefacts (zmm refs must be 0 -- Zen3 has no AVX-512):"
SITE=$(python -c "import site;print(site.getsitepackages()[0])")
for so in $(find "$SITE" -maxdepth 2 -name "*.so" 2>/dev/null | head -20); do
    ymm=$(objdump -d --no-show-raw-insn "$so" 2>/dev/null | grep -coE '%ymm[0-9]+')
    zmm=$(objdump -d --no-show-raw-insn "$so" 2>/dev/null | grep -coE '%zmm[0-9]+')
    printf '  %-46s ymm=%-8s zmm=%s\n' "$(basename "$so")" "$ymm" "$zmm"
done
echo "========================================================"
