#!/bin/bash
# Shared benchmark environment -- NERSC Perlmutter.

# --- thread count ----------------------------------------------------------
# Perlmutter CPU nodes are 2 x AMD EPYC 7763: 128 physical cores / 256
# hyperthreads. SLURM counts hyperthreads, so --cpus-per-task is 2x threads.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    BENCH_THREADS="${BENCH_THREADS:-$(( SLURM_CPUS_PER_TASK / 2 ))}"
else
    BENCH_THREADS="${BENCH_THREADS:-128}"
fi

export BENCH_THREADS
export OMP_NUM_THREADS="$BENCH_THREADS"
export RAYON_NUM_THREADS="$BENCH_THREADS"   # Rust (kANNolo, SEISMIC)
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- pinning ---------------------------------------------------------------
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# --- NUMA ------------------------------------------------------------------
if command -v numactl >/dev/null 2>&1; then
    if [ "$BENCH_THREADS" -le 64 ]; then
        BENCH_LAUNCH="numactl --cpunodebind=0-3 --interleave=0-3"
    else
        BENCH_LAUNCH="numactl --interleave=all"
    fi
else
    BENCH_LAUNCH=""
    echo "bench_env: WARNING numactl not found; NUMA policy is NOT controlled." >&2
fi
export BENCH_LAUNCH

export PYTHONUNBUFFERED=1

# --- OpenMP runtime for the C++ binaries -----------------------------------
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/AMD/aocc-compiler-4.1.0/lib:${LD_LIBRARY_PATH:-}

# --- libstdc++ ABI ---------------------------------------------------------
if [ -f /usr/lib64/libstdc++.so.6 ]; then
    export LD_PRELOAD=/usr/lib64/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}
fi

# --- target ISA ------------------------------------------------------------
export SPKNN_ARCH=znver3

# --- canonical paths -------------------------------------------------------
# The repo root is two levels above this file (sparse/scripts/), wherever the repo is cloned.
export SPKNN_HNSW_REPO=${SPKNN_HNSW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export SPKNN_BIN=${SPKNN_BIN:-"$SPKNN_HNSW_REPO/build/bin"}
# Python venv for the baselines (built by sparse/baselines/common/build_bench_venv_perlmutter.sh).
export SPKNN_VENV=${SPKNN_VENV:-$SCRATCH/benchmarks/minimal_hnsw/bench-venv-perlmutter}

# --- dataset selection -----------------------------------------------------
export SPKNN_DATASET=${SPKNN_DATASET:-msmarco_full}
case "$SPKNN_DATASET" in
    msmarco_full)
        export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_full"
        export SPKNN_QSET="${SPKNN_QSET:-dev}"
        export SPKNN_BASE="$SPKNN_DATA/base_full.csr"
        export SPKNN_QUERIES="$SPKNN_DATA/queries.$SPKNN_QSET.csr"
        export SPKNN_GT="$SPKNN_DATA/base_full.$SPKNN_QSET.gt"
        export SPKNN_NDOCS=8841823          # 6980 queries, dim 30109
        export SPKNN_NNZ=1121199371
        ;;
    nq_splade)
        export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/nq_splade"
        export SPKNN_QSET="${SPKNN_QSET:-test}"
        export SPKNN_BASE="$SPKNN_DATA/base_nq.csr"
        export SPKNN_QUERIES="$SPKNN_DATA/queries.$SPKNN_QSET.csr"
        export SPKNN_GT="$SPKNN_DATA/base_nq.$SPKNN_QSET.gt"
        export SPKNN_NDOCS=2680893          # 3452 queries, dim 30522
        export SPKNN_NNZ=411886108
        ;;
    msmarco_v2_splade)
        export SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_v2_splade"
        export SPKNN_QSET="${SPKNN_QSET:-dev}"
        export SPKNN_BASE="$SPKNN_DATA/base_v2_splade.csr"
        export SPKNN_QUERIES="$SPKNN_DATA/queries.$SPKNN_QSET.csr"
        export SPKNN_GT="$SPKNN_DATA/base_v2_splade.$SPKNN_QSET.gt"
        export SPKNN_NDOCS=138364198        # dim 30522
        export SPKNN_NNZ=17479047748        # 4.07x uint32 -- see SPKNN_BIG below
        ;;
    *)
        echo "bench_env: unknown SPKNN_DATASET='$SPKNN_DATASET'" >&2
        echo "           known: msmarco_full, nq_splade, msmarco_v2_splade" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

# --- 2^32 nnz flag ---------------------------------------------------------
if [ "${SPKNN_NNZ:-0}" -gt 4294967295 ]; then
    export SPKNN_BIG=1
else
    export SPKNN_BIG=0
fi
export SPKNN_OUT_ROOT=${SPKNN_OUT_ROOT:-$SCRATCH/datasets/SpKNN/$SPKNN_DATASET}

for _f in "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT"; do
    [ -s "$_f" ] || { echo "bench_env: FATAL missing dataset file $_f" >&2
                      return 1 2>/dev/null || exit 1; }
done
unset _f
# Baseline sources, cloned at pinned commits by sparse/baselines/fetch_baselines.sh.
export SPKNN_PYANNS_SRC=$SPKNN_HNSW_REPO/sparse/pyanns
export SPKNN_VSAG_SRC=$SPKNN_HNSW_REPO/sparse/vsag
export SPKNN_GRASSRMA_SRC=$SPKNN_HNSW_REPO/sparse/GrassRMA

export SPKNN_MODULES="python/3.11-24.1.0"

bench_load_modules() {
    module load python/3.11-24.1.0
}

# --- provenance ------------------------------------------------------------
bench_provenance() {
    echo "=================== BENCH PROVENANCE ==================="
    echo "date          : $(date -Is)"
    echo "host          : $(hostname)"
    echo "slurm job     : ${SLURM_JOB_ID:-<none>}  partition=${SLURM_JOB_PARTITION:-<none>}"
    echo "nodelist      : ${SLURM_JOB_NODELIST:-<none>}"
    echo "cpus-per-task : ${SLURM_CPUS_PER_TASK:-<unset>}"
    echo "bench threads : $BENCH_THREADS"
    echo "launcher      : ${BENCH_LAUNCH:-<none>}"
    echo "target arch   : $SPKNN_ARCH"
    echo "cpu           : $(lscpu | sed -n 's/^Model name: *//p')"
    echo "sockets/cores : $(lscpu | sed -n 's/^Socket(s): *//p') sockets x $(lscpu | sed -n 's/^Core(s) per socket: *//p') cores"
    echo "avx           : $(lscpu | grep -o -E 'avx[0-9a-z_]*' | sort -u | tr '\n' ' ')"
    echo "-- numactl -H --"
    numactl -H 2>/dev/null | head -12
    echo "-- git --"
    for r in "$SPKNN_HNSW_REPO" "$SPKNN_GRASSRMA_SRC" "$SPKNN_PYANNS_SRC" "$SPKNN_VSAG_SRC"; do
        printf '%-28s %s %s\n' "$(basename "$r")" \
            "$(git -C "$r" rev-parse --short HEAD 2>/dev/null || echo '?')" \
            "$(git -C "$r" diff --quiet 2>/dev/null && echo clean || echo DIRTY)"
    done
    echo "-- modules --"
    (module list) 2>&1 | head -12
    echo "-- binaries --"
    for b in "$SPKNN_BIN"/*; do
        [ -f "$b" ] || continue
        printf '%-24s %s  %s\n' "$(basename "$b")" \
            "$(date -r "$b" '+%Y-%m-%d %H:%M')" "$(sha256sum "$b" | cut -c1-16)"
    done
    echo "========================================================"
}

bench_assert_optimized() {
    local bin="$1"
    local ymm zmm
    ymm=$(objdump -d --no-show-raw-insn "$bin" 2>/dev/null | grep -coE '%ymm[0-9]+' || true)
    zmm=$(objdump -d --no-show-raw-insn "$bin" 2>/dev/null | grep -coE '%zmm[0-9]+' || true)
    echo "bench_env: $(basename "$bin") vector-register refs: ymm=$ymm zmm=$zmm"
    if [ "${ymm:-0}" -eq 0 ]; then
        echo "bench_env: FATAL $bin contains no vector instructions -- it is an" >&2
        echo "           unoptimized build. Refusing to benchmark it." >&2
        exit 1
    fi
    if [ "${zmm:-0}" -gt 0 ]; then
        echo "bench_env: FATAL $bin references AVX-512 (%zmm) registers, which" >&2
        echo "           Zen3 cannot execute. This is a Grace binary or a" >&2
        echo "           -march=cascadelake build; it will SIGILL. Rebuild with" >&2
        echo "           -march=$SPKNN_ARCH." >&2
        exit 1
    fi
}
