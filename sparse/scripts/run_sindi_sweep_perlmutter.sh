#!/bin/bash
#SBATCH --job-name=sindi_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 physical cores x 2 hyperthreads -> BENCH_THREADS=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

module load python/3.11-24.1.0

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance

SINDI_BIN=${SINDI_BIN:-$SPKNN_HNSW_REPO/build-gnu/bin/sindi_sweep}
bench_assert_optimized "$SINDI_BIN"
if ldd "$SINDI_BIN" | grep -q libiomp5; then
    echo "FATAL $SINDI_BIN links libiomp5 as well as libgomp; its search" >&2
    echo "      throughput under OMP_PROC_BIND is not trustworthy. Rebuild it" >&2
    echo "      in build-gnu (cmake -DCMAKE_CXX_COMPILER=g++-14)." >&2
    exit 1
fi
echo "### sindi binary: $SINDI_BIN"
ldd "$SINDI_BIN" | grep -E 'omp|vsag' | sed 's/^/###   /'

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/sindi}
mkdir -p "$OUT"

CSV="$OUT/sindi_results${TAG:+_$TAG}.csv"
rm -f "$CSV"

$BENCH_LAUNCH stdbuf -oL -eL "$SINDI_BIN" \
  "${DPR:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}" \
  "${QPR:-0.0,0.1,0.2,0.3,0.5,0.7,0.9}" \
  "${NCAND:-10,20,50,200,1000,5000}" \
  "$SPKNN_BASE" \
  "$SPKNN_QUERIES" \
  "$SPKNN_GT" \
  "$CSV" \
  "${TPR:-0}" "${WINDOW:-50000}" "${REORDER:-1}" "${QUANT:-0}" \
  SINDI "${REPEATS:-5}" "${WARMUP:-1}"
