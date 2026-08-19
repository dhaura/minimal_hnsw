#!/bin/bash
#SBATCH --job-name=hnsw_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 physical cores x 2 hyperthreads -> BENCH_THREADS=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

module load python/3.11-24.1.0

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/sparse_hnsw}
mkdir -p "$OUT"

ALPHA=${ALPHA:-0.8}
BETA=${BETA:-3}
CSV="$OUT/sparse_hnsw_results${TAG:+_$TAG}.csv"
MODEL=${MODEL:-SparseHNSW}
rm -f "$CSV"          # writer appends; keep reruns idempotent
echo "alpha=$ALPHA beta=$BETA -> model='$MODEL' csv=$CSV"

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
  "${M:-32}" "${EFC:-200}" \
  "${EF_LIST:-10,20,50,100,200,400,800,1600,3200}" \
  1 0 0 "$ALPHA" "$BETA" \
  "$SPKNN_BASE" \
  "$SPKNN_QUERIES" \
  "$SPKNN_GT" \
  "$CSV" \
  "$MODEL" "${REPEATS:-5}" "${WARMUP:-1}"
