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
bench_assert_optimized "$SPKNN_BIN/sindi_sweep"

OUT=${SPKNN_OUT:-$SCRATCH/datasets/SpKNN/sindi}
mkdir -p "$OUT"

CSV="$OUT/sindi_results${TAG:+_$TAG}.csv"
rm -f "$CSV"

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sindi_sweep" \
  "${DPR:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7}" \
  "${QPR:-0.5,0.4,0.3,0.2,0.1,0.0}" \
  "${NCAND:-10,20,50}" \
  "$SPKNN_DATA/base_full.csr" \
  "$SPKNN_DATA/queries.dev.csr" \
  "$SPKNN_DATA/base_full.dev.gt" \
  "$CSV" \
  "${TPR:-0}" "${WINDOW:-50000}" "${REORDER:-1}" "${QUANT:-0}" \
  SINDI "${REPEATS:-5}" "${WARMUP:-1}"
