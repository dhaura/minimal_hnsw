#!/bin/bash
#SBATCH --job-name=hnsw_sweep
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source /scratch/user/dhaura/repos/spknn-playground/common/bench_env.sh
bench_provenance

bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

OUT=/scratch/user/dhaura/datasets/SpKNN/sparse_hnsw
mkdir -p "$OUT"

ALPHA="${ALPHA:-0.8}"
BETA="${BETA:-3}"

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
  "${M:-16}" "${EFC:-200}" \
  "${EF_LIST:-10,20,50,100,200,400,800,1600,3200}" \
  1 0 0 "$ALPHA" "$BETA" \
  "$SPKNN_DATA/base_full.csr" \
  "$SPKNN_DATA/queries.dev.csr" \
  "$SPKNN_DATA/base_full.dev.gt" \
  "$OUT/sparse_hnsw_results.csv" \
  SparseHNSW "${REPEATS:-5}" "${WARMUP:-1}"
