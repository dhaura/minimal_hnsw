#!/bin/bash
#SBATCH --job-name=sindi_sweep
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
bench_assert_optimized "$SPKNN_BIN/sindi_sweep"

OUT=/scratch/user/dhaura/datasets/SpKNN/sindi
mkdir -p "$OUT"

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sindi_sweep" \
  "${DPR:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7}" \
  "${QPR:-0.5,0.4,0.3,0.2,0.1,0.0}" \
  "${NCAND:-10,20,50}" \
  "$SPKNN_DATA/base_full.csr" \
  "$SPKNN_DATA/queries.dev.csr" \
  "$SPKNN_DATA/base_full.dev.gt" \
  "$OUT/sindi_results.csv" \
  "${TPR:-0}" "${WINDOW:-50000}" "${REORDER:-1}" "${QUANT:-0}" \
  SINDI "${REPEATS:-5}" "${WARMUP:-1}"
