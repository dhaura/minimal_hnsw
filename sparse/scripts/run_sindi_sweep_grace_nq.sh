#!/bin/bash
#SBATCH --job-name=sindi_nq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#
# SINDI baseline on Grace for nq_splade.  Grace port of run_sindi_sweep.sh with
# the nq_splade grid from the Perlmutter run: doc_prune x query_prune x n_cand
# = 10 x 7 x 6 = 420 points.  No graph build; SINDI indexes per (dpr,qpr).
#
#   cd sparse/scripts && mkdir -p logs && sbatch run_sindi_sweep_grace_nq.sh

set -euo pipefail
PG=/scratch/user/dhaura/repos/spknn-playground
cd /scratch/user/dhaura/repos/minimal_hnsw/sparse/scripts
mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

export SPKNN_DATASET=${SPKNN_DATASET:-nq_splade}
source "$PG/common/bench_env.sh"
source "$PG/common/dataset_env_grace.sh"
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sindi_sweep"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/sindi}
mkdir -p "$OUT"
CSV="$OUT/sindi_results_${SPKNN_DATASET}_${SLURM_JOB_ID}.csv"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS"
echo "### csv     : $CSV"

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sindi_sweep" \
  "${DPR:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}" \
  "${QPR:-0.0,0.1,0.2,0.3,0.5,0.7,0.9}" \
  "${NCAND:-10,20,50,200,1000,5000}" \
  "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" "$CSV" \
  "${TPR:-0}" "${WINDOW:-50000}" "${REORDER:-1}" "${QUANT:-0}" \
  SINDI "${REPEATS:-3}" "${WARMUP:-1}"
echo "########## done -> $CSV ##########"
