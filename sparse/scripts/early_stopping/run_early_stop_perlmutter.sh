#!/bin/bash
#SBATCH --job-name=early_stop
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Early-stopping study on one shared msmarco_full graph (M=32 efC=200
# alpha=0.85 beta=4 u8, 8x4 seeds). Each timing = 1 warm-up + median of 3.
# Binary must be built with icpx (PrgEnv-intel) in build/.
#
#   cd sparse/scripts/early_stopping
#   sbatch --export=ALL,STEP=step6_early_stop,FLAG=--work run_early_stop_perlmutter.sh
#
# FLAG=--work: per-query oracle stop vs fixed ef (50..1600) vs patience (step 6).
# FLAG=--patience-k: patience watching the best 10/20/40 results vs fixed ef (step 7).
#
# STEP names the results subfolder (single value; --export splits on commas).

set -euo pipefail
STEP=${STEP:?set STEP, e.g. --export=ALL,STEP=step6_early_stop}
FLAG=${FLAG:?set FLAG, e.g. --export=ALL,FLAG=--work}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/early_stopping"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance

BIN="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/sparse/sparse_bloom_sweep"
bench_assert_optimized "$BIN"

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/early_stopping/${STEP}/msmarco_full_icpx${FLAG#-}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET  ndocs=$SPKNN_NDOCS nnz=$SPKNN_NNZ"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### out     : $OUT"

$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$OUT/results.csv" "$FLAG"

echo "########## done -> $OUT ##########"
