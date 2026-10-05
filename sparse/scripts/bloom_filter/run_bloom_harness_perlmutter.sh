#!/bin/bash
#SBATCH --job-name=bloom_harness
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Bloom gate harness, rerun after each implementation step: one shared msmarco_full graph (M=32 efC=200
# alpha=0.85 beta=4 u8, 8x4 seeds), ef=1600. Per (hashes, T): shadow mode
# (check runs, never rejects), gate mode, exact audit. Filter-off baseline at
# start and end. Each timing = 1 warm-up + median of 3.
#
# Grid: hashes 1-5 x T 5..35; the h=2-5 rows pair with the step 1/2 runs.
# Binary must be built with icpx (PrgEnv-intel) in build/.
#
#   cd sparse/scripts/bloom_filter
#   sbatch --export=ALL,STEP=step2_two_pass run_bloom_harness_perlmutter.sh
#   sbatch --export=ALL,STEP=step3_ef_sweep,FLAG=--ef-sweep run_bloom_harness_perlmutter.sh
#
# FLAG=--grid-1-5 (default): ef=1600, hashes x T grid with shadow + audit.
# FLAG=--ef-sweep: filter off vs h=1 gate (T 20-35) at ef 50..1600.
# FLAG=--trace: lossless-gate headroom + bound tightness at ef 100/400/1600 (step 4).
#
# STEP names the results subfolder (single value; --export splits on commas).

set -euo pipefail
STEP=${STEP:?set STEP, e.g. --export=ALL,STEP=step2_two_pass}
FLAG=${FLAG:---grid-1-5}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/bloom_filter"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance

BIN="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/sparse/sparse_bloom_sweep"
bench_assert_optimized "$BIN"

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/bloom_filter/${STEP}/msmarco_full_icpx${FLAG#-}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET  ndocs=$SPKNN_NDOCS nnz=$SPKNN_NNZ"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### out     : $OUT"
cat /sys/kernel/mm/transparent_hugepage/enabled || true

$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$OUT/results.csv" "$FLAG"

echo "########## done -> $OUT ##########"
