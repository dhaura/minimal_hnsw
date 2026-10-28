#!/bin/bash
#SBATCH --job-name=stop_online
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Adaptive stopping step 1: the replay study's best rules run online in the
# engine and timed (QPS), against the fixed-ef curve of the same graph.
#   "0"           fixed ef (baseline curve; dense ef grid)
#   "gG"          score gap: stop before expanding a node scoring below (1-G) x the k_hat-th best
#   "P"           plain patience (top-k_hat), for reference
# Each rule caps the search at the ef it runs with; only ef >= 800 rows matter for rules.
# results/adaptive_stopping/online_stop_msmarco (job 60337143) was run on the
# adaptive-stopping-online worktree and also has combo rows "gG:mM:pP" (gap OR P
# expansions without a top-M improvement); the engine no longer has top-M patience.
#
#   cd sparse/scripts/adaptive_stopping
#   sbatch --export=ALL,STEP=online_stop_msmarco run_stop_online_perlmutter.sh
set -euo pipefail
STEP=${STEP:?set STEP, e.g. online_stop_msmarco}
M=${M:-32}; EFC=${EFC:-800}; ALPHA=${ALPHA:-0.85}; BETA=${BETA:-4}
REPO="$SCRATCH/repos/sparse_hnsw/minimal_hnsw"
cd "$REPO/sparse/scripts/adaptive_stopping"
mkdir -p logs
source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN=${BIN:-"$REPO/build/bin/sparse_hnsw_sweep"}  # pass a frozen copy for long runs
bench_assert_optimized "$BIN"

EF_LIST=50,75,100,150,200,300,400,600,800,1200,1600,2400,3200
RULES=0,64,128,g0,g0.02,g0.05,g0.075,g0.1,g0.15,g0.2,g0.3

OUT="$REPO/sparse/scripts/results/adaptive_stopping/${STEP}"
mkdir -p "$OUT"
CSV="$OUT/stop_${SPKNN_DATASET}_M${M}_efc${EFC}_a${ALPHA}_b${BETA}_${SLURM_JOB_ID}.csv"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### config  : M=$M efC=$EFC alpha=$ALPHA beta=$BETA rules=$RULES -> $CSV"
$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETA" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$CSV" SparseHNSW 3 1 1 8 8:4 "$RULES"
echo "########## done -> $OUT ##########"
