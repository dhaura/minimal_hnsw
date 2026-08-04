#!/bin/bash
#SBATCH --job-name=ef_ndist
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=logs/ef_ndist_%j.out

# The ef experiment, both halves in one job:
#
#   pass 1  sparse_hnsw_sweep   the EXACT benchmark binary and env used for the
#                               spknn-playground msmarco_full results (-O3
#                               -march=cascadelake, no profiling code) -> the
#                               authoritative recall / search time / QPS.
#   pass 2  sparse_diag_sweep   same flags PLUS the SPARSE_HNSW_PROFILE counters
#                               -> ndist per query per ef. Its own timings are a
#                               cross-check only (counters + noinline cost a few
#                               percent). diag_queries=10 so the containment /
#                               oracle diagnostics add no meaningful time.

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
DATA=$REPO/sparse/data/msmarco_full
OUT=$REPO/sparse/scripts/results/ef_ndist
mkdir -p "$OUT"

EF_LIST=10,20,50,100,200,400,800,1600,3200
ALPHA=0.8
BETA=3

# Benchmark env, mirrored from spknn-playground/common/bench_env.sh
export OMP_NUM_THREADS=48
export OMP_PROC_BIND=false
unset OMP_PLACES

echo "########## pass 1: benchmark binary (timing + recall) ##########"
numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL \
  "$REPO/build-release/bin/sparse_hnsw_sweep" \
  16 200 "$EF_LIST" 1 0 0 $ALPHA $BETA \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt" \
  "$OUT/bench_timing.csv" SparseHNSW 5 1

echo ""
echo "########## pass 2: profile binary (distance-call counts) ##########"
numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL \
  "$REPO/build-release/bin/sparse_diag_sweep" \
  16 200 "$EF_LIST" 1 0 0 $ALPHA $BETA \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt" \
  "$OUT/prof_timing.csv" "$OUT/prof_ndist.csv" SparseHNSW 3 1 10
