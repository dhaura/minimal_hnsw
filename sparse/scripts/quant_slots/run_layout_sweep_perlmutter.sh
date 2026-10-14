#!/bin/bash
#SBATCH --job-name=layout_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Full SparseHNSW grid on msmarco_full with BOTH row layouts per point:
# sparse_hnsw_sweep with slot_size>0 times the original offset-indexed rows
# and fixed-size slots alternately on every (ef, beta, patience) point of
# the same graph, and aborts if their predictions ever differ.
#
# One job per M; it builds each efC in EFCS in turn (dash-separated, since
# --export splits on commas). Grid mirrors the Sep-7 baseline grid
# (alpha 0.85, beta 2-8, patience 128-2048, 8x4 seeds), plus ef 30/40 and
# patience 0 to cover the low-recall bins.
#
#   cd sparse/scripts/quant_slots
#   sbatch --export=ALL,M=32,EFCS=200-400-800-1600 run_layout_sweep_perlmutter.sh
#   sbatch --export=ALL,M=48,EFCS=800-1600         run_layout_sweep_perlmutter.sh
#   sbatch --export=ALL,M=64,EFCS=800-1600         run_layout_sweep_perlmutter.sh

set -euo pipefail
M=${M:?set M}
EFCS=${EFCS:?set EFCS, e.g. 200-400-800-1600}
SLOT_SIZE=${SLOT_SIZE:-320}
STEP=${STEP:-step2_full_sweep}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/quant_slots"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

ALPHA=0.85
BETAS=2,3,4,5,8
EF_LIST=10,20,30,40,50,100,200,400,800,1600,3200
PATIENCE_LIST=0,128,256,512,2048

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/quant_slots/${STEP}"
mkdir -p "$OUT"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### grid    : M=$M efC=$EFCS alpha=$ALPHA beta=$BETAS ef=$EF_LIST pat=$PATIENCE_LIST slot_size=$SLOT_SIZE"

for EFC in ${EFCS//-/ }; do
    CSV="$OUT/layout_sweep_M${M}_efc${EFC}_${SLURM_JOB_ID}.csv"
    echo "########## M=$M efC=$EFC -> $CSV ##########"
    $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
        "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
        "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
        "$CSV" SparseHNSW 3 1 1 8 8:4 "$PATIENCE_LIST" "$SLOT_SIZE"
done
echo "########## done -> $OUT ##########"
