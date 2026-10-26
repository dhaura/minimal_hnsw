#!/bin/bash
#SBATCH --job-name=slots_lowrec
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Low-recall (80-91%) SparseHNSW points on msmarco_full with the default
# slot layout (slots only, so PeakRSSGB is the slotted footprint). Mirrors the
# region the old playground front covered: M=32 efC=200, small ef, alpha
# 0.75-0.9, beta 2-3, with and without inverted seeding (seed spec 0:1 = off).
# One graph build per alpha.
#
#   cd sparse/scripts/quant_slots && sbatch run_low_recall_perlmutter.sh

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/quant_slots"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

SLOT_SIZE=${SLOT_SIZE:-320}
EF_LIST=10,15,20,25,30,40
BETAS=2,3
OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/quant_slots/step5_low_recall"
mkdir -p "$OUT"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH   out: $OUT"

for ALPHA in 0.75 0.8 0.85 0.9; do
    CSV="$OUT/low_recall_M32_efc200_a${ALPHA}_${SLURM_JOB_ID}.csv"
    echo "########## alpha=$ALPHA -> $CSV ##########"
    $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
        32 200 "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
        "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
        "$CSV" SparseHNSW 3 1 1 8 0:1,8:4 0 "-$SLOT_SIZE"
done
echo "########## done -> $OUT ##########"
