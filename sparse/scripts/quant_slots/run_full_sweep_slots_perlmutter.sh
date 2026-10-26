#!/bin/bash
#SBATCH --job-name=slots_full
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Full SparseHNSW grid with the slot layout only (slots-only processes, so
# PeakRSSGB is the slotted footprint) for the spknn-playground comparison.
# One process (graph build) per (M, efC, alpha). Presets hold the list-valued
# knobs because --export splits on commas.
#
#   alpha_lo / alpha_hi : M=32 efC=200, the alpha sweep (low/mid recall)
#   grid                : alpha 0.85, M from --export, efC 200..1600 (high recall)
#
#   cd sparse/scripts/quant_slots
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,SLOT_SIZE=448,PRESET=alpha_lo run_full_sweep_slots_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,SLOT_SIZE=448,PRESET=grid,M=48 run_full_sweep_slots_perlmutter.sh

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha_lo, alpha_hi or grid}
SLOT_SIZE=${SLOT_SIZE:?set SLOT_SIZE, e.g. 448}
STEP=${STEP:-step6_full_sweep_${SPKNN_DATASET:-msmarco_full}}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/quant_slots"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

case "$PRESET" in
    alpha_lo|alpha_hi)
        M=32; EFCS="200"
        [ "$PRESET" = alpha_lo ] && ALPHAS="0.6 0.65 0.7 0.725" || ALPHAS="0.75 0.775 0.8 0.85 0.9"
        EF_LIST=10,20,25,30,35,40,50,60,75,100,200,400,800,1600,3200
        BETAS=2,3,4,5,8; PATIENCE_LIST=0,128,256; SEEDS=0:1,8:4 ;;
    grid)
        M=${M:?set M for PRESET=grid}; EFCS="200 400 800 1600"; ALPHAS="0.85"
        EF_LIST=10,20,50,100,200,400,800,1600,3200
        BETAS=2,3,4,5,8; PATIENCE_LIST=0,128,256,512,2048; SEEDS=8:4 ;;
    *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/quant_slots/${STEP}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### preset  : $PRESET M=$M efC=$EFCS alpha=$ALPHAS slot_size=$SLOT_SIZE"

for EFC in $EFCS; do
    for ALPHA in $ALPHAS; do
        CSV="$OUT/slots_${SPKNN_DATASET}_M${M}_efc${EFC}_a${ALPHA}_${SLURM_JOB_ID}.csv"
        echo "########## M=$M efC=$EFC alpha=$ALPHA -> $CSV ##########"
        $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
            "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" SparseHNSW 3 1 1 8 "$SEEDS" "$PATIENCE_LIST" "-$SLOT_SIZE"
    done
done
echo "########## done -> $OUT ##########"
