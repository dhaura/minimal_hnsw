#!/bin/bash
#SBATCH --job-name=gap_top16
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Full SparseHNSW sweep for the playground, same grids as the Bloom full sweeps,
# on the default slot layout (slots only). Each graph runs four series in one job:
#   SparseHNSW          sep layout, no filter, patience list   (same-job baseline)
#   SparseHNSW+Gap      sep layout, no filter, score-gap stop list
#   SparseHNSW+Top16    list16 layout (top-16 terms per neighbor), patience list
#   SparseHNSW+Top16+Gap list16 layout, score-gap stop list
# The Bloom filter is not built. One graph build per (M, efC, alpha); the top-16
# sketch build time is in IndexSec of the list16 rows only.
# Presets hold the lists (--export splits on commas).
#
#   alpha : M=32 efC=200, alpha from ALPHAS_LIST (dash list), small ef  (low recall)
#   grid  : alpha 0.85, M from --export, efC from EFCS (dash list)      (mid/high recall)
#
#   cd sparse/scripts/gap_top16
#   sbatch --export=ALL,PRESET=alpha,ALPHAS_LIST=0.75-0.8-0.85-0.9 run_gap_top16_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600 run_gap_top16_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=48,EFCS=800-1600 run_gap_top16_sweep_perlmutter.sh
#   nq_splade: add SPKNN_DATASET=nq_splade,SLOT_SIZE=448 (alpha 0.65-0.7-0.725-0.75-0.8; M=64 EFCS=400-800)

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha or grid}
SLOT_SIZE=${SLOT_SIZE:-320}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/gap_top16"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
# Own copy of the binary so rebuilding build/ does not touch running jobs.
BIN=${BIN:-"$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build-gaptop16/bin/sparse_hnsw_sweep"}
bench_assert_optimized "$BIN"

GAPS=g0.05,g0.1,g0.15,g0.2,g0.3
case "$PRESET" in
    alpha)
        M=32; EFCS="200"; ALPHAS=${ALPHAS_LIST:?set ALPHAS_LIST}; ALPHAS=${ALPHAS//-/ }
        EF_LIST=10,15,20,25,30,40,50,75,100,200,400
        BETAS=2,3,4; STOPS=0,128,256,$GAPS; SEEDS=0:1,8:4 ;;
    grid)
        M=${M:?set M}; EFCS=${EFCS:?set EFCS, e.g. 200-400-800-1600}; EFCS=${EFCS//-/ }; ALPHAS="0.85"
        EF_LIST=10,20,30,40,50,100,200,400,800,1600
        BETAS=2,3,4,5,8; STOPS=0,128,256,512,$GAPS; SEEDS=8:4 ;;
    *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/gap_top16/full_sweep_${SPKNN_DATASET}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### preset  : $PRESET M=$M efC=$EFCS alpha=$ALPHAS stops=$STOPS slot_size=$SLOT_SIZE bin=$BIN"

for EFC in $EFCS; do
    for ALPHA in $ALPHAS; do
        CSV="$OUT/gaptop16_${SPKNN_DATASET}_M${M}_efc${EFC}_a${ALPHA}_${SLURM_JOB_ID}.csv"
        echo "########## M=$M efC=$EFC alpha=$ALPHA -> $CSV ##########"
        $BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
            "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" SparseHNSW 3 1 1 8 "$SEEDS" "$STOPS" "-$SLOT_SIZE" off 1 sep,list16
    done
done
echo "########## done -> $OUT ##########"
