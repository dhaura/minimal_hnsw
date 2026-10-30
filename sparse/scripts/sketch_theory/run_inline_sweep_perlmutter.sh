#!/bin/bash
#SBATCH --job-name=inline_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Full SparseHNSW sweep with the inline top-K gate, same grids as the gap_top16 full
# sweeps, slots only. Each graph runs six series in one job:
#   SparseHNSW               sep, rows in dimension order, no gate   (same-job baseline)
#   SparseHNSW_inline16      rows top-16 first, score a neighbor only if one of its first
#   SparseHNSW_inline24      16 / 24 slot IDs is a query term (0 B extra memory)
# each with the patience list and the score-gap stop list (+Gap rows carry gap= in
# Params). The Bloom filter is not built. One graph build per (M, efC, alpha); the
# top-K-first slot rebuild replaces the slot build time in IndexSec of the inline rows.
# Presets hold the lists (--export splits on commas).
#
#   alpha : M=32 efC=200, alpha from ALPHAS_LIST (dash list), small ef  (low recall)
#   grid  : alpha 0.85, M from --export, efC from EFCS (dash list)      (mid/high recall)
#
# Frozen binary, so rebuilding build-inline/ does not touch running jobs (PrgEnv-intel):
#   cmake -S . -B build-inlinesweep -DCMAKE_BUILD_TYPE=Release && cmake --build build-inlinesweep --target sparse_hnsw_sweep
#
#   cd sparse/scripts/sketch_theory
#   sbatch --export=ALL,PRESET=alpha,ALPHAS_LIST=0.75-0.8-0.85-0.9 run_inline_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600 run_inline_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=48,EFCS=800-1600 run_inline_sweep_perlmutter.sh
#   nq_splade: add SPKNN_DATASET=nq_splade,SLOT_SIZE=448 (alpha 0.65-0.7-0.725-0.75-0.8; M=64 EFCS=400-800)

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha or grid}
SLOT_SIZE=${SLOT_SIZE:-320}
LAYOUTS=sep,inline16,inline24
REPO="$SCRATCH/repos/sparse_hnsw/minimal_hnsw"
cd "$REPO/sparse/scripts/sketch_theory"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN=${BIN:-"$REPO/build-inlinesweep/bin/sparse_hnsw_sweep"}
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

OUT="$REPO/sparse/scripts/results/sketch_theory/full_sweep_${SPKNN_DATASET}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### preset  : $PRESET M=$M efC=$EFCS alpha=$ALPHAS stops=$STOPS layouts=$LAYOUTS slot_size=$SLOT_SIZE bin=$BIN"

for EFC in $EFCS; do
    for ALPHA in $ALPHAS; do
        CSV="$OUT/inline_${SPKNN_DATASET}_M${M}_efc${EFC}_a${ALPHA}_${SLURM_JOB_ID}.csv"
        echo "########## M=$M efC=$EFC alpha=$ALPHA -> $CSV ##########"
        $BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
            "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" SparseHNSW 3 1 1 8 "$SEEDS" "$STOPS" "-$SLOT_SIZE" off 1 "$LAYOUTS"
    done
done
echo "########## done -> $OUT ##########"
