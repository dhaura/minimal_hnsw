#!/bin/bash
#SBATCH --job-name=bloom_slots
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

# Full SparseHNSW sweep with the Bloom distance gate ON (h=1 popcount check,
# two-pass gate, survivor-only row prefetch) on the default slot layout
# (slots only, 320 B), for the spknn-playground comparison. Every grid point
# is run at each Bloom threshold T (% of query terms that must look present;
# -1 = gate off, a same-job baseline). One graph build per (M, efC, alpha); the
# filter is built once per graph and counted in IndexSec. Presets hold the
# lists (--export splits on commas).
#
#   alpha     : M=32 efC=200, alpha 0.75-0.9 (ALPHAS_LIST, dash list), small ef  (low recall)
#   grid      : alpha 0.85, M from --export, efC from EFCS (dash list)           (mid/high recall)
# Top-16 list-sketch and score-gap sweeps: ../gap_top16/run_gap_top16_sweep_perlmutter.sh.
#
#   cd sparse/scripts/bloom_filter
#   sbatch --export=ALL,PRESET=alpha,STEP=threshold_sweeps/perlmutter_msmarco_full run_bloom_slots_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600,STEP=... run_bloom_slots_sweep_perlmutter.sh
#   nq_splade: add SPKNN_DATASET=nq_splade,SLOT_SIZE=448
#
# GATE_SET (alpha or grid): full (default) T = 5,10,...,35; schedule = off,0,1,2,3,4,5,8,10;
# schedule_high = off,25,30,...,50; schedule_all = both lists in one job.

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha or grid}
STEP=${STEP:?set STEP, the results subfolder, e.g. threshold_sweeps/perlmutter_msmarco_full}
SLOT_SIZE=${SLOT_SIZE:-320}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/bloom_filter"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
# Pass BIN= a frozen copy of the binary so rebuilding build/ does not touch running jobs.
BIN=${BIN:-"$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin/sparse_hnsw_sweep"}
bench_assert_optimized "$BIN"

THRESHOLDS=5,10,15,20,25,30,35; HASHES=1
case "$PRESET" in
    alpha)
        M=32; EFCS="200"; ALPHAS="0.75 0.8 0.85 0.9"
        EF_LIST=10,15,20,25,30,40,50,75,100,200,400
        BETAS=2,3,4; PATIENCE_LIST=0,128,256; SEEDS=0:1,8:4 ;;
    grid)
        M=${M:?set M}; EFCS=${EFCS:?set EFCS, e.g. 200-400-800-1600}; EFCS=${EFCS//-/ }; ALPHAS="0.85"
        EF_LIST=10,20,30,40,50,100,200,400,800,1600
        BETAS=2,3,4,5,8; PATIENCE_LIST=0,128,256,512; SEEDS=8:4 ;;
    *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac
[ -n "${ALPHAS_LIST:-}" ] && ALPHAS=${ALPHAS_LIST//-/ }
case "${GATE_SET:-full}" in
    full) ;;
    schedule) THRESHOLDS=-1,0,1,2,3,4,5,8,10 ;;
    schedule_high) THRESHOLDS=-1,25,30,35,40,45,50 ;;
    schedule_all) THRESHOLDS=-1,0,1,2,3,4,5,8,10,25,30,35,40,45,50 ;;
    *) echo "unknown GATE_SET=$GATE_SET" >&2; exit 1 ;;
esac

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/bloom_filter/${STEP}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### preset  : $PRESET M=$M efC=$EFCS alpha=$ALPHAS T=$THRESHOLDS slot_size=$SLOT_SIZE bin=$BIN"

for EFC in $EFCS; do
    for ALPHA in $ALPHAS; do
        CSV="$OUT/bloom_${SPKNN_DATASET}_M${M}_efc${EFC}_a${ALPHA}_${SLURM_JOB_ID}.csv"
        echo "########## M=$M efC=$EFC alpha=$ALPHA -> $CSV ##########"
        $BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
            "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" SparseHNSW 3 1 1 8 "$SEEDS" "$PATIENCE_LIST" "-$SLOT_SIZE" "$THRESHOLDS" "$HASHES"
    done
done
echo "########## done -> $OUT ##########"
