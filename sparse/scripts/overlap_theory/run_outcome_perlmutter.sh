#!/bin/bash
#SBATCH --job-name=outcome
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Overlap-check theory, step 4c: the exact-overlap (ideal) gate inside the search, and
# what becomes of every checked neighbor. Config of step 3 (M=32 efC=200 alpha 0.85
# beta 3 seeds 8:4); gate off (-1) and ideal T = 0..30, on a dense ef grid so gated
# points can be matched to the gate-off curve at equal recall. Each point gets an
# untimed outcome pass -> <csv>.outcome.csv (calls, row lines, gate checks/skips, and
# checked neighbors by class: not added / added never expanded / expanded / returned).
# Needs the outcome build (SPARSE_HNSW_OUTCOME), removed from the engine 2026-10-30; re-apply
# it from the repo root first, then (PrgEnv-intel):
#   git apply ../cleanup_backup_2026-10-30/engine_outcome/outcome_instrumentation.patch
#   cmake --build build-outcome --target sparse_hnsw_sweep_outcome
# QPS is a single instrumented pass and not meant for timing.
#
#   cd sparse/scripts/overlap_theory
#   sbatch run_outcome_perlmutter.sh                                    # msmarco_full
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,SLOT_SIZE=448 run_outcome_perlmutter.sh

set -euo pipefail
SLOT_SIZE=${SLOT_SIZE:-320}
REPO="$SCRATCH/repos/sparse_hnsw/minimal_hnsw"
cd "$REPO/sparse/scripts/overlap_theory"
mkdir -p logs
source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN="$REPO/build-outcome/bin/sparse_hnsw_sweep_outcome"
bench_assert_optimized "$BIN"

OUT="$REPO/sparse/scripts/results/overlap_theory/outcome_${SPKNN_DATASET}"
mkdir -p "$OUT"
CSV="$OUT/outcome_${SPKNN_DATASET}_M32_efc200_a0.85_${SLURM_JOB_ID}.csv"
EF_LIST=20,30,50,75,100,150,200,300,400,600,800,1200,1600
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### ideal gate off + T 0..30, ef $EF_LIST, slot_size=$SLOT_SIZE, bin=$BIN -> $CSV"
$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    32 200 "$EF_LIST" 1 0 0 0.85 3 \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$CSV" SparseHNSW 1 0 1 8 8:4 0 "-$SLOT_SIZE" -1,0,2,5,10,15,20,25,30 1 ideal
echo "########## done -> $OUT ##########"
