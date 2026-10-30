#!/bin/bash
#SBATCH --job-name=sketch_outcome
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Top-K sketch theory, inline layout: each row slot is written top-K first (its K heaviest
# IDs, ties in dimension order, then the rest, each block in dimension order), so the gate
# tests the row's own first K IDs (line 0 holds the header and 30 IDs). The sweep rebuilds
# the slots for each inline<K> (< 1 s). The outcome_inline_<ds> results (jobs 60509218/220)
# predate that order (rows heaviest first): same top-K sets, so same gate decisions and
# line counts. Config of overlap_theory step 4c (M=32 efC=200 alpha 0.85 beta 3
# seeds 8:4, stop rules off), dense ef grid; LAYOUTS = sep (no gate, same-job baseline,
# rows in dimension order) and inline<K>. Each point gets an untimed outcome pass ->
# <csv>.outcome.csv: calls, row lines, gate checks/skips, layer-0 expansions, checked
# neighbors by class, and per-class histograms of query hits among each neighbor's
# top-16 and top-32 pruned terms. Inline traffic = row lines + one line per skip.
# Needs the inline outcome build (SPARSE_HNSW_OUTCOME), removed from the engine 2026-10-30;
# re-apply it from the repo root first, then (PrgEnv-intel):
#   git apply ../cleanup_backup_2026-10-30/engine_outcome/outcome_instrumentation.patch
#   cmake --build build-inline --target sparse_hnsw_sweep_outcome
# QPS is a single instrumented pass and not meant for timing.
#
#   cd sparse/scripts/sketch_theory
#   sbatch run_sketch_outcome_perlmutter.sh                                    # msmarco_full
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,SLOT_SIZE=448 run_sketch_outcome_perlmutter.sh

set -euo pipefail
SLOT_SIZE=${SLOT_SIZE:-320}
LAYOUTS=sep,inline16,inline24,inline30
REPO="$SCRATCH/repos/sparse_hnsw/minimal_hnsw"
cd "$REPO/sparse/scripts/sketch_theory"
mkdir -p logs
source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN="$REPO/build-inline/bin/sparse_hnsw_sweep_outcome"
bench_assert_optimized "$BIN"

OUT="$REPO/sparse/scripts/results/sketch_theory/outcome_inline_${SPKNN_DATASET}"
mkdir -p "$OUT"
CSV="$OUT/outcome_inline_${SPKNN_DATASET}_M32_efc200_a0.85_${SLURM_JOB_ID}.csv"
EF_LIST=20,30,50,75,100,150,200,300,400,600,800,1200,1600
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### $LAYOUTS, ef $EF_LIST, slot_size=$SLOT_SIZE, bin=$BIN -> $CSV"
$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    32 200 "$EF_LIST" 1 0 0 0.85 3 \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$CSV" SparseHNSW 1 0 1 8 8:4 0 "-$SLOT_SIZE" off 1 "$LAYOUTS"
echo "########## done -> $OUT ##########"
