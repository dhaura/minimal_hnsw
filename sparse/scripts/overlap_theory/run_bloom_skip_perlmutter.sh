#!/bin/bash
#SBATCH --job-name=bloom_skip
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# Overlap-check theory, step 3: how many checked neighbors the real h=1 Bloom filter
# skips at each T = 0..30 (the same grid as the ideal filter), next to the ideal
# exact-overlap filter, on the config of the ideal-gate run (job 60345323: M=32
# efC=200 alpha 0.85 beta 3, ef 20..800). Each gate-on point gets an untimed
# gate-stats pass -> <csv>.gate.csv (gate_rej = Bloom skips, ideal_rej_T* = ideal).
# Needs the pre-cleanup build with gate stats (build-idealT/); its argv has
# bypass, schedules, layouts and gate_stats at 23-26. QPS is a single pass and
# not meant for timing.
#
#   cd sparse/scripts/overlap_theory
#   sbatch run_bloom_skip_perlmutter.sh                                    # msmarco_full
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,SLOT_SIZE=448 run_bloom_skip_perlmutter.sh

set -euo pipefail
SLOT_SIZE=${SLOT_SIZE:-320}
REPO="$SCRATCH/repos/sparse_hnsw/minimal_hnsw"
cd "$REPO/sparse/scripts/overlap_theory"
mkdir -p logs
source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN="$REPO/build-idealT/bin/sparse_hnsw_sweep"
bench_assert_optimized "$BIN"

OUT="$REPO/sparse/scripts/results/overlap_theory/bloom_skip_${SPKNN_DATASET}"
mkdir -p "$OUT"
CSV="$OUT/bloom_${SPKNN_DATASET}_M32_efc200_a0.85_${SLURM_JOB_ID}.csv"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### T 0..30, slot_size=$SLOT_SIZE, bin=$BIN -> $CSV"
$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    32 200 20,50,100,200,400,800 1 0 0 0.85 3 \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$CSV" SparseHNSW 1 0 1 8 8:4 0 "-$SLOT_SIZE" 0,2,5,10,15,20,25,30 1 0 "" sep 1
echo "########## done -> $OUT ##########"
