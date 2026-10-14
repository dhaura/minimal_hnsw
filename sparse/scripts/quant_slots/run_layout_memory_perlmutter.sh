#!/bin/bash
#SBATCH --job-name=layout_memory
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# Peak RSS of the original (offset-indexed) layout vs slots-only on
# msmarco_full, one process each so PeakRSSGB (lifetime ru_maxrss) belongs to
# one layout. Slots-only never allocates the offset-indexed copy.
# efC=200 for every M: graph memory depends on M, not efC, and builds are short.
#
#   cd sparse/scripts/quant_slots && sbatch run_layout_memory_perlmutter.sh

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/quant_slots"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin/sparse_hnsw_sweep"
bench_assert_optimized "$BIN"

SLOT_SIZE=${SLOT_SIZE:-320}
OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/quant_slots/step3_peak_memory"
mkdir -p "$OUT"
CSV="$OUT/layout_memory_${SLURM_JOB_ID}.csv"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH   csv: $CSV"

for M in 32 48 64; do
    for SLOT in 0 "-$SLOT_SIZE"; do
        echo "########## M=$M slot_arg=$SLOT ##########"
        $BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
            "$M" 200 100 1 0 0 0.85 3 \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" "SparseHNSW_M${M}_slot${SLOT}" 3 1 1 8 8:4 0 "$SLOT"
    done
done
echo "########## done -> $CSV ##########"
