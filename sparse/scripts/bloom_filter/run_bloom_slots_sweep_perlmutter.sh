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
# two-pass gate, survivor-only row prefetch) on the slot layout (slots only,
# SLOT_SIZE, default 320 B; 448 for nq_splade), for the spknn-playground
# comparison. Every grid point
# is run at each Bloom threshold T (% of query terms that must look present).
# One graph build per (M, efC, alpha); the filter is built once per graph and
# counted in IndexSec. Presets hold the lists (--export splits on commas).
#
#   alpha : M=32 efC=200, alpha 0.75-0.9, small ef, seeds on/off  (low recall)
#   grid  : alpha 0.85, M from --export, efC from EFCS (dash list)  (mid/high recall)
#
#   cd sparse/scripts/bloom_filter
#   sbatch --export=ALL,PRESET=alpha run_bloom_slots_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600 run_bloom_slots_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=48,EFCS=800-1600 run_bloom_slots_sweep_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,SLOT_SIZE=448,PRESET=alpha,ALPHAS_LIST=0.65-0.7-0.725-0.75-0.8 run_bloom_slots_sweep_perlmutter.sh
#
# GATE_SET=schedule (with alpha or grid): low thresholds T = off,0,1,2,3,4,5,8,10,
# with a gate-off pass on the same graph, for the "lower T towards 0 as recall
# rises" question. ALPHAS_LIST (dash list) overrides the alpha preset; BIN
# overrides the binary (default: $SPKNN_BIN, i.e. build/bin).
#   sbatch --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600,GATE_SET=schedule,STEP=threshold_schedule_msmarco_full run_bloom_slots_sweep_perlmutter.sh
# GATE_SET=schedule_high: the same with T = off,25,30,35,40,45,50 (adds to a
# schedule run's folder; each job keeps its own gate-off pass).
# GATE_SET=schedule_all: both lists in one job (one graph build, one gate-off pass).

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha or grid}
STEP=${STEP:-full_sweep_slots_h1}
SLOT_SIZE=${SLOT_SIZE:-320}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/bloom_filter"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
BIN=${BIN:-"$SPKNN_BIN/sparse_hnsw_sweep"}
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
if [ "${GATE_SET:-full}" = schedule_all ]; then  # schedule + schedule_high in one job
    THRESHOLDS=-1,0,1,2,3,4,5,8,10,25,30,35,40,45,50
elif [ "${GATE_SET:-full}" = schedule_high ]; then
    THRESHOLDS=-1,25,30,35,40,45,50
elif [ "${GATE_SET:-full}" = schedule ]; then
    THRESHOLDS=-1,0,1,2,3,4,5,8,10
fi

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
