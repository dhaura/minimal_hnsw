#!/bin/bash
#SBATCH --job-name=slots_ab
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Packed rows vs fixed-size slots on any dataset bench_env knows: for each
# alpha x slot size, one sparse_hnsw_sweep process builds a graph and times both
# layouts alternately on every (ef, beta, patience) point, aborting if their
# predictions differ. Results: results/quant_slots/<STEP>/slot<size>/.
# List-valued knobs are dash-separated (--export splits on commas).
#
#   cd sparse/scripts/quant_slots
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,STEP=step4_nq_splade,ALPHAS=0.7-0.85,SLOT_SIZES=320-384-448 \
#          run_layout_ab_perlmutter.sh

set -euo pipefail
STEP=${STEP:?set STEP, e.g. step4_nq_splade}
M=${M:-32}
EFC=${EFC:-200}
ALPHAS=${ALPHAS:-0.85}
SLOT_SIZES=${SLOT_SIZES:-320}
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/quant_slots"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

BETAS=3,5,8
EF_LIST=10,20,30,50,100,200,400,800,1600
PATIENCE_LIST=0,256

OUT_ROOT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/quant_slots/${STEP}"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   ndocs=$SPKNN_NDOCS   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### grid    : M=$M efC=$EFC alpha=$ALPHAS slot_size=$SLOT_SIZES beta=$BETAS ef=$EF_LIST pat=$PATIENCE_LIST"

for SLOT in ${SLOT_SIZES//-/ }; do
    OUT="$OUT_ROOT/slot$SLOT"
    mkdir -p "$OUT"
    for ALPHA in ${ALPHAS//-/ }; do
        CSV="$OUT/layout_sweep_${SPKNN_DATASET}_M${M}_efc${EFC}_a${ALPHA}_${SLURM_JOB_ID}.csv"
        echo "########## slot=$SLOT alpha=$ALPHA -> $CSV ##########"
        $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
            "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETAS" \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" SparseHNSW 3 1 1 8 8:4 "$PATIENCE_LIST" "$SLOT"
    done
done
echo "########## done -> $OUT_ROOT ##########"
