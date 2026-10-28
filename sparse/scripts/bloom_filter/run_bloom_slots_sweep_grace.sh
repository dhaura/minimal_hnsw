#!/bin/bash
#SBATCH --job-name=bloom_slots
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# GRACE port of run_bloom_slots_sweep_perlmutter.sh (TAMU Grace, Intel Xeon
# Gold 6248R Cascade Lake, 48 threads, GCC 13.2 -march=cascadelake).
# Same presets, same CSV naming; only the paths, the SBATCH header and the
# per-dataset threshold lists differ.  Results land in NEW grace_* folders so
# the Perlmutter results are never touched.
#
# Full SparseHNSW sweep with the Bloom distance gate ON (h=1 popcount check,
# two-pass gate, survivor-only row prefetch) on the slot layout (slots only,
# SLOT_SIZE: 320 B msmarco_full, 448 B nq_splade).  Every grid point is run at
# each Bloom threshold T (% of query terms that must look present).  T=-1 is a
# gate-OFF pass on the SAME graph -- that is the matched-recall baseline.
# One graph build per (M, efC, alpha); the filter is built once per graph and
# counted in IndexSec.  Presets hold the lists (--export splits on commas).
#
#   alpha : M=32 efC=200, small ef, seeds on/off          (low recall, quick)
#   grid  : alpha 0.85, M from --export, efC from EFCS    (mid/high recall)
#
#   cd sparse/scripts/bloom_filter && mkdir -p logs
#   sbatch --export=ALL,PRESET=alpha,ALPHAS_LIST=0.75-0.8-0.85-0.9 run_bloom_slots_sweep_grace.sh
#   sbatch --time=24:00:00 --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600 run_bloom_slots_sweep_grace.sh
#   sbatch --time=24:00:00 --export=ALL,PRESET=grid,M=48,EFCS=800-1600 run_bloom_slots_sweep_grace.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,PRESET=alpha,ALPHAS_LIST=0.65-0.7-0.725-0.75-0.8 run_bloom_slots_sweep_grace.sh
#
# THRESHOLDS/EF_OVERRIDE/ALPHAS_LIST take dash lists (sbatch --export splits on
# commas).  BIN overrides the binary (default: $SPKNN_BIN, i.e. build-release/bin).

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha or grid}
cd /scratch/user/dhaura/repos/minimal_hnsw/sparse/scripts/bloom_filter
mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

source /scratch/user/dhaura/repos/spknn-playground/common/bench_env.sh
bench_provenance
BIN=${BIN:-"$SPKNN_BIN/sparse_hnsw_sweep"}
bench_assert_optimized "$BIN"

# --- dataset (Grace bench_env.sh only sets SPKNN_DATA; be explicit) --------
export SPKNN_DATASET=${SPKNN_DATASET:-msmarco_full}
case "$SPKNN_DATASET" in
    msmarco_full)
        SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_full"; QSET=dev
        SPKNN_BASE="$SPKNN_DATA/base_full.csr"
        SPKNN_GT="$SPKNN_DATA/base_full.$QSET.gt"
        DEF_SLOT=320; DEF_T=-1,5,10,15,20,25,30,35 ;;
    nq_splade)
        SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/nq_splade"; QSET=test
        SPKNN_BASE="$SPKNN_DATA/base_nq.csr"
        SPKNN_GT="$SPKNN_DATA/base_nq.$QSET.gt"
        DEF_SLOT=448; DEF_T=-1,0,5,10,15,20,25 ;;
    msmarco_small)   # smoke test only
        SPKNN_DATA="$SPKNN_HNSW_REPO/sparse/data/msmarco_small"; QSET=dev
        SPKNN_BASE="$SPKNN_DATA/base_small.csr"
        SPKNN_GT="$SPKNN_DATA/base_small.$QSET.gt"
        DEF_SLOT=320; DEF_T=-1,0,10,20 ;;
    *) echo "unknown SPKNN_DATASET='$SPKNN_DATASET'" >&2; exit 1 ;;
esac
SPKNN_QUERIES="$SPKNN_DATA/queries.$QSET.csr"
for f in "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT"; do
    [ -r "$f" ] || { echo "FATAL missing data file: $f" >&2; exit 1; }
done
SLOT_SIZE=${SLOT_SIZE:-$DEF_SLOT}
THRESHOLDS=${THRESHOLDS:-$DEF_T}; THRESHOLDS=${THRESHOLDS//-/,}; THRESHOLDS=${THRESHOLDS/#,/-}
HASHES=${HASHES:-1}

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
[ -n "${EF_OVERRIDE:-}" ] && EF_LIST=${EF_OVERRIDE//-/,}
[ -n "${BETAS_OVERRIDE:-}" ] && BETAS=${BETAS_OVERRIDE//-/,}
[ -n "${PATIENCE_OVERRIDE:-}" ] && PATIENCE_LIST=${PATIENCE_OVERRIDE//-/,}

STEP=${STEP:-threshold_sweeps/grace_$SPKNN_DATASET}
OUT="$SPKNN_HNSW_REPO/sparse/scripts/results/bloom_filter/${STEP}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET / $QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
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
