#!/bin/bash
#SBATCH --job-name=dw_efc
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Distance-call dead-weight histogram on msmarco_full at one ef_construction.
# Submit one job per efC value:
#
#   for e in 400 1600 3200; do
#     sbatch --job-name=dw_efc$e --export=ALL,EFC=$e run_deadweight_efc_perlmutter.sh
#   done
#
# Single values only in --export; it splits on commas (see the sweep script).
#
# Params are the `frontier` preset -- the msmarco_full best-in-band config:
# M=32 alpha=0.85 u8-quantized, 8x4 inverted seeding, heuristic on. Only efC
# moves. beta does NOT affect the histogram (the refine pass calls
# distanceDense, which is not instrumented) but is kept at the best value so
# the reported recall stays comparable.
#
# BUILD COST scales linearly in efC on this dataset (measured, M=32:
# efC=200 -> 431s, efC=400 -> 879s), so expect roughly 15min / 1h / 2h for
# 400 / 1600 / 3200 plus ~5min load+prune+quantize.

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"
mkdir -p logs

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance

BIN="$SPKNN_BIN/sparse_hnsw_demo_distlog"
bench_assert_optimized "$BIN"

# --- knobs -----------------------------------------------------------------
EFC=${EFC:?set EFC (e.g. --export=ALL,EFC=1600)}
EF=${EF:-3200}                 # search ef, held fixed across the three runs
M=${M:-32}
ALPHA=${ALPHA:-0.85}
BETA=${BETA:-4}
QUANTIZE=${QUANTIZE:-1}        # MUST be 1: distanceQuant is the instrumented kernel
SEED_TOP_K=${SEED_TOP_K:-8}
SEED_TERMS=${SEED_TERMS:-8}
SEED_PER_TERM=${SEED_PER_TERM:-4}
BINS=${BINS:-100}
ROW_STRIDE=${ROW_STRIDE:-500}  # per-row CSV is a sample; the histogram is not

OUT="$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/results/deadweight_efc"
mkdir -p "$OUT"
STEM="${SPKNN_DATASET}_M${M}_efc${EFC}_ef${EF}_a${ALPHA}"
HIST="$OUT/${STEM}_hist.csv"
ROWS="$OUT/${STEM}_rows.csv"
RES="$OUT/${STEM}_result.csv"

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET  ndocs=$SPKNN_NDOCS nnz=$SPKNN_NNZ"
echo "### params  : M=$M efC=$EFC ef=$EF alpha=$ALPHA beta=$BETA q=$QUANTIZE seed=${SEED_TOP_K}x${SEED_TERMS}:${SEED_PER_TERM}"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### hist    : $HIST"
echo "### rows    : $ROWS (every $ROW_STRIDE queries)"
free -g | head -2

$BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
    "$M" "$EFC" "$EF" \
    1 0 0 \
    "$ALPHA" "$BETA" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$RES" \
    "$QUANTIZE" "$SEED_TOP_K" "$SEED_TERMS" "$SEED_PER_TERM" \
    "$ROWS" "$ROW_STRIDE" \
    "$HIST" "$BINS"

echo "########## done -> $HIST ##########"
