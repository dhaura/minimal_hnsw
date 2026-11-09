#!/bin/bash
#SBATCH --job-name=final_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Final method sweep: the QPS / recall points behind the paper's SparseHNSW results.
#
# Each job builds one HNSW graph per (M, efC, alpha) and searches it with every combination of
# ef, beta (refine factor), stop rule and gate layout, so four series share one graph and one node:
#   SparseHNSW            no gate, no early stop   (the baseline the other three are compared with)
#   SparseHNSW +Gap       no gate, score-gap stop  (Params carry gap=)
#   SparseHNSW_inline16   inline top-16 gate, no early stop
#   SparseHNSW_inline16 +Gap
# All series use the method's defaults: u8 quantized rows in fixed-size slots, inverted seeds
# (top_k 8, 8 query terms x 4 docs per term), 5 timed runs after 1 warmup run.
# Slot size per dataset: msmarco_full 320 B, nq_splade 448 B.
#
# Presets hold the grids (sbatch --export splits values on commas, so lists use dashes):
#   alpha  : M=32 efC=200, alpha from ALPHAS_LIST, small ef           (low/mid recall)
#   grid   : alpha 0.85, M from --export, efC from EFCS               (mid/high recall)
#   lowrec : as alpha, but every point with seeds off and on. Seeds raise the lowest reachable
#            recall to ~0.87 (msmarco_full) / ~0.82 (nq_splade); the unseeded runs cover 0.75-0.87.
# Output: results/final_sweep/full_sweep_<dataset>/final_<dataset>_[lowrec_]M.._efc.._a.._<jobid>.csv
#
# Binary: sparse_hnsw_sweep from a Release build made with PrgEnv-intel (see the README, "Build").
# Set BIN to its path; the default is a copy of that build kept aside so that rebuilding build/
# while jobs are queued cannot change the binary under them. Each log records the binary's md5.
#
#   cd sparse/scripts/final_sweep
#   sbatch --export=ALL,PRESET=alpha,ALPHAS_LIST=0.75-0.8-0.85-0.9 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=32,EFCS=200-400-800-1600 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=grid,M=48,EFCS=800-1600 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,PRESET=lowrec,ALPHAS_LIST=0.65-0.7-0.75-0.8-0.85 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,PRESET=alpha,ALPHAS_LIST=0.65-0.7-0.725-0.75-0.8 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,PRESET=grid,M=32,EFCS=200-400-800-1600 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,PRESET=grid,M=64,EFCS=400-800 run_final_sweep_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade,PRESET=lowrec,ALPHAS_LIST=0.6-0.65-0.7-0.75-0.8 run_final_sweep_perlmutter.sh
# Then: analyze_final_sweep.py (tables) and make_final_figures.sh (figures vs baselines).

set -euo pipefail
PRESET=${PRESET:?set PRESET: alpha, grid or lowrec}
LAYOUTS=off,inline16
REPO="$SCRATCH/repos/sparse_hnsw/minimal_hnsw"
cd "$REPO/sparse/scripts/final_sweep"
mkdir -p logs

source "${SPKNN_HNSW_REPO:-$SCRATCH/repos/sparse_hnsw/minimal_hnsw}/sparse/scripts/bench_env_perlmutter.sh"
bench_provenance
case "$SPKNN_DATASET" in
    msmarco_full) DEF_SLOT=320 ;;
    nq_splade) DEF_SLOT=448 ;;
    *) echo "unsupported SPKNN_DATASET=$SPKNN_DATASET" >&2; exit 1 ;;
esac
SLOT_SIZE=${SLOT_SIZE:-$DEF_SLOT}
BIN=${BIN:-/pscratch/sd/d/dhaura/cleanup_6b/bin/sparse_hnsw_sweep}
bench_assert_optimized "$BIN"
echo "### bin md5 : $(md5sum "$BIN" | cut -d' ' -f1)"

GAPS=g0.05,g0.1,g0.15,g0.2,g0.3
SEEDS=8:4
NAME_TAG=""   # lowrec files are named final_<dataset>_lowrec_M...
case "$PRESET" in
    lowrec)
        M=32; EFCS="200"; ALPHAS=${ALPHAS_LIST:?set ALPHAS_LIST}; ALPHAS=${ALPHAS//-/ }
        EF_LIST=10,15,20,25,30,40,50,75,100,200,400
        BETAS=2,3,4; STOPS=0,$GAPS; SEEDS=0:1,8:4; NAME_TAG=lowrec_ ;;
    alpha)
        M=32; EFCS="200"; ALPHAS=${ALPHAS_LIST:?set ALPHAS_LIST}; ALPHAS=${ALPHAS//-/ }
        EF_LIST=10,15,20,25,30,40,50,75,100,200,400
        BETAS=2,3,4; STOPS=0,$GAPS ;;
    grid)
        M=${M:?set M}; EFCS=${EFCS:?set EFCS, e.g. 200-400-800-1600}; EFCS=${EFCS//-/ }; ALPHAS="0.85"
        EF_LIST=10,20,30,40,50,100,200,400,800,1600
        BETAS=2,3,4,5,8; STOPS=0,$GAPS ;;
    *) echo "unknown PRESET=$PRESET" >&2; exit 1 ;;
esac

OUT="$REPO/sparse/scripts/results/final_sweep/full_sweep_${SPKNN_DATASET}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   threads=$BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### preset  : $PRESET M=$M efC=$EFCS alpha=$ALPHAS seeds=$SEEDS stops=$STOPS layouts=$LAYOUTS slot_size=$SLOT_SIZE"

for EFC in $EFCS; do
    for ALPHA in $ALPHAS; do
        CSV="$OUT/final_${SPKNN_DATASET}_${NAME_TAG}M${M}_efc${EFC}_a${ALPHA}_${SLURM_JOB_ID}.csv"
        echo "########## M=$M efC=$EFC alpha=$ALPHA -> $CSV ##########"
        # args after the output CSV: model name, repeats 5, warmup 1, quantize 1, seed table top_k 8,
        # seeds (terms:per_term, 0:1 = no seeds), stops (0 = none, gG = score gap G),
        # -slot size (negative = every row in a fixed slot of that many bytes), gate layouts.
        $BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
            "$M" "$EFC" "$EF_LIST" "$ALPHA" "$BETAS" \
            "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            "$CSV" SparseHNSW 5 1 1 8 "$SEEDS" "$STOPS" "-$SLOT_SIZE" "$LAYOUTS"
    done
done
echo "########## done -> $OUT ##########"
