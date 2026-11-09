#!/bin/bash
#SBATCH --job-name=final_profile
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

# Work per query for the final method sweep (run_final_sweep_perlmutter.sh): how much each
# series reads and computes, which QPS alone does not show.
#
# One graph, the headline configuration (M32 efC200 alpha 0.85, beta 3, seeds 8:4, u8 slots), and
# four series (off, Gap 0.1, inline16, inline16+Gap 0.1) at ef 50, 100, 200, 400. sparse_profile
# runs each point in batch mode and prints PROF lines: recall, distance calls, row bytes and
# 64 B cache lines per query, inline gate checks and rejects per query, and the per-query recall
# distribution; PROF_PERQ_CSV also writes each query's recall. sparse_profile takes one config per
# run, so the graph is rebuilt for every point. Its counters cost time, so its QPS is not reported.
# Output: results/final_sweep/profile_<dataset>_<jobid>/<series>_ef<ef>.{log,perq.csv}
#
# Binary: sparse_profile, built with the rest of sparse/ (PrgEnv-intel Release, see the README,
# "Build"); its CMake target compiles the counters in (SPARSE_HNSW_PROFILE). Set BIN to its path;
# the default is a copy kept aside so that rebuilding build/ while the job is queued cannot change
# it. Each log records the binary's md5.
#
#   cd sparse/scripts/final_sweep
#   sbatch run_final_profile_perlmutter.sh
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade run_final_profile_perlmutter.sh
# Then: analyze_final_profile.py <dataset>.

set -euo pipefail
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
BIN=${BIN:-/pscratch/sd/d/dhaura/cleanup_6b/bin/sparse_profile}
bench_assert_optimized "$BIN"
echo "### bin md5 : $(md5sum "$BIN" | cut -d' ' -f1)"

EFS=${EFS:-50-100-200-400}
GAP=0.1
OUT="$REPO/sparse/scripts/results/final_sweep/profile_${SPKNN_DATASET}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "### dataset : $SPKNN_DATASET   threads=$BENCH_THREADS   efs=$EFS gap=$GAP slot_size=$SLOT_SIZE -> $OUT"

for EF in ${EFS//-/ }; do
    for SERIES in off gap inline16 inline16gap; do
        case $SERIES in
            off) K=0; G=-1 ;;    gap) K=0; G=$GAP ;;
            inline16) K=16; G=-1 ;;    inline16gap) K=16; G=$GAP ;;
        esac
        TAG="${SERIES}_ef${EF}"
        echo "########## $TAG ##########"
        # args: M efC ef alpha beta base queries gt modes nq gate quantize seed_top_k seed_terms
        #       seed_per_term patience slot_size inline_k stop_gap
        PROF_PERQ_CSV="$OUT/$TAG.perq.csv" $BENCH_LAUNCH stdbuf -oL -eL "$BIN" \
            32 200 "$EF" 0.85 3 "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
            batch 0 cold 1 8 8 4 0 "$SLOT_SIZE" "$K" "$G" > "$OUT/$TAG.log" 2>&1
        grep -E "^PROF (search_s|ndist=|lines_per|perq)" "$OUT/$TAG.log"
    done
done
echo "########## done -> $OUT ##########"
