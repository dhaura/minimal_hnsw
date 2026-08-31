#!/bin/bash
#SBATCH --job-name=decouple
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out

# Decoupled build/search sweep on msmarco_full (see decouple_main.cpp).
#
# Select a grid with a single comma-free variable:
#     sbatch --export=ALL,PRESET=highrecall run_decouple_perlmutter.sh

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

PRESET=${PRESET:-highrecall}
case "$PRESET" in
    baseline)
        BUILD_ALPHA=0.85
        SEARCH_ALPHAS=0.85
        BETAS=3,4,5
        EF_LIST=100,200,400,800,1600,3200
        PAT_LIST=64,128,256,512,2048
        MODEL=DecoupleBase
        ;;
    highrecall)
        BUILD_ALPHA=0.85
        SEARCH_ALPHAS=0.7,0.85
        BETAS=10,20,40
        EF_LIST=3200,6400,12800
        PAT_LIST=2048,8192
        MODEL=DecoupleHR
        ;;
    *)
        echo "unknown PRESET='$PRESET' (known: baseline, highrecall)" >&2
        exit 1
        ;;
esac

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_decouple_sweep"

OUT=$SPKNN_OUT_ROOT/decouple/decouple_${PRESET}_$SLURM_JOB_ID.csv
mkdir -p "$(dirname "$OUT")" logs

echo "### preset=$PRESET build_alpha=$BUILD_ALPHA search_alphas=$SEARCH_ALPHAS"
echo "### betas=$BETAS ef=$EF_LIST pat=$PAT_LIST"

$BENCH_LAUNCH stdbuf -oL "$SPKNN_BIN/sparse_decouple_sweep" \
    "${M:-32}" "${EFC:-200}" "$EF_LIST" 1 \
    "$BUILD_ALPHA" "$SEARCH_ALPHAS" "$BETAS" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$OUT" "$MODEL" "${REPEATS:-3}" "${WARMUP:-1}" \
    "${QUANTIZE:-1}" "${SEED_TOP_K:-8}" "${SEED_SPEC:-8:4}" \
    "$PAT_LIST"

echo "########## done -> $OUT ##########"
