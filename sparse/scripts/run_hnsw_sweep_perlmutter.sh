#!/bin/bash
#SBATCH --job-name=hnsw_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 threads; override for big datasets
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# sparse_hnsw ef sweep on any dataset bench_env knows.
#
#   sbatch run_hnsw_sweep_perlmutter.sh                                  # msmarco_full
#   sbatch --export=ALL,SPKNN_DATASET=nq_splade run_hnsw_sweep_perlmutter.sh
#   sbatch --cpus-per-task=256 \
#          --export=ALL,SPKNN_DATASET=msmarco_v2_splade,PRESET=frontier \
#          run_hnsw_sweep_perlmutter.sh
#
# Paths come from bench_env's SPKNN_DATASET/SPKNN_QSET, so there is nothing
# dataset-specific in here. SPKNN_QSET picks the query set where a dataset has
# more than one (msmarco_v2_splade: dev, dev2, dl21, dl22, dl23).
#
# THREADS: bench_env derives BENCH_THREADS from --cpus-per-task/2, and at <=64
# threads it pins with `numactl --cpunodebind=0-3 --interleave=0-3`, which caps
# allocation at 256 GB. msmarco_v2_splade needs ~241 GB, so it needs
# --cpus-per-task=256 (128 threads, --interleave=all). The guard below refuses
# to start rather than let it die 80 minutes in.
#
# COMMAS: `sbatch --export` splits on commas, so EF_LIST=50,100 as an --export
# value does NOT survive. Use a PRESET name for anything list-valued; the
# defaults below are set inside the script where commas are safe.

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"
mkdir -p logs

module load python/3.11-24.1.0

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

# --- presets (list-valued knobs live here, not in --export) ----------------
case "${PRESET:-default}" in
    frontier)   # the msmarco_full best-in-band config: u8 + 8x4 seeding
        M=${M:-32}; EFC=${EFC:-200}; ALPHA=${ALPHA:-0.85}
        BETAS=${BETAS:-3,4}
        EF_LIST=${EF_LIST:-50,100,200,400,800,1600,3200}
        PATIENCE_LIST=${PATIENCE_LIST:-0,256}
        QUANTIZE=${QUANTIZE:-1}; SEED_TOP_K=${SEED_TOP_K:-8}; SEED_SPEC=${SEED_SPEC:-8:4}
        ;;
    highrecall)
        M=${M:-32}; EFC=${EFC:-200}; ALPHA=${ALPHA:-0.9}
        BETAS=${BETAS:-4,6}
        EF_LIST=${EF_LIST:-400,800,1600,3200,6400}
        PATIENCE_LIST=${PATIENCE_LIST:-0,256}
        QUANTIZE=${QUANTIZE:-1}; SEED_TOP_K=${SEED_TOP_K:-8}; SEED_SPEC=${SEED_SPEC:-8:4}
        ;;
    exhaustive) # full v2 grid: vary M/EFC/ALPHA per job via --export, sweep the
                # rest in-process. beta up to 8 -- v1's 99% bin needed it.
        M=${M:-32}; EFC=${EFC:-200}; ALPHA=${ALPHA:-0.85}
        BETAS=${BETAS:-2,3,4,8}
        EF_LIST=${EF_LIST:-20,50,100,200,400,800,1600,3200}
        PATIENCE_LIST=${PATIENCE_LIST:-0,256}
        QUANTIZE=${QUANTIZE:-1}; SEED_TOP_K=${SEED_TOP_K:-8}; SEED_SPEC=${SEED_SPEC:-8:4}
        ;;
    quick)      # smoke test: one build, three points
        M=${M:-32}; EFC=${EFC:-200}; ALPHA=${ALPHA:-0.85}
        BETAS=${BETAS:-3}
        EF_LIST=${EF_LIST:-50,200,800}
        PATIENCE_LIST=${PATIENCE_LIST:-0}
        QUANTIZE=${QUANTIZE:-1}; SEED_TOP_K=${SEED_TOP_K:-8}; SEED_SPEC=${SEED_SPEC:-8:4}
        ;;
    default)    # unquantized, unseeded -- the plain graph
        M=${M:-32}; EFC=${EFC:-200}; ALPHA=${ALPHA:-0.8}
        BETAS=${BETAS:-3}
        EF_LIST=${EF_LIST:-10,20,50,100,200,400,800,1600,3200}
        PATIENCE_LIST=${PATIENCE_LIST:-0}
        QUANTIZE=${QUANTIZE:-0}; SEED_TOP_K=${SEED_TOP_K:-0}; SEED_SPEC=${SEED_SPEC:-off}
        ;;
    *)
        echo "unknown PRESET='$PRESET'; known: default, frontier, highrecall, exhaustive, quick" >&2
        exit 1
        ;;
esac

# --- guards ----------------------------------------------------------------
# A >2^32-nnz dataset needs a 64-bit-clean distance kernel AND the full node.
if [ "${SPKNN_BIG:-0}" = "1" ] && [ "$BENCH_THREADS" -le 64 ]; then
    echo "FATAL $SPKNN_DATASET has $SPKNN_NNZ nnz and needs --interleave=all;" >&2
    echo "      resubmit with --cpus-per-task=256 (got BENCH_THREADS=$BENCH_THREADS)." >&2
    exit 1
fi

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/sparse_hnsw}
mkdir -p "$OUT"
CSV="$OUT/sparse_hnsw_${SPKNN_DATASET}_${SPKNN_QSET}${TAG:+_$TAG}_${SLURM_JOB_ID}.csv"
MODEL=${MODEL:-SparseHNSW_a${ALPHA}}

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET   ndocs=$SPKNN_NDOCS nnz=$SPKNN_NNZ"
echo "### base    : $SPKNN_BASE"
echo "### queries : $SPKNN_QUERIES"
echo "### gt      : $SPKNN_GT"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### preset  : ${PRESET:-default}  M=$M efC=$EFC alpha=$ALPHA beta=$BETAS"
echo "### search  : ef=$EF_LIST patience=$PATIENCE_LIST quantize=$QUANTIZE seed=${SEED_TOP_K}x${SEED_SPEC}"
echo "### csv     : $CSV"
free -g | head -2

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
    "$M" "$EFC" \
    "$EF_LIST" \
    1 0 0 \
    "$ALPHA" "$BETAS" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$CSV" \
    "$MODEL" \
    "${REPEATS:-3}" "${WARMUP:-1}" "$QUANTIZE" \
    "$SEED_TOP_K" "$SEED_SPEC" "$PATIENCE_LIST"

echo "########## done -> $CSV ##########"
