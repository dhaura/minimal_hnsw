#!/bin/bash
#SBATCH --job-name=v2_hnsw
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256   # 128 physical cores x 2 HT -> BENCH_THREADS=128
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# One SparseHNSW build on msmarco_v2 SPLADE++ ED (138.4M passages), swept over
# ef x beta x patience.
#
#   sbatch run_v2_splade_hnsw_perlmutter.sh
#   sbatch --export=ALL,QSET=dl21 run_v2_splade_hnsw_perlmutter.sh

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"
mkdir -p logs

module load python/3.11-24.1.0

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

QSET=${QSET:-dev}
DATA=$SPKNN_HNSW_REPO/sparse/data/msmarco_v2_splade
export SPKNN_BASE=$DATA/base_v2_splade.csr
export SPKNN_QUERIES=$DATA/queries.$QSET.csr
export SPKNN_GT=$DATA/base_v2_splade.$QSET.gt
for f in "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT"; do
    [ -s "$f" ] || { echo "FATAL missing $f" >&2; exit 1; }
done

OUT=${SPKNN_OUT:-$SCRATCH/datasets/SpKNN/msmarco_v2_splade/sparse_hnsw}
mkdir -p "$OUT"
CSV="$OUT/sparse_hnsw_v2${TAG:+_$TAG}_${SLURM_JOB_ID}.csv"

ALPHA=${ALPHA:-0.85}
BETAS=${BETAS:-3,4}
EF_LIST=${EF_LIST:-50,100,200,400,800,1600,3200}
PATIENCE_LIST=${PATIENCE_LIST:-0,256}

echo "### dataset  : msmarco_v2_splade / $QSET"
echo "### base     : $SPKNN_BASE"
echo "### queries  : $SPKNN_QUERIES"
echo "### gt       : $SPKNN_GT"
echo "### threads  : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### M=${M:-32} efC=${EFC:-200} alpha=$ALPHA beta=$BETAS"
echo "### ef=$EF_LIST patience=$PATIENCE_LIST quantize=${QUANTIZE:-1} seed=8x${SEED_SPEC:-8:4}"
echo "### csv      : $CSV"
free -g | head -2

$BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
    "${M:-32}" "${EFC:-200}" \
    "$EF_LIST" \
    1 0 0 \
    "$ALPHA" "$BETAS" \
    "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
    "$CSV" \
    "${MODEL:-SparseHNSW_v2_a${ALPHA}}" \
    "${REPEATS:-3}" "${WARMUP:-1}" "${QUANTIZE:-1}" \
    "${SEED_TOP_K:-8}" "${SEED_SPEC:-8:4}" "$PATIENCE_LIST"

echo "########## done -> $CSV ##########"
