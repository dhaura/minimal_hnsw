#!/bin/bash
#SBATCH --job-name=grassRMA_v2
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256   # 128 physical cores x 2 HT -> BENCH_THREADS=128
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=12:00:00

# GrassRMA on msmarco_v2 SPLADE++ ED (138.4M passages).
#
#   sbatch run_grassRMA_v2_perlmutter.sh
#
# REQUIRES the int64 offset fix in GrassRMA/hnswlib/space_ip.h. That file had the
# identical uint32 truncation of CSR indptr that sparse_hnsw.cpp did, and v2 is
# the first dataset big enough to trip it (17.5e9 nnz = 4.07x the uint32 range).

set -euo pipefail
BASELINES=${SPKNN_HNSW_REPO:-$SCRATCH/repos/sparse_hnsw/minimal_hnsw}/sparse/baselines
cd "$BASELINES/grassRMA"
mkdir -p logs

module load python/3.11-24.1.0

export SPKNN_DATASET=${SPKNN_DATASET:-msmarco_v2_splade}
export SPKNN_QSET=${SPKNN_QSET:-dev}

source "$BASELINES/../scripts/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

for f in "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT"; do
    [ -s "$f" ] || { echo "FATAL missing $f" >&2; exit 1; }
done

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/grassRMA}
mkdir -p "$OUT"
CSV="$OUT/grassRMA_${SPKNN_DATASET}${TAG:+_$TAG}_${SLURM_JOB_ID}.csv"

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET  ndocs=$SPKNN_NDOCS"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### params  : M=${M:-32} efC=${EFC:-200} ef=${EF_LIST:-20,50,100,200,400,800,1600,3200,6400}"
echo "### csv     : $CSV"
python3 -c "import sparse_hnswlib,os; print('### module  :', sparse_hnswlib.__file__, os.path.getmtime(sparse_hnswlib.__file__))"
free -g | head -2

$BENCH_LAUNCH stdbuf -oL -eL python3 grassRMA_ex.py \
    -n "$SPKNN_NDOCS" \
    -M "${M:-32}" -ef_construction "${EFC:-200}" \
    -ef_list "${EF_LIST:-20,50,100,200,400,800,1600,3200,6400}" \
    -repeats "${REPEATS:-3}" -warmup "${WARMUP:-1}" \
    -input  "$SPKNN_BASE" \
    -query  "$SPKNN_QUERIES" \
    -gt     "$SPKNN_GT" \
    -csv    "$CSV"

echo "########## done -> $CSV ##########"
