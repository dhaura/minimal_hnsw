#!/bin/bash
#SBATCH --job-name=kannolo
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 physical cores x 2 hyperthreads -> BENCH_THREADS=64
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=12:00:00

set -euo pipefail
BASELINES=${SPKNN_HNSW_REPO:-$SCRATCH/repos/sparse_hnsw/minimal_hnsw}/sparse/baselines
cd "$BASELINES/kannolo"

module load python/3.11-24.1.0

source "$BASELINES/../scripts/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/kannolo}
mkdir -p "$OUT/indices"
rm -f "$OUT/kannolo_results${TAG:+_$TAG}.csv"

$BENCH_LAUNCH stdbuf -oL -eL python3 kannolo_mt_ex.py \
    -m "${M:-32}" -ef_construction "${EFC:-200}" \
    -ef_list "${EF_LIST:-10,20,50,100,200,400,800,1600,3200,6400,12800,25600}" \
    -early_exit_list "${EARLY_EXIT_LIST:-none,0.005,0.01,0.05,0.1,0.25}" \
    -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
    -input  "$SPKNN_BASE" \
    -query  "$SPKNN_QUERIES" \
    -gt     "$SPKNN_GT" \
    -index  "$OUT/indices/${SPKNN_DATASET}_m${M:-32}_efc${EFC:-200}.index" \
    -csv    "$OUT/kannolo_results${TAG:+_$TAG}.csv"
