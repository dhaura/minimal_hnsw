#!/bin/bash
#SBATCH --job-name=pyanns
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
cd "$BASELINES/pyanns"

module load python/3.11-24.1.0

source "$BASELINES/../scripts/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/pyanns}
mkdir -p "$OUT/indices"
rm -f "$OUT/pyanns_results${TAG:+_$TAG}.csv"

$BENCH_LAUNCH stdbuf -oL -eL python3 pyanns_ex.py \
    -n "$SPKNN_NDOCS" -ef_list "${EF_LIST:-80,150,300,600,1200,2400,4800}" \
    -budgets "${BUDGETS:-0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5,0.8,0.9,1.0}" \
    -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" \
    -input  "$SPKNN_BASE" \
    -query  "$SPKNN_QUERIES" \
    -gt     "$SPKNN_GT" \
    -R "${PYANNS_R:-32}" -L "${PYANNS_L:-200}" \
    -index  "${PYANNS_INDEX:-$OUT/indices/${SPKNN_DATASET}_R${PYANNS_R:-32}_L${PYANNS_L:-200}}" \
    -csv    "$OUT/pyanns_results${TAG:+_$TAG}.csv"
