#!/bin/bash
#SBATCH --job-name=ab_sweep
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 physical cores x 2 hyperthreads -> BENCH_THREADS=64
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

module load python/3.11-24.1.0

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_hnsw_sweep"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/alpha_beta}
mkdir -p "$OUT"
CSV="$OUT/alpha_beta_sweep${TAG:+_$TAG}.csv"
rm -f "$CSV"

M=${M:-32}
EFC=${EFC:-200}
EF_LIST=${EF_LIST:-10,20,50,100,200,400,800,1600,3200}
ALPHAS=${ALPHAS:-0.5 0.6 0.7 0.8 0.9 1.0}
BETAS=${BETAS:-1 2 3 4 5}

echo "dataset=$SPKNN_DATASET  M=$M efC=$EFC  ef=[$EF_LIST]  threads=$BENCH_THREADS"
echo "alphas=[$ALPHAS]  betas=[$BETAS]"
echo "=========================================================================="

for ALPHA in $ALPHAS; do
  if [ "$ALPHA" = "1.0" ]; then combo_betas="1"; else combo_betas="$BETAS"; fi
  for BETA in $combo_betas; do
    echo
    echo "######## alpha=$ALPHA beta=$BETA ########"
    $BENCH_LAUNCH stdbuf -oL -eL "$SPKNN_BIN/sparse_hnsw_sweep" \
      "$M" "$EFC" "$EF_LIST" 1 0 0 "$ALPHA" "$BETA" \
      "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
      "$CSV" "SparseHNSW_a${ALPHA}_b${BETA}" \
      "${REPEATS:-5}" "${WARMUP:-1}"
  done
done

echo
echo "=========================================================================="
echo "wrote $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
echo "analyse with: python3 analyze_alpha_beta.py $CSV"
