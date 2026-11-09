#!/bin/bash
#SBATCH --job-name=seismic
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
cd "$BASELINES/seismic"

module load python/3.11-24.1.0

source "$BASELINES/../scripts/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/seismic}
mkdir -p "$OUT/indices" "$OUT/bin"
CSV="$OUT/seismic_results${TAG:+_$TAG}.csv"
rm -f "$CSV"
case "${PRESET:-}" in
  edge)
    NPOST_LIST=${NPOST_LIST:-500 1000 2000 6000}
    SENERGY_LIST=${SENERGY_LIST:-0.6 0.7 0.8}
    CFRAC_LIST=${CFRAC_LIST:-0.2 0.3 0.4}
    SORTED=${SORTED:-both}
    ;;
  knn)
    NPOST_LIST=${NPOST_LIST:-2000 6000}
    SENERGY_LIST=${SENERGY_LIST:-0.6}
    CFRAC_LIST=${CFRAC_LIST:-0.2}
    NKNN=${NKNN:-16}
    N_KNN_LIST=${N_KNN_LIST:-0,4,8,16}
    SORTED=${SORTED:-both}
    ;;
esac

for NPOST in ${NPOST_LIST:-2000 3000 6000}; do
  for SENERGY in ${SENERGY_LIST:-0.4 0.5 0.6}; do
    for CFRAC in ${CFRAC_LIST:-0.05 0.1 0.2}; do
    echo
    echo "######## seismic index: n_postings=$NPOST summary_energy=$SENERGY centroid_fraction=$CFRAC ########"
    $BENCH_LAUNCH stdbuf -oL -eL python3 seismic_mt_ex.py \
        -n_postings "$NPOST" -summary_energy "$SENERGY" \
        -centroid_fraction "$CFRAC" \
        -min_cluster_size "${MIN_CLUSTER:-2}" \
        -max_fraction "${MAX_FRACTION:-6}" \
        -sweep "${SEISMIC_SWEEP:-1:1.0,2:1.0,3:1.0,3:0.9,4:0.9,5:0.9,5:1.0,6:0.9,7:0.9,8:0.9,10:0.9,10:0.8,12:0.8,14:0.8,20:0.8,20:0.7,30:0.7,30:0.8,50:0.7}" \
        -nknn "${NKNN:-0}" -n_knn_list "${N_KNN_LIST:-0}" \
        -sorted "${SORTED:-true}" \
        -repeats "${REPEATS:-5}" -warmup "${WARMUP:-1}" ${RM_INDEX:+-rm_index} \
        -input   "$SPKNN_BASE" \
        -query   "$SPKNN_QUERIES" \
        -gt      "$SPKNN_GT" \
        -bin_dir "$OUT/bin" \
        -index   "$OUT/indices/${SPKNN_DATASET}_np${NPOST}_se${SENERGY}_cf${CFRAC}.index" \
        -csv     "$CSV"
    done
  done
done
