#!/bin/bash
#SBATCH --job-name=seismic_v2
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256   # 128 physical cores x 2 HT -> BENCH_THREADS=128
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=12:00:00

# Seismic on msmarco_v2 SPLADE++ ED (138.4M passages), at the msmarco_full
# optimum: n_postings=6000 summary_energy=0.8 centroid_fraction=0.4, which
# reached 99.90% recall@10 there (job 57842801).
#
#   sbatch run_seismic_v2_perlmutter.sh

set -euo pipefail
BASELINES=${SPKNN_HNSW_REPO:-$SCRATCH/repos/sparse_hnsw/minimal_hnsw}/sparse/baselines
cd "$BASELINES/seismic"
mkdir -p logs

module load python/3.11-24.1.0

# bench_env resolves every path from these two.
export SPKNN_DATASET=${SPKNN_DATASET:-msmarco_v2_splade}
export SPKNN_QSET=${SPKNN_QSET:-dev}

source "$BASELINES/../scripts/bench_env_perlmutter.sh"
bench_provenance
source "$SPKNN_VENV/bin/activate"

for f in "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT"; do
    [ -s "$f" ] || { echo "FATAL missing $f" >&2; exit 1; }
done

# PRESET carries the list-valued knobs -- `sbatch --export` splits on commas, so
# N_KNN_LIST cannot be passed that way. `knn` adds the kNN-graph refinement on
# top of the inverted index; it beat edge-only at the top bin on msmarco_full.
case "${PRESET:-edge}" in
    edge) NKNN=${NKNN:-0};      N_KNN_LIST=${N_KNN_LIST:-0} ;;
    knn)  NKNN=${NKNN:-16};     N_KNN_LIST=${N_KNN_LIST:-0,4,8,16} ;;
    *) echo "unknown PRESET='$PRESET'; known: edge, knn" >&2; exit 1 ;;
esac

OUT=${SPKNN_OUT:-$SPKNN_OUT_ROOT/seismic}
mkdir -p "$OUT/bin"
CSV="$OUT/seismic_${SPKNN_DATASET}${TAG:+_$TAG}_${SLURM_JOB_ID}.csv"

echo "### dataset : $SPKNN_DATASET / $SPKNN_QSET"
echo "### threads : $BENCH_THREADS   launch: $BENCH_LAUNCH"
echo "### params  : n_postings=${NPOST:-6000} summary_energy=${SENERGY:-0.8} centroid_fraction=${CFRAC:-0.4} doc_cut=${DOC_CUT:-15} preset=${PRESET:-edge} nknn=$NKNN n_knn_list=$N_KNN_LIST"
echo "### csv     : $CSV"
df -h "$OUT" | tail -1
free -g | head -2

$BENCH_LAUNCH stdbuf -oL -eL python3 seismic_mt_ex.py \
    -n_postings "${NPOST:-6000}" -summary_energy "${SENERGY:-0.8}" \
    -centroid_fraction "${CFRAC:-0.4}" \
    -min_cluster_size "${MIN_CLUSTER:-2}" -doc_cut "${DOC_CUT:-15}" \
    -max_fraction "${MAX_FRACTION:-6}" \
    -sweep "${SEISMIC_SWEEP:-1:1.0,2:1.0,3:1.0,3:0.9,4:0.9,5:0.9,5:1.0,6:0.9,7:0.9,8:0.9,10:0.9,10:0.8,12:0.8,14:0.8,20:0.8,20:0.7,30:0.7,30:0.8,50:0.7}" \
    -nknn "$NKNN" -n_knn_list "$N_KNN_LIST" \
    -sorted "${SORTED:-both}" \
    -repeats "${REPEATS:-3}" -warmup "${WARMUP:-1}" \
    -input   "$SPKNN_BASE" \
    -query   "$SPKNN_QUERIES" \
    -gt      "$SPKNN_GT" \
    -bin_dir "$OUT/bin" \
    -csv     "$CSV"

echo "########## done -> $CSV ##########"
