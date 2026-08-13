#!/bin/bash
#SBATCH --job-name=ab_figures
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

module load python/3.11-24.1.0 2>/dev/null

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
source "$SPKNN_VENV/bin/activate"

AB="$SPKNN_OUT_ROOT/alpha_beta"
OUT="$SPKNN_PLAYGROUND/results/${SPKNN_DATASET}_perlmutter/alpha_beta"
mkdir -p "$OUT"

shopt -s nullglob
CSVS=("$AB"/alpha_beta_sweep*.csv)
if [ ${#CSVS[@]} -eq 0 ]; then
    echo "FATAL: no alpha_beta_sweep*.csv under $AB" >&2
    exit 1
fi
echo "inputs:"; for f in "${CSVS[@]}"; do echo "  $f ($(($(wc -l < "$f") - 1)) rows)"; done

echo
echo "=================== iso-recall analysis ==================="
python3 analyze_alpha_beta.py "${CSVS[@]}" \
    --targets "${TARGETS:-0.90,0.95,0.98,0.99}" \
    --baseline "${BASELINE:-0.8/3}" | tee "$OUT/analysis.txt"

echo
echo "=================== figures ==================="
python3 plot_alpha_beta_perlmutter.py "${CSVS[@]}" \
    -o "$OUT" \
    --candidates "${CANDIDATES:-auto}" \
    --baseline "${BASELINE:-0.8/3}" \
    --label "${SPKNN_DATASET} (${SPKNN_NDOCS} docs, ${BENCH_THREADS} threads, M=${M:-32} efC=${EFC:-200})"

echo
echo "wrote $OUT"
ls -la "$OUT"
