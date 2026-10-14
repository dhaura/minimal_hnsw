#!/bin/bash
#SBATCH --job-name=diag_perq
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

#   concentrated  -> a cheap per-query fallback (detect a hard query, route it
#                    to an inverted scan)
#   even          -> no routing trick helps; only a better graph will do, which
#                    points back at the M/efC and heuristic experiments.
#
# Emits one row per query per ef: containment@ef, containment@khat, candidate
# set size, and the query's term count (the obvious hardness covariate).

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

module load python/3.11-24.1.0
source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance

OUT=$SPKNN_OUT_ROOT/diag_perquery
mkdir -p "$OUT"

$BENCH_LAUNCH stdbuf -oL "$SPKNN_BIN/sparse_diag_sweep" \
  32 200 "${EF_LIST:-400,1600,3200}" 1 0 0 "${ALPHA:-0.85}" "${BETA:-8}" \
  "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
  "$OUT/diag_results.csv" "$OUT/diag_decomp.csv" \
  DiagPerQuery "${REPEATS:-1}" "${WARMUP:-0}" 0 1 \
  "$OUT/perquery"

echo "########## per-query files ##########"
ls -la "$OUT"/perquery.ef*.csv
