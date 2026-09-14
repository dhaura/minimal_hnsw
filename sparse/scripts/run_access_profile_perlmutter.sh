#!/bin/bash
#SBATCH --job-name=access_prof
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Records which node's row each distance call fetches, then reports the reuse
# factor and a coverage curve at cache sizes matched to this hardware (L2 1 MiB
# per core, L3 32 MiB per CCD on the EPYC 7763, 256 MiB across both sockets).
# Reported in BYTES as well as fetch counts, because rows vary in length and a
# cache is sized in bytes.

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

module load python/3.11-24.1.0
source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
bench_provenance
bench_assert_optimized "$SPKNN_BIN/sparse_access_profile"

OUT=$SPKNN_OUT_ROOT/access_profile
mkdir -p "$OUT"
TAG=${TAG:-efc${EFC:-200}_ef${EF:-3200}}

$BENCH_LAUNCH stdbuf -oL "$SPKNN_BIN/sparse_access_profile" \
  "${M:-32}" "${EFC:-200}" "${EF:-3200}" "${ALPHA:-0.85}" "${BETA:-4}" "${PAT:-2048}" \
  "$SPKNN_BASE" "$SPKNN_QUERIES" "$SPKNN_GT" \
  "$OUT/$TAG" 1 8 8:4

echo "########## done -> $OUT/$TAG ##########"
