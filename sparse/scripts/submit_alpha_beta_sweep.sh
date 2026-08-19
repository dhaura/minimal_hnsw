#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/alpha_beta_sweep_template.sh"
TMP_DIR="$SCRIPT_DIR/tmp"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$TMP_DIR" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/results"

echo "Building sparse_hnsw_demo before submitting (so every job picks up the current source)..."
module load GCC/13.2.0
cmake --build "$REPO_ROOT/build-release" --target sparse_hnsw_demo -j8

ALPHAS=(0.5 0.6 0.7 0.8 0.9 1.0)
BETAS=(1 2 3)

for alpha in "${ALPHAS[@]}"; do
  if [ "$alpha" = "1.0" ]; then
    combo_betas=(1)
  else
    combo_betas=("${BETAS[@]}")
  fi
  for beta in "${combo_betas[@]}"; do
    job_script="$TMP_DIR/alpha_beta_a${alpha}_b${beta}.sh"
    sed -e "s/{{ALPHA}}/${alpha}/g" -e "s/{{BETA}}/${beta}/g" "$TEMPLATE" > "$job_script"

    # Submit from SCRIPT_DIR so the relative logs/ path in #SBATCH --output works.
    job_id=$(cd "$SCRIPT_DIR" && sbatch --parsable "$job_script")
    echo "Submitted job $job_id for alpha=$alpha beta=$beta ($job_script)"
  done
done
