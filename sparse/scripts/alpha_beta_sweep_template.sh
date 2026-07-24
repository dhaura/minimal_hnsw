#!/bin/bash
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=logs/alpha_beta_a{{ALPHA}}_b{{BETA}}_%j.out

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
BIN=$REPO/build/bin/sparse_hnsw_demo
DATA=$REPO/sparse/data/msmarco_full
RESULTS_DIR=$REPO/sparse/scripts/results

export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close

numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL "$BIN" \
  16 200 150 1 0 0 {{ALPHA}} {{BETA}} \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt" \
  "$RESULTS_DIR/raw_a{{ALPHA}}_b{{BETA}}.csv"
