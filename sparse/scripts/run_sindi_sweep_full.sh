#!/bin/bash
#SBATCH --job-name=sindi_sweep
#SBATCH --partition=medium
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --output=logs/sindi_sweep_%j.out

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
BIN=$REPO/build-release/bin/sindi_sweep
DATA=$REPO/sparse/data/msmarco_full
RESULTS_DIR=$REPO/sparse/scripts/results/sweeps

mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

DOC_PRUNE_RATIOS=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7
QUERY_PRUNE_RATIOS=0.5,0.4,0.3,0.2,0.1,0.0
# n_candidate only matters via use_reorder rescoring; >=50 was already saturated
# at doc_prune_ratio=0, so the grid stops there.
N_CANDIDATES=10,20,50
TERM_PRUNE_RATIO=0
WINDOW_SIZE=50000
USE_REORDER=1
USE_QUANTIZATION=0

numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL "$BIN" \
  $DOC_PRUNE_RATIOS $QUERY_PRUNE_RATIOS $N_CANDIDATES \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt" \
  "$RESULTS_DIR/sindi_sweep_full.csv" \
  $TERM_PRUNE_RATIO $WINDOW_SIZE $USE_REORDER $USE_QUANTIZATION SINDI
