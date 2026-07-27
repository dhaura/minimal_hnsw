#!/bin/bash
#SBATCH --job-name=hnsw_ef_sweep
#SBATCH --partition=medium
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --output=logs/hnsw_ef_sweep_%j.out

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
BIN=$REPO/build-release/bin/sparse_hnsw_sweep
DATA=$REPO/sparse/data/msmarco_full
RESULTS_DIR=$REPO/sparse/scripts/results/sweeps

mkdir -p "$RESULTS_DIR"

export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=close

M=16
EF_CONSTRUCTION=200
ALPHA=0.8
BETA=3
EF_LIST=10,20,30,50,75,100,150,200,300,500,750,1000,1500,2000

numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL "$BIN" \
  $M $EF_CONSTRUCTION $EF_LIST 1 0 0 $ALPHA $BETA \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt" \
  "$RESULTS_DIR/hnsw_ef_sweep_full.csv" SparseHNSW
