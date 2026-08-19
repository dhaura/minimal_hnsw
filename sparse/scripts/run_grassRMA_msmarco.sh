#!/bin/bash
#SBATCH --qos=normal
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --output=logs/grassRMA_full_%j.out

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
BIN=$REPO/build-release/bin/grassRMA_demo
DATA=$REPO/sparse/data/msmarco_full
SCRIPTS=$REPO/sparse/scripts
LOG=$SCRIPTS/logs/grassRMA_full_${SLURM_JOB_ID}.out

export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL "$BIN" 16 200 150 \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt"

"$SCRIPTS/append_competitor_result.sh" "$LOG" grassRMA "M=16 efC=200 ef=150" 48 \
  "$SCRIPTS/results/competitors.csv"
