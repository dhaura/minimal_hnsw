#!/bin/bash
#SBATCH --qos=normal
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --output=logs/sindi_full_%j.out

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
BIN=$REPO/build/bin/sindi_demo
DATA=$REPO/sparse/data/msmarco_full
SCRIPTS=$REPO/sparse/scripts
LOG=$SCRIPTS/logs/sindi_full_${SLURM_JOB_ID}.out

export OMP_NUM_THREADS=48
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

DOC_PRUNE_RATIO=0.35
QUERY_PRUNE_RATIO=0.5
N_CANDIDATE=20

numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL "$BIN" \
  $DOC_PRUNE_RATIO $QUERY_PRUNE_RATIO $N_CANDIDATE \
  "$DATA/base_full.csr" "$DATA/queries.dev.csr" "$DATA/base_full.dev.gt"

"$SCRIPTS/append_competitor_result.sh" "$LOG" sindi \
  "doc_prune=${DOC_PRUNE_RATIO} query_prune=${QUERY_PRUNE_RATIO} n_cand=${N_CANDIDATE}" 48 \
  "$SCRIPTS/results/competitors.csv"
