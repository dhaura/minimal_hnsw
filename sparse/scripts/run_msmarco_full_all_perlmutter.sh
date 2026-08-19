#!/bin/bash
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=06:00:00
#SBATCH --job-name=msmarco_full_all
#SBATCH --output=logs/msmarco_full_all_%j.out

set -u
REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw
BUILD=${BUILD_DIR:-build}
BIN=$REPO/$BUILD/bin
DATA=$REPO/sparse/data/msmarco_full
SCRIPTS=$REPO/sparse/scripts

BASE=$DATA/base_full.csr
QUERIES=$DATA/queries.dev.csr
GT=$DATA/base_full.dev.gt

M=16; EFC=200; EF=150
DOC_PRUNE=0.35; QUERY_PRUNE=0.5; N_CAND=20

THREADS=${THREADS:-128}
export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MKL_NUM_THREADS=1

export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/AMD/aocc-compiler-4.1.0/lib:${LD_LIBRARY_PATH:-}

LAUNCH=""
if command -v numactl >/dev/null; then
  if [ "$THREADS" -le 64 ]; then
    LAUNCH="numactl --cpunodebind=0-3 --interleave=0-3"
  else
    LAUNCH="numactl --interleave=all"
  fi
fi

echo "host=$(hostname) build=$BUILD threads=$THREADS launch='$LAUNCH'"
echo "params: M=$M efC=$EFC ef=$EF | sindi doc=$DOC_PRUNE query=$QUERY_PRUNE ncand=$N_CAND"
echo "================================================================"

echo; echo "########## sparse_hnsw ##########"
$LAUNCH stdbuf -oL -eL "$BIN/sparse_hnsw_demo" $M $EFC $EF 1 0 0 1 0 \
  "$BASE" "$QUERIES" "$GT" "$SCRIPTS/results/msmarco_full_sparse_hnsw_${SLURM_JOB_ID}.csv"

echo; echo "########## grassRMA ##########"
$LAUNCH stdbuf -oL -eL "$BIN/grassRMA_demo" $M $EFC $EF \
  "$BASE" "$QUERIES" "$GT"

echo; echo "########## SINDI (vsag) ##########"
$LAUNCH stdbuf -oL -eL "$BIN/sindi_demo" \
  $DOC_PRUNE $QUERY_PRUNE $N_CAND "$BASE" "$QUERIES" "$GT"

echo; echo "================================================================"
echo "done"
