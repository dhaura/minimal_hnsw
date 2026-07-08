#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --constraint=cpu
#SBATCH --output=logs/grassRMA_1M_t{{THREADS}}_%j.out

module load intel

export OMP_NUM_THREADS={{THREADS}}
export MKL_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

LAUNCH=""
if command -v numactl >/dev/null; then
  if [ {{THREADS}} -le 64 ]; then
    LAUNCH="numactl --cpunodebind=0-3 --interleave=0-3"
  else
    LAUNCH="numactl --interleave=all"
  fi
fi

$LAUNCH $SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin/grassRMA_demo 16 200 150 \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/base_1M.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/queries.dev.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/base_1M.dev.gt
