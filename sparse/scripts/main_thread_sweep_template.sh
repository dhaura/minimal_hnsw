#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --constraint=cpu
#SBATCH --output=logs/main_full_t{{THREADS}}_%j.out

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

$LAUNCH $SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin/sparse_hnsw_demo 16 200 150 1 0 0 1 0 \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full/base_full.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full/queries.dev.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full/base_full.dev.gt
