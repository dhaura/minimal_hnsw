#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{THREADS}}
#SBATCH --constraint=cpu
#SBATCH --output=logs/main_1M_t{{THREADS}}_%j.out

module load intel

export OMP_NUM_THREADS={{THREADS}}
export MKL_NUM_THREADS=1

$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin/sparse_hnsw_demo 16 200 150 1 0 0 1 0 \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/base_1M.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/queries.dev.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/base_1M.dev.gt
