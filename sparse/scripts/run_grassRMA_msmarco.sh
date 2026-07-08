#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/sparse_hnsw_1M_%j.out

module load intel

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin/grassRMA_demo 16 200 150 \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/base_1M.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/queries.dev.csr \
  $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_1M/base_1M.dev.gt

