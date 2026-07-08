#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/hnswlib_gist_1M_t1_%j.out

export OMP_NUM_THREADS=1
srun $SCRATCH/repos/minimal_hnsw/build/bin/hnswlib_demo 16 200 150 $SCRATCH/repos/minimal_hnsw/dense/data/gist/gist_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/gist/gist_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/gist/gist_groundtruth.ivecs fvecs

