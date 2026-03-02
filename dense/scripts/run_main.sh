#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/hnsw_1M_111_%j.out

srun -u -n 1 $SCRATCH/repos/minimal_hnsw/build/bin/hnsw_demo 16 200 150 sqr 1 1 1 $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_groundtruth.ivecs fvecs
