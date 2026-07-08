# minimal_hnsw

A minimal implementation of Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search.

## Project Structure

```
minimal_hnsw/
├── dense/           # Dense vector HNSW implementation
│   ├── hnsw.h                # HNSW header file
│   ├── hnsw.cpp              # HNSW implementation
│   ├── hilbert_ordering.h    # Hilbert curve dataset reordering header
│   ├── hilbert_ordering.cpp  # Faiss PCA projection + Hilbert curve ordering
│   ├── main.cpp              # Demo application
│   ├── hnswlib_main.cpp      # hnswlib benchmark driver
│   ├── scripts/              # Run scripts and plotting utilities
│   ├── perf/                 # Captured perf reports
│   ├── output/               # Timing stats, distributions, plots
│   └── CMakeLists.txt
├── sparse/          # Sparse vector HNSW implementation
│   ├── sparse_hnsw.h    # Sparse HNSW header file
│   ├── sparse_hnsw.cpp  # Sparse HNSW implementation
│   ├── csr_matrix.h     # Compact CSR matrix loader (uint16 indices, fp16 values)
│   ├── main.cpp         # Demo application
│   ├── grassRMA_main.cpp# grassRMA benchmark driver
│   ├── scripts/         # SLURM run and thread-sweep scripts
│   └── CMakeLists.txt
└── CMakeLists.txt   # Root CMake configuration
```

## Building the Project

### Requirements
- CMake 3.10 or higher
- C++11 compatible compiler (GCC, Clang, MSVC)

### Build and Run Instructions

```bash
# Clone hnswlib (for benchmarking)
cd dense
git clone https://github.com/nmslib/hnswlib.git
cd ../

# Clone grassRMA (for benchmarking)
cd sparse
git clone https://github.com/Leslie-Chung/GrassRMA.git
cd ../

# Clone and build faiss (for PCA calculations used for hilbert ordering)
git clone https://github.com/facebookresearch/faiss.git
cmake -S $SCRATCH/repos/minimal_hnsw/dense/faiss -B $SCRATCH/repos/minimal_hnsw/dense/faiss/build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$SCRATCH/repos/minimal_hnsw/dense/faiss/install
cmake --build $SCRATCH/repos/minimal_hnsw/dense/faiss/build -j4
cmake --install $SCRATCH/repos/minimal_hnsw/dense/faiss/build

# Create build directory
mkdir build
cd build

# Configure and build
modue load intel # For MKL support
cmake -S $SCRATCH/repos/minimal_hnsw -B $SCRATCH/repos/minimal_hnsw/build -DFAISS_ROOT=$SCRATCH/repos/minimal_hnsw/dense/faiss/install
cmake --build $SCRATCH/repos/minimal_hnsw/build -j4

# Download the dataset (dense)
cd ../dense
mkdir data
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz

# Run the demo (dense)
cd ../build
./bin/hnsw_demo 16 200 150 1 0 0 1 0 $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_groundtruth.ivecs fvecs $SCRATCH/repos/minimal_hnsw/dense/output/timing_stats/mkl_stats.csv

# Usage
# ./bin/hnsw_demo <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <use_mkl> <mklThreshold> <input_filepath> <query_filepath> <gt_filepath> <file_type> <stat_file>

# Download the dataset (sparse)
cd ../sparse
mkdir data
cd data
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_1M.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.dev.gt
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_1M.dev.gt
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.dev.gt

# Run the demo (sparse)
cd ../build
export OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close MKL_NUM_THREADS=1
numactl --cpunodebind=0-3 --interleave=0-3 ./bin/sparse_hnsw_demo 16 200 150 1 0 0 1 0 $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/queries.dev.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.dev.gt  

export OMP_NUM_THREADS=128 OMP_PROC_BIND=spread
numactl --interleave=all ./bin/sparse_hnsw_demo 16 200 150 1 0 0 1 0 $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/queries.dev.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.dev.gt  

export OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close MKL_NUM_THREADS=1
numactl --cpunodebind=0-3 --interleave=0-3 ./bin/grassRMA_demo 16 200 150 $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/queries.dev.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.dev.gt

export OMP_NUM_THREADS=128 OMP_PROC_BIND=spread
numactl --interleave=all ./bin/grassRMA_demo 16 200 150 $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/queries.dev.csr $SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_small/base_small.dev.gt 

# Usage
# ./bin/sparse_hnsw_demo <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <use_mkl> <mklThreshold> <input_filepath> <query_filepath> <gt_filepath>
```

## Dense HNSW Implementation

The dense folder contains a simple HNSW implementation for dense vectors with the following features:

- **Basic HNSW structure**: Multi-layer graph with hierarchical navigation
- **Point insertion**: Add vectors to the index with automatic layer assignment
- **K-NN search**: Find k nearest neighbors for a query vector
- **Euclidean distance**: Uses L2 distance metric

### Usage Example

```cpp
#include "hnsw.h"

// Create index: dimension=2, M=16, ef_construction=200
HNSW index(2, 16, 200, 1000);

// Add points
std::vector<float> point = {1.0f, 2.0f};
index.addPoint(point, 0);

// Search for k nearest neighbors
std::vector<float> query = {1.0f, 2.0f};
auto results = index.searchKNN(query, 5);
```

## Sparse HNSW Implementation

The sparse folder contains an HNSW implementation for high-dimensional sparse vectors stored in CSR format, targeting inner-product (MIPS) retrieval on datasets such as MS MARCO. It shares the dense algorithm structure but replaces the vector storage and distance kernel:

- **CSR-backed storage**: Vectors are never materialized densely; the index stores only row ids and reads `(index, value)` pairs straight out of the shared `CSRMatrix`
- **Compact 4-byte entries**: Column indices are `uint16_t` (valid while `ncol <= 65536`, checked at load) and values are IEEE `fp16`, halving the bytes streamed per distance call
- **Sparse inner-product distance**: Branchless merge over two sorted index lists, accumulating in fp32; returns `1 - <q, p>`
- **Flat neighbor lists**: Level 0 uses a fixed-size contiguous block per node; upper layers share one packed buffer with per-node offsets
- **Parallel construction and search**: OpenMP-parallel `addPointsBatch` / `searchKNNBatch`, with one mutex per element guarding its neighbor lists and a global lock for the entry point
- **Prefetching**: Software prefetch of visited bits, `indptr`, and neighbor vectors, driven by a neighbor filtering loop
- **Neighbor selection**: Both naive top-M and the HNSW heuristic, with optional `extend_candidates` and `keep_pruned`

### Usage Example

```cpp
#include "sparse_hnsw.h"

// Load the base and query matrices in CSR format
CSRMatrix* base  = new CSRMatrix("base_small.csr", true);
CSRMatrix* query = new CSRMatrix("queries.dev.csr", true);

// Create index: M=16, ef_construction=200, heuristic neighbor selection
sparse_hnsw::SPARSE_HNSW index(base->ncol, base, 16, 200, base->nrow,
                               /*use_heuristic=*/true);

// Insert all rows of the base matrix (OpenMP-parallel)
index.addPointsBatch(base->nrow);

// Search: k nearest neighbors for every query row (OpenMP-parallel)
std::vector<uint32_t> labels;
index.searchKNNBatch(query, query->nrow, /*k=*/10, /*ef=*/150, labels);

// Or a single query row
auto results = index.searchKNN(/*query_id=*/0, query, 10, 150);
```

## Future Work

- **SIMD sparse intersection**: Vectorize the merge loop (AVX-512 gather/conflict detection) instead of the scalar branchless walk
- **MKL sparse BLAS path**: `use_mkl` / `mklThreshold` are plumbed through but the distance kernel does not yet dispatch to MKL for long rows
- **Locality-improving reordering**: Finish the PCA + Hilbert curve pre-ordering (currently stubbed out in `sparse/main.cpp`) so neighbor lists touch nearby cache lines
- **Support for `ncol > 65536`**: Widen `IndiceDataPair::indice` or block-partition the column space
- **Serialization/deserialization**: Persist a built index instead of rebuilding per run
- **Additional distance metrics**: Currently inner product only (sparse) and L2 (dense)
