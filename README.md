# minimal_hnsw

A minimal implementation of Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search.

## Project Structure

```
minimal_hnsw/
├── dense/           # Dense vector HNSW implementation
│   ├── hnsw.h      # HNSW header file
│   ├── hnsw.cpp    # HNSW implementation
│   ├── main.cpp    # Demo application
│   └── CMakeLists.txt
├── sparse/          # Sparse vector implementation (to be added)
└── CMakeLists.txt   # Root CMake configuration
```

## Building the Project

### Requirements
- CMake 3.10 or higher
- C++11 compatible compiler (GCC, Clang, MSVC)

### Build and Run Instructions

```bash
# Clone hnswlib (for benchmarking)
git clone https://github.com/nmslib/hnswlib.git

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

# Download the dataset
cd ../
mkdir data
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz

# Run the demo 
cd ../build
./bin/hnsw_demo 16 200 150 1 0 0 1 0 $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_groundtruth.ivecs fvecs $SCRATCH/repos/minimal_hnsw/dense/output/timing_stats/mkl_stats.csv

# Usage
# ./bin/hnsw_demo <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <use_mkl> <mklThreshold> <input_filepath> <query_filepath> <gt_filepath> <file_type> <stat_file>
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

## Future Work

- Sparse vector HNSW implementation (sparse/ folder)
- Performance optimizations
- Additional distance metrics
- Serialization/deserialization