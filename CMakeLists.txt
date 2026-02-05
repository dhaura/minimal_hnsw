cmake_minimum_required(VERSION 3.10)
project(minimal_hnsw VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add subdirectory for dense implementation
add_subdirectory(dense)
