# Create library for HNSW
add_library(hnsw_lib hnsw.cpp hilbert_ordering.cpp hnsw.h hilbert_ordering.h)

# Create executables
add_executable(hnsw_demo main.cpp)
add_executable(hnswlib_demo hnswlib_main.cpp)

# Link the library to the executable
target_link_libraries(hnsw_demo hnsw_lib)

# Link OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hnsw_demo OpenMP::OpenMP_CXX)
    target_link_libraries(hnswlib_demo OpenMP::OpenMP_CXX)
endif()

# ---- Perf-friendly build option ----
option(ENABLE_PERF "Build with perf-friendly flags (symbols + frame pointers)" OFF)

if(ENABLE_PERF)
    foreach(tgt hnsw_lib hnsw_demo hnswlib_demo)
        # Keep debug symbols even in Release; good stacks for perf
        target_compile_options(${tgt} PRIVATE -g -fno-omit-frame-pointer)
        target_link_options(${tgt} PRIVATE -g)
    endforeach()

    # Optional: if you use Release, ensure optimization still happens
    # (CMake already sets -O3/-O2 depending on your toolchain)
endif()

# Set output directory
set_target_properties(hnsw_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(hnswlib_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
