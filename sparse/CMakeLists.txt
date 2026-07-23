# Create library for HNSW
add_library(sparse_hnsw_lib sparse_hnsw.cpp sparse_hnsw.h)

# Create executables
add_executable(sparse_hnsw_demo main.cpp)
add_executable(grassRMA_demo grassRMA_main.cpp)

# Link the library to the executable
target_link_libraries(sparse_hnsw_demo sparse_hnsw_lib)

# Link OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sparse_hnsw_lib PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(sparse_hnsw_demo OpenMP::OpenMP_CXX)
    target_link_libraries(grassRMA_demo OpenMP::OpenMP_CXX)
endif()

# ---- Optional SINDI (vsag) benchmark ----
# Needs a prebuilt vsag shared library. One-time setup (own toolchain, gcc):
#   cmake -S vsag -Bvsag/build-release -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
#         -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_TOOLS=OFF \
#         -DCMAKE_INSTALL_PREFIX=$PWD/vsag/install
#   cmake --build vsag/build-release --parallel 16
#   cmake --install vsag/build-release
find_library(VSAG_LIBRARY NAMES vsag
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install/lib64
    NO_DEFAULT_PATH)
find_path(VSAG_INCLUDE_DIR NAMES vsag/vsag.h
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install/include
    NO_DEFAULT_PATH)
if(VSAG_LIBRARY AND VSAG_INCLUDE_DIR)
    add_executable(sindi_demo sindi_main.cpp)
    target_include_directories(sindi_demo PRIVATE ${VSAG_INCLUDE_DIR})
    target_link_libraries(sindi_demo ${VSAG_LIBRARY})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(sindi_demo OpenMP::OpenMP_CXX)
    endif()
    get_filename_component(VSAG_LIBRARY_DIR ${VSAG_LIBRARY} DIRECTORY)
    set_target_properties(sindi_demo PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        BUILD_RPATH ${VSAG_LIBRARY_DIR}
        INSTALL_RPATH ${VSAG_LIBRARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    message(STATUS "vsag found: building sindi_demo")
else()
    message(STATUS "vsag not installed under sparse/vsag/install: skipping sindi_demo")
endif()

# ---- Profiling ladder (see profiling/README.md) ----
add_executable(bench_distance profiling/bench_distance.cpp)
add_executable(bench_scale profiling/bench_scale.cpp)
add_executable(sparse_profile profiling/profile_search.cpp sparse_hnsw.cpp)
target_compile_definitions(sparse_profile PRIVATE SPARSE_HNSW_PROFILE)

foreach(tgt bench_distance bench_scale sparse_profile)
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(${tgt} PRIVATE -O3 -g -fno-omit-frame-pointer)
    set_target_properties(${tgt} PROPERTIES
        CXX_STANDARD 17
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} OpenMP::OpenMP_CXX)
    endif()
endforeach()

# ---- Perf-friendly build option ----
option(ENABLE_PERF "Build with perf-friendly flags (symbols + frame pointers)" OFF)

if(ENABLE_PERF)
    foreach(tgt sparse_hnsw_lib sparse_hnsw_demo grassRMA_demo)
        # Keep debug symbols even in Release; good stacks for perf
        target_compile_options(${tgt} PRIVATE -g -fno-omit-frame-pointer)
        target_link_options(${tgt} PRIVATE -g)
    endforeach()

    # Optional: if you use Release, ensure optimization still happens
    # (CMake already sets -O3/-O2 depending on your toolchain)
endif()

# Set output directory
set_target_properties(sparse_hnsw_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(grassRMA_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
