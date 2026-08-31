# Create library for HNSW
add_library(sparse_hnsw_lib sparse_hnsw.cpp sparse_hnsw.h)

# Create executables
add_executable(sparse_hnsw_demo main.cpp)
add_executable(grassRMA_demo grassRMA_main.cpp)
add_executable(sparse_hnsw_sweep sweep_main.cpp)
add_executable(sparse_decouple_sweep decouple_main.cpp)

# Link the library to the executable
target_link_libraries(sparse_hnsw_demo sparse_hnsw_lib)
target_link_libraries(sparse_hnsw_sweep sparse_hnsw_lib)
target_link_libraries(sparse_decouple_sweep sparse_hnsw_lib)
target_include_directories(sparse_decouple_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(sparse_decouple_sweep PROPERTIES
    CXX_STANDARD 17
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Link OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sparse_hnsw_lib PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(sparse_hnsw_demo OpenMP::OpenMP_CXX)
    target_link_libraries(sparse_hnsw_sweep OpenMP::OpenMP_CXX)
    target_link_libraries(sparse_decouple_sweep OpenMP::OpenMP_CXX)
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
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install-znver3/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install-znver3/lib64
          ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install/lib64
    NO_DEFAULT_PATH)
find_path(VSAG_INCLUDE_DIR NAMES vsag/vsag.h
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install-znver3/include
          ${CMAKE_CURRENT_SOURCE_DIR}/vsag/install/include
    NO_DEFAULT_PATH)
if(VSAG_LIBRARY AND VSAG_INCLUDE_DIR)
    add_executable(sindi_demo sindi_main.cpp)
    add_executable(sindi_sweep sindi_sweep_main.cpp)
    get_filename_component(VSAG_LIBRARY_DIR ${VSAG_LIBRARY} DIRECTORY)
    foreach(tgt sindi_demo sindi_sweep)
        target_include_directories(${tgt} PRIVATE ${VSAG_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${tgt} ${VSAG_LIBRARY})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${tgt} OpenMP::OpenMP_CXX)
        endif()
        set_target_properties(${tgt} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            BUILD_RPATH ${VSAG_LIBRARY_DIR}
            INSTALL_RPATH ${VSAG_LIBRARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endforeach()
    message(STATUS "vsag found: building sindi_demo and sindi_sweep")
else()
    message(STATUS "vsag not installed under sparse/vsag/install: skipping sindi_demo")
endif()

# ---- Profiling ladder (see profiling/PROFILING_GUIDE.md) ----
add_executable(bench_distance profiling/bench_distance.cpp)
add_executable(bench_scale profiling/bench_scale.cpp)
add_executable(sparse_profile profiling/profile_search.cpp sparse_hnsw.cpp)
target_compile_definitions(sparse_profile PRIVATE SPARSE_HNSW_PROFILE)

# Recall-loss decomposition across ef, sharing one index build. Needs its own
# copy of sparse_hnsw.cpp because the counters are behind SPARSE_HNSW_PROFILE.
add_executable(sparse_diag_sweep diag_sweep_main.cpp sparse_hnsw.cpp)
target_compile_definitions(sparse_diag_sweep PRIVATE SPARSE_HNSW_PROFILE)

foreach(tgt bench_distance bench_scale sparse_profile sparse_diag_sweep)
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    # -gdwarf-4, not plain -g: GCC 13 defaults to DWARF 5, which the RHEL 8
    # binutils `as` on Grace rejects with "unrecognized option '--gdwarf-5'".
    target_compile_options(${tgt} PRIVATE -O3 -gdwarf-4 -fno-omit-frame-pointer)
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
        # Keep debug symbols even in Release; good stacks for perf.
        # -gdwarf-4 for the same binutils reason as the profiling targets above.
        target_compile_options(${tgt} PRIVATE -gdwarf-4 -fno-omit-frame-pointer)
        target_link_options(${tgt} PRIVATE -gdwarf-4)
    endforeach()

    # Optional: if you use Release, ensure optimization still happens
    # (CMake already sets -O3/-O2 depending on your toolchain)
endif()

# Set output directory
set_target_properties(sparse_hnsw_demo sparse_hnsw_sweep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(grassRMA_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
