# Create library for HNSW
add_library(sparse_hnsw_lib sparse_hnsw.cpp sparse_hnsw.h)


# Optional MKL for batched distance path in sparse_hnsw.cpp
find_package(MKL CONFIG QUIET)
if(MKL_FOUND)
    target_link_libraries(sparse_hnsw_lib PUBLIC MKL::MKL)
    message(STATUS "MKL found: enabling MKL path in sparse_hnsw_lib")
else()
    # Common on HPC systems: MKL module loaded, but no MKLConfig.cmake provided.
    if(DEFINED ENV{MKLROOT})
        set(MKLROOT_DIR "$ENV{MKLROOT}")
        find_path(MKL_INCLUDE_DIR
            NAMES mkl.h
            HINTS
                "${MKLROOT_DIR}/include")
        find_library(MKL_RT_LIBRARY
            NAMES mkl_rt
            HINTS
                "${MKLROOT_DIR}/lib"
                "${MKLROOT_DIR}/lib/intel64")

        if(MKL_INCLUDE_DIR AND MKL_RT_LIBRARY)
            find_package(Threads QUIET)
            target_include_directories(sparse_hnsw_lib PRIVATE ${MKL_INCLUDE_DIR})
            target_link_libraries(sparse_hnsw_lib PUBLIC ${MKL_RT_LIBRARY})
            if(Threads_FOUND)
                target_link_libraries(sparse_hnsw_lib PUBLIC Threads::Threads)
            endif()
            if(UNIX AND NOT APPLE)
                target_link_libraries(sparse_hnsw_lib PUBLIC dl m)
            endif()
            message(STATUS "MKL found via MKLROOT: enabling MKL path in sparse_hnsw_lib")
        else()
            message(STATUS "MKL package config missing and MKLROOT lookup failed: using non-MKL distance path")
        endif()
    else()
        message(STATUS "MKL not found and MKLROOT is unset: using non-MKL distance path")
    endif()
endif()

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

set_target_properties(hnswlib_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(grassRMA_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
