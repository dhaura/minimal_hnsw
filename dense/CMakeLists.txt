# Create library for HNSW
add_library(hnsw_lib hnsw.cpp hilbert_ordering.cpp hnsw.h hilbert_ordering.h)

set(FAISS_ROOT "" CACHE PATH "Faiss install prefix (optional, e.g. <prefix> with lib/cmake/faiss)")

if(FAISS_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${FAISS_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH "${FAISS_ROOT}/lib/cmake/faiss")
endif()

find_package(faiss CONFIG QUIET)
if(NOT faiss_FOUND)
    find_package(Faiss CONFIG QUIET)
endif()

if(TARGET faiss::faiss)
    target_link_libraries(hnsw_lib PUBLIC faiss::faiss)
elseif(TARGET faiss)
    target_link_libraries(hnsw_lib PUBLIC faiss)
elseif(faiss_FOUND AND DEFINED faiss_LIBRARIES)
    if(DEFINED faiss_INCLUDE_DIRS)
        target_include_directories(hnsw_lib PRIVATE ${faiss_INCLUDE_DIRS})
    endif()
    target_link_libraries(hnsw_lib PUBLIC ${faiss_LIBRARIES})
else()
    message(FATAL_ERROR "Prebuilt Faiss not found. Set FAISS_ROOT or CMAKE_PREFIX_PATH to your Faiss install.")
endif()

message(STATUS "Using prebuilt Faiss")

# Optional MKL for batched distance path in hnsw.cpp
find_package(MKL CONFIG QUIET)
if(MKL_FOUND)
    target_link_libraries(hnsw_lib PUBLIC MKL::MKL)
    message(STATUS "MKL found: enabling MKL path in hnsw_lib")
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
            target_include_directories(hnsw_lib PRIVATE ${MKL_INCLUDE_DIR})
            target_link_libraries(hnsw_lib PUBLIC ${MKL_RT_LIBRARY})
            if(Threads_FOUND)
                target_link_libraries(hnsw_lib PUBLIC Threads::Threads)
            endif()
            if(UNIX AND NOT APPLE)
                target_link_libraries(hnsw_lib PUBLIC dl m)
            endif()
            message(STATUS "MKL found via MKLROOT: enabling MKL path in hnsw_lib")
        else()
            message(STATUS "MKL package config missing and MKLROOT lookup failed: using non-MKL distance path")
        endif()
    else()
        message(STATUS "MKL not found and MKLROOT is unset: using non-MKL distance path")
    endif()
endif()

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
