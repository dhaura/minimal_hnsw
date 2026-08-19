cmake_minimum_required(VERSION 3.10)
project(minimal_hnsw VERSION 1.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type; benchmarks require an optimized one" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE was empty -- defaulting to Release.")
endif()

option(ALLOW_UNOPTIMIZED_BUILD
       "Permit a Debug/-O0 configure (never use such a tree for benchmarks)" OFF)

if(NOT ALLOW_UNOPTIMIZED_BUILD)
    if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
        message(FATAL_ERROR
            "CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}' is unoptimized and would "
            "produce benchmark numbers 3-6x too slow. Configure with "
            "-DCMAKE_BUILD_TYPE=Release, or pass -DALLOW_UNOPTIMIZED_BUILD=ON "
            "if you are genuinely debugging.")
    endif()
endif()

# ISA flags for optimized builds. These are machine-specific: a binary built
# with a -march= the run host cannot execute dies with SIGILL, so the default
# is auto-detected from the build host rather than hardcoded. Override with
# -DSPARSE_ARCH_FLAGS="..." when cross-targeting.
#
#   TAMU Grace  : Intel Xeon Cascade Lake -> -march=cascadelake (has AVX-512)
#   Perlmutter  : AMD EPYC 7763/7713 Zen3 -> -march=znver3      (AVX2 only,
#                 no AVX-512 in hardware -- Zen3 predates AMD's AVX-512, so
#                 no compiler or library can provide it here.)
if(NOT SPARSE_ARCH_FLAGS)
    include(CheckCXXCompilerFlag)
    if(DEFINED ENV{CRAY_CPU_TARGET})
        if("$ENV{CRAY_CPU_TARGET}" STREQUAL "x86-milan" OR
           "$ENV{CRAY_CPU_TARGET}" STREQUAL "x86-milan-x")
            set(_sparse_arch "znver3")
        elseif("$ENV{CRAY_CPU_TARGET}" STREQUAL "x86-rome")
            set(_sparse_arch "znver2")
        elseif("$ENV{CRAY_CPU_TARGET}" STREQUAL "x86-turin")
            set(_sparse_arch "znver5")
        endif()
    endif()
    if(NOT _sparse_arch)
        set(_sparse_arch "native")
    endif()
    check_cxx_compiler_flag("-march=${_sparse_arch}" SPARSE_ARCH_SUPPORTED)
    if(NOT SPARSE_ARCH_SUPPORTED)
        message(WARNING "Compiler rejects -march=${_sparse_arch}; falling back "
                        "to -march=native.")
        set(_sparse_arch "native")
    endif()
    set(SPARSE_ARCH_FLAGS "-march=${_sparse_arch} -mtune=${_sparse_arch}")
endif()
set(SPARSE_ARCH_FLAGS "${SPARSE_ARCH_FLAGS}" CACHE STRING
    "ISA flags applied to optimized builds")
message(STATUS "Arch flags : ${SPARSE_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${SPARSE_ARCH_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O2 -gdwarf-4 -DNDEBUG ${SPARSE_ARCH_FLAGS}" CACHE STRING "" FORCE)

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Release CXX: ${CMAKE_CXX_FLAGS_RELEASE}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add subdirectory for dense implementation
# add_subdirectory(dense)
# Add subdirectory for sparse implementation
add_subdirectory(sparse)
