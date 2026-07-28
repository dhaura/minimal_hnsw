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

set(SPARSE_ARCH_FLAGS "-march=cascadelake -mtune=cascadelake" CACHE STRING
    "ISA flags applied to optimized builds")
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
