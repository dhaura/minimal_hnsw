# Create library for HNSW
add_library(hnsw_lib hnsw.cpp hnsw.h)

# Create executable
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

# Set output directory
set_target_properties(hnsw_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(hnswlib_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
