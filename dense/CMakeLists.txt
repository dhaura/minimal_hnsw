# Create library for HNSW
add_library(hnsw_lib hnsw.cpp hnsw.h)

# Create executable
add_executable(hnsw_demo main.cpp)

# Link the library to the executable
target_link_libraries(hnsw_demo hnsw_lib)

# Set output directory
set_target_properties(hnsw_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
