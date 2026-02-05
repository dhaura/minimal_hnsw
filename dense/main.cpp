#include "hnsw.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Minimal HNSW Demo\n";
    std::cout << "=================\n\n";
    
    // Create HNSW index with 2D vectors
    int dim = 2;
    HNSW index(dim, 16, 200, 1000);
    
    // Add some sample points
    std::cout << "Adding points to the index...\n";
    std::vector<std::vector<float>> points = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {0.5f, 0.5f},
        {1.5f, 1.5f},
        {3.0f, 3.0f},
        {0.2f, 0.3f},
        {2.5f, 2.6f}
    };
    
    for (size_t i = 0; i < points.size(); ++i) {
        index.addPoint(points[i], i);
        std::cout << "  Added point " << i << ": (" 
                  << points[i][0] << ", " << points[i][1] << ")\n";
    }
    
    // Search for nearest neighbors
    std::cout << "\nSearching for k-nearest neighbors...\n";
    std::vector<float> query = {1.0f, 1.0f};
    int k = 3;
    
    std::cout << "Query point: (" << query[0] << ", " << query[1] << ")\n";
    std::cout << "k = " << k << "\n\n";
    
    auto results = index.searchKNN(query, k);
    
    std::cout << "Results:\n";
    for (const auto& result : results) {
        std::cout << "  ID: " << result.first 
                  << ", Distance: " << std::fixed << std::setprecision(4) 
                  << result.second << "\n";
    }
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
