#include "hnsw.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

int readfvecs(const std::string& filename, std::vector<std::vector<float>>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1;
    }   

    int dim = 0;
    while (file) {
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break; // Check if we reached the end of the file

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file) break; // Check if we successfully read the vector

        data.push_back(std::move(vec));
    }
    return dim;
}

int readivecs(const std::string& filename, std::vector<std::vector<int>>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1;
    }
    
    int dim = 0;
    while (file) {
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break; // Check if we reached the end of the file

        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        if (!file) break; // Check if we successfully read the vector

        data.push_back(std::move(vec));
    }
    return dim;
}

int main(int argc, char* argv[]) {
    std::cout << "Minimal HNSW Demo\n";
    std::cout << "=================\n\n";

     if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <input_filepath> <query_filepath> <gt_filepath>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    int ef = std::stoi(argv[3]);
    std::string input_filepath = argv[4];
    std::string query_filepath = argv[5];
    std::string gt_filepath = argv[6];

    // Read a dense dataset from file
    std::vector<std::vector<float>> points;
    int dim = readfvecs(input_filepath, points);
    
    // Create HNSW index with 2D vectors
    HNSW index(dim, M, ef_construction, points.size());
    
    // Add points from the dataset to the index
    std::cout << "Adding points to the index...\n";
    
    for (size_t i = 0; i < points.size(); ++i) {
        index.addPoint(points[i], i);
    }

    std::cout << "Added " << points.size() << " points to the index.\n";
    
    // Search for nearest neighbors
    std::cout << "\nSearching for k-nearest neighbors...\n";
    
    std::vector<std::vector<float>> query;
    int dim_query = readfvecs(query_filepath, query);
    int query_count = static_cast<int>(query.size());

    std::vector<std::vector<int>> true_labels;
    int k = readivecs(gt_filepath, true_labels);
    
    int correct = 0;
    for (int i = 0; i < query_count; i++) {
        auto nns = index.searchKNN(query[i], k, ef);
        for (size_t j = 0; j < nns.size(); j++) {
            if (std::find(true_labels[i].begin(), true_labels[i].end(), nns[j].first) != true_labels[i].end()) {
                correct++;
            }
        }
    }
    
    float recall = static_cast<float>(correct) / (query_count * k) * 100.0f;
    std::cout << "Recall@100: " << std::fixed << std::setprecision(2) << recall << "%\n";
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
