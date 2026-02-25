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
        if (!file) break; // Check if we reached the end of the file.

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file) break; // Check if we successfully read the vector.

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
        if (!file) break; // Check if we reached the end of the file.

        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        if (!file) break; // Check if we successfully read the vector.

        data.push_back(std::move(vec));
    }
    return dim;
}

int readbvecs(const std::string& filename, std::vector<std::vector<float>>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return -1;
    }

    int dim = 0;
    while (file) {
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (!file) break; // Check if we reached the end of the file    
        std::vector<unsigned char> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(unsigned char));
        if (!file) break; // Check if we successfully read the vector   

        // Convert unsigned char vector to float vector
        std::vector<float> float_vec(vec.begin(), vec.end());
        data.push_back(std::move(float_vec));
    }
    return dim;
}

int main(int argc, char* argv[]) {
    std::cout << "Minimal HNSW Demo\n";
    std::cout << "=================\n\n";

     if (argc < 9)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <distance_metric> <input_filepath> <query_filepath> <gt_filepath> <file_type>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    int ef = std::stoi(argv[3]);
    std::string distance_metric = argv[4];
    std::string input_filepath = argv[5];
    std::string query_filepath = argv[6];
    std::string gt_filepath = argv[7];
    std::string file_type = argv[8];
    // Read a dense dataset from file.
    std::vector<std::vector<float>> points;
    int dim = 0;
    if (file_type == "fvecs") {
        dim = readfvecs(input_filepath, points);
    } else if (file_type == "bvecs") {
        dim = readbvecs(input_filepath, points);
    } else {
        std::cerr << "Unsupported file type: " << file_type << std::endl;
        return -1;
    }
    
    // Create HNSW index with 2D vectors.
    HNSW index(dim, M, ef_construction, points.size(), distance_metric);
    
    // Add points from the dataset to the index.
    std::cout << "Adding points to the index...\n";
    
    for (size_t i = 0; i < points.size(); ++i) {
        index.addPoint(points[i], i);
    }

    std::cout << "Added " << points.size() << " points to the index.\n";
    
    // Search for nearest neighbors.
    std::cout << "\nSearching for k-nearest neighbors...\n";
    
    std::vector<std::vector<float>> query;
    int dim_query = 0;
    if (file_type == "fvecs") {
        dim_query = readfvecs(query_filepath, query);
    } else if (file_type == "bvecs") {
        dim_query = readbvecs(query_filepath, query);
    } else {
        std::cerr << "Unsupported file type: " << file_type << std::endl;
        return -1;
    }
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

        // // For debugging.
        // if (i == 0) {
        //     std::cout << "Query 0: Found neighbors (label, distance):\n";
        //     for (const auto& nn : nns) {
        //         std::cout << "  Label: " << nn.first << ", Distance: " << nn.second << "\n";
        //     }
        //     std::cout << "True neighbors: \n";
        //     for (int label : true_labels[i]) {
        //         std::cout << "  Label: " << label << ", Distance: " << index.distance(query[i], points[label]) << "\n";
        //     }
        // }
    }
    
    float recall = static_cast<float>(correct) / (query_count * k) * 100.0f;
    std::cout << "Recall@k: " << std::fixed << std::setprecision(2) << recall << "%\n";

    index.printInfo();
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
