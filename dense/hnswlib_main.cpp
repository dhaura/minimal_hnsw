#include "hnswlib/hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <chrono>

using namespace hnswlib;

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
    std::cout << "HNSWLIB HNSW Demo\n";
    std::cout << "=================\n\n";

     if (argc < 8)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <input_filepath> <query_filepath> <gt_filepath> <file_type>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    int ef = std::stoi(argv[3]);
    std::string input_filepath = argv[4];
    std::string query_filepath = argv[5];
    std::string gt_filepath = argv[6];
    std::string file_type = argv[7];

    // Read a dense dataset from file
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
    
    auto start_index_time = std::chrono::steady_clock::now();

    // Create HNSW index with 2D vectors
    L2Space l2space(dim);
    HierarchicalNSW<float>* index = new HierarchicalNSW<float>(&l2space, points.size(), M, ef_construction);
    index->ef_ = ef; // Set ef for search
    
    // Add points from the dataset to the index
    std::cout << "Adding points to the index...\n";
    
    #pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i) {
        index->addPoint(points[i].data(), static_cast<labeltype>(i));
    }

    auto end_index_time = std::chrono::steady_clock::now();
    auto index_time = std::chrono::duration_cast<std::chrono::microseconds>(end_index_time - start_index_time).count();

    std::cout << "Added " << points.size() << " points to the index in " << index_time << " microseconds.\n";
    
    // Search for nearest neighbors
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

    // reduce gts to top 10
    // for (auto& labels : true_labels) {
    //     if (labels.size() > 10) {
    //         labels.resize(10);
    //     }
    // }

    // k = 10;

    auto start_query_time = std::chrono::steady_clock::now();
    
    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (int i = 0; i < query_count; i++) {
        std::priority_queue<std::pair<float, labeltype>> nns = index->searchKnn(query[i].data(), k);
        while (!nns.empty()) {
            int label = nns.top().second;
            nns.pop();
            if (std::find(true_labels[i].begin(), true_labels[i].end(), label) != true_labels[i].end()) {
                correct++;
            }
        }

    }

    auto end_query_time = std::chrono::steady_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(end_query_time - start_query_time).count();

    std::cout << "\nTotal correct neighbors found: " << correct << " out of " << (query_count * k) << "\n";
    
    float recall = static_cast<float>(correct) / (query_count * k) * 100.0f;
    std::cout << "Recall@k: " << std::fixed << std::setprecision(2) << recall << "%\n";
    std::cout << "Query time: " << query_time << " microseconds\n";
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
