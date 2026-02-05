#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

class HNSW {
public:
    struct Node {
        int id;
        std::vector<float> data;
        std::vector<std::vector<int>> neighbors; // neighbors at each layer
    };

    HNSW(int dim, int M = 16, int ef_construction = 200, int max_elements = 1000);
    
    void addPoint(const std::vector<float>& point, int label);
    std::vector<std::pair<int, float>> searchKNN(const std::vector<float>& query, int k, int ef = 50);
    
private:
    int dim_;
    int M_;  // maximum number of connections per layer
    int ef_construction_;
    int max_elements_;
    int max_level_;
    std::vector<Node> nodes_;
    int entry_point_;
    
    std::mt19937 rng_;
    std::uniform_real_distribution<double> level_generator_;
    
    float distance(const std::vector<float>& a, const std::vector<float>& b) const;
    int getRandomLevel();
    std::vector<int> searchLayer(const std::vector<float>& query, std::vector<int> entry_points, int ef, int layer);
    std::vector<int> connectNeighbors(int node_id, const std::vector<int>& candidates, int level, int M);
    std::vector<int> sortCandidates(int node_id, const std::vector<int>& candidates);
};

#endif // HNSW_H
