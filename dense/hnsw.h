#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

namespace hnsw {
    using MinPQ = std::priority_queue<
        std::pair<float, uint32_t>,
        std::vector<std::pair<float, uint32_t>>,
        std::greater<std::pair<float,uint32_t>>
    >;

    class HNSW {
    public:
        HNSW(int dim, int M = 16, int ef_construction = 200, int max_elements = 1000, 
            bool use_heuristic = false, bool extend_candidates = false, bool keep_pruned = false);
        
        float distance(float * a, float * b) const;
        void addPoint(std::vector<float> point, uint32_t label);
        std::priority_queue<std::pair<float, uint32_t>> searchKNN(std::vector<float> query, int k, int ef = 50);
        void printInfo() const;
        
    private:
        int dim_;
        int M_;  // maximum number of connections per layer
        int ef_construction_;
        int max_elements_;
        int max_level_;
        uint32_t entry_point_;

        std::vector<float> data_;
        std::vector<std::vector<std::vector<uint32_t>>> neighbors_; // neighbors_[node_id][layer] gives the neighbors of node_id at that layer

        bool use_heuristic_;
        bool extend_candidates_;
        bool keep_pruned_;
        
        std::mt19937 rng_;
        std::uniform_real_distribution<double> level_generator_;
        
        int getRandomLevel();
        std::priority_queue<std::pair<float, uint32_t>> searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer);
        std::vector<uint32_t> connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level, int M);
        std::vector<uint32_t> selectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M);
        std::vector<uint32_t> selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M, int level);
    };
}

#endif // HNSW_H
