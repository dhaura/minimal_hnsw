#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <string>
#include <utility>

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
        void finalizeIndex();
        void relabelGroundTruth(std::vector<std::vector<uint32_t>>& groundtruth) const;
        void printInfo() const;

        // For profiling layer 0 metrics.
        bool dumpLayer0Counts(const std::string& output_path, const std::string param) const {
            return false;
        }
        
    private:
        int dim_;
        int M_;  // maximum number of connections per layer
        int ef_construction_;
        int max_elements_;
        int max_level_;
        uint32_t entry_point_;

        std::vector<float> data_;
        uint32_t size_neighbor_list_level0_;
        uint32_t size_neighbor_list_per_element_;
        std::vector<uint32_t> level0_neighbor_lists_;       // [count, n1, n2, ...] per node, fixed-size block
        std::vector<uint32_t> neighbor_lists_;         // upper layers only: one contiguous buffer
        std::vector<uint32_t> neighbor_list_offsets_;       // per-node start offset into neighbor_lists_flat_
        std::vector<int> element_levels_;
        
        // For Hilbert curve ordering
        std::vector<uint32_t> old_to_new_labels_;
        std::vector<uint32_t> new_to_old_labels_;

        bool pca_projection_ready_;
        std::vector<float> pca_mean_;
        std::vector<float> pca_component_1_;
        std::vector<float> pca_component_2_;

        // Profiling metrics for layer 0.
        std::vector<uint32_t> num_dist_calc_layer0_insertion_;
        std::vector<uint32_t> num_cand_elements_layer0_insertion_;
        std::vector<uint32_t> max_hops_layer0_insertion_;

        std::vector<uint32_t> num_dist_calc_layer0_search_;
        std::vector<uint32_t> num_cand_elements_layer0_search_;
        std::vector<uint32_t> max_hops_layer0_search_;

        bool use_heuristic_;
        bool extend_candidates_;
        bool keep_pruned_;
        enum class Phase {
            Insertion,
            Search
        };

        Phase current_phase_;
        
        std::mt19937 rng_;
        std::uniform_real_distribution<double> level_generator_;
        
        int getRandomLevel();
        uint32_t* get_neighbor_list0(uint32_t node_id);
        const uint32_t* get_neighbor_list0(uint32_t node_id) const;
        uint32_t* get_neighbor_list(uint32_t node_id, int level);
        const uint32_t* get_neighbor_list(uint32_t node_id, int level) const;
        uint32_t* get_neighbor_list_at_level(uint32_t node_id, int level);
        const uint32_t* get_neighbor_list_at_level(uint32_t node_id, int level) const;
        uint32_t getListCount(const uint32_t* ptr) const;
        void setListCount(uint32_t* ptr, uint32_t size);
        std::vector<uint32_t> getNeighborsAtLevel(uint32_t node_id, int level) const;
        void setNeighborsAtLevel(uint32_t node_id, int level, const std::vector<uint32_t>& neighbors, int max_degree);
        std::priority_queue<std::pair<float, uint32_t>> searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer);
        std::vector<uint32_t> connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level, int M);
        std::vector<uint32_t> selectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M);
        std::vector<uint32_t> selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M, int level);
        
        // Hilbert curve ordering helpers
        uint64_t xy2d(uint32_t n, uint32_t x, uint32_t y);
        void fitPcaProjection();
        std::pair<float, float> projectTo2D(const std::vector<float>& point) const;
        uint64_t computeHilbertIndex(const std::vector<float>& point, 
                                     float min_x, float max_x, 
                                     float min_y, float max_y);
    };
}

#endif // HNSW_H
