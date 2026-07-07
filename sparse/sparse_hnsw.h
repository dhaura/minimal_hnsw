#ifndef SPARSE_HNSW_H
#define SPARSE_HNSW_H

#include "csr_matrix.h"
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <string>
#include <utility>
#include <mutex>

namespace sparse_hnsw {
    using MinPQ = std::priority_queue<
        std::pair<float, uint32_t>,
        std::vector<std::pair<float, uint32_t>>,
        std::greater<std::pair<float,uint32_t>>
    >;

    struct SearchScratch {
        std::vector<uint8_t> visited_bits;
        std::vector<uint32_t> visited_list;
        std::vector<uint32_t> filtered_neighbors;

        void prepare(int max_elements) {
            const size_t num_words = (static_cast<size_t>(max_elements) + 7) / 8;
            if (visited_bits.size() < num_words) {
                visited_bits.assign(num_words, 0);
            }
            visited_list.clear();
            filtered_neighbors.clear();
        }

        bool isVisited(uint32_t id) const {
            return (visited_bits[static_cast<size_t>(id) >> 3] & (1U << (id & 7U))) != 0;
        }

        void markVisited(uint32_t id) {
            const size_t word = static_cast<size_t>(id) >> 3;
            const uint8_t mask = 1U << (id & 7U);
            if ((visited_bits[word] & mask) == 0) {
                visited_bits[word] |= mask;
                visited_list.push_back(id);
            }
        }

        void clear() {
            for (uint32_t id : visited_list) {
                visited_bits[static_cast<size_t>(id) >> 3] &= ~(1U << (id & 7U));
            }
            visited_list.clear();
        }
    };

    class SPARSE_HNSW {
    public:
        SPARSE_HNSW(int dim, CSRMatrix *data_matrix, int M = 16, int ef_construction = 200, int max_elements = 1000, 
            bool use_heuristic = false, bool extend_candidates = false, bool keep_pruned = false, 
            bool use_mkl = false, size_t mklThreshold = 256);
        
        float distance(const void *pVect1, const void *pVect2, const void *qty_ptr, const void *other_ptr) const;
        void addPoint(uint32_t node_id, uint32_t label);
        void addPointsBatch(int num_points);
        std::priority_queue<std::pair<float, uint32_t>> searchKNN(uint32_t query_id, CSRMatrix *query_matrix, int k, int ef = 50) const;
        void searchKNNBatch(CSRMatrix *query_matrix, int num_queries, int k, int ef,
                            std::vector<uint32_t>& out_labels) const;
        void setLabelRemapping(std::vector<uint32_t> old_to_new, std::vector<uint32_t> new_to_old);
        void relabelGroundTruth(std::vector<std::vector<uint32_t>>& groundtruth) const;
        void printInfo() const;

    private:
        CSRMatrix *data_matrix_;    
        int dim_;
        int M_;  // maximum number of connections per layer
        int ef_construction_;
        int max_elements_;
        int max_level_;
        uint32_t entry_point_;

        std::vector<float> data_;
        std::vector<float> norms_; // Precomputed norms for inner product distance 
        uint32_t size_neighbor_list_level0_;
        uint32_t size_neighbor_list_per_element_;
        std::vector<uint32_t> level0_neighbor_lists_;       // [count, n1, n2, ...] per node, fixed-size block
        std::vector<uint32_t> neighbor_lists_;         // upper layers only: one contiguous buffer
        std::vector<uint32_t> neighbor_list_offsets_;       // per-node start offset into neighbor_lists_flat_
        std::vector<int> element_levels_;

        // Scratch reused across the serial insertion path.
        SearchScratch insert_scratch_;

        // Oone mutex per element guarding its neighbor lists.
        // A global mutex for entry_point_ / max_level_.
        mutable std::vector<std::mutex> link_locks_;
        std::mutex global_lock_;

        // For Hilbert curve ordering
        std::vector<uint32_t> old_to_new_labels_;
        std::vector<uint32_t> new_to_old_labels_;

        bool use_heuristic_;
        bool extend_candidates_;
        bool keep_pruned_;
        bool use_mkl_;
        size_t mklThreshold_;

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
        std::priority_queue<std::pair<float, uint32_t>> searchLayer(uint32_t query_id, const void *qty_ptr, std::vector<uint32_t> entry_points, int ef, int layer, SearchScratch& scratch, bool lock_links = false) const;
        std::priority_queue<std::pair<float, uint32_t>> searchKNN(uint32_t query_id, CSRMatrix *query_matrix, int k, int ef, SearchScratch& scratch) const;
        void addPointInternal(uint32_t node_id, uint32_t label, SearchScratch& scratch);
        void connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level, int M);
        std::vector<uint32_t> selectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M);
        std::vector<uint32_t> selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M, int level, bool allow_extend);
    };
}

#endif // SPARSE_HNSW_H
