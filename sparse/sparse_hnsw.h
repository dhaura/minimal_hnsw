#ifndef SPARSE_HNSW_H
#define SPARSE_HNSW_H

#include "csr_matrix.h"
#include "prune.h"
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

// Adds distance-call and byte accounting to the search path.
#ifdef SPARSE_HNSW_PROFILE
#include <atomic>
#endif

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

        std::vector<float> q_dense;
        bool dense_query = false;

        void scatterQuery(int dim, const IndiceDataPair* q_indices, uint32_t q_num) {
            if (q_dense.size() < static_cast<size_t>(dim)) {
                q_dense.assign(static_cast<size_t>(dim), 0.0f);
            }
            for (uint32_t i = 0; i < q_num; ++i) {
                q_dense[q_indices[i].indice] = static_cast<float>(q_indices[i].data);
            }
            dense_query = true;
        }

        void unscatterQuery(const IndiceDataPair* q_indices, uint32_t q_num) {
            for (uint32_t i = 0; i < q_num; ++i) {
                q_dense[q_indices[i].indice] = 0.0f;
            }
            dense_query = false;
        }

#ifdef SPARSE_HNSW_PROFILE
        // Accumulated across searchLayer calls (never reset by prepare();
        // callers snapshot-and-diff). bytes = doc-row bytes each distance
        // call must pull; the query row is resident and not counted.
        // graph_bytes = the neighbor lists walked, which are memory traffic too
        // and are NOT part of any distance call.
        uint64_t prof_ndist = 0;
        uint64_t prof_bytes = 0;
        uint64_t prof_graph_bytes = 0;

        // The beta>1 refine pass re-scores k*beta candidates against the
        // UNPRUNED matrix using the old merge kernel. It runs in
        // searchKNNBatch, not searchLayer, and streams different (longer)
        // rows -- so it is counted apart from the traversal above.
        uint64_t prof_refine_ndist = 0;
        uint64_t prof_refine_bytes = 0;

        // Record/replay, used to measure what share of search time distance()
        // actually owns (mode=replay). searchLayer is deterministic given the
        // distance VALUES, so replaying a recorded sequence reproduces the
        // traversal, the heap operations and the visited set exactly -- while
        // touching no CSR row at all. The delta is distance()'s true cost.
        std::vector<float>* record = nullptr;   // set: append each computed distance
        const float* replay = nullptr;          // set: return these instead of computing
        size_t replay_idx = 0;
#endif

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
            bool use_heuristic = false, bool extend_candidates = false, bool keep_pruned = false, float alpha = 1.0, int beta = 1);
        
        // Kept out-of-line in a profile build so `perf report` can attribute
        // cycles to distance() as its own symbol instead of folding them into
        // searchLayer.
#ifdef SPARSE_HNSW_PROFILE
        __attribute__((noinline))
#endif
        float distance(const void *pVect1, const void *pVect2, const void *qty_ptr, const void *other_ptr) const;

#ifdef SPARSE_HNSW_PROFILE
        __attribute__((noinline))
#endif
        float distanceDense(uint32_t p_idx, const std::vector<float>& q_dense) const;

        void addPoint(uint32_t node_id, uint32_t label);
        void addPointsBatch(int num_points);
        std::priority_queue<std::pair<float, uint32_t>> searchKNN(uint32_t query_id, CSRMatrix *query_matrix, int k, int ef = 50) const;
        void searchKNNBatch(CSRMatrix *query_matrix, int num_queries, int k, int ef,
                            std::vector<uint32_t>& out_labels) const;
        CSRMatrix* pruneMatrix(const CSRMatrix *m);
        void setPrunedDataMatrix(CSRMatrix *pruned_data_matrix);
        void setLabelRemapping(std::vector<uint32_t> old_to_new, std::vector<uint32_t> new_to_old);
        void relabelGroundTruth(std::vector<std::vector<uint32_t>>& groundtruth) const;
        void printInfo() const;

#ifdef SPARSE_HNSW_PROFILE
        std::priority_queue<std::pair<float, uint32_t>> searchKNNProf(
                uint32_t query_id, CSRMatrix *query_matrix, int k, int ef,
                SearchScratch& scratch) const {
            return searchKNN(query_id, query_matrix, k, ef, scratch);
        }
        // Totals aggregated by searchKNNBatch across its worker scratches.
        uint64_t profNDist() const { return prof_ndist_.load(std::memory_order_relaxed); }
        uint64_t profBytes() const { return prof_bytes_.load(std::memory_order_relaxed); }
        uint64_t profGraphBytes() const { return prof_graph_bytes_.load(std::memory_order_relaxed); }
        uint64_t profRefineNDist() const { return prof_refine_ndist_.load(std::memory_order_relaxed); }
        uint64_t profRefineBytes() const { return prof_refine_bytes_.load(std::memory_order_relaxed); }
        void profReset() const {
            prof_ndist_.store(0); prof_bytes_.store(0); prof_graph_bytes_.store(0);
            prof_refine_ndist_.store(0); prof_refine_bytes_.store(0);
        }
#endif

    private:
        CSRMatrix *data_matrix_;    
        CSRMatrix *original_data_matrix_;
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

#ifdef SPARSE_HNSW_PROFILE
        mutable std::atomic<uint64_t> prof_ndist_{0};
        mutable std::atomic<uint64_t> prof_bytes_{0};
        mutable std::atomic<uint64_t> prof_graph_bytes_{0};
        mutable std::atomic<uint64_t> prof_refine_ndist_{0};
        mutable std::atomic<uint64_t> prof_refine_bytes_{0};

        float profDistance(uint32_t q, uint32_t p, const void* qty,
                           SearchScratch& s) const {
            if (s.replay) {
                return s.replay[s.replay_idx++];
            }
            float d = distance(&q, &p, data_matrix_, qty);
            if (s.record) {
                s.record->push_back(d);
            }
            return d;
        }

        float profDistanceDense(uint32_t p, SearchScratch& s) const {
            if (s.replay) {
                return s.replay[s.replay_idx++];
            }
            float d = distanceDense(p, s.q_dense);
            if (s.record) {
                s.record->push_back(d);
            }
            return d;
        }
#endif

        // For Hilbert curve ordering
        std::vector<uint32_t> old_to_new_labels_;
        std::vector<uint32_t> new_to_old_labels_;

        bool use_heuristic_;
        bool extend_candidates_;
        bool keep_pruned_;
        float alpha_;
        int beta_;

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
