#ifndef SPARSE_HNSW_H
#define SPARSE_HNSW_H

#include "csr_matrix.h"
#include "prune.h"
#include "quant_csr.h"
#include "inverted_seed.h"
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

// SPARSE_HNSW_PROFILE (compile time) adds distance-call / byte counters and distance record/replay.
#ifdef SPARSE_HNSW_PROFILE
#include <atomic>
#include <memory>
#endif

namespace sparse_hnsw {
    using MinPQ = std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>,
                                      std::greater<std::pair<float, uint32_t>>>;

    // Per-thread search state, reused across queries so the hot path never allocates.
    struct SearchScratch {
        // Visited bitset plus the list of set ids, so clear() costs O(visited), not O(n).
        std::vector<uint8_t> visited_bits;
        std::vector<uint32_t> visited_list;
        std::vector<uint32_t> filtered_neighbors;

        // Query scattered into a dense vector: distances become gathers over the doc's indices.
        std::vector<float> q_dense;
        bool dense_query = false;

        std::vector<std::pair<float, uint32_t>> seed_terms;
        std::vector<uint32_t> seed_ids;

        // Max-heap of the patience_k smallest layer-0 distances (front = k-th best), read by the
        // patience and score-gap stops.
        std::vector<float> kbest;
        int patience_k = 0;

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

        // Inline gate and 64 B line counts. lines = distinct row lines touched: every scored row's
        // lines, plus the line(s) the gate read for each rejected neighbor.
        uint64_t prof_gate_checks = 0;
        uint64_t prof_gate_rejects = 0;
        uint64_t prof_lines = 0;

        // Record/replay, used to measure what share of search time distance()
        // actually owns (mode=replay). searchLayer is deterministic given the
        // distance VALUES, so replaying a recorded sequence reproduces the
        // traversal, the heap operations and the visited set exactly -- while
        // touching no CSR row at all. The delta is distance()'s true cost.
        std::vector<float>* record = nullptr; // set: append each computed distance
        const float* replay = nullptr;        // set: return these instead of computing
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
        SPARSE_HNSW(int dim, CSRMatrix* data_matrix, int M = 16, int ef_construction = 200, int max_elements = 1000,
                    float alpha = 1.0, int beta = 1);

        // Kept out-of-line in a profile build so `perf report` can attribute
        // cycles to distance() as its own symbol instead of folding them into
        // searchLayer. (clang-format off: it would glue the attribute onto each declaration.)
        // clang-format off
#ifdef SPARSE_HNSW_PROFILE
        __attribute__((noinline))
#endif
        float distance(const void* pVect1, const void* pVect2, const void* qty_ptr, const void* other_ptr) const;

#ifdef SPARSE_HNSW_PROFILE
        __attribute__((noinline))
#endif
        float distanceDense(const CSRMatrix* m, uint32_t p_idx, const std::vector<float>& q_dense) const;

#ifdef SPARSE_HNSW_PROFILE
        __attribute__((noinline))
        float distanceDenseRefine(const CSRMatrix* m, uint32_t p_idx, const std::vector<float>& q_dense) const;
#endif

#ifdef SPARSE_HNSW_PROFILE
        __attribute__((noinline))
#endif
        float distanceQuant(uint32_t p_idx, const std::vector<float>& q_dense,
                            uint32_t* log_total_nnz = nullptr,
                            uint32_t* log_unused_nnz = nullptr) const;
        // clang-format on

        // u8-quantized traversal rows. Default: fixed-size, 64-byte-aligned slots
        // (no per-call row-offset lookup). slot_size = 0 keeps the packed,
        // offset-indexed layout (QuantCSR), e.g. for layout A/B runs.
        static constexpr uint32_t kDefaultSlotSize = 320;
        void enableQuantizedTraversal(uint32_t slot_size = kDefaultSlotSize);
        bool quantized() const { return quantized_; }
        size_t quantizedBytes() const { return quant_.bytes() + qslots_.bytes(); }
        // Bytes a distance call on quantized row id streams, in either layout.
        uint32_t quantRowBytes(uint32_t id) const {
            return slotted_ ? QuantSlots::rowBytes(qslots_.row(id)) : quant_.rowBytes(id);
        }

        void buildRowSlots(uint32_t slot_size, uint32_t top_k = 0) { qslots_.build(*data_matrix_, slot_size, top_k); }
        uint32_t slotSize() const { return qslots_.slot_size; }
        uint32_t slotsTopK() const { return qslots_.top_k; }

        // Inline top-k gate. With slots written top-k first (buildRowSlots(size, k)), a row's first k
        // IDs are its k heaviest terms (in slot line 0 for k <= 30). Once the layer-0 result list is full, a
        // neighbor is scored only if one of those IDs is a query term. k = 0 (or slots not built
        // with this k): off.
        void setInlineGate(int k) {
            inline_k_ = k > 0 && qslots_.top_k == static_cast<uint32_t>(k) ? qslots_.top_k : 0;
        }
        // Rebuilds the existing slots top-k first and turns the gate on (k = 0: dimension order, off).
        // Needs slots: enableQuantizedTraversal(slot_size > 0) or enableSlottedTraversal.
        void enableInlineGate(uint32_t k) {
            buildRowSlots(qslots_.slot_size, k);
            setInlineGate(static_cast<int>(k));
        }

        // Slots vs packed rows when both exist (A/B runs); slots-only builds stay slotted.
        void setSlottedRows(bool on) { slotted_ = slots_only_ || (on && quantized_ && !qslots_.empty()); }
        // Slots only, no packed QuantCSR copy.
        void enableSlottedTraversal(uint32_t slot_size) {
            buildRowSlots(slot_size);
            quantized_ = true;
            slots_only_ = true;
            slotted_ = true;
        }
        bool slottedRows() const { return slotted_; }
        int64_t slotOverflowRows() const { return qslots_.overflow_rows; }
        size_t slotBytes() const { return qslots_.bytes(); }

        void buildSeedTable(uint32_t top_k);
        void setSeedParams(int terms, int per_term);

        void setPatience(int patience) { patience_ = std::max(0, patience); }
        int patience() const { return patience_; }
        // Score gap stop: stop before expanding a node scoring below (1 - gamma) x
        // the k-th best score so far (k x beta when pruned rows are reranked). gamma < 0: off.
        void setStopGap(double gamma) { stop_gap_ = gamma; }
        bool seedingEnabled() const { return seed_terms_ > 0 && !seed_table_.empty(); }
        size_t seedTableBytes() const { return seed_table_.bytes(); }
        uint32_t seedTableTopK() const { return seed_table_.top_k; }

        void addPoint(uint32_t node_id, uint32_t label);
        void addPointsBatch(int num_points);
        std::priority_queue<std::pair<float, uint32_t>> searchKNN(uint32_t query_id, CSRMatrix* query_matrix, int k,
                                                                  int ef = 50) const;
        void searchKNNBatch(CSRMatrix* query_matrix, int num_queries, int k, int ef,
                            std::vector<uint32_t>& out_labels) const;
        CSRMatrix* pruneMatrix(const CSRMatrix* m);
        void setPrunedDataMatrix(CSRMatrix* pruned_data_matrix);
        void setSearchMatrix(CSRMatrix* search_matrix);

        void setAlpha(float alpha) { alpha_ = alpha; }
        float getAlpha() const { return alpha_; }

        void setBeta(int beta) { beta_ = beta; }
        int getBeta() const { return beta_; }
#ifdef SPARSE_HNSW_DIST_LOG
        // Select whether overlap is measured against the matrix traversed by
        // search (normally alpha-pruned) or the original unpruned matrix.
        void setDistLogUseOriginalMatrix(bool use_original) { dist_log_use_original_matrix_ = use_original; }
#endif
        void setLabelRemapping(std::vector<uint32_t> old_to_new, std::vector<uint32_t> new_to_old);
        void relabelGroundTruth(std::vector<std::vector<uint32_t>>& groundtruth) const;
        void printInfo() const;

#ifdef SPARSE_HNSW_PROFILE
        std::priority_queue<std::pair<float, uint32_t>> searchKNNProf(uint32_t query_id, CSRMatrix* query_matrix, int k,
                                                                      int ef, SearchScratch& scratch) const {
            return searchKNN(query_id, query_matrix, k, ef, scratch);
        }

        void enableAccessProfile();
        const std::atomic<uint32_t>* nodeHits() const { return node_hits_.get(); }
        bool accessProfiling() const { return node_hits_ != nullptr; }
        int elementLevel(uint32_t id) const { return element_levels_[id]; }
        uint32_t entryPoint() const { return entry_point_; }
        int maxLevel() const { return max_level_; }
        uint64_t rowBytesOf(uint32_t id) const {
            return quantized_ ? static_cast<uint64_t>(quantRowBytes(id))
                              : static_cast<uint64_t>(data_matrix_->indptr[id + 1] - data_matrix_->indptr[id]) *
                                    sizeof(IndiceDataPair);
        }

        // Totals aggregated by searchKNNBatch across its worker scratches.
        uint64_t profNDist() const { return prof_ndist_.load(std::memory_order_relaxed); }
        uint64_t profBytes() const { return prof_bytes_.load(std::memory_order_relaxed); }
        uint64_t profGraphBytes() const { return prof_graph_bytes_.load(std::memory_order_relaxed); }
        uint64_t profRefineNDist() const { return prof_refine_ndist_.load(std::memory_order_relaxed); }
        uint64_t profRefineBytes() const { return prof_refine_bytes_.load(std::memory_order_relaxed); }
        uint64_t profGateChecks() const { return prof_gate_checks_.load(std::memory_order_relaxed); }
        uint64_t profGateRejects() const { return prof_gate_rejects_.load(std::memory_order_relaxed); }
        uint64_t profLines() const { return prof_lines_.load(std::memory_order_relaxed); }
        void profReset() const {
            prof_ndist_.store(0);
            prof_bytes_.store(0);
            prof_graph_bytes_.store(0);
            prof_refine_ndist_.store(0);
            prof_refine_bytes_.store(0);
            prof_gate_checks_.store(0);
            prof_gate_rejects_.store(0);
            prof_lines_.store(0);
        }
#endif

    private:
        CSRMatrix* data_matrix_;
        CSRMatrix* original_data_matrix_;
        int dim_;
        int M_; // maximum number of connections per layer
        int ef_construction_;
        int max_elements_;
        int max_level_;
        uint32_t entry_point_;

        uint32_t size_neighbor_list_level0_;
        uint32_t size_neighbor_list_per_element_;
        std::vector<uint32_t> level0_neighbor_lists_; // [count, n1, n2, ...] per node, fixed-size block
        std::vector<uint32_t> neighbor_lists_;        // upper layers only: one contiguous buffer
        std::vector<uint32_t> neighbor_list_offsets_; // per-node start offset into neighbor_lists_
        std::vector<int> element_levels_;

        // Scratch reused across the serial insertion path.
        SearchScratch insert_scratch_;

        // u8 copy of the data matrix for traversal: packed rows (QuantCSR) ...
        QuantCSR quant_;
        bool quantized_ = false;

        // ... or fixed-size slots (QuantSlots, the default). slots_only_: no packed copy exists.
        QuantSlots qslots_;
        bool slotted_ = false;
        bool slots_only_ = false;

        // Inverted seeds (setSeedParams) and the layer-0 stops (setPatience, setStopGap).
        InvertedSeedTable seed_table_;
        int seed_terms_ = 0;
        int seed_per_term_ = 1;
        int patience_ = 0;
        double stop_gap_ = -1.0;
#ifdef SPARSE_HNSW_DIST_LOG
        bool dist_log_use_original_matrix_ = false;
#endif

        // Parallel build: one mutex per element for its neighbor lists, one for entry_point_ / max_level_.
        mutable std::vector<std::mutex> link_locks_;
        std::mutex global_lock_;

        uint32_t inline_k_ = 0; // setInlineGate
        bool inlinePasses(uint32_t id, const float* q) const;

#ifdef SPARSE_HNSW_PROFILE
        mutable std::atomic<uint64_t> prof_ndist_{0};
        mutable std::atomic<uint64_t> prof_bytes_{0};
        mutable std::atomic<uint64_t> prof_graph_bytes_{0};
        mutable std::atomic<uint64_t> prof_refine_ndist_{0};
        mutable std::atomic<uint64_t> prof_refine_bytes_{0};
        mutable std::atomic<uint64_t> prof_gate_checks_{0};
        mutable std::atomic<uint64_t> prof_gate_rejects_{0};
        mutable std::atomic<uint64_t> prof_lines_{0};
        mutable std::unique_ptr<std::atomic<uint32_t>[]> node_hits_;

        // 64 B lines spanned by n bytes at p.
        static uint64_t profSpan(const void* p, uint64_t n) {
            const uintptr_t a = reinterpret_cast<uintptr_t>(p);
            return n ? ((a + n - 1) >> 6) - (a >> 6) + 1 : 1;
        }
        // Lines a distance call on traversal row id touches; an overflowed slot adds its slot line.
        uint64_t profRowLines(uint32_t id, bool quant) const {
            if (!quant) {
                const int64_t s = data_matrix_->indptr[id];
                return profSpan(data_matrix_->indices_data + s,
                                (data_matrix_->indptr[id + 1] - s) * sizeof(IndiceDataPair));
            }
            if (!slotted_) return profSpan(quant_.blob.data() + quant_.row_off[id], quant_.rowBytes(id));
            const uint8_t* p = qslots_.row(id);
            return profSpan(p, QuantSlots::rowBytes(p)) + (p != qslots_.base + size_t(id) * qslots_.slot_size);
        }
        // Lines inlinePasses reads: the header and the first inline_k_ IDs (+ the slot line if overflowed).
        uint64_t profGateLines(uint32_t id) const {
            const uint8_t* p = qslots_.row(id);
            uint16_t n;
            std::memcpy(&n, p, sizeof(n));
            return profSpan(p, QuantCSR::kHeader + 2u * std::min<uint32_t>(n, inline_k_)) +
                   (p != qslots_.base + size_t(id) * qslots_.slot_size);
        }

        float profDistance(uint32_t q, uint32_t p, const void* qty, SearchScratch& s) const {
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
            float d = distanceDense(data_matrix_, p, s.q_dense);
            if (s.record) {
                s.record->push_back(d);
            }
            return d;
        }

        float profDistanceQuant(uint32_t p, SearchScratch& s) const {
            if (s.replay) {
                return s.replay[s.replay_idx++];
            }
            float d = distanceQuant(p, s.q_dense);
            if (s.record) {
                s.record->push_back(d);
            }
            return d;
        }
#endif

        // Label remapping after a node reordering (setLabelRemapping); empty = identity.
        std::vector<uint32_t> old_to_new_labels_;
        std::vector<uint32_t> new_to_old_labels_;

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
        std::priority_queue<std::pair<float, uint32_t>> searchLayer(uint32_t query_id, const void* qty_ptr,
                                                                    std::vector<uint32_t> entry_points, int ef,
                                                                    int layer, SearchScratch& scratch,
                                                                    bool lock_links = false) const;
        std::priority_queue<std::pair<float, uint32_t>> searchKNN(uint32_t query_id, CSRMatrix* query_matrix, int k,
                                                                  int ef, SearchScratch& scratch) const;
        void collectSeeds(CSRMatrix* query_matrix, uint32_t query_id, SearchScratch& scratch) const;
        void addPointInternal(uint32_t node_id, uint32_t label, SearchScratch& scratch);
        void connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level,
                              int M);
        std::vector<uint32_t>
        selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M);
    };
} // namespace sparse_hnsw

#endif // SPARSE_HNSW_H
