#include "sparse_hnsw.h"
#include <iostream>
#include <cstring>

#if defined(__has_include)
#if __has_include(<mkl.h>)
#include <mkl.h>
#define SPARSE_HNSW_HAS_MKL 1
#endif
#endif

#ifndef SPARSE_HNSW_HAS_MKL
#define SPARSE_HNSW_HAS_MKL 0
#endif

using namespace sparse_hnsw;

#ifdef SPARSE_HNSW_PROFILE
#define SPARSE_HNSW_DISTANCE(q, p) profDistance((q), (p), qty_ptr, scratch)
#define SPARSE_HNSW_DISTANCE_DENSE(p) profDistanceDense((p), scratch)
#define SPARSE_HNSW_DISTANCE_QUANT(p) profDistanceQuant((p), scratch)
#define SPARSE_HNSW_DISTANCE_DENSE_REFINE(m, p) distanceDenseRefine((m), (p), scratch.q_dense)
#define SPARSE_HNSW_REPLAYING (scratch.replay != nullptr)
#else
#define SPARSE_HNSW_DISTANCE(q, p) distance(&(q), &(p), data_matrix_, qty_ptr)
#define SPARSE_HNSW_DISTANCE_DENSE(p) distanceDense(data_matrix_, (p), scratch.q_dense)
#define SPARSE_HNSW_DISTANCE_QUANT(p) distanceQuant((p), scratch.q_dense)
#define SPARSE_HNSW_DISTANCE_DENSE_REFINE(m, p) distanceDense((m), (p), scratch.q_dense)
#define SPARSE_HNSW_REPLAYING false
#endif

#define SPARSE_HNSW_TRAVERSAL_DISTANCE(p)                       \
    (use_quant ? SPARSE_HNSW_DISTANCE_QUANT(p)                  \
               : (scratch.dense_query ? SPARSE_HNSW_DISTANCE_DENSE(p) \
                                      : SPARSE_HNSW_DISTANCE(query_id, p)))



SPARSE_HNSW::SPARSE_HNSW(int dim, CSRMatrix *data_matrix, int M, int ef_construction, int max_elements, 
    bool use_heuristic, bool extend_candidates, bool keep_pruned, float alpha, int beta)
    : data_matrix_(data_matrix), original_data_matrix_(nullptr), dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements),
      use_heuristic_(use_heuristic), extend_candidates_(extend_candidates), keep_pruned_(keep_pruned),
      alpha_(alpha), beta_(beta), max_level_(0), entry_point_(-1),
      rng_(42), level_generator_(0.0, 1.0), link_locks_(max_elements) {
    size_neighbor_list_level0_ = static_cast<uint32_t>(2 * M_ + 1);  // count + maxM0 neighbors
    size_neighbor_list_per_element_ = static_cast<uint32_t>(M_ + 1); // count + maxM neighbors
    level0_neighbor_lists_.assign(static_cast<size_t>(max_elements_) * size_neighbor_list_level0_, 0);
    neighbor_list_offsets_.assign(max_elements_, std::numeric_limits<uint32_t>::max());
    element_levels_.assign(max_elements_, 0);

    // Precompute the max level for each element and allocate the upper layer neighbor lists.
    size_t upper_words = 0;
    for (int i = 0; i < max_elements_; ++i) {
        int level = getRandomLevel();
        element_levels_[i] = level;
        if (level > 0) {
            neighbor_list_offsets_[i] = static_cast<uint32_t>(upper_words);
            upper_words += static_cast<size_t>(level) * size_neighbor_list_per_element_;
        }
    }
    neighbor_lists_.assign(upper_words, 0);

    insert_scratch_.prepare(max_elements_);
    insert_scratch_.visited_list.reserve(static_cast<size_t>(max_elements_));
}

uint32_t* SPARSE_HNSW::get_neighbor_list0(uint32_t node_id) {
    return level0_neighbor_lists_.data() + static_cast<size_t>(node_id) * size_neighbor_list_level0_;
}

const uint32_t* SPARSE_HNSW::get_neighbor_list0(uint32_t node_id) const {
    return level0_neighbor_lists_.data() + static_cast<size_t>(node_id) * size_neighbor_list_level0_;
}

uint32_t* SPARSE_HNSW::get_neighbor_list(uint32_t node_id, int level) {
    if (level <= 0 || level > element_levels_[node_id]) {
        return nullptr;
    }
    uint32_t base = neighbor_list_offsets_[node_id];
    if (base == std::numeric_limits<uint32_t>::max()) {
        return nullptr;
    }
    return neighbor_lists_.data() + static_cast<size_t>(base) + static_cast<size_t>(level - 1) * size_neighbor_list_per_element_;
}

const uint32_t* SPARSE_HNSW::get_neighbor_list(uint32_t node_id, int level) const {
    if (level <= 0 || level > element_levels_[node_id]) {
        return nullptr;
    }
    uint32_t base = neighbor_list_offsets_[node_id];
    if (base == std::numeric_limits<uint32_t>::max()) {
        return nullptr;
    }
    return neighbor_lists_.data() + static_cast<size_t>(base) + static_cast<size_t>(level - 1) * size_neighbor_list_per_element_;
}

uint32_t* SPARSE_HNSW::get_neighbor_list_at_level(uint32_t node_id, int level) {
    return level == 0 ? get_neighbor_list0(node_id) : get_neighbor_list(node_id, level);
}

const uint32_t* SPARSE_HNSW::get_neighbor_list_at_level(uint32_t node_id, int level) const {
    return level == 0 ? get_neighbor_list0(node_id) : get_neighbor_list(node_id, level);
}

uint32_t SPARSE_HNSW::getListCount(const uint32_t* ptr) const {
    return ptr ? ptr[0] : 0;
}

void SPARSE_HNSW::setListCount(uint32_t* ptr, uint32_t size) {
    if (ptr) {
        ptr[0] = size;
    }
}

std::vector<uint32_t> SPARSE_HNSW::getNeighborsAtLevel(uint32_t node_id, int level) const {
    const uint32_t* ll = get_neighbor_list_at_level(node_id, level);
    if (!ll) {
        return {};
    }
    uint32_t count = getListCount(ll);
    std::vector<uint32_t> neighbors;
    neighbors.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        neighbors.push_back(ll[1 + i]);
    }
    return neighbors;
}

void SPARSE_HNSW::setNeighborsAtLevel(uint32_t node_id, int level, const std::vector<uint32_t>& neighbors, int max_degree) {
    uint32_t* ll = get_neighbor_list_at_level(node_id, level);
    if (!ll) {
        return;
    }
    uint32_t block_size = (level == 0) ? size_neighbor_list_level0_ : size_neighbor_list_per_element_;
    uint32_t capped = std::min<uint32_t>(static_cast<uint32_t>(neighbors.size()), static_cast<uint32_t>(max_degree));
    setListCount(ll, capped);
    for (uint32_t i = 0; i < capped; ++i) {
        ll[1 + i] = neighbors[i];
    }
    for (uint32_t i = capped + 1; i < block_size; ++i) {
        ll[i] = 0;
    }
}

float SPARSE_HNSW::distance(const void *pVect1, const void *pVect2, const void *qty_ptr, const void *other_ptr) const {

    CSRMatrix *csr_matrix = reinterpret_cast<CSRMatrix *>(const_cast<void *>(qty_ptr));
        
    const uint32_t q_idx = *((uint32_t *) pVect1);
    const uint32_t p_idx = *((uint32_t *) pVect2);

    const uint32_t p_start = csr_matrix->indptr[p_idx];
    const uint32_t p_end = csr_matrix->indptr[p_idx + 1];
    IndiceDataPair *p_indices = csr_matrix->indices_data + p_start;

    uint32_t q_end, q_start = 0;
    IndiceDataPair *q_indices;
    if (other_ptr == nullptr) {
        q_start = csr_matrix->indptr[q_idx];
        q_end = csr_matrix->indptr[q_idx + 1];
        q_indices = csr_matrix->indices_data + q_start;
    } else {
        CSRMatrix *csr_matrix_query = reinterpret_cast<CSRMatrix *>(const_cast<void *>(other_ptr));
        q_start = csr_matrix_query->indptr[q_idx];
        q_end = csr_matrix_query->indptr[q_idx + 1];
        q_indices = csr_matrix_query->indices_data + q_start;
    }

    uint32_t p_num = p_end - p_start;
    uint32_t q_num = q_end - q_start;

    float res = 0;

    IndiceDataPair* q_indices_end = q_indices + q_num;
    IndiceDataPair* p_indices_end = p_indices + p_num;

    while (q_indices < q_indices_end && p_indices < p_indices_end)
    {
        const int32_t q_col = q_indices->indice;
        const int32_t p_col = p_indices->indice;
        // Convert both fp16 operands up front so the product is computed and
        // accumulated in fp32 rather than rounded through fp16.
        res += (q_col == p_col) ? static_cast<float>(q_indices->data) * static_cast<float>(p_indices->data) : 0.0f;
        q_indices += (q_col <= p_col);
        p_indices += (p_col <= q_col);
    }
    return 1.0f - res;
}

__attribute__((always_inline))
static inline float denseDotRow(const CSRMatrix* m, uint32_t p_idx, const std::vector<float>& q_dense) {
    const int64_t p_start = m->indptr[p_idx];
    const int64_t p_end = m->indptr[p_idx + 1];
    const IndiceDataPair* p = m->indices_data + p_start;
    const uint32_t p_num = static_cast<uint32_t>(p_end - p_start);

    float res = 0.0f;
    #pragma omp simd reduction(+:res)
    for (uint32_t i = 0; i < p_num; ++i) {
        res += static_cast<float>(p[i].data) * q_dense[p[i].indice];
    }
    return 1.0f - res;
}

float SPARSE_HNSW::distanceDense(const CSRMatrix* m, uint32_t p_idx, const std::vector<float>& q_dense) const {
    return denseDotRow(m, p_idx, q_dense);
}

#ifdef SPARSE_HNSW_PROFILE
float SPARSE_HNSW::distanceDenseRefine(const CSRMatrix* m, uint32_t p_idx, const std::vector<float>& q_dense) const {
    return denseDotRow(m, p_idx, q_dense);
}
#endif

float SPARSE_HNSW::distanceQuant(uint32_t p_idx, const std::vector<float>& q_dense) const {
    const uint8_t* p = quant_.blob.data() + quant_.row_off[p_idx];

    uint16_t n16;
    _Float16 scale16;
    std::memcpy(&n16, p, sizeof(n16));
    std::memcpy(&scale16, p + 2, sizeof(scale16));
    const uint32_t p_num = n16;

    const uint16_t* idx = reinterpret_cast<const uint16_t*>(p + QuantCSR::kHeader);
    const uint8_t* code = p + QuantCSR::kHeader + 2 * static_cast<size_t>(p_num);
    const float* q = q_dense.data();

    float res = 0.0f;
    #pragma omp simd reduction(+:res)
    for (uint32_t i = 0; i < p_num; ++i) {
        res += static_cast<float>(code[i]) * q[idx[i]];
    }
    return 1.0f - res * static_cast<float>(scale16);
}

void SPARSE_HNSW::enableQuantizedTraversal() {
    quant_.build(*data_matrix_);
    quantized_ = true;
}

void SPARSE_HNSW::buildSeedTable(uint32_t top_k) {
    const CSRMatrix* src = original_data_matrix_ ? original_data_matrix_ : data_matrix_;
    seed_table_.build(*src, top_k);
}

void SPARSE_HNSW::setSeedParams(int terms, int per_term) {
    seed_terms_ = std::max(0, terms);
    seed_per_term_ = std::max(1, per_term);
}

void SPARSE_HNSW::collectSeeds(CSRMatrix* query_matrix, uint32_t query_id,
                               SearchScratch& scratch) const {
    scratch.seed_ids.clear();
    if (!seedingEnabled()) {
        return;
    }

    const int64_t q_start = query_matrix->indptr[query_id];
    const int64_t q_end = query_matrix->indptr[query_id + 1];
    scratch.seed_terms.clear();
    scratch.seed_terms.reserve(static_cast<size_t>(q_end - q_start));
    for (int64_t i = q_start; i < q_end; ++i) {
        scratch.seed_terms.push_back({static_cast<float>(query_matrix->indices_data[i].data),
                                      query_matrix->indices_data[i].indice});
    }

    const size_t h = std::min<size_t>(static_cast<size_t>(seed_terms_), scratch.seed_terms.size());
    std::partial_sort(scratch.seed_terms.begin(), scratch.seed_terms.begin() + h,
                      scratch.seed_terms.end(),
                      std::greater<std::pair<float, uint32_t>>());

    const uint32_t take = std::min<uint32_t>(static_cast<uint32_t>(seed_per_term_),
                                             seed_table_.top_k);
    for (size_t t = 0; t < h; ++t) {
        const uint32_t* col = seed_table_.column(scratch.seed_terms[t].second);
        for (uint32_t u = 0; u < take; ++u) {
            if (col[u] != InvertedSeedTable::kNone) {
                scratch.seed_ids.push_back(col[u]);
            }
        }
    }
}

int SPARSE_HNSW::getRandomLevel() {
    double r = level_generator_(rng_);
    // Ensure r is not too close to 0 to avoid log(0).
    r = std::max(r, std::numeric_limits<double>::min());
    return static_cast<int>(-log(r) * (1.0 / log(M_)));
}

std::priority_queue<std::pair<float, uint32_t>> SPARSE_HNSW::searchLayer(uint32_t query_id, const void *qty_ptr, std::vector<uint32_t> entry_points, int ef, int layer, SearchScratch& scratch, bool lock_links) const {
    scratch.prepare(max_elements_);

    MinPQ candidates;
    std::priority_queue<std::pair<float, uint32_t>> top_candidates;

    const bool use_quant = quantized_ && scratch.dense_query;
    const bool use_patience = patience_ > 0 && layer == 0 &&
                              scratch.dense_query && scratch.patience_k > 0;
    uint32_t stale_expansions = 0;
    if (use_patience) {
        scratch.kbest.clear();
        scratch.kbest.reserve(static_cast<size_t>(scratch.patience_k));
    }

    auto touchTopK = [&](float d) -> bool {
        if (!use_patience) {
            return false;
        }
        if (static_cast<int>(scratch.kbest.size()) < scratch.patience_k) {
            scratch.kbest.push_back(d);
            std::push_heap(scratch.kbest.begin(), scratch.kbest.end());
            return true;
        }
        if (d < scratch.kbest.front()) {
            std::pop_heap(scratch.kbest.begin(), scratch.kbest.end());
            scratch.kbest.back() = d;
            std::push_heap(scratch.kbest.begin(), scratch.kbest.end());
            return true;
        }
        return false;
    };
    const int64_t* qrow_off = quant_.row_off.empty() ? nullptr : quant_.row_off.data();
    const uint8_t* qblob = quant_.blob.empty() ? nullptr : quant_.blob.data();

    for (uint32_t entry_point : entry_points) {
#ifdef SPARSE_HNSW_PROFILE
        scratch.prof_ndist++;
        scratch.prof_bytes += use_quant
            ? static_cast<uint64_t>(quant_.rowBytes(entry_point))
            : static_cast<uint64_t>(
                  data_matrix_->indptr[entry_point + 1] - data_matrix_->indptr[entry_point]) * sizeof(IndiceDataPair);
#endif
        float d = SPARSE_HNSW_TRAVERSAL_DISTANCE(entry_point);
        touchTopK(d);
        candidates.push({d, entry_point});
        top_candidates.push({d, entry_point});
        scratch.markVisited(entry_point);
    }

    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();

        // Compare with the farthest in nearest neighbors.
        if (top_candidates.size() >= static_cast<size_t>(ef) && current.first > top_candidates.top().first) {
            break;
        }

        uint32_t current_node = current.second;

        const uint32_t* ll = get_neighbor_list_at_level(current_node, layer);
        if (!ll) {
            continue;
        }
        const uint32_t* neighbors = ll + 1;
        const int64_t* indptr = data_matrix_->indptr;
        const IndiceDataPair* vec_base = data_matrix_->indices_data;
        const uint8_t* visited_bytes = scratch.visited_bits.data();

        scratch.filtered_neighbors.clear();

        {
            std::unique_lock<std::mutex> link_guard(link_locks_[current_node], std::defer_lock);
            if (lock_links) {
                link_guard.lock();
            }
            uint32_t count = getListCount(ll);
            scratch.filtered_neighbors.reserve(count);
#ifdef SPARSE_HNSW_PROFILE
            scratch.prof_graph_bytes += static_cast<uint64_t>(count + 1) * sizeof(uint32_t);
#endif

            for (uint32_t i = 0; i < count && i < 4; ++i) {
                __builtin_prefetch(visited_bytes + (neighbors[i] >> 3), 0, 3);
            }

            for (uint32_t i = 0; i < count; ++i) {
                if (i + 4 < count) {
                    __builtin_prefetch(visited_bytes + (neighbors[i + 4] >> 3), 0, 3);
                }
                uint32_t neighbor_id = neighbors[i];
                if (!scratch.isVisited(neighbor_id)) {
                    scratch.markVisited(neighbor_id);
                    scratch.filtered_neighbors.push_back(neighbor_id);
                    if (!SPARSE_HNSW_REPLAYING) {
                        __builtin_prefetch(use_quant ? static_cast<const void*>(qrow_off + neighbor_id)
                                                     : static_cast<const void*>(indptr + neighbor_id), 0, 3);
                    }
                }
            }
        }

        const uint32_t nfilter = static_cast<uint32_t>(scratch.filtered_neighbors.size());
        const uint32_t* filtered = scratch.filtered_neighbors.data();

        bool improved = false;

        if (!SPARSE_HNSW_REPLAYING) {
            for (uint32_t j = 0; j < nfilter; ++j) {
                const char* vec = use_quant
                    ? reinterpret_cast<const char*>(qblob + qrow_off[filtered[j]])
                    : reinterpret_cast<const char*>(vec_base + indptr[filtered[j]]);
                __builtin_prefetch(vec, 0, 2);
                __builtin_prefetch(vec + 64, 0, 2);
            }
        }

        // Process unvisited neighbors.
        for (uint32_t j = 0; j < nfilter; ++j) {
            uint32_t neighbor_id = filtered[j];

            if (j + 1 < nfilter && !SPARSE_HNSW_REPLAYING) {
                const char* next_vec = use_quant
                    ? reinterpret_cast<const char*>(qblob + qrow_off[filtered[j + 1]])
                    : reinterpret_cast<const char*>(vec_base + indptr[filtered[j + 1]]);
                __builtin_prefetch(next_vec, 0, 3);
                __builtin_prefetch(next_vec + 64, 0, 3);
                __builtin_prefetch(next_vec + 128, 0, 3);
                __builtin_prefetch(next_vec + 192, 0, 3);
            }

#ifdef SPARSE_HNSW_PROFILE
            scratch.prof_ndist++;
            if (!scratch.replay) {
                scratch.prof_bytes += use_quant
                    ? static_cast<uint64_t>(quant_.rowBytes(neighbor_id))
                    : static_cast<uint64_t>(
                          indptr[neighbor_id + 1] - indptr[neighbor_id]) * sizeof(IndiceDataPair);
            }
#endif
            float dist = SPARSE_HNSW_TRAVERSAL_DISTANCE(neighbor_id);

            if (touchTopK(dist)) {
                improved = true;
            }

            if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
                candidates.push({dist, neighbor_id});
                // The best pending candidate is the likely next expansion.
                __builtin_prefetch(get_neighbor_list_at_level(candidates.top().second, layer), 0, 3);
                top_candidates.push({dist, neighbor_id});

                if (top_candidates.size() > static_cast<size_t>(ef)) {
                    top_candidates.pop();
                }
            }
        }

        if (use_patience) {
            if (improved) {
                stale_expansions = 0;
            } else if (++stale_expansions >= static_cast<uint32_t>(patience_)) {
                break;
            }
        }
    }

    scratch.clear();
    return top_candidates;
}

std::vector<uint32_t> SPARSE_HNSW::selectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M) {

    if (candidates.size() <= static_cast<size_t>(M)) {
        std::vector<uint32_t> result;
        while (!candidates.empty()) {
            result.push_back(candidates.top().second);
            candidates.pop();
        }
        return result;
    }

    std::vector<uint32_t> selected_candidates;
    while (!candidates.empty()) {
        if (static_cast<int>(candidates.size()) > M) {
            candidates.pop();
            continue;
        }
        uint32_t candidate = candidates.top().second;
        candidates.pop();
        selected_candidates.push_back(candidate);
    }
    return selected_candidates;
}

std::vector<uint32_t> SPARSE_HNSW::selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M, int level, bool allow_extend) {

   if (candidates.size() <= static_cast<size_t>(M)) {
        std::priority_queue<std::pair<float, uint32_t>> temp = candidates;
        std::vector<uint32_t> result;
        while (!temp.empty()) {
            result.push_back(temp.top().second);
            temp.pop();
        }
        return result;
    }

    MinPQ working_set;
    while (!candidates.empty()) {
        working_set.push(candidates.top());
        candidates.pop();
    }
    std::vector<uint32_t> results_set;
    
    if (allow_extend && extend_candidates_) {
        MinPQ temp = working_set;
        while (!temp.empty()) {
            uint32_t candidate = temp.top().second;
            temp.pop();
            std::vector<uint32_t> candidate_neighbors;
            {
                std::lock_guard<std::mutex> link_guard(link_locks_[candidate]);
                candidate_neighbors = getNeighborsAtLevel(candidate, level);
            }
            for (uint32_t neighbor : candidate_neighbors) {
                if (neighbor != node_id) { // Ideally should check in working_set if the neighbor is already there.
                    float dist = distance(&node_id, &neighbor, data_matrix_, nullptr);
                    working_set.push({dist, neighbor});
                }
            }
        }
    }

    MinPQ discarded_set;
    while (!working_set.empty() && static_cast<int>(results_set.size()) < M) {
        auto current = working_set.top();
        working_set.pop();
        
        bool good = true;
        for (int result : results_set) {
            float dist = distance(&current.second, &result, data_matrix_, nullptr);
            if (dist < current.first) {
                good = false;
                break;
            }
        }
        
        if (good) {
            results_set.push_back(current.second);
        } else if (keep_pruned_) {
            discarded_set.push(current);
        }
    }

    if (keep_pruned_) {
        while (!discarded_set.empty() && static_cast<int>(results_set.size()) < M) {
            results_set.push_back(discarded_set.top().second);
            discarded_set.pop();
        }
    }

    return results_set;
}

void SPARSE_HNSW::connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level, int M) {

    std::vector<uint32_t> selected_neighbors = use_heuristic_
        ? selectNeighborsHeuristic(node_id, std::move(candidates), M, level, true)
        : selectNeighbors(node_id, std::move(candidates), M);

    {
        std::lock_guard<std::mutex> link_guard(link_locks_[node_id]);
        setNeighborsAtLevel(node_id, level, selected_neighbors, M);
    }

    const int neighbor_max_degree = (level == 0) ? (2 * M_) : M_;

    // Add the back-links, operating directly on each neighbor's raw link list
    // instead of copying it out to a std::vector.
    for (uint32_t neighbor : selected_neighbors) {
        uint32_t* ll = get_neighbor_list_at_level(neighbor, level);
        if (!ll) {
            continue;
        }
        std::lock_guard<std::mutex> link_guard(link_locks_[neighbor]);
        uint32_t cnt = getListCount(ll);

        // Skip if node_id is already linked.
        bool present = false;
        for (uint32_t i = 0; i < cnt; ++i) {
            if (ll[1 + i] == node_id) {
                present = true;
                break;
            }
        }
        if (present) {
            continue;
        }

        if (static_cast<int>(cnt) < neighbor_max_degree) {
            ll[1 + cnt] = node_id;
            setListCount(ll, cnt + 1);
        } else {
            std::priority_queue<std::pair<float, uint32_t>> econn_candidates;
            econn_candidates.push({distance(&neighbor, &node_id, data_matrix_, nullptr), node_id});
            for (uint32_t i = 0; i < cnt; ++i) {
                uint32_t other = ll[1 + i];
                econn_candidates.push({distance(&neighbor, &other, data_matrix_, nullptr), other});
            }
            std::vector<uint32_t> reduced_neighbors = use_heuristic_
                ? selectNeighborsHeuristic(neighbor, std::move(econn_candidates), neighbor_max_degree, level, false)
                : selectNeighbors(neighbor, std::move(econn_candidates), neighbor_max_degree);
            setNeighborsAtLevel(neighbor, level, reduced_neighbors, neighbor_max_degree);
        }
    }
}

void SPARSE_HNSW::addPoint(uint32_t node_id, uint32_t label) {
    addPointInternal(node_id, label, insert_scratch_);
}

void SPARSE_HNSW::addPointInternal(uint32_t node_id, uint32_t label, SearchScratch& scratch) {
    int level = element_levels_[label];

    std::unique_lock<std::mutex> global_guard(global_lock_);
    if (entry_point_ == -1) {
        entry_point_ = label;
        max_level_ = level;
        return;
    }
    int max_level_copy = max_level_;
    uint32_t entry_point_copy = entry_point_;
    if (level <= max_level_copy) {
        global_guard.unlock();
    }

    std::vector<uint32_t> entry_points = {entry_point_copy};

    // Search from top layer to target layer
    for (int lc = max_level_copy; lc > level; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> nearest = searchLayer(node_id, data_matrix_, entry_points, 1, lc, scratch, true);
        if (!nearest.empty()) {
            entry_points.clear();
            while (!nearest.empty()) {
                entry_points.push_back(nearest.top().second);
                nearest.pop();
            }
        }
    }

    // Insert at all layers from level to 0
    for (int lc = level; lc >= 0; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> candidates = searchLayer(node_id, data_matrix_, entry_points, ef_construction_, lc, scratch, true);
        if (!candidates.empty()) {
            entry_points.clear();
            std::priority_queue<std::pair<float, uint32_t>> temp = candidates;
            while (!temp.empty()) {
                entry_points.push_back(temp.top().second);
                temp.pop();
            }
        }

        connectNeighbors(label, std::move(candidates), lc, M_);
    }

    if (level > max_level_copy) {
        max_level_ = level;
        entry_point_ = label;
    }
}

void SPARSE_HNSW::addPointsBatch(int num_points) {
    int start = 0;
    if (entry_point_ == -1 && num_points > 0) {
        // Seed the graph serially so every parallel insert has an entry point.
        addPointInternal(0, 0, insert_scratch_);
        start = 1;
    }

    #pragma omp parallel
    {
        SearchScratch scratch;
        scratch.prepare(max_elements_);

        #pragma omp for schedule(dynamic, 16)
        for (int i = start; i < num_points; ++i) {
            addPointInternal(static_cast<uint32_t>(i), static_cast<uint32_t>(i), scratch);
        }
    }
}

std::priority_queue<std::pair<float, uint32_t>> SPARSE_HNSW::searchKNN(uint32_t query_id, CSRMatrix *query_matrix, int k, int ef, SearchScratch& scratch) const {
    if (entry_point_ == -1) {
        return {};
    }

    const int64_t q_start = query_matrix->indptr[query_id];
    const int64_t q_end = query_matrix->indptr[query_id + 1];
    const IndiceDataPair* q_indices = query_matrix->indices_data + q_start;
    const uint32_t q_num = static_cast<uint32_t>(q_end - q_start);
    scratch.scatterQuery(dim_, q_indices, q_num);

    std::vector<uint32_t> entry_points = {entry_point_};

    // Search from top layer to layer 0
    for (int lc = max_level_; lc > 0; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> nearest = searchLayer(query_id, query_matrix, entry_points, 1, lc, scratch);
        if (!nearest.empty()) {
            entry_points.clear();
            while (!nearest.empty()) {
                entry_points.push_back(nearest.top().second);
                nearest.pop();
            }
        }
    }

    // Inverted-list seeding: enter layer 0 additionally at the strongest
    // documents of the query's heaviest terms.
    collectSeeds(query_matrix, query_id, scratch);
    if (!scratch.seed_ids.empty()) {
        entry_points.insert(entry_points.end(),
                            scratch.seed_ids.begin(), scratch.seed_ids.end());
        std::sort(entry_points.begin(), entry_points.end());
        entry_points.erase(std::unique(entry_points.begin(), entry_points.end()),
                           entry_points.end());
    }
    scratch.patience_k = k;
    std::priority_queue<std::pair<float, uint32_t>> result =
        searchLayer(query_id, query_matrix, entry_points, std::max(ef, k), 0, scratch);

    scratch.unscatterQuery(q_indices, q_num);

    // result is a max-heap on distance; pop the farthest until k remain.
    while (static_cast<int>(result.size()) > k) {
        result.pop();
    }
    return result;
}

std::priority_queue<std::pair<float, uint32_t>> SPARSE_HNSW::searchKNN(uint32_t query_id, CSRMatrix *query_matrix, int k, int ef) const {
    SearchScratch scratch;
    return searchKNN(query_id, query_matrix, k, ef, scratch);
}

void SPARSE_HNSW::searchKNNBatch(CSRMatrix *query_matrix, int num_queries, int k, int ef,
                                 std::vector<uint32_t>& out_labels) const {
    
    int k_hat = k;
    if (alpha_ < 1.0f) {
        k_hat = k * beta_;
    }
    out_labels.assign(static_cast<size_t>(num_queries) * static_cast<size_t>(k), 0);

    const bool refine = (alpha_ < 1.0f && beta_ > 1);
    std::vector<uint32_t> approx_labels(static_cast<size_t>(num_queries) * static_cast<size_t>(k_hat), 0);
    #pragma omp parallel
    {
        SearchScratch scratch;
        scratch.prepare(max_elements_);

        std::vector<std::pair<float, uint32_t>> heap;
        if (refine) heap.reserve(static_cast<size_t>(k));

        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < num_queries; ++i) {
            std::priority_queue<std::pair<float, uint32_t>> nns =
                searchKNN(static_cast<uint32_t>(i), query_matrix, k_hat, ef, scratch);
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k_hat);
            while (!nns.empty()) {
                approx_labels[base + (static_cast<size_t>(k_hat) - nns.size())] = nns.top().second;
                nns.pop();
            }
        }

        if (refine) {
            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < num_queries; ++i) {
                const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k_hat);
                const uint32_t query_id = static_cast<uint32_t>(i);

                const int64_t q_start = query_matrix->indptr[query_id];
                const int64_t q_end = query_matrix->indptr[query_id + 1];
                const IndiceDataPair* q_indices = query_matrix->indices_data + q_start;
                const uint32_t q_num = static_cast<uint32_t>(q_end - q_start);
                scratch.scatterQuery(dim_, q_indices, q_num);

                const int64_t* refine_indptr = original_data_matrix_->indptr;
                const IndiceDataPair* refine_base = original_data_matrix_->indices_data;
                {
                    const char* vec = reinterpret_cast<const char*>(refine_base + refine_indptr[approx_labels[base]]);
                    __builtin_prefetch(vec, 0, 3);
                    __builtin_prefetch(vec + 64, 0, 3);
                    __builtin_prefetch(vec + 128, 0, 3);
                    __builtin_prefetch(vec + 192, 0, 3);
                }

                heap.clear();
                for (int j = 0; j < k_hat; ++j) {
                    const uint32_t label = approx_labels[base + j];
                    if (j + 1 < k_hat) {
                        const char* next_vec = reinterpret_cast<const char*>(refine_base + refine_indptr[approx_labels[base + j + 1]]);
                        __builtin_prefetch(next_vec, 0, 3);
                        __builtin_prefetch(next_vec + 64, 0, 3);
                        __builtin_prefetch(next_vec + 128, 0, 3);
                        __builtin_prefetch(next_vec + 192, 0, 3);
                    }
#ifdef SPARSE_HNSW_PROFILE
                    scratch.prof_refine_ndist++;
                    scratch.prof_refine_bytes += static_cast<uint64_t>(
                        original_data_matrix_->indptr[label + 1] -
                        original_data_matrix_->indptr[label]) * sizeof(IndiceDataPair);
#endif
                    const float d = SPARSE_HNSW_DISTANCE_DENSE_REFINE(original_data_matrix_, label);
                    if (static_cast<int>(heap.size()) < k) {
                        heap.emplace_back(d, label);
                        std::push_heap(heap.begin(), heap.end());
                    } else if (d < heap.front().first) {
                        std::pop_heap(heap.begin(), heap.end());
                        heap.back() = {d, label};
                        std::push_heap(heap.begin(), heap.end());
                    }
                }
                scratch.unscatterQuery(q_indices, q_num);
                const size_t out_base = static_cast<size_t>(i) * static_cast<size_t>(k);
                while (!heap.empty()) {
                    std::pop_heap(heap.begin(), heap.end());
                    out_labels[out_base + heap.size() - 1] = heap.back().second;
                    heap.pop_back();
                }
            }
        } else {
            #pragma omp single
            {
                out_labels = std::move(approx_labels);
            }
        }

#ifdef SPARSE_HNSW_PROFILE
        prof_ndist_.fetch_add(scratch.prof_ndist, std::memory_order_relaxed);
        prof_bytes_.fetch_add(scratch.prof_bytes, std::memory_order_relaxed);
        prof_graph_bytes_.fetch_add(scratch.prof_graph_bytes, std::memory_order_relaxed);
        prof_refine_ndist_.fetch_add(scratch.prof_refine_ndist, std::memory_order_relaxed);
        prof_refine_bytes_.fetch_add(scratch.prof_refine_bytes, std::memory_order_relaxed);
#endif
    }
}

void SPARSE_HNSW::setPrunedDataMatrix(CSRMatrix *pruned_data_matrix) {
    original_data_matrix_ = data_matrix_;
    data_matrix_ = pruned_data_matrix;
}

CSRMatrix* SPARSE_HNSW::pruneMatrix(const CSRMatrix *m) {
    return sparse_hnsw::pruneMatrixWithAlpha(m, alpha_);
}

void SPARSE_HNSW::setLabelRemapping(std::vector<uint32_t> old_to_new, std::vector<uint32_t> new_to_old) {
    old_to_new_labels_ = std::move(old_to_new);
    new_to_old_labels_ = std::move(new_to_old);
}


void SPARSE_HNSW::relabelGroundTruth(std::vector<std::vector<uint32_t>>& groundtruth) const {
    if (old_to_new_labels_.empty()) {
        return;
    }

    for (size_t i = 0; i < groundtruth.size(); ++i) {
        std::vector<uint32_t>& labels = groundtruth[i];
        for (size_t j = 0; j < labels.size(); ++j) {
            uint32_t label = labels[j];
            if (label < old_to_new_labels_.size()) {
                labels[j] = old_to_new_labels_[label];
            }
        }
    }
}

void SPARSE_HNSW::printInfo() const {
    std::cout << "\nSPARSE_HNSW Index Info:\n";
    std::cout << "Dimension: " << dim_ << "\n";
    std::cout << "M (max connections per layer): " << M_ << "\n";
    std::cout << "ef_construction: " << ef_construction_ << "\n";
    std::cout << "Max elements: " << max_elements_ << "\n";
    std::cout << "Current number of nodes: " << data_matrix_->nrow << "\n";
    std::cout << "Max level: " << max_level_ << "\n";
    std::cout << "Entry point ID: " << entry_point_ << "\n";

    std::vector<int> layer_counts;
    for (int level_id = 0; level_id <= max_level_; ++level_id) {
        int count = 0;
        size_t num_nodes = static_cast<size_t>(data_matrix_->nrow);
        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            if (!getNeighborsAtLevel(static_cast<uint32_t>(node_id), level_id).empty()) {
                count++;
            }
        }
        layer_counts.push_back(count);
    }
    for (size_t i = 0; i < layer_counts.size(); ++i) {
        std::cout << "Layer " << i << " has " << layer_counts[i] << " nodes\n";
    }
}
