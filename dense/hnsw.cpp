#include "hnsw.h"
#include <iostream>
#include <immintrin.h>
#include <unordered_set>
#include <omp.h>

using namespace hnsw;

HNSW::HNSW(int dim, int M, int ef_construction, int max_elements, 
    bool use_heuristic, bool extend_candidates, bool keep_pruned)
    : dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements), 
      use_heuristic_(use_heuristic), extend_candidates_(extend_candidates), keep_pruned_(keep_pruned),
      max_level_(0), entry_point_(-1), rng_(42), level_generator_(0.0, 1.0) {
    data_.reserve(max_elements * dim);
    neighbors_.resize(max_elements);
}

// L2 Euclidean distance
float HNSW::distance(float * a, float * b) const {
    float dist = 0.0f;
    size_t size = dim_;
    size_t i = 0;
    
    // Process 8 floats at a time with AVX
    #ifdef __AVX__
    __m256 sum_v = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_v = _mm256_fmadd_ps(diff, diff, sum_v);  // fused multiply-add
    }
    // Horizontal sum
    float tmp[8];
    _mm256_storeu_ps(tmp, sum_v);
    for (int j = 0; j < 8; ++j) dist += tmp[j];
    #endif
    
    // Remaining elements
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;

    // float dist = 0.0f;
    // size_t size = a.size();
    
    // #pragma omp simd reduction(+:dist)
    // for (; i < size; ++i) {
    //     float diff = a[i] - b[i];
    //     dist += diff * diff;
    // }
    // return dist;
}

int HNSW::getRandomLevel() {
    double r = level_generator_(rng_);
    // Ensure r is not too close to 0 to avoid log(0).
    r = std::max(r, std::numeric_limits<double>::min());
    return static_cast<int>(-log(r) * (1.0 / log(M_)));
}

std::priority_queue<std::pair<float, uint32_t>> HNSW::searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer) {
    std::vector<bool> visited(max_elements_, false);
    
    MinPQ candidates;
    std::priority_queue<std::pair<float, uint32_t>> top_candidates;
    
    for (uint32_t entry_point : entry_points) {
        float d = distance(query.data(), data_.data() + entry_point * dim_);
        candidates.push({d, entry_point});
        top_candidates.push({d, entry_point});
        visited[entry_point] = true;
    }
    
    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();
        
        // Compare with the farthest in nearest neighbors.
        if (top_candidates.size() >= static_cast<size_t>(ef) && current.first > top_candidates.top().first) {
            break;
        }
        
        uint32_t current_node = current.second;
        // Ideally this check shouldn't fail.
        if (layer < static_cast<int>(neighbors_[current_node].size())) {
            for (uint32_t neighbor_id : neighbors_[current_node][layer]) {
                if (!visited[neighbor_id]) {
                    visited[neighbor_id] = true;
                    float dist = distance(query.data(), data_.data() + neighbor_id * dim_);
                    
                    if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
                        candidates.push({dist, neighbor_id});
                        top_candidates.push({dist, neighbor_id});
                        
                        if (top_candidates.size() > static_cast<size_t>(ef)) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }
    }
    
    return top_candidates;
}

std::vector<uint32_t> HNSW::selectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M) {

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

std::vector<uint32_t> HNSW::selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M, int level) {
    
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
    
    if (extend_candidates_) {
        MinPQ temp = working_set;
        while (!temp.empty()) {
            uint32_t candidate = temp.top().second;
            temp.pop();
            if (static_cast<int>(neighbors_[candidate].size()) > level) {
                for (uint32_t neighbor : neighbors_[candidate][level]) {
                    if (neighbor != node_id) { // Ideally should check in working_set if the neighbor is already there.
                        float dist = distance(data_.data() + node_id * dim_, data_.data() + neighbor * dim_);
                        working_set.push({dist, neighbor});
                    }
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
            float dist = distance(data_.data() + current.second * dim_, data_.data() + result * dim_);
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

std::vector<uint32_t> HNSW::connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level, int M) {    
    
    if (static_cast<int>(neighbors_[node_id].size()) <= level) {
        neighbors_[node_id].resize(level + 1);
    }

    std::vector<uint32_t> selected_neighbors;
    if (use_heuristic_) {
        selected_neighbors = selectNeighborsHeuristic(node_id, candidates, M, level);
    } else {
        selected_neighbors = selectNeighbors(node_id, candidates, M);
    }
    
    // Clear existing neighbors at this level
    neighbors_[node_id][level].clear();

    for (uint32_t neighbor : selected_neighbors) {
        neighbors_[node_id][level].push_back(neighbor);
        
        // Add bidirectional connection
        if (static_cast<int>(neighbors_[neighbor].size()) <= level) {
            neighbors_[neighbor].resize(level + 1);
        }
        if (std::find(neighbors_[neighbor][level].begin(), neighbors_[neighbor][level].end(), node_id) == neighbors_[neighbor][level].end()) {
            neighbors_[neighbor][level].push_back(node_id);
        }
    }
    return neighbors_[node_id][level];
}

void HNSW::addPoint(std::vector<float> point, uint32_t label) {
    int level = getRandomLevel();
    
    neighbors_.emplace_back();
    data_.insert(data_.end(), point.begin(), point.end());
    
    if (entry_point_ == -1) {
        entry_point_ = label;
        max_level_ = level;
        return;
    }
    
    std::vector<uint32_t> entry_points = {entry_point_};
    
    // Search from top layer to target layer
    for (int lc = max_level_; lc > level; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> nearest = searchLayer(point, entry_points, 1, lc);
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
        int M_max = (lc == 0) ? M_ * 2 : M_;
        std::priority_queue<std::pair<float, uint32_t>> candidates = searchLayer(point, entry_points, ef_construction_, lc);
        if (!candidates.empty()) {
            entry_points.clear();
            std::priority_queue<std::pair<float, uint32_t>> temp = candidates;
            while (!temp.empty()) {
                entry_points.push_back(temp.top().second);
                temp.pop();
            }
        }

        auto neighbors = connectNeighbors(label, candidates, lc, M_);
        for (uint32_t neighbor : neighbors) {
            std::vector<uint32_t> econn = neighbors_[neighbor][lc];
            int neighborhood_size = static_cast<int>(econn.size());
            if (neighborhood_size > M_max) {
                std::priority_queue<std::pair<float, uint32_t>> econn_candidates;
                for (uint32_t econn_neighbor : econn) {
                    float dist = distance(data_.data() + neighbor * dim_, data_.data() + econn_neighbor * dim_);
                    econn_candidates.push({dist, econn_neighbor});
                }
                connectNeighbors(neighbor, econn_candidates, lc, M_max);
            }
        }
    }
    
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = label;
    }
}

std::priority_queue<std::pair<float, uint32_t>> HNSW::searchKNN(std::vector<float> query, int k, int ef) {
    if (entry_point_ == -1) {
        return {};
    }

    std::vector<uint32_t> entry_points = {entry_point_};

    // Search from top layer to layer 0
    for (int lc = max_level_; lc > 0; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> nearest = searchLayer(query, entry_points, 1, lc);
        if (!nearest.empty()) {
            entry_points.clear();
            while (!nearest.empty()) {
                entry_points.push_back(nearest.top().second);
                nearest.pop();
            }
        }
    }
    
    // Search at layer 0
    return searchLayer(query, entry_points, std::max(ef, k), 0);
}

void HNSW::printInfo() const {
    std::cout << "\nHNSW Index Info:\n";
    std::cout << "Dimension: " << dim_ << "\n";
    std::cout << "M (max connections per layer): " << M_ << "\n";
    std::cout << "ef_construction: " << ef_construction_ << "\n";
    std::cout << "Max elements: " << max_elements_ << "\n";
    std::cout << "Current number of nodes: " << data_.size() / dim_ << "\n";
    std::cout << "Max level: " << max_level_ << "\n";
    std::cout << "Entry point ID: " << entry_point_ << "\n";

    std::vector<int> layer_counts;
    for (size_t node_id = 0; node_id < neighbors_.size(); ++node_id) {
        int levels = static_cast<int>(neighbors_[node_id].size());
        if (layer_counts.size() < static_cast<size_t>(levels)) {
            layer_counts.resize(levels, 0);
        }
        for (int l = 0; l < levels; ++l) {
            layer_counts[l]++;
        }
    }
    for (size_t i = 0; i < layer_counts.size(); ++i) {
        std::cout << "Layer " << i << " has " << layer_counts[i] << " nodes\n";
    }
}
