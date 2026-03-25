#include "hnsw.h"
#include <iostream>
#include <immintrin.h>

using namespace hnsw;

HNSW::HNSW(int dim, int M, int ef_construction, int max_elements, 
    bool use_heuristic, bool extend_candidates, bool keep_pruned)
    : dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements), 
      use_heuristic_(use_heuristic), extend_candidates_(extend_candidates), keep_pruned_(keep_pruned),
      max_level_(0), entry_point_(-1), rng_(42), level_generator_(0.0, 1.0) {
    nodes_.reserve(max_elements);
}

// L2 Euclidean distance
float HNSW::distance(const std::vector<float>& a, const std::vector<float>& b) const {
    float dist = 0.0f;
    size_t size = a.size();
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
}

int HNSW::getRandomLevel() {
    double r = level_generator_(rng_);
    // Ensure r is not too close to 0 to avoid log(0).
    r = std::max(r, std::numeric_limits<double>::min());
    return static_cast<int>(-log(r) * (1.0 / log(M_)));
}

MinPQ HNSW::searchLayer(const std::vector<float>& query, const std::vector<int>& entry_points, int ef, int layer) {
    std::unordered_set<int> visited;
    
    MinPQ candidates;
    std::priority_queue<std::pair<float, int>> top_candidates;
    
    for (int entry_point : entry_points) {
        float d = distance(query, nodes_[entry_point].data);
        candidates.push({d, entry_point});
        top_candidates.push({d, entry_point});
        visited.insert(entry_point);
    }
    
    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();
        
        // Compare with the farthest in nearest neighbors.
        if (top_candidates.size() >= static_cast<size_t>(ef) && current.first > top_candidates.top().first) {
            break;
        }
        
        int current_node = current.second;
        // Ideally this check shouldn't fail.
        if (layer < static_cast<int>(nodes_[current_node].neighbors.size())) {
            for (int neighbor_id : nodes_[current_node].neighbors[layer]) {
                if (visited.find(neighbor_id) == visited.end()) {
                    visited.insert(neighbor_id);
                    float dist = distance(query, nodes_[neighbor_id].data);
                    
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
    
    MinPQ result;
    while (!top_candidates.empty()) {
        result.push(top_candidates.top());
        top_candidates.pop();
    }
    return result;
}

std::vector<int> HNSW::selectNeighbors(int node_id, const MinPQ& candidates, int M) {

    if (candidates.size() <= static_cast<size_t>(M)) {
        MinPQ temp = candidates;
        std::vector<int> result;
        while (!temp.empty()) {
            result.push_back(temp.top().second);
            temp.pop();
        }
        return result;
    }

    std::vector<int> selected_candidates;
    MinPQ temp = candidates;
    while (!temp.empty()) {
        int candidate = temp.top().second;
        temp.pop();
        if (static_cast<int>(selected_candidates.size()) >= M) {
            break;
        }
        selected_candidates.push_back(candidate);
    }
    return selected_candidates;
}

std::vector<int> HNSW::selectNeighborsHeuristic(int node_id, const MinPQ& candidates, int M, int level) {
    
   if (candidates.size() <= static_cast<size_t>(M)) {
        MinPQ temp = candidates;
        std::vector<int> result;
        while (!temp.empty()) {
            result.push_back(temp.top().second);
            temp.pop();
        }
        return result;
    }

    MinPQ working_set = candidates;
    std::vector<int> results_set;
    
    if (extend_candidates_) {
        MinPQ temp = candidates;
        while (!temp.empty()) {
            int candidate = temp.top().second;
            temp.pop();
            if (static_cast<int>(nodes_[candidate].neighbors.size()) > level) {
                for (int neighbor : nodes_[candidate].neighbors[level]) {
                    if (neighbor != node_id) { // Ideally should check in working_set if the neighbor is already there.
                        float dist = distance(nodes_[node_id].data, nodes_[neighbor].data);
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
            float dist = distance(nodes_[current.second].data, nodes_[result].data);
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

std::vector<int> HNSW::connectNeighbors(int node_id, const MinPQ& candidates, int level, int M) {    
    
    if (static_cast<int>(nodes_[node_id].neighbors.size()) <= level) {
        nodes_[node_id].neighbors.resize(level + 1);
    }

    std::vector<int> selected_neighbors;
    if (use_heuristic_) {
        selected_neighbors = selectNeighborsHeuristic(node_id, candidates, M, level);
    } else {
        selected_neighbors = selectNeighbors(node_id, candidates, M);
    }
    
    // Clear existing neighbors at this level
    nodes_[node_id].neighbors[level].clear();

    for (int neighbor : selected_neighbors) {
        nodes_[node_id].neighbors[level].push_back(neighbor);
        
        // Add bidirectional connection
        if (static_cast<int>(nodes_[neighbor].neighbors.size()) <= level) {
            nodes_[neighbor].neighbors.resize(level + 1);
        }
        if (std::find(nodes_[neighbor].neighbors[level].begin(), nodes_[neighbor].neighbors[level].end(), node_id) == nodes_[neighbor].neighbors[level].end()) {
            nodes_[neighbor].neighbors[level].push_back(node_id);
        }
    }
    return nodes_[node_id].neighbors[level];
}

void HNSW::addPoint(const std::vector<float>& point, int label) {
    Node new_node;
    new_node.label = label;
    new_node.data = point;
    
    int node_id = nodes_.size();
    int level = getRandomLevel();
    
    new_node.neighbors.resize(level + 1);
    nodes_.push_back(new_node);
    
    if (entry_point_ == -1) {
        entry_point_ = node_id;
        max_level_ = level;
        return;
    }
    
    std::vector<int> entry_points = {entry_point_};
    
    // Search from top layer to target layer
    for (int lc = max_level_; lc > level; --lc) {
        MinPQ nearest = searchLayer(point, entry_points, 1, lc);
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
        MinPQ candidates = searchLayer(point, entry_points, ef_construction_, lc);
        auto neighbors = connectNeighbors(node_id, candidates, lc, M_);
        for (int neighbor : neighbors) {
            std::vector<int> econn = nodes_[neighbor].neighbors[lc];
            int neighborhood_size = static_cast<int>(econn.size());
            if (neighborhood_size > M_max) {
                MinPQ econn_candidates;
                for (int econn_neighbor : econn) {
                    float dist = distance(nodes_[neighbor].data, nodes_[econn_neighbor].data);
                    econn_candidates.push({dist, econn_neighbor});
                }
                connectNeighbors(neighbor, econn_candidates, lc, M_max);
            }
        }
        if (!candidates.empty()) {
            entry_points.clear();
            while (!candidates.empty()) {
                entry_points.push_back(candidates.top().second);
                candidates.pop();
            }
        }
    }
    
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = node_id;
    }
}

MinPQ HNSW::searchKNN(const std::vector<float>& query, int k, int ef) {
    if (entry_point_ == -1) {
        return {};
    }
    
    std::vector<int> entry_points = {entry_point_};
    
    // Search from top layer to layer 0
    for (int lc = max_level_; lc > 0; --lc) {
        MinPQ nearest = searchLayer(query, entry_points, 1, lc);
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
    std::cout << "Current number of nodes: " << nodes_.size() << "\n";
    std::cout << "Max level: " << max_level_ << "\n";
    std::cout << "Entry point ID: " << entry_point_ << "\n";

    std::vector<int> layer_counts;
    for (const auto& node : nodes_) {
        int levels = static_cast<int>(node.neighbors.size());
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
