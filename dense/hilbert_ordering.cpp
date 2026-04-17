#include "hilbert_ordering.h"

#include <iostream>
#include <limits>

namespace hnsw {
namespace {

struct PcaProjectionModel {
    bool ready;
    int dim;
    std::vector<float> mean;
    std::vector<float> component_1;
    std::vector<float> component_2;
};

bool normalizeVector(std::vector<float>& vector) {
    double norm_sq = 0.0;
    for (float value : vector) {
        norm_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    if (norm_sq <= std::numeric_limits<double>::epsilon()) {
        return false;
    }
    const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm_sq));
    for (float& value : vector) {
        value *= inv_norm;
    }
    return true;
}

PcaProjectionModel fitPcaProjection(int dim, const std::vector<float>& data) {
    PcaProjectionModel model;
    model.ready = false;
    model.dim = dim;
    model.mean.assign(static_cast<size_t>(dim), 0.0f);
    model.component_1.assign(static_cast<size_t>(dim), 0.0f);
    model.component_2.assign(static_cast<size_t>(dim), 0.0f);

    const size_t num_nodes = (dim > 0) ? (data.size() / static_cast<size_t>(dim)) : 0;
    if (num_nodes == 0 || dim <= 0) {
        return model;
    }

    if (dim == 1) {
        model.component_1[0] = 1.0f;
        model.ready = true;
        return model;
    }

    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        const float* point = data.data() + node_id * static_cast<size_t>(dim);
        for (int dim_id = 0; dim_id < dim; ++dim_id) {
            model.mean[static_cast<size_t>(dim_id)] += point[dim_id];
        }
    }
    const float inv_num_nodes = 1.0f / static_cast<float>(num_nodes);
    for (int dim_id = 0; dim_id < dim; ++dim_id) {
        model.mean[static_cast<size_t>(dim_id)] *= inv_num_nodes;
    }

    std::vector<double> covariance(static_cast<size_t>(dim) * static_cast<size_t>(dim), 0.0);
    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        const float* point = data.data() + node_id * static_cast<size_t>(dim);
        for (int row = 0; row < dim; ++row) {
            const double centered_row = static_cast<double>(point[row]) - static_cast<double>(model.mean[static_cast<size_t>(row)]);
            for (int col = row; col < dim; ++col) {
                const double centered_col = static_cast<double>(point[col]) - static_cast<double>(model.mean[static_cast<size_t>(col)]);
                covariance[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col)] += centered_row * centered_col;
            }
        }
    }

    const double inv_denom = (num_nodes > 1) ? (1.0 / static_cast<double>(num_nodes - 1)) : 1.0;
    for (int row = 0; row < dim; ++row) {
        for (int col = row; col < dim; ++col) {
            const double value = covariance[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col)] * inv_denom;
            covariance[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col)] = value;
            covariance[static_cast<size_t>(col) * static_cast<size_t>(dim) + static_cast<size_t>(row)] = value;
        }
    }

    auto multiplyCovariance = [&](const std::vector<float>& vector) {
        std::vector<double> result(static_cast<size_t>(dim), 0.0);
        for (int row = 0; row < dim; ++row) {
            double sum = 0.0;
            for (int col = 0; col < dim; ++col) {
                sum += covariance[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col)] * static_cast<double>(vector[static_cast<size_t>(col)]);
            }
            result[static_cast<size_t>(row)] = sum;
        }
        return result;
    };

    auto computeComponent = [&](const std::vector<std::vector<float>>& basis) {
        std::vector<float> vector(static_cast<size_t>(dim), 0.0f);
        vector[0] = 1.0f;
        if (!basis.empty()) {
            for (int dim_id = 0; dim_id < dim; ++dim_id) {
                vector[static_cast<size_t>(dim_id)] = static_cast<float>((dim_id == 0) ? 0.5f : 0.0f);
            }
        }

        if (!normalizeVector(vector)) {
            vector.assign(static_cast<size_t>(dim), 0.0f);
            vector[0] = 1.0f;
        }

        for (int iteration = 0; iteration < 64; ++iteration) {
            std::vector<double> next_double = multiplyCovariance(vector);
            for (const std::vector<float>& prior : basis) {
                double dot = 0.0;
                for (int dim_id = 0; dim_id < dim; ++dim_id) {
                    dot += next_double[static_cast<size_t>(dim_id)] * static_cast<double>(prior[static_cast<size_t>(dim_id)]);
                }
                for (int dim_id = 0; dim_id < dim; ++dim_id) {
                    next_double[static_cast<size_t>(dim_id)] -= dot * static_cast<double>(prior[static_cast<size_t>(dim_id)]);
                }
            }

            std::vector<float> next(static_cast<size_t>(dim), 0.0f);
            for (int dim_id = 0; dim_id < dim; ++dim_id) {
                next[static_cast<size_t>(dim_id)] = static_cast<float>(next_double[static_cast<size_t>(dim_id)]);
            }

            if (!normalizeVector(next)) {
                break;
            }

            double diff_sq = 0.0;
            for (int dim_id = 0; dim_id < dim; ++dim_id) {
                const double delta = static_cast<double>(next[static_cast<size_t>(dim_id)]) - static_cast<double>(vector[static_cast<size_t>(dim_id)]);
                diff_sq += delta * delta;
            }
            vector = std::move(next);
            if (diff_sq <= 1e-10) {
                break;
            }
        }

        return vector;
    };

    model.component_1 = computeComponent({});
    if (!normalizeVector(model.component_1)) {
        model.component_1.assign(static_cast<size_t>(dim), 0.0f);
        model.component_1[0] = 1.0f;
    }

    std::vector<std::vector<float>> basis;
    basis.push_back(model.component_1);
    model.component_2 = computeComponent(basis);
    if (!normalizeVector(model.component_2)) {
        model.component_2.assign(static_cast<size_t>(dim), 0.0f);
        model.component_2[1] = 1.0f;
    }

    double dot = 0.0;
    for (int dim_id = 0; dim_id < dim; ++dim_id) {
        dot += static_cast<double>(model.component_1[static_cast<size_t>(dim_id)]) * static_cast<double>(model.component_2[static_cast<size_t>(dim_id)]);
    }
    for (int dim_id = 0; dim_id < dim; ++dim_id) {
        model.component_2[static_cast<size_t>(dim_id)] -= static_cast<float>(dot * static_cast<double>(model.component_1[static_cast<size_t>(dim_id)]));
    }
    if (!normalizeVector(model.component_2)) {
        model.component_2.assign(static_cast<size_t>(dim), 0.0f);
        model.component_2[1] = 1.0f;
    }

    model.ready = true;
    return model;
}

std::pair<float, float> projectTo2D(const std::vector<float>& point, const PcaProjectionModel& model) {
    if (!model.ready || static_cast<int>(model.mean.size()) != model.dim ||
        static_cast<int>(model.component_1.size()) != model.dim || static_cast<int>(model.component_2.size()) != model.dim) {
        const float x = (point.size() > 0) ? point[0] : 0.0f;
        const float y = (point.size() > 1) ? point[1] : 0.0f;
        return std::make_pair(x, y);
    }

    double x = 0.0;
    double y = 0.0;
    for (int dim_id = 0; dim_id < model.dim; ++dim_id) {
        const double centered = static_cast<double>(point[static_cast<size_t>(dim_id)]) - static_cast<double>(model.mean[static_cast<size_t>(dim_id)]);
        x += centered * static_cast<double>(model.component_1[static_cast<size_t>(dim_id)]);
        y += centered * static_cast<double>(model.component_2[static_cast<size_t>(dim_id)]);
    }
    return std::make_pair(static_cast<float>(x), static_cast<float>(y));
}

uint64_t xy2d(uint32_t n, uint32_t x, uint32_t y) {
    uint64_t rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

uint64_t computeHilbertIndex(const std::vector<float>& point,
                             const PcaProjectionModel& model,
                             float min_x, float max_x,
                             float min_y, float max_y) {
    std::pair<float, float> proj = projectTo2D(point, model);
    float x = proj.first;
    float y = proj.second;

    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    if (range_x < 1e-6f) range_x = 1.0f;
    if (range_y < 1e-6f) range_y = 1.0f;

    float norm_x = (x - min_x) / range_x;
    float norm_y = (y - min_y) / range_y;

    norm_x = std::max(0.0f, std::min(1.0f, norm_x));
    norm_y = std::max(0.0f, std::min(1.0f, norm_y));

    const uint32_t grid_size = 65536;
    uint32_t grid_x = static_cast<uint32_t>(norm_x * (grid_size - 1));
    uint32_t grid_y = static_cast<uint32_t>(norm_y * (grid_size - 1));

    return xy2d(grid_size, grid_x, grid_y);
}

} // namespace

void HilbertOrdering::reorder(HNSW& index) {
    size_t num_nodes = index.data_.size() / index.dim_;
    if (num_nodes <= 1) {
        return;
    }

    const PcaProjectionModel pca_model = fitPcaProjection(index.dim_, index.data_);

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < num_nodes; ++i) {
        std::vector<float> point(index.data_.begin() + i * index.dim_,
                                 index.data_.begin() + (i + 1) * index.dim_);
        std::pair<float, float> proj = projectTo2D(point, pca_model);
        float x = proj.first;
        float y = proj.second;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    std::vector<std::pair<uint64_t, uint32_t>> hilbert_indices;
    hilbert_indices.reserve(num_nodes);

    for (size_t i = 0; i < num_nodes; ++i) {
        std::vector<float> point(index.data_.begin() + i * index.dim_,
                                 index.data_.begin() + (i + 1) * index.dim_);
        uint64_t h_idx = computeHilbertIndex(point, pca_model, min_x, max_x, min_y, max_y);
        hilbert_indices.push_back(std::make_pair(h_idx, static_cast<uint32_t>(i)));
    }

    std::sort(hilbert_indices.begin(), hilbert_indices.end());

    std::vector<uint32_t> old_to_new(num_nodes);
    std::vector<uint32_t> new_to_old(num_nodes);
    for (uint32_t new_id = 0; new_id < num_nodes; ++new_id) {
        uint32_t old_id = hilbert_indices[new_id].second;
        old_to_new[old_id] = new_id;
        new_to_old[new_id] = old_id;
    }
    index.old_to_new_labels_ = old_to_new;
    index.new_to_old_labels_ = new_to_old;

    std::vector<float> new_data;
    new_data.reserve(index.data_.size());
    for (uint32_t new_id = 0; new_id < num_nodes; ++new_id) {
        uint32_t old_id = new_to_old[new_id];
        new_data.insert(new_data.end(),
                        index.data_.begin() + old_id * index.dim_,
                        index.data_.begin() + (old_id + 1) * index.dim_);
    }
    index.data_ = new_data;

    std::vector<int> new_element_levels(num_nodes);
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t new_id = old_to_new[old_id];
        new_element_levels[new_id] = index.element_levels_[old_id];
    }
    index.element_levels_ = new_element_levels;

    std::vector<uint32_t> new_neighbor_list_offsets(num_nodes, std::numeric_limits<uint32_t>::max());
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t new_id = old_to_new[old_id];
        new_neighbor_list_offsets[new_id] = index.neighbor_list_offsets_[old_id];
    }
    index.neighbor_list_offsets_ = new_neighbor_list_offsets;

    std::vector<uint32_t> new_level0_neighbor_lists(index.level0_neighbor_lists_.size(), 0);
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t new_id = old_to_new[old_id];

        const uint32_t* old_list = index.level0_neighbor_lists_.data() + old_id * index.size_neighbor_list_level0_;
        uint32_t* new_list = new_level0_neighbor_lists.data() + new_id * index.size_neighbor_list_level0_;

        uint32_t neighbor_count = old_list[0];
        new_list[0] = neighbor_count;

        for (uint32_t i = 0; i < neighbor_count; ++i) {
            uint32_t old_neighbor = old_list[1 + i];
            uint32_t new_neighbor = old_to_new[old_neighbor];
            new_list[1 + i] = new_neighbor;
        }
    }
    index.level0_neighbor_lists_ = new_level0_neighbor_lists;

    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t offset = index.neighbor_list_offsets_[old_to_new[old_id]];
        if (offset != std::numeric_limits<uint32_t>::max()) {
            int level = index.element_levels_[old_to_new[old_id]];
            for (int lv = 1; lv <= level; ++lv) {
                uint32_t* neighbor_list = index.neighbor_lists_.data() + offset + (lv - 1) * index.size_neighbor_list_per_element_;
                uint32_t neighbor_count = neighbor_list[0];

                for (uint32_t i = 0; i < neighbor_count; ++i) {
                    uint32_t old_neighbor = neighbor_list[1 + i];
                    uint32_t new_neighbor = old_to_new[old_neighbor];
                    neighbor_list[1 + i] = new_neighbor;
                }
            }
        }
    }

    index.entry_point_ = old_to_new[index.entry_point_];

    std::cout << "Hilbert curve optimization applied: " << num_nodes << " points reordered\n";
}

} // namespace hnsw
