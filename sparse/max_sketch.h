#ifndef SPARSE_MAX_SKETCH_H
#define SPARSE_MAX_SKETCH_H

#include "csr_matrix.h"
#include "quant_csr.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace sparse_hnsw {

// One cache line per document bounding its quantized traversal score from above.
// Dimensions hash into kBuckets buckets; a row keeps the largest u8 code in each
// bucket plus the row's fp16 scale. For q >= 0 nothing is lost:
//   scale * sum_i code_i * q_i  <=  scale * sum_b max_b * qsum_b,
// qsum_b = sum of the positive query weights in bucket b.
class MaxSketch {
public:
    static constexpr uint32_t kBuckets = 62;  // 62 codes + 2-byte scale = 64 B

    struct alignas(64) Row {
        uint8_t max[kBuckets];
        _Float16 scale;
    };
    static_assert(sizeof(Row) == 64, "Sketch rows must occupy one cache line");

    static uint32_t bucket(uint32_t dim) {
        return static_cast<uint32_t>((uint64_t{dim * 2654435761u} * kBuckets) >> 32);
    }

    struct Query {
        alignas(64) std::array<float, 64> qsum{};  // [62], [63] stay 0: they meet the scale bytes

        void build(const IndiceDataPair* entries, uint32_t count) {
            qsum.fill(0.0f);
            for (uint32_t i = 0; i < count; ++i) {
                const float v = static_cast<float>(entries[i].data);
                if (v > 0.0f) { qsum[bucket(entries[i].indice)] += v; }
            }
        }
    };

    void build(const QuantCSR& quant) {
        rows_.assign(static_cast<size_t>(quant.nrow), Row{});
        #pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < quant.nrow; ++r) {
            const uint8_t* p = quant.blob.data() + quant.row_off[r];
            uint16_t n;
            std::memcpy(&n, p, sizeof(n));
            const uint16_t* idx = reinterpret_cast<const uint16_t*>(p + QuantCSR::kHeader);
            const uint8_t* code = p + QuantCSR::kHeader + 2 * static_cast<size_t>(n);
            Row& row = rows_[r];
            std::memcpy(&row.scale, p + 2, sizeof(row.scale));
            for (uint32_t i = 0; i < n; ++i) {
                uint8_t& m = row.max[bucket(idx[i])];
                m = std::max(m, code[i]);
            }
        }
    }

    // Upper bound on scale * sum_i code_i * q_i for this document.
    float upperBound(size_t document, const Query& query) const {
        const Row& row = rows_[document];
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&row);
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (uint32_t b = 0; b < 64; ++b) {
            sum += static_cast<float>(bytes[b]) * query.qsum[b];
        }
        return sum * static_cast<float>(row.scale);
    }

    const Row* row(size_t document) const { return &rows_[document]; }
    size_t bytes() const { return rows_.size() * sizeof(Row); }
    bool empty() const { return rows_.empty(); }
    void clear() { rows_.clear(); }

private:
    std::vector<Row> rows_;
};

}  // namespace sparse_hnsw

#endif  // SPARSE_MAX_SKETCH_H
