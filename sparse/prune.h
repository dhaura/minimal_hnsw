#ifndef SPARSE_PRUNE_H
#define SPARSE_PRUNE_H

#include "csr_matrix.h"
#include <algorithm>
#include <vector>

namespace sparse_hnsw {

// Mass-ratio pruning (MRP): per row, keep the fewest largest-magnitude entries
// whose running sum first reaches alpha * row_weight, then restore index order
// so the row stays a valid sorted CSR row.
inline CSRMatrix* pruneMatrixWithAlpha(const CSRMatrix* m, float alpha) {
    CSRMatrix* pruned = new CSRMatrix(*m);

    int64_t write = 0;
    std::vector<IndiceDataPair> buf;
    for (int64_t row = 0; row < pruned->nrow; ++row) {
        const int64_t s = pruned->indptr[row];
        const int64_t e = pruned->indptr[row + 1];
        pruned->indptr[row] = write;

        float weight = 0;
        for (int64_t i = s; i < e; ++i) weight += static_cast<float>(pruned->indices_data[i].data);

        buf.assign(pruned->indices_data + s, pruned->indices_data + e);
        std::sort(buf.begin(), buf.end(),
                  [](const IndiceDataPair& a, const IndiceDataPair& b) {
                      return static_cast<float>(a.data) > static_cast<float>(b.data);
                  });

        float prefix_sum = 0; size_t kept = 0;
        while (kept < buf.size()) {
            prefix_sum += static_cast<float>(buf[kept].data);
            ++kept;
            if (prefix_sum >= alpha * weight) break;
        }

        std::sort(buf.begin(), buf.begin() + kept);
        for (size_t i = 0; i < kept; ++i) pruned->indices_data[write++] = buf[i];
    }
    pruned->indptr[pruned->nrow] = write;
    pruned->nnz = write;
    return pruned;
}

} // namespace sparse_hnsw

#endif // SPARSE_PRUNE_H
