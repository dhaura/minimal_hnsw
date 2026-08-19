#pragma once
#ifndef INVERTED_SEED_H_
#define INVERTED_SEED_H_

#include "csr_matrix.h"

#include <cstdint>
#include <vector>

namespace sparse_hnsw
{

  // Per-column list of the top-T documents by stored value: the layer-0 entry
  // points a query's heaviest terms suggest.

  struct InvertedSeedTable
  {
    // An unscoped enum rather than `static constexpr uint32_t`: this struct is
    // header-only, and an ODR-used constexpr member would need an out-of-line
    // definition to link.
    enum : uint32_t { kNone = 0xFFFFFFFFu };

    std::vector<uint32_t> top_doc_ids;
    uint32_t top_k{0};
    int64_t ncol{0};

    bool empty() const { return top_doc_ids.empty(); }
    size_t bytes() const { return top_doc_ids.size() * sizeof(uint32_t); }

    const uint32_t *column(uint32_t c) const
    {
      return top_doc_ids.data() + static_cast<size_t>(c) * top_k;
    }

    void build(const CSRMatrix &m, uint32_t t);
  };

  inline void InvertedSeedTable::build(const CSRMatrix &m, uint32_t t)
  {
    top_k = t ? t : 1;
    ncol = m.ncol;
    top_doc_ids.assign(static_cast<size_t>(ncol) * top_k, kNone);

    std::vector<float> top_values(static_cast<size_t>(ncol) * top_k, -1.0f);

    for (int64_t r = 0; r < m.nrow; ++r)
    {
      for (int64_t i = m.indptr[r]; i < m.indptr[r + 1]; ++i)
      {
        const uint32_t c = m.indices_data[i].indice;
        const float v = static_cast<float>(m.indices_data[i].data);
        float *col_values = top_values.data() + static_cast<size_t>(c) * top_k;
        uint32_t *col_doc_ids = top_doc_ids.data() + static_cast<size_t>(c) * top_k;
        for (uint32_t s = 0; s < top_k; ++s)
        {
          if (v > col_values[s])
          {
            for (uint32_t u = top_k - 1; u > s; --u)
            {
              col_values[u] = col_values[u - 1];
              col_doc_ids[u] = col_doc_ids[u - 1];
            }
            col_values[s] = v;
            col_doc_ids[s] = static_cast<uint32_t>(r);
            break;
          }
        }
      }
    }
  }

} // namespace sparse_hnsw

#endif // INVERTED_SEED_H_
