#pragma once
#ifndef QUANT_CSR_H_
#define QUANT_CSR_H_

#include "csr_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace sparse_hnsw
{
  struct QuantCSR
  {
    std::vector<uint8_t> blob;
    std::vector<int64_t> row_off; // nrow+1 byte offsets into blob
    int64_t nrow{0}, ncol{0}, nnz{0};

    static constexpr uint32_t kHeader = 4; // u16 count + fp16 scale

    bool empty() const { return blob.empty(); }

    size_t bytes() const
    {
      return blob.size() + row_off.size() * sizeof(int64_t);
    }

    // Bytes a distance call on this row streams (header + payload + padding).
    uint32_t rowBytes(uint32_t r) const
    {
      return static_cast<uint32_t>(row_off[r + 1] - row_off[r]);
    }

    void build(const CSRMatrix &m);
  };

  inline void QuantCSR::build(const CSRMatrix &m)
  {
    nrow = m.nrow;
    ncol = m.ncol;
    nnz = m.nnz;

    // Pass 1: per-row byte size, padded to 4 so the NEXT row's u16 index
    // array is still 2-byte aligned.
    row_off.assign(static_cast<size_t>(nrow) + 1, 0);
    for (int64_t r = 0; r < nrow; ++r)
    {
      const int64_t n = m.indptr[r + 1] - m.indptr[r];
      if (n > 65535)
      {
        std::cerr << "QuantCSR: row " << r << " has " << n
                  << " entries, which does not fit the u16 row header."
                  << std::endl;
        std::exit(1);
      }
      const int64_t sz = static_cast<int64_t>(kHeader) + 3 * n;
      row_off[r + 1] = (sz + 3) & ~static_cast<int64_t>(3);
    }
    for (int64_t r = 0; r < nrow; ++r)
    {
      row_off[r + 1] += row_off[r];
    }
    blob.assign(static_cast<size_t>(row_off[nrow]), 0);

    // Pass 2: quantize.
    int64_t neg_entries = 0;
#pragma omp parallel for schedule(static) reduction(+ : neg_entries)
    for (int64_t r = 0; r < nrow; ++r)
    {
      const int64_t s = m.indptr[r];
      const int64_t e = m.indptr[r + 1];
      const uint32_t n = static_cast<uint32_t>(e - s);

      float rmax = 0.0f;
      for (int64_t i = s; i < e; ++i)
      {
        const float v = static_cast<float>(m.indices_data[i].data);
        if (v < 0.0f)
        {
          neg_entries++;
        }
        rmax = std::max(rmax, v);
      }

      const _Float16 scale16 = static_cast<_Float16>(rmax > 0.0f ? rmax / 255.0f : 0.0f);
      const float scale = static_cast<float>(scale16);
      const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;

      uint8_t *p = blob.data() + row_off[r];
      const uint16_t n16 = static_cast<uint16_t>(n);
      std::memcpy(p, &n16, sizeof(n16));
      std::memcpy(p + 2, &scale16, sizeof(scale16));

      uint16_t *idx = reinterpret_cast<uint16_t *>(p + kHeader);
      uint8_t *code = p + kHeader + 2 * static_cast<size_t>(n);
      for (uint32_t i = 0; i < n; ++i)
      {
        const IndiceDataPair &src = m.indices_data[s + i];
        idx[i] = src.indice;
        const int q = static_cast<int>(static_cast<float>(src.data) * inv + 0.5f);
        code[i] = static_cast<uint8_t>(std::min(255, std::max(0, q)));
      }
    }

    if (neg_entries > 0)
    {
      std::cerr << "QuantCSR: matrix has " << neg_entries
                << " negative values; the unsigned u8 layout cannot represent "
                   "them. Use the fp16 path for this dataset."
                << std::endl;
      std::exit(1);
    }
  }

} // namespace sparse_hnsw

#endif // QUANT_CSR_H_
