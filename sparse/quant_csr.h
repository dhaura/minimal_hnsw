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

    // Bytes row r of m occupies (header + 3 B per entry, padded to 4).
    static int64_t rowSize(const CSRMatrix &m, int64_t r)
    {
      const int64_t sz = static_cast<int64_t>(kHeader) + 3 * (m.indptr[r + 1] - m.indptr[r]);
      return (sz + 3) & ~static_cast<int64_t>(3);
    }

    // Quantize row r of m into p; returns the number of negative entries.
    static int64_t writeRow(const CSRMatrix &m, int64_t r, uint8_t *p);
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
      if (n >= 65535) // 65535 marks an overflowed QuantSlots slot
      {
        std::cerr << "QuantCSR: row " << r << " has " << n
                  << " entries, which does not fit the u16 row header (max 65534)."
                  << std::endl;
        std::exit(1);
      }
      row_off[r + 1] = rowSize(m, r);
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
      neg_entries += writeRow(m, r, blob.data() + row_off[r]);
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

  inline int64_t QuantCSR::writeRow(const CSRMatrix &m, int64_t r, uint8_t *p)
  {
    const int64_t s = m.indptr[r];
    const int64_t e = m.indptr[r + 1];
    const uint32_t n = static_cast<uint32_t>(e - s);
    int64_t neg = 0;
    float rmax = 0.0f;
    for (int64_t i = s; i < e; ++i)
    {
      const float v = static_cast<float>(m.indices_data[i].data);
      neg += v < 0.0f;
      rmax = std::max(rmax, v);
    }
    const _Float16 scale16 = static_cast<_Float16>(rmax > 0.0f ? rmax / 255.0f : 0.0f);
    const float scale = static_cast<float>(scale16);
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
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
    return neg;
  }

  // Fixed-size quantized rows: row r starts at base + r * slot_size, so a
  // distance call needs no row_off lookup. Slots are 64-byte aligned. A row
  // longer than a slot stores kOverflow in its count and, at byte 8, its
  // offset into this struct's own overflow blob (one extra, dependent read).
  struct QuantSlots
  {
    static constexpr uint16_t kOverflow = 0xFFFF;
    uint32_t slot_size{0};
    int64_t overflow_rows{0};
    std::vector<uint8_t> storage;
    std::vector<uint8_t> overflow;
    uint8_t *base{nullptr};

    bool empty() const { return base == nullptr; }
    size_t bytes() const { return storage.size() + overflow.size(); }

    // Quantize m straight into slots (no QuantCSR copy is ever allocated).
    void build(const CSRMatrix &m, uint32_t size_bytes)
    {
      if (size_bytes < 64 || size_bytes % 64 != 0)
      {
        std::cerr << "QuantSlots: slot_size must be a positive multiple of 64." << std::endl;
        std::exit(1);
      }
      slot_size = size_bytes;
      std::vector<int64_t> over_off(static_cast<size_t>(m.nrow), -1);
      int64_t total = 0;
      for (int64_t r = 0; r < m.nrow; ++r)
      {
        if (m.indptr[r + 1] - m.indptr[r] >= 65535)
        {
          std::cerr << "QuantSlots: row " << r << " exceeds the u16 row header." << std::endl;
          std::exit(1);
        }
        const int64_t sz = QuantCSR::rowSize(m, r);
        if (sz > static_cast<int64_t>(slot_size))
        {
          over_off[r] = total;
          total += sz;
        }
      }
      storage.assign(static_cast<size_t>(m.nrow) * slot_size + 64, 0);
      overflow.assign(static_cast<size_t>(total), 0);
      const uintptr_t raw = reinterpret_cast<uintptr_t>(storage.data());
      base = storage.data() + ((64 - raw % 64) % 64);
      int64_t over = 0, neg = 0;
#pragma omp parallel for schedule(static) reduction(+ : over, neg)
      for (int64_t r = 0; r < m.nrow; ++r)
      {
        uint8_t *slot = base + static_cast<size_t>(r) * slot_size;
        if (over_off[r] < 0)
        {
          neg += QuantCSR::writeRow(m, r, slot);
        }
        else
        {
          neg += QuantCSR::writeRow(m, r, overflow.data() + over_off[r]);
          std::memcpy(slot, &kOverflow, sizeof(kOverflow));
          std::memcpy(slot + 8, &over_off[r], sizeof(int64_t));
          over++;
        }
      }
      overflow_rows = over;
      if (neg > 0)
      {
        std::cerr << "QuantSlots: matrix has negative values; u8 layout cannot hold them." << std::endl;
        std::exit(1);
      }
    }

    // Bytes the row at p occupies (header + 3 B per entry, padded to 4).
    static uint32_t rowBytes(const uint8_t *p)
    {
      uint16_t n;
      std::memcpy(&n, p, sizeof(n));
      return (QuantCSR::kHeader + 3u * n + 3u) & ~3u;
    }

    const uint8_t *row(uint32_t r) const
    {
      const uint8_t *s = base + static_cast<size_t>(r) * slot_size;
      uint16_t n;
      std::memcpy(&n, s, sizeof(n));
      if (n != kOverflow)
      {
        return s;
      }
      int64_t off;
      std::memcpy(&off, s + 8, sizeof(off));
      return overflow.data() + off;
    }
  };

} // namespace sparse_hnsw

#endif // QUANT_CSR_H_
