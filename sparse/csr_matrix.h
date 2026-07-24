
#pragma once
#ifndef MATRIX_HPP_
#define MATRIX_HPP_
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <assert.h>

// Compact 4-byte entry: column indices fit uint16_t as long as ncol <= 65536
// (checked at load time), and values are stored as IEEE fp16 (the F16C unit
// converts to fp32 on use). Halves the bytes streamed per distance call.
struct IndiceDataPair
{
  uint16_t indice;
  _Float16 data;
  bool operator>(const IndiceDataPair &s2)
  {
    return this->indice > s2.indice;
  }
  bool operator<(const IndiceDataPair &s2)
  {
    return this->indice < s2.indice;
  }

  bool operator==(const IndiceDataPair &s2)
  {
    return this->indice == s2.indice;
  }
};

class CSRMatrix
{
private:
public:
  int64_t nrow{0};
  int64_t ncol{0};
  int64_t nnz{0};
  int64_t global_nrow{0};
  int64_t global_nnz{0};
  int64_t *indptr{nullptr};

  IndiceDataPair *indices_data{nullptr};

  CSRMatrix(const std::string data_file_path, bool ori = true)
  {

    std::ifstream infile(data_file_path, std::ios::binary);

    if (infile.fail())
    {
      std::cerr << std::string("Failed to open file ") + data_file_path;
      exit(1);
    }

    // Detect a gzip-compressed file (magic bytes 0x1f 0x8b). Reading a .gz as a
    // raw CSR file decodes the header into garbage and blows up new[].
    unsigned char magic[2] = {0, 0};
    infile.read((char *)magic, sizeof(magic));
    if (magic[0] == 0x1f && magic[1] == 0x8b)
    {
      std::cerr << "File '" << data_file_path
                << "' looks gzip-compressed; decompress it first (e.g. gunzip -k "
                << data_file_path << ")." << std::endl;
      exit(1);
    }
    infile.seekg(0, std::ios::beg);

    infile.read((char *)&nrow, sizeof(int64_t));
    infile.read((char *)&ncol, sizeof(int64_t));
    infile.read((char *)&nnz, sizeof(int64_t));

    global_nrow = nrow;
    global_nnz = nnz;

    if (ncol > 65536)
    {
      std::cerr << "CSRMatrix: ncol=" << ncol
                << " exceeds the uint16_t index range of IndiceDataPair." << std::endl;
      exit(1);
    }

    indptr = new int64_t[nrow + 1];
    infile.read((char *)indptr, (nrow + 1) * sizeof(int64_t));
    indices_data = new IndiceDataPair[nnz];

    if (ori)
    {
      int32_t *indices = new int32_t[nnz];
      infile.read((char *)indices, nnz * sizeof(int32_t));

      for (int64_t i = 0; i < nnz; ++i)
      {
        indices_data[i].indice = static_cast<uint16_t>(indices[i]);
      }
      delete[] indices;

      float *data = new float[nnz];
      infile.read((char *)data, nnz * sizeof(float));
      infile.close();

      for (int64_t i = 0; i < nnz; ++i)
      {
        indices_data[i].data = static_cast<_Float16>(data[i]);
      }

      delete[] data;
    }
    else
    {
      // Raw layout read/write (see save()): only compatible with files saved
      // by a build using the same 4-byte IndiceDataPair.
      infile.read((char *)indices_data, nnz * sizeof(IndiceDataPair));
    }
  }

  CSRMatrix(int64_t nrow_, int64_t ncol_, int64_t nnz_, int64_t global_nrow_, int64_t global_nnz_, int64_t *indptr_, const int32_t *indices_, const float *data_)
  {
    nrow = nrow_;
    ncol = ncol_;
    nnz = nnz_;
    global_nrow = global_nrow_;
    global_nnz = global_nnz_;
    indptr = new int64_t[nrow + 1];

    indices_data = new IndiceDataPair[nnz];

    for (int64_t i = 0; i < nnz; ++i)
    {
      indices_data[i].indice = static_cast<uint16_t>(indices_[i]);
      indices_data[i].data = static_cast<_Float16>(data_[i]);
    }

    memcpy(indptr, indptr_, sizeof(uint64_t) * (nrow + 1));
  }

  // std::pair<uint32_t, uint32_t> get_row(uint32_t i)
  // {
  //   uint32_t start = indptr[i];
  //   uint32_t end = indptr[i + 1];
  //   return {start, end};
  // }

  // Deep copy: allocates its own indptr/indices_data so the copy can be
  // mutated (e.g. pruned) independently of the original.
  CSRMatrix(const CSRMatrix &other)
  {
    nrow = other.nrow;
    ncol = other.ncol;
    nnz = other.nnz;
    global_nrow = other.global_nrow;
    global_nnz = other.global_nnz;

    indptr = new int64_t[nrow + 1];
    memcpy(indptr, other.indptr, (nrow + 1) * sizeof(int64_t));

    indices_data = new IndiceDataPair[nnz];
    memcpy(indices_data, other.indices_data, nnz * sizeof(IndiceDataPair));
  }

  ~CSRMatrix()
  {
    delete[] indptr;
    delete[] indices_data;
  }

  void save(const std::string location)
  {
    std::ofstream outfile(location, std::ios::binary);
    outfile.write((char *)&nrow, sizeof(int64_t));
    outfile.write((char *)&ncol, sizeof(int64_t));
    outfile.write((char *)&nnz, sizeof(int64_t));

    outfile.write((char *)indptr, (nrow + 1) * sizeof(int64_t));

    outfile.write((char *)indices_data, nnz * sizeof(IndiceDataPair));
    outfile.close();
  }
};

#endif
