"""Convert BigANN .csr files to the Seismic/kANNolo .bin layout.
"""

import argparse

import numpy as np


def convert_csr_to_bin(csr_file_path, bin_file_path, chunk_rows=500_000):
    with open(csr_file_path, "rb") as f:
        n_rows, n_cols, nnz = np.fromfile(f, dtype=np.int64, count=3)
        print(f"n_rows = {n_rows}, n_cols = {n_cols}, nnz = {nnz}")
        indptr = np.fromfile(f, dtype=np.int64, count=n_rows + 1)
        indices_off = 24 + indptr.nbytes
    indices = np.memmap(csr_file_path, dtype=np.int32, mode="r", offset=indices_off, shape=nnz)
    data = np.memmap(csr_file_path, dtype=np.float32, mode="r", offset=indices_off + indices.nbytes, shape=nnz)

    with open(bin_file_path, "wb") as out:
        np.array([n_rows], dtype=np.uint32).tofile(out)

        for lo in range(0, n_rows, chunk_rows):
            hi = min(lo + chunk_rows, n_rows)
            ptr = indptr[lo : hi + 1] - indptr[lo]
            counts = np.diff(ptr).astype(np.int64)
            c_nnz = int(ptr[-1])
            c_ind = np.asarray(indices[indptr[lo] : indptr[hi]], dtype=np.uint32)
            c_val = np.asarray(data[indptr[lo] : indptr[hi]], dtype=np.float32)

            # Per row r: [count, ids..., values...] laid out in one u32 buffer.
            n = hi - lo
            buf = np.empty(n + 2 * c_nnz, dtype=np.uint32)
            starts = np.arange(n, dtype=np.int64) + 2 * ptr[:-1]  # row start in buf
            buf[starts] = counts.astype(np.uint32)
            within = np.arange(c_nnz, dtype=np.int64) - np.repeat(ptr[:-1], counts)
            pos_ids = np.repeat(starts + 1, counts) + within
            buf[pos_ids] = c_ind
            buf[pos_ids + np.repeat(counts, counts)] = c_val.view(np.uint32)
            buf.tofile(out)
            print(f"  rows {lo}..{hi} written ({c_nnz} nnz)")

    print(f"Wrote {bin_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BigANN .csr files to Seismic/kANNolo .bin files.")
    parser.add_argument("-file_path", type=str, required=True)
    parser.add_argument("-output_path", type=str, required=True)
    args = parser.parse_args()
    convert_csr_to_bin(args.file_path, args.output_path)
