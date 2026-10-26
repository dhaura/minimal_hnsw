# Bloom gate on Grace vs Perlmutter - nq_splade

Grace: Intel Xeon Gold 6248R (Cascade Lake), 48 threads, 15,330 points, 210 configs.
Perlmutter reference: AMD EPYC 7763 (Zen3), 64 threads.

## 1. Matched-recall QPS ratio (filter on vs filter off, same config)

| T | points | Grace ratio median [p10, p90] | Perlmutter ratio | Grace - PM | recall change (pp) |
|---|---|---|---|---|---|
| 0 | 2190 | 0.769 [0.72, 0.79] | ~0.755 | +0.014 | +0.00 |
| 5 | 1976 | 0.771 [0.73, 0.79] | ~0.755 | +0.016 | +0.00 |
| 10 | 1691 | 0.779 [0.74, 0.80] | ~0.755 | +0.024 | -0.00 |
| 15 | 1331 | 0.766 [0.64, 0.81] | 0.760 | +0.006 | -0.08 |
| 20 | 1068 | 0.659 [0.42, 0.78] | 0.690 | -0.031 | -0.41 |
| 25 | 716 | 0.458 [0.27, 0.70] | 0.490 | -0.032 | -1.63 |

## 2. Best filter-on vs best filter-off QPS per recall bin

Perlmutter reference range for this ratio: 0.64-0.86x.

| recall | winning T | filter-on QPS | filter-off QPS | Grace ratio |
|---|---|---|---|---|
| 0.80 | 20 | 219,719 | 263,532 | 0.834 |
| 0.85 | 20 | 179,194 | 224,564 | 0.798 |
| 0.90 | 15 | 154,259 | 197,573 | 0.781 |
| 0.91 | 15 | 151,870 | 188,459 | 0.806 |
| 0.92 | 15 | 131,826 | 168,284 | 0.783 |
| 0.93 | 15 | 114,692 | 144,055 | 0.796 |
| 0.94 | 15 | 96,980 | 122,620 | 0.791 |
| 0.95 | 10 | 76,176 | 98,634 | 0.772 |
| 0.96 | 15 | 62,137 | 77,267 | 0.804 |
| 0.97 | 10 | 44,955 | 57,144 | 0.787 |
| 0.98 | 10 | 21,354 | 27,301 | 0.782 |
