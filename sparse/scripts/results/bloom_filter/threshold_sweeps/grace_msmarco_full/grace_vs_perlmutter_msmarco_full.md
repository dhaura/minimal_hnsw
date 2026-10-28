# Bloom gate on Grace vs Perlmutter - msmarco_full

Grace: Intel Xeon Gold 6248R (Cascade Lake), 48 threads, 15,936 points, 183 configs.
Perlmutter reference: AMD EPYC 7763 (Zen3), 64 threads.

## 1. Matched-recall QPS ratio (filter on vs filter off, same config)

| T | points | Grace ratio median [p10, p90] | Perlmutter ratio | Grace - PM | recall change (pp) |
|---|---|---|---|---|---|
| 5 | 1748 | 0.772 [0.75, 0.80] | ~0.750 | +0.022 | +0.00 |
| 10 | 1330 | 0.782 [0.76, 0.81] | 0.759 | +0.023 | -0.01 |
| 15 | 1152 | 0.749 [0.65, 0.81] | 0.720 | +0.029 | -0.15 |
| 20 | 1045 | 0.657 [0.42, 0.80] | 0.610 | +0.047 | -0.68 |
| 25 | 787 | 0.495 [0.24, 0.75] | 0.450 | +0.045 | -2.14 |
| 30 | 501 | 0.367 [0.16, 0.65] | n/a | n/a | -5.42 |
| 35 | 297 | 0.311 [0.12, 0.54] | n/a | n/a | -11.54 |

## 2. Best filter-on vs best filter-off QPS per recall bin

Perlmutter reference range for this ratio: 0.67-0.85x.

| recall | winning T | filter-on QPS | filter-off QPS | Grace ratio |
|---|---|---|---|---|
| 0.80 | 15 | 262,042 | 335,948 | 0.780 |
| 0.85 | 20 | 228,545 | 269,863 | 0.847 |
| 0.90 | 10 | 200,506 | 262,465 | 0.764 |
| 0.91 | 15 | 180,848 | 229,620 | 0.788 |
| 0.92 | 20 | 167,478 | 195,694 | 0.856 |
| 0.93 | 20 | 160,741 | 195,694 | 0.821 |
| 0.94 | 20 | 151,443 | 176,463 | 0.858 |
| 0.95 | 15 | 144,078 | 176,463 | 0.816 |
| 0.96 | 10 | 128,182 | 165,615 | 0.774 |
| 0.97 | 10 | 109,550 | 137,667 | 0.796 |
| 0.98 | 15 | 73,144 | 91,114 | 0.803 |
| 0.99 | 10 | 39,372 | 50,966 | 0.773 |
