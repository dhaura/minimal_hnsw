# Filters vs no filter by recall (median QPS ratio per 0.01 recall bin, T=10)

## msmarco_full

| recall bin | top-16 terms per neighbor (32 B) | top-8 terms per neighbor (16 B) | Bloom, filter in the row | Bloom, separate filter |
|---|---|---|---|---|
| 0.92 | 1.05 | 1.14 | 0.93 | 0.73 |
| 0.93 | 1.08 | 1.17 | 0.96 | 0.74 |
| 0.94 | - | 1.22 | - | - |
| 0.95 | 1.13 | 1.23 | 1.00 | 0.76 |
| 0.96 | 1.17 | 1.23 | 1.05 | 0.81 |
| 0.97 | 1.20 | 1.29 | 1.05 | 0.83 |
| 0.98 | 1.21 | 1.26 | 1.04 | 0.82 |

## nq_splade

| recall bin | top-16 terms per neighbor (32 B) | top-8 terms per neighbor (16 B) | Bloom, filter in the row | Bloom, separate filter |
|---|---|---|---|---|
| 0.92 | 1.19 | 0.83 | - | 0.75 |
| 0.93 | 1.17 | 0.73 | 0.85 | 0.75 |
| 0.94 | - | 0.68 | 0.85 | 0.76 |
| 0.95 | 1.16 | 0.64 | 0.84 | 0.77 |
| 0.96 | 1.13 | 0.73 | 0.84 | 0.80 |
| 0.97 | 1.17 | 0.73 | 0.83 | 0.82 |
| 0.98 | 1.10 | - | 0.82 | 0.81 |
