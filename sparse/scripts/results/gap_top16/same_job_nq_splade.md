# Same-job comparison, nq_splade (10,680 points)

Best QPS at recall >= r. rerun = no filter, patience list, same jobs as the variants;
existing = the playground SparseHNSW front (older build, other jobs).

| recall | existing | rerun | rerun / existing | Gap / rerun | Top16 / rerun | Top16+Gap / rerun |
|---|---|---|---|---|---|---|
| 0.8 | 335,896 | 344,477 | 1.03 | 0.98 | 1.16 | 1.17 |
| 0.85 | 267,867 | 276,072 | 1.03 | 1.06 | 1.17 | 1.24 |
| 0.9 | 250,526 | 258,306 | 1.03 | 1.05 | 1.17 | 1.19 |
| 0.92 | 225,209 | 221,879 | 0.99 | 1.01 | 1.13 | 1.15 |
| 0.94 | 167,882 | 169,390 | 1.01 | 1.05 | 1.17 | 1.19 |
| 0.95 | 139,548 | 119,820 | 0.86 | 1.19 | 1.25 | 1.36 |
| 0.96 | 106,291 | 92,527 | 0.87 | 1.12 | 1.35 | 1.44 |
| 0.97 | 76,544 | 68,904 | 0.90 | 1.12 | 1.20 | 1.44 |
| 0.98 | 60,961 | 57,913 | 0.95 | 1.08 | 0.98 | 1.24 |
| 0.99 | 32,653 | 28,468 | 0.87 | 1.19 | 1.16 | 1.25 |
