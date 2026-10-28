# Same-job comparison, msmarco_full (9,624 points)

Best QPS at recall >= r. rerun = no filter, patience list, same jobs as the variants;
existing = the playground SparseHNSW front (older build, other jobs).

| recall | existing | rerun | rerun / existing | Gap / rerun | Top16 / rerun | Top16+Gap / rerun |
|---|---|---|---|---|---|---|
| 0.8 | 519,307 | 529,068 | 1.02 | 0.93 | 1.01 | 1.00 |
| 0.85 | 393,107 | 397,381 | 1.01 | 1.04 | 1.07 | 1.09 |
| 0.9 | 393,107 | 394,484 | 1.00 | 1.05 | 1.07 | 1.09 |
| 0.92 | 332,761 | 332,460 | 1.00 | 1.11 | 1.07 | 1.11 |
| 0.94 | 311,342 | 296,693 | 0.95 | 1.06 | 1.08 | 1.11 |
| 0.95 | 277,979 | 259,325 | 0.93 | 1.14 | 1.11 | 1.19 |
| 0.96 | 259,496 | 240,366 | 0.93 | 1.08 | 1.12 | 1.17 |
| 0.97 | 219,266 | 201,612 | 0.92 | 1.13 | 1.12 | 1.24 |
| 0.98 | 152,123 | 145,232 | 0.95 | 1.11 | 1.16 | 1.22 |
| 0.99 | 85,264 | 81,807 | 0.96 | 1.23 | 1.14 | 1.33 |
