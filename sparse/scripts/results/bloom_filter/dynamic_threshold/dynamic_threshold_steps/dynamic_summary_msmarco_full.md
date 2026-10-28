# Per-query threshold schedules, msmarco_full (576 points)

Percentages are of the search after the result list fills (a query runs about ef expansions there). Ratio = QPS / gate-off QPS at the same recall (same config).

| group | schedule | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |
|---|---|---|---|---|
| fixed T | fixed T=10 | 63 | 0.747 [0.73, 0.78] | +0.01 |
| off early, on late | no filter for first 50%, then T=25 | 58 | 0.908 [0.88, 0.99] | -0.24 |
| off early, on late | no filter for first 25%, then T=20 | 58 | 0.808 [0.79, 0.90] | -0.20 |
| stepped (3+ steps) | no filter for first 50%, then T=20, T=25 from 67%, T=30 from 83% | 58 | 0.914 [0.89, 0.99] | -0.23 |
| stepped (3+ steps) | no filter for first 50%, then T=25, T=30 from 75% | 58 | 0.909 [0.88, 0.99] | -0.31 |
| stepped (3+ steps) | no filter for first 25%, then T=10, T=15 from 50%, T=20 from 75% | 56 | 0.819 [0.81, 0.87] | -0.06 |
| stepped (3+ steps) | no filter for first 25%, then T=15, T=20 from 50%, T=25 from 75% | 58 | 0.819 [0.80, 0.88] | -0.19 |

## Best QPS at recall >= r (best series per group by median ratio)

| recall | gate off | fixed T: fixed T=10 | off early, on late: no filter for first 50%, then T=25 | stepped (3+ steps): no filter for first 50%, then T=20, T=25 from 67%, T=30 from 83% |
|---|---|---|---|---|
| 0.85 | 351,868 | 256,316 | 331,403 | 343,453 |
| 0.90 | 351,868 | 256,316 | 331,403 | 343,453 |
| 0.92 | 301,825 | 223,503 | 294,291 | 295,137 |
| 0.94 | 179,813 | 132,335 | 176,669 | 178,462 |
| 0.95 | 178,503 | 131,287 | 138,057 | 138,407 |
| 0.96 | 139,063 | 102,435 | 98,292 | 99,208 |
| 0.97 | 76,939 | 59,585 | 59,366 | 58,867 |
| 0.98 | 46,980 | 36,555 | 46,664 | 46,693 |
