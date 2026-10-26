# Per-query threshold schedules, msmarco_full (1,224 points)

Percentages are of the search after the result list fills (a query runs about ef expansions there). Ratio = QPS / gate-off QPS at the same recall (same config).

| group | schedule | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |
|---|---|---|---|---|
| fixed T | fixed T=10 | 56 | 0.775 [0.74, 0.81] | -0.00 |
| fixed T | fixed T=20 | 58 | 0.674 [0.60, 0.73] | -0.61 |
| fixed T | fixed T=25 | 48 | 0.480 [0.43, 0.60] | -1.79 |
| fixed T | fixed T=30 | 34 | 0.310 [0.28, 0.42] | -4.45 |
| off early, on late | no filter for first 50%, then T=25 | 58 | 0.943 [0.91, 1.02] | -0.24 |
| off early, on late | no filter for first 50%, then T=20 | 58 | 0.937 [0.91, 0.99] | -0.10 |
| off early, on late | no filter for first 50%, then T=30 | 54 | 0.929 [0.90, 1.01] | -0.44 |
| off early, on late | no filter for first 50%, then T=35 | 52 | 0.908 [0.85, 1.01] | -0.79 |
| off early, on late | no filter for first 25%, then T=20 | 58 | 0.832 [0.81, 0.94] | -0.17 |
| off early, on late | no filter for first 25%, then T=25 | 58 | 0.821 [0.79, 0.97] | -0.43 |
| off early, on late | no filter for first 10%, then T=20 | 58 | 0.757 [0.73, 0.86] | -0.30 |
| off early, on late | no filter for first 25%, then T=30 | 48 | 0.750 [0.72, 0.98] | -0.87 |
| on early, off late | T=20 for first 10%, then no filter | 58 | 0.926 [0.86, 0.96] | -0.23 |
| on early, off late | T=20 for first 25%, then no filter | 58 | 0.868 [0.78, 0.90] | -0.36 |
| on early, off late | T=30 for first 10%, then no filter | 48 | 0.521 [0.49, 0.68] | -1.72 |
| on early, off late | T=30 for first 25%, then no filter | 46 | 0.466 [0.42, 0.58] | -2.51 |

## Best QPS at recall >= r (best series per group by median ratio)

| recall | gate off | fixed T: fixed T=10 | off early, on late: no filter for first 50%, then T=25 | on early, off late: T=20 for first 10%, then no filter |
|---|---|---|---|---|
| 0.85 | 350,824 | 259,268 | 336,986 | 349,805 |
| 0.90 | 350,824 | 259,268 | 336,986 | 349,805 |
| 0.92 | 303,597 | 224,857 | 294,019 | 302,492 |
| 0.94 | 181,813 | 135,911 | 183,515 | 182,703 |
| 0.95 | 180,385 | 135,389 | 143,743 | 143,121 |
| 0.96 | 140,127 | 106,510 | 102,432 | 100,433 |
| 0.97 | 76,866 | 60,828 | 78,525 | 77,453 |
| 0.98 | 45,699 | 38,316 | 48,451 | 44,729 |
