# Gate layouts, msmarco_full (432 points)

Ratio = QPS / sep gate-off QPS at the same recall (same config and job).

| series | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |
|---|---|---|---|
| list16 T=10 | 57 | 1.169 [1.08, 1.21] | -0.01 |
| list16 off | 72 | 0.999 [0.99, 1.01] | +0.00 |
| list8 T=10 | 58 | 1.229 [1.16, 1.31] | -0.33 |
| list8 off | 72 | 1.000 [0.99, 1.01] | +0.00 |
| sep T=10 | 55 | 0.812 [0.74, 0.84] | +0.01 |

## Median matched-recall ratio by recall bin

| series | [0.90, 0.94) | [0.94, 0.96) | [0.96, 0.97) | [0.97, 0.98) | [0.98, 0.99) |
|---|---|---|---|---|---|
| list16 T=10 | 1.07 | 1.12 | 1.17 | 1.20 | 1.21 |
| list16 off | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| list8 T=10 | 1.16 | 1.23 | 1.23 | 1.29 | 1.26 |
| list8 off | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 |
| sep T=10 | 0.74 | 0.76 | 0.81 | 0.83 | 0.82 |

## Gate decisions vs the ideal exact-overlap gate (all gated neighbors, all points)

Ideal gate: pass iff the neighbor's exact query-term overlap > T% of the query's terms. Useful: the neighbor would enter the result list (distance < current worst).

| series | checked | gate rejects % | ideal rejects % | false passes % of checked (% of ideal rejects) | false rejects % | useful % | useful rejected: gate % / ideal % |
|---|---|---|---|---|---|---|---|
| list16 T=10 | 1,665,422,639 | 22.4 | 28.6 | 14.2 (49.7) | 8.0 | 12.1 | 2.8 / 9.8 |
| list8 T=10 | 1,656,523,091 | 40.2 | 28.8 | 9.8 (34.1) | 21.3 | 12.8 | 12.0 / 10.1 |
| sep T=10 | 1,668,153,169 | 3.8 | 28.6 | 24.8 (86.8) | 0.0 | 12.0 | 0.8 / 9.7 |
