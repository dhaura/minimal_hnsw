# Gate layouts, nq_splade (432 points)

Ratio = QPS / sep gate-off QPS at the same recall (same config and job).

| series | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |
|---|---|---|---|
| list16 T=10 | 58 | 1.158 [1.09, 1.26] | -0.35 |
| list16 off | 72 | 1.001 [0.99, 1.01] | +0.00 |
| list8 T=10 | 40 | 0.722 [0.64, 0.89] | -3.33 |
| list8 off | 72 | 1.001 [0.99, 1.01] | +0.00 |
| sep T=10 | 63 | 0.800 [0.75, 0.82] | +0.01 |

## Median matched-recall ratio by recall bin

| series | [0.90, 0.94) | [0.94, 0.96) | [0.96, 0.97) | [0.97, 0.98) | [0.98, 0.99) | [0.99, 1.00) |
|---|---|---|---|---|---|---|
| list16 T=10 | 1.17 | 1.17 | 1.13 | 1.17 | 1.10 | - |
| list16 off | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| list8 T=10 | 0.73 | 0.67 | 0.73 | 0.73 | - | - |
| list8 off | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sep T=10 | 0.75 | 0.77 | 0.80 | 0.82 | 0.81 | - |

## Gate decisions vs the ideal exact-overlap gate (all gated neighbors, all points)

Ideal gate: pass iff the neighbor's exact query-term overlap > T% of the query's terms. Useful: the neighbor would enter the result list (distance < current worst).

| series | checked | gate rejects % | ideal rejects % | false passes % of checked (% of ideal rejects) | false rejects % | useful % | useful rejected: gate % / ideal % |
|---|---|---|---|---|---|---|---|
| list16 T=10 | 1,192,692,706 | 40.3 | 27.8 | 10.3 (37.0) | 22.8 | 10.0 | 12.7 / 8.6 |
| list8 T=10 | 1,170,837,792 | 63.1 | 28.5 | 6.2 (21.8) | 40.8 | 12.1 | 35.8 / 9.2 |
| sep T=10 | 1,202,342,384 | 2.1 | 27.7 | 25.5 (92.3) | 0.0 | 9.3 | 0.4 / 8.3 |
