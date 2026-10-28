# Gate layouts, msmarco_full (432 points)

Ratio = QPS / sep gate-off QPS at the same recall (same config and job).

| series | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |
|---|---|---|---|
| inline T=10 | 52 | 1.038 [0.95, 1.07] | -0.00 |
| inline T=25 | 48 | 0.663 [0.56, 0.76] | -1.87 |
| inline off | 72 | 1.052 [1.00, 1.08] | +0.00 |
| sep T=10 | 52 | 0.758 [0.71, 0.81] | -0.00 |
| sep T=25 | 48 | 0.476 [0.42, 0.57] | -1.87 |

## Median matched-recall ratio by recall bin

| series | [0.90, 0.94) | [0.94, 0.96) | [0.96, 0.97) | [0.97, 0.98) | [0.98, 0.99) |
|---|---|---|---|---|---|
| inline T=10 | 0.95 | 1.00 | 1.05 | 1.05 | 1.04 |
| inline T=25 | 0.75 | 0.66 | 0.59 | 0.56 | - |
| inline off | 1.05 | 1.08 | 1.07 | 1.02 | 0.96 |
| sep T=10 | 0.71 | 0.72 | 0.76 | 0.81 | 0.80 |
| sep T=25 | 0.53 | 0.47 | 0.44 | 0.42 | - |

## Gate decisions vs the ideal exact-overlap gate (all gated neighbors, all points)

Ideal gate: pass iff the neighbor's exact query-term overlap > T% of the query's terms. Useful: the neighbor would enter the result list (distance < current worst).

| series | checked | gate rejects % | ideal rejects % | false passes % of checked (% of ideal rejects) | false rejects % | useful % | useful rejected: gate % / ideal % |
|---|---|---|---|---|---|---|---|
| inline T=10 | 1,668,235,272 | 3.8 | 28.6 | 24.8 (86.9) | 0.0 | 12.0 | 0.8 / 9.7 |
| inline T=25 | 1,708,448,254 | 42.1 | 74.0 | 31.8 (43.0) | 0.0 | 14.3 | 27.2 / 55.5 |
| sep T=10 | 1,668,235,272 | 3.8 | 28.6 | 24.8 (86.9) | 0.0 | 12.0 | 0.8 / 9.7 |
| sep T=25 | 1,708,448,254 | 42.1 | 74.0 | 31.8 (43.0) | 0.0 | 14.3 | 27.2 / 55.5 |
