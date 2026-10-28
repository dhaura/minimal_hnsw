# Gate layouts, nq_splade (432 points)

Ratio = QPS / sep gate-off QPS at the same recall (same config and job).

| series | points | QPS ratio median [p10, p90] | recall change at same ef (pp) |
|---|---|---|---|
| inline T=10 | 60 | 0.834 [0.82, 0.85] | -0.00 |
| inline T=25 | 52 | 0.612 [0.53, 0.77] | -1.18 |
| inline off | 72 | 0.964 [0.93, 0.99] | +0.00 |
| sep T=10 | 60 | 0.810 [0.77, 0.82] | -0.00 |
| sep T=25 | 52 | 0.577 [0.51, 0.69] | -1.18 |

## Median matched-recall ratio by recall bin

| series | [0.90, 0.94) | [0.94, 0.96) | [0.96, 0.97) | [0.97, 0.98) | [0.98, 0.99) | [0.99, 1.00) |
|---|---|---|---|---|---|---|
| inline T=10 | 0.85 | 0.85 | 0.84 | 0.83 | 0.82 | 0.82 |
| inline T=25 | 0.77 | 0.70 | 0.59 | 0.57 | 0.53 | - |
| inline off | 0.98 | 0.99 | 0.99 | 0.95 | 0.93 | 0.93 |
| sep T=10 | 0.76 | 0.78 | 0.80 | 0.82 | 0.81 | 0.81 |
| sep T=25 | 0.68 | 0.64 | 0.57 | 0.55 | 0.51 | - |

## Gate decisions vs the ideal exact-overlap gate (all gated neighbors, all points)

Ideal gate: pass iff the neighbor's exact query-term overlap > T% of the query's terms. Useful: the neighbor would enter the result list (distance < current worst).

| series | checked | gate rejects % | ideal rejects % | false passes % of checked (% of ideal rejects) | false rejects % | useful % | useful rejected: gate % / ideal % |
|---|---|---|---|---|---|---|---|
| inline T=10 | 1,202,069,364 | 2.1 | 27.7 | 25.5 (92.3) | 0.0 | 9.3 | 0.4 / 8.3 |
| inline T=25 | 1,220,086,812 | 33.9 | 76.8 | 42.9 (55.9) | 0.0 | 10.5 | 19.1 / 56.0 |
| sep T=10 | 1,202,069,364 | 2.1 | 27.7 | 25.5 (92.3) | 0.0 | 9.3 | 0.4 / 8.3 |
| sep T=25 | 1,220,086,812 | 33.9 | 76.8 | 42.9 (55.9) | 0.0 | 10.5 | 19.1 / 56.0 |
