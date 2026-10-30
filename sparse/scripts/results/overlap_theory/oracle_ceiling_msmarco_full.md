# Lossless oracle vs overlap filters, msmarco_full

Break-even alpha* = 1/C = 28.8%. Oracle = skip every checked neighbor that would not enter the result list (no effect on the search); bytes vs no filter = 1 - o + 1/C.

| ef | 20 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| oracle skip share | 84.6% | 87.4% | 88.7% | 89.5% | 89.9% | 89.7% |
| oracle bytes | 0.44x | 0.41x | 0.40x | 0.39x | 0.39x | 0.39x |

## Share of unused neighbors caught (ideal / Bloom), and true top-10 at risk (overlap <= T)

| T | ef=20 | ef=50 | ef=100 | ef=200 | ef=400 | ef=800 | top-10 at risk |
|---|---|---|---|---|---|---|---|
| 0 | 2.0% / 0.0% | 2.1% / 0.0% | 2.5% / 0.0% | 3.0% / 0.0% | 3.8% / 0.1% | 4.8% / 0.1% | 0.0% |
| 2 | 3.5% / 0.1% | 3.6% / 0.1% | 4.4% / 0.1% | 5.4% / 0.1% | 6.7% / 0.1% | 8.4% / 0.1% | 0.0% |
| 5 | 8.8% / 0.4% | 9.3% / 0.4% | 10.9% / 0.4% | 13.1% / 0.5% | 15.7% / 0.6% | 18.8% / 0.7% | 0.1% |
| 10 | 20.8% / 2.5% | 21.9% / 2.6% | 25.0% / 3.1% | 28.5% / 3.7% | 32.5% / 4.4% | 37.2% / 5.2% | 0.6% |
| 15 | 34.9% / 8.5% | 36.7% / 9.0% | 40.7% / 10.3% | 45.0% / 11.9% | 49.7% / 13.8% | 54.9% / 15.9% | 2.0% |
| 20 | 50.7% / 19.4% | 52.8% / 20.5% | 57.1% / 22.9% | 61.6% / 25.8% | 66.3% / 28.8% | 71.1% / 32.2% | 5.2% |
| 25 | 65.2% / 33.5% | 67.3% / 35.2% | 71.2% / 38.6% | 75.2% / 42.2% | 79.2% / 46.1% | 83.1% / 50.1% | 10.9% |
| 30 | 76.7% / 48.5% | 78.5% / 50.6% | 81.7% / 54.2% | 84.9% / 58.1% | 87.8% / 62.1% | 90.5% / 66.1% | 19.6% |

## In-engine exact-overlap gate, ef=1600 (September runs; each run its own build)

Bytes estimate = scored-call ratio + 1/C (assumes the gate checks no more neighbors than the ungated run scores).

| gate | scored calls | vs no gate | recall | change (pp) | beta | build (s) | bytes est |
|---|---|---|---|---|---|---|---|
| off | 173,145,500 | 1.00x | 98.49% | +0.00 | 4 | 484 | 1.00x |
| 5 | 142,552,557 | 0.82x | 98.51% | +0.02 | 4 | 485 | 1.11x |
| 10 | 116,231,140 | 0.67x | 98.28% | -0.20 | 4 | 484 | 0.96x |
| 15 | 88,815,907 | 0.51x | 97.95% | -0.54 | 4 | 488 | 0.80x |
