# Lossless oracle vs overlap filters, nq_splade

Break-even alpha* = 1/C = 24.3%. Oracle = skip every checked neighbor that would not enter the result list (no effect on the search); bytes vs no filter = 1 - o + 1/C.

| ef | 20 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| oracle skip share | 88.5% | 90.7% | 91.4% | 91.9% | 92.0% | 91.8% |
| oracle bytes | 0.36x | 0.34x | 0.33x | 0.32x | 0.32x | 0.33x |

## Share of unused neighbors caught (ideal / Bloom), and true top-10 at risk (overlap <= T)

| T | ef=20 | ef=50 | ef=100 | ef=200 | ef=400 | ef=800 | top-10 at risk |
|---|---|---|---|---|---|---|---|
| 0 | 1.6% / 0.0% | 1.7% / 0.0% | 2.1% / 0.0% | 2.5% / 0.0% | 3.2% / 0.0% | 4.0% / 0.0% | 0.0% |
| 2 | 3.0% / 0.0% | 3.2% / 0.0% | 3.7% / 0.0% | 4.6% / 0.0% | 5.6% / 0.0% | 6.9% / 0.0% | 0.0% |
| 5 | 8.2% / 0.2% | 8.7% / 0.2% | 10.0% / 0.2% | 11.7% / 0.3% | 13.7% / 0.3% | 16.3% / 0.4% | 0.0% |
| 10 | 21.0% / 1.5% | 22.1% / 1.6% | 24.4% / 1.8% | 27.3% / 2.1% | 30.6% / 2.4% | 34.4% / 2.8% | 0.3% |
| 15 | 36.9% / 5.7% | 38.5% / 5.9% | 41.5% / 6.6% | 45.0% / 7.4% | 48.8% / 8.4% | 52.8% / 9.6% | 1.7% |
| 20 | 54.4% / 14.6% | 56.2% / 15.3% | 59.4% / 16.6% | 62.8% / 18.3% | 66.3% / 20.3% | 69.8% / 22.4% | 5.4% |
| 25 | 70.2% / 28.2% | 71.9% / 29.4% | 74.8% / 31.4% | 77.7% / 33.8% | 80.4% / 36.5% | 83.0% / 39.3% | 12.9% |
| 30 | 82.2% / 44.2% | 83.6% / 45.6% | 85.8% / 48.1% | 87.9% / 50.8% | 89.7% / 53.7% | 91.4% / 56.7% | 24.0% |

## In-engine exact-overlap gate, ef=1600 (September runs; each run its own build)

Bytes estimate = scored-call ratio + 1/C (assumes the gate checks no more neighbors than the ungated run scores).

| gate | scored calls | vs no gate | recall | change (pp) | beta | build (s) | bytes est |
|---|---|---|---|---|---|---|---|
| off | 122,257,457 | 1.00x | 98.09% | +0.00 | 5 | 216 | 1.00x |
| 5 | 102,240,498 | 0.84x | 98.19% | +0.10 | 4 | 6929 | 1.08x |
| 10 | 82,105,190 | 0.67x | 98.11% | +0.02 | 4 | 6929 | 0.91x |
| 15 | 60,951,021 | 0.50x | 97.82% | -0.27 | 4 | 6929 | 0.74x |
