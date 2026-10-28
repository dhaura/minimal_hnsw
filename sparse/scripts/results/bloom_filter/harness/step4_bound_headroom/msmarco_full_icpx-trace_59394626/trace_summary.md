# Lossless-gate trace: msmarco_full_icpx-trace_59394626

Calls traced after top_candidates filled, every 10th query. 'Within c x' = a hypothetical bound never more than c times the true score.

| ef | calls | perfect gate (%) | within 1.25x | 1.5x | 2x | 3x | 5x | 10x | 62-bucket sketch (%) | sketch violations |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 1,322,596 | 88.9 | 73.6 | 59.7 | 42.3 | 27.6 | 18.0 | 11.1 | 14.8 | 0 |
| 400 | 4,523,416 | 89.8 | 75.2 | 62.1 | 45.9 | 32.0 | 22.0 | 14.2 | 9.2 | 0 |
| 1600 | 15,994,846 | 89.3 | 76.0 | 64.5 | 50.1 | 37.0 | 26.9 | 18.4 | 5.2 | 0 |

One-cache-line head+tail designs (K exact heaviest terms, tail bounded by B bucket max codes; B=0 = one max code). Skip share of traced calls:

| ef | K | B | bytes | skipped (%) | violations |
|---|---|---|---|---|---|
| 100 | 20 | 0 | 63 | 3.0 | 0 |
| 100 | 16 | 12 | 62 | 6.5 | 0 |
| 100 | 12 | 24 | 62 | 10.4 | 0 |
| 100 | 8 | 36 | 62 | 13.2 | 0 |
| 100 | 4 | 48 | 62 | 14.8 | 0 |
| 400 | 20 | 0 | 63 | 1.4 | 0 |
| 400 | 16 | 12 | 62 | 3.5 | 0 |
| 400 | 12 | 24 | 62 | 5.9 | 0 |
| 400 | 8 | 36 | 62 | 7.7 | 0 |
| 400 | 4 | 48 | 62 | 9.0 | 0 |
| 1600 | 20 | 0 | 63 | 0.7 | 0 |
| 1600 | 16 | 12 | 62 | 1.7 | 0 |
| 1600 | 12 | 24 | 62 | 3.0 | 0 |
| 1600 | 8 | 36 | 62 | 4.0 | 0 |
| 1600 | 4 | 48 | 62 | 4.9 | 0 |
