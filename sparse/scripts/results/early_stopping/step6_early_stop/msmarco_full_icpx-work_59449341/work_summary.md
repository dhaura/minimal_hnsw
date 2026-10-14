# Early-stopping headroom: msmarco_full_icpx-work_59449341

Calls = layer-0 distance calls per query (mean). Oracle = stop each query right after the last member of its final top-k_hat was scored: identical predictions. Oracle QPS is an estimate (time assumed proportional to calls).

| ef | recall (%) | QPS | calls | after list filled (%) | oracle calls | top-10 oracle calls | oracle saving | oracle QPS (est.) |
|---|---|---|---|---|---|---|---|---|
| 50 | 93.65 | 175,888 | 1,122 | 95.5 | 887 | 454 | 1.27x | 222,610 |
| 100 | 96.03 | 103,772 | 2,009 | 95.0 | 1,181 | 546 | 1.70x | 176,550 |
| 200 | 97.27 | 58,050 | 3,690 | 94.6 | 1,567 | 656 | 2.35x | 136,689 |
| 400 | 98.00 | 33,175 | 6,895 | 94.2 | 2,119 | 813 | 3.25x | 107,952 |
| 800 | 98.43 | 18,070 | 13,038 | 93.9 | 3,035 | 1,056 | 4.30x | 77,642 |
| 1200 | 98.66 | 12,191 | 18,958 | 93.7 | 3,810 | 1,293 | 4.98x | 60,661 |
| 1600 | 98.81 | 9,199 | 24,734 | 93.5 | 4,467 | 1,443 | 5.54x | 50,930 |

Patience (stop after N expansions without improving the top-k_hat), measured QPS; 'vs fixed ef' interpolates the fixed-ef curve at the same recall (blank = outside its range).

| ef | patience | recall (%) | QPS | fixed-ef QPS at same recall | vs fixed ef |
|---|---|---|---|---|---|
| 400 | 16 | 94.67 | 152,111 | 144,771 | 1.05x |
| 400 | 32 | 95.93 | 114,445 | 106,893 | 1.07x |
| 400 | 64 | 96.95 | 81,097 | 69,639 | 1.16x |
| 400 | 128 | 97.59 | 54,964 | 47,113 | 1.17x |
| 400 | 256 | 97.95 | 38,109 | 34,869 | 1.09x |
| 800 | 16 | 94.67 | 145,787 | 144,771 | 1.01x |
| 800 | 32 | 95.93 | 108,365 | 106,893 | 1.01x |
| 800 | 64 | 96.95 | 77,731 | 69,639 | 1.12x |
| 800 | 128 | 97.63 | 52,088 | 45,758 | 1.14x |
| 800 | 256 | 98.13 | 34,708 | 28,830 | 1.20x |
| 1600 | 16 | 94.67 | 139,846 | 144,771 | 0.97x |
| 1600 | 32 | 95.93 | 104,755 | 106,893 | 0.98x |
| 1600 | 64 | 96.95 | 75,142 | 69,639 | 1.08x |
| 1600 | 128 | 97.63 | 51,882 | 45,709 | 1.14x |
| 1600 | 256 | 98.14 | 34,139 | 28,527 | 1.20x |

Per query at ef=1600 (6980 queries): oracle work / total work p10 0.02, median 0.06, p90 0.54, max 1.00.
