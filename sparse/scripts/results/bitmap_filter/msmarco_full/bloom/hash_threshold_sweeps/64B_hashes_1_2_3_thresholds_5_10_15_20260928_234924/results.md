MS MARCO Full, 64-byte Bloom filters from pruned documents; M=32, efC=200, ef=1600, alpha=0.85, 64 threads. Recall/time/QPS are from counter-free searches; rejection statistics are from separate audit searches on the same graph, with identical predictions. Missed rejection rate = exact-only rejections / all exact rejections.

| Hashes | Bitmap bytes | Threshold (%) | Recall (%) | Search time (s) | QPS | Skipped candidates | Exact rejections | Missed rejection (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 5 | 98.5057 | 1.0867 | 6,423.11 | 1,336,793 | 32,886,413 | 95.9351 |
| 1 | 64 | 10 | 98.4957 | 1.0984 | 6,354.53 | 9,133,880 | 63,271,129 | 85.5639 |
| 1 | 64 | 15 | 98.4169 | 1.0475 | 6,663.24 | 27,391,605 | 92,115,627 | 70.2639 |
| 2 | 64 | 5 | 98.5057 | 1.0979 | 6,357.77 | 9,828,710 | 32,662,884 | 69.9086 |
| 2 | 64 | 10 | 98.4327 | 1.0561 | 6,609.20 | 31,979,549 | 62,444,038 | 48.7869 |
| 2 | 64 | 15 | 98.2436 | 0.9618 | 7,257.51 | 59,765,274 | 91,757,177 | 34.8658 |
| 3 | 64 | 5 | 98.4914 | 1.1120 | 6,276.96 | 15,469,474 | 32,366,625 | 52.2055 |
| 3 | 64 | 10 | 98.3754 | 1.0648 | 6,555.36 | 40,598,468 | 61,721,751 | 34.2234 |
| 3 | 64 | 15 | 98.1791 | 0.9810 | 7,115.41 | 69,449,142 | 91,368,682 | 23.9902 |
