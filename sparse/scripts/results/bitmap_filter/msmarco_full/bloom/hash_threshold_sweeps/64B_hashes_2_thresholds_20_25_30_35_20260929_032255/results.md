MS MARCO Full, 64-byte Bloom filters from pruned documents; M=32, efC=200, ef=1600, alpha=0.85, 64 threads. Recall/time/QPS are from counter-free searches; rejection statistics are from separate audit searches on the same graph, with identical predictions. Missed rejection rate = exact-only rejections / all exact rejections.

| Hashes | Bitmap bytes | Threshold (%) | Recall (%) | Search time (s) | QPS | Skipped candidates | Exact rejections | Missed rejection (%) |
|---|---|---|---|---|---|---|---|---|
| 2 | 64 | 20 | 97.8381 | 0.8614 | 8,103.53 | 91,957,401 | 121,220,020 | 24.1401 |
| 2 | 64 | 25 | 97.3152 | 0.7743 | 9,014.19 | 124,199,115 | 147,224,508 | 15.6396 |
| 2 | 64 | 30 | 96.7063 | 0.7172 | 9,732.91 | 153,483,734 | 169,158,768 | 9.2665 |
| 2 | 64 | 35 | 95.8524 | 0.6854 | 10,183.32 | 180,558,598 | 189,607,226 | 4.7723 |
